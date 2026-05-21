#!/usr/bin/env python3
"""
build_kb.py  --  one-time, fully scripted Knowledge Base provisioning.

Run this script ONCE before the talk to create every AWS resource the demo
needs. When it finishes, it prints a block of IDs to copy into config.py.

What this script creates (in order):
  1. IAM role  -- a service role that Bedrock assumes to read S3 and write
                  to the vector store on your behalf.
  2. S3 Vectors bucket + index  -- the serverless vector store.  No hourly
                  charge; you pay only for storage and query calls.
  3. Bedrock Knowledge Base + S3 data source  -- connects the KB to your S3
                  corpus bucket so Bedrock can find the papers.
  4. Ingestion job  -- reads every paper from S3, splits it into 512-token
                  chunks, embeds each chunk with Titan Embed v2, and writes
                  the vectors into the S3 Vectors index.  Takes ~5 minutes.
  5. Bedrock Guardrail  -- a regex policy that intercepts any https:// URL
                  in model output and replaces it with a local corpus link.
  6. AgentCore Gateway + Cedar policy engine  -- demonstrates policy-based
                  tool access control.  The Cedar "ForbidWeb" policy denies
                  the web_fetch tool call that Q4 attempts.

Re-running safely:
  The script is idempotent for the IAM role and S3 Vectors steps: if the
  resource already exists, those steps print a notice and continue.
  The Knowledge Base, Guardrail, and Gateway are NOT idempotent -- running
  twice will create a second set.  If you need to re-run, run teardown.py
  first, then re-run this script.

>>> VERIFY BEFORE RUNNING <<<
S3 Vectors is a recent service (GA 2025).  The `s3vectors` boto3 client
method names and the `create_knowledge_base` storageConfiguration shape for
S3 Vectors are the newest APIs in this repo.  The two spots marked
``# VERIFY`` below should be checked against current AWS docs if you
encounter UnknownServiceException or validation errors.

What to do if something goes wrong:
  - "EntityAlreadyExistsException" on the IAM role: harmless, continuing.
  - "ResourceConflictException" on the vector bucket/index: harmless.
  - Ingestion job status "FAILED": check the KB data source in the console
    to see which documents failed (usually large PDFs or XML parse errors).
  - 400 / ValidationException on create_knowledge_base: the storageConfiguration
    shape changed -- check the ``# VERIFY`` comment in create_kb() below.
  - "AccessDeniedException" anywhere: your IAM user/role is missing permissions;
    see the trust policy and inline policy in create_kb_role().

Requires: boto3 (installed via uv pip install -e ".[dev]")
"""

import json
import time

import boto3

import config as cfg

# All three clients share the region from config.py.
iam = boto3.client("iam")
agent = boto3.client("bedrock-agent", region_name=cfg.REGION)

# VERIFY: "s3vectors" is the boto3 service name as of 2026-05; confirm it
# hasn't changed if you get an UnknownServiceException.
s3v = boto3.client("s3vectors", region_name=cfg.REGION)


# ----------------------------------------------------------------------
# 1. IAM role for the knowledge base
# ----------------------------------------------------------------------
def create_kb_role() -> str:
    """Create (or reuse) the IAM role that the Bedrock Knowledge Base assumes.

    The role needs three permissions:
      - s3:GetObject / s3:ListBucket on your corpus bucket (to read papers)
      - bedrock:InvokeModel on the embedding model (to embed chunks)
      - s3vectors:* (to write and read vectors)

    AWS IAM propagation takes ~10 seconds after role creation; we sleep to
    avoid a ConditionFailure when create_knowledge_base is called right after.

    Returns:
        The role ARN (e.g. "arn:aws:iam::123456789012:role/inside-the-lines-kb-role").
    """
    # Trust policy: only the bedrock.amazonaws.com service can assume this role.
    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "bedrock.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    try:
        arn = iam.create_role(
            RoleName=cfg.KB_ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="Inside the Lines demo -- Bedrock KB execution role",
        )["Role"]["Arn"]
    except iam.exceptions.EntityAlreadyExistsException:
        # Role was created on a previous run -- that's fine, reuse it.
        arn = iam.get_role(RoleName=cfg.KB_ROLE_NAME)["Role"]["Arn"]

    # Inline policy: grant the minimum permissions Bedrock needs.
    # s3vectors:* is broad -- narrow to s3vectors:PutVectors / GetVectors /
    # ListVectors once you confirm the exact action names in the IAM reference.
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:ListBucket"],
                "Resource": [f"arn:aws:s3:::{cfg.BUCKET}", f"arn:aws:s3:::{cfg.BUCKET}/*"],
            },
            {
                # Bedrock must call the embedding model to vectorise each chunk.
                "Effect": "Allow",
                "Action": "bedrock:InvokeModel",
                "Resource": (
                    f"arn:aws:bedrock:{cfg.REGION}::foundation-model/{cfg.EMBED_MODEL_ID}"
                ),
            },
            # VERIFY: scope s3vectors actions per current S3 Vectors IAM reference.
            # As of 2026-05-20, "s3vectors:*" is the safe catch-all.
            {"Effect": "Allow", "Action": "s3vectors:*", "Resource": "*"},
        ],
    }
    iam.put_role_policy(
        RoleName=cfg.KB_ROLE_NAME, PolicyName="kb-access", PolicyDocument=json.dumps(policy)
    )
    print(f"  IAM role: {arn}")

    # IAM roles take ~10 seconds to propagate globally.  If create_knowledge_base
    # is called immediately, it may fail with "Role cannot be assumed."
    time.sleep(10)
    return arn


# ----------------------------------------------------------------------
# 2. S3 Vectors bucket + index
# ----------------------------------------------------------------------
def create_vector_store() -> str:
    """Create an S3 Vectors bucket and a vector index inside it.

    S3 Vectors is the vector store backing the Bedrock Knowledge Base.
    It is serverless -- no hourly charge, no OpenSearch collection.
    You pay per GB stored and per 1,000 query calls.

    The ``nonFilterableMetadataKeys`` configuration is critical:
    Bedrock KB attaches several metadata fields to each vector during
    ingestion.  Some of these fields (e.g. AMAZON_BEDROCK_TEXT, which
    holds the raw chunk text) are large enough to push the filterable
    metadata payload over S3 Vectors' 2,048-byte limit, causing ingestion
    to fail with a validation error.  Marking them non-filterable excludes
    them from the index metadata while still storing them on the vector
    object -- so the KB can still retrieve the text.
    (Verified 2026-05-20.)

    Returns:
        The ARN of the created vector index.
    """
    # VERIFY: confirm create_vector_bucket / create_index method names match
    # the current s3vectors boto3 client.
    try:
        s3v.create_vector_bucket(vectorBucketName=cfg.VECTOR_BUCKET_NAME)
        print(f"  vector bucket: {cfg.VECTOR_BUCKET_NAME}")
    except Exception as e:  # noqa: BLE001 -- already-exists is fine
        # ResourceConflictException = bucket already exists from a previous run.
        print(f"  vector bucket: {type(e).__name__} (continuing)")

    # All Bedrock KB metadata keys must be non-filterable.  If ANY of these
    # are left filterable, ingestion will fail with a metadata size error.
    _NON_FILTERABLE = [
        "x-amz-bedrock-kb-source-uri",
        "x-amz-bedrock-kb-chunk-id",
        "x-amz-bedrock-kb-data-source-id",
        "x-amz-bedrock-kb-document-id",
        "x-amz-bedrock-kb-index-id",
        "x-amz-bedrock-kb-knowledge-base-id",
        "AMAZON_BEDROCK_TEXT",  # raw chunk text -- large, must be non-filterable
        "AMAZON_BEDROCK_METADATA",  # source metadata -- also large
    ]
    try:
        s3v.create_index(
            vectorBucketName=cfg.VECTOR_BUCKET_NAME,
            indexName=cfg.VECTOR_INDEX_NAME,
            dataType="float32",
            dimension=cfg.EMBED_DIM,  # 1024 for Titan Embed v2
            distanceMetric="cosine",  # cosine similarity for semantic search
            metadataConfiguration={"nonFilterableMetadataKeys": _NON_FILTERABLE},
        )
        print(f"  vector index: {cfg.VECTOR_INDEX_NAME}")
    except Exception as e:  # noqa: BLE001
        print(f"  vector index: {type(e).__name__} (continuing)")

    # The index ARN is what create_knowledge_base needs in storageConfiguration.
    return (
        f"arn:aws:s3vectors:{cfg.REGION}:{cfg.ACCOUNT_ID}:"
        f"bucket/{cfg.VECTOR_BUCKET_NAME}/index/{cfg.VECTOR_INDEX_NAME}"
    )


# ----------------------------------------------------------------------
# 3. Knowledge base + data source
# ----------------------------------------------------------------------
def create_kb(role_arn: str, index_arn: str) -> tuple[str, str]:
    """Create the Bedrock Knowledge Base and attach an S3 data source.

    The Knowledge Base is the logical container Bedrock uses for retrieval.
    It links three things together:
      - the IAM role (so it can call S3 and the embed model)
      - the vector index (where it stores and searches embeddings)
      - the data source (the S3 prefix where the papers live)

    Chunking strategy: fixed-size, 512 tokens, 15% overlap.  This gives
    Bedrock enough context per chunk without making chunks so large that
    retrieval dilutes relevance.

    Returns:
        (kb_id, data_source_id) -- paste both into config.py.

    Raises:
        ValidationException: most likely a storageConfiguration shape mismatch.
            Check the ``# VERIFY`` comment and the current boto3 docs.
    """
    embed_arn = f"arn:aws:bedrock:{cfg.REGION}::foundation-model/{cfg.EMBED_MODEL_ID}"

    # VERIFY: storageConfiguration shape for S3 Vectors against current boto3.
    # As of 2026-05-20, type "S3_VECTORS" with s3VectorsConfiguration.indexArn works.
    kb = agent.create_knowledge_base(
        name=cfg.KB_NAME,
        roleArn=role_arn,
        knowledgeBaseConfiguration={
            "type": "VECTOR",
            "vectorKnowledgeBaseConfiguration": {"embeddingModelArn": embed_arn},
        },
        storageConfiguration={
            "type": "S3_VECTORS",
            "s3VectorsConfiguration": {"indexArn": index_arn},
        },
    )
    kb_id = kb["knowledgeBase"]["knowledgeBaseId"]

    # The data source tells Bedrock which S3 prefix to read papers from.
    # inclusionPrefixes limits ingestion to corpus/ -- other S3 objects are ignored.
    ds = agent.create_data_source(
        knowledgeBaseId=kb_id,
        name="pmc-corpus",
        dataSourceConfiguration={
            "type": "S3",
            "s3Configuration": {
                "bucketArn": f"arn:aws:s3:::{cfg.BUCKET}",
                "inclusionPrefixes": [cfg.CORPUS_PREFIX],
            },
        },
        vectorIngestionConfiguration={
            "chunkingConfiguration": {
                "chunkingStrategy": "FIXED_SIZE",
                "fixedSizeChunkingConfiguration": {
                    "maxTokens": 512,  # tokens per chunk
                    "overlapPercentage": 15,  # 15% overlap so sentences don't split cold
                },
            }
        },
    )
    return kb_id, ds["dataSource"]["dataSourceId"]


# ----------------------------------------------------------------------
# 4. Ingestion
# ----------------------------------------------------------------------
def ingest(kb_id: str, ds_id: str) -> None:
    """Start an ingestion job and poll until it completes or fails.

    The ingestion job reads every .txt file under corpus/ in S3, splits each
    into chunks, calls Titan Embed v2 to produce a 1024-float32 vector per
    chunk, and writes all vectors to the S3 Vectors index.

    For ~650 papers this typically takes 5--10 minutes.  The script polls
    every 15 seconds and prints the current status.

    Note: the ingestion statistics only report document counts -- there is
    NO token count in the API response.  Ingestion cost is computed in
    estimate_ingestion_cost() from the local corpus character count instead.
    (Verified 2026-05-20.)

    Args:
        kb_id: the Knowledge Base ID returned by create_kb().
        ds_id: the data source ID returned by create_kb().
    """
    job_id = agent.start_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id)["ingestionJob"][
        "ingestionJobId"
    ]
    print("  ingestion started -- embedding + indexing every paper...")

    while True:
        st = agent.get_ingestion_job(
            knowledgeBaseId=kb_id, dataSourceId=ds_id, ingestionJobId=job_id
        )["ingestionJob"]
        print(f"    status: {st['status']}")

        if st["status"] in ("COMPLETE", "FAILED"):
            if st["status"] == "COMPLETE":
                s = st.get("statistics", {})
                print(
                    f"    indexed {s.get('numberOfNewDocumentsIndexed')} "
                    f"of {s.get('numberOfDocumentsScanned')} documents"
                )
            break
        time.sleep(15)


# ----------------------------------------------------------------------
# 5. Bedrock Guardrail
# ----------------------------------------------------------------------
def create_guardrail() -> tuple[str, str]:
    """Create a Bedrock Guardrail that anonymises external URLs in model output.

    The demo's system prompts instruct models to cite papers as full
    PubMed Central URLs (https://www.ncbi.nlm.nih.gov/pmc/articles/PMCxxx/).
    The guardrail intercepts those URLs before they reach the browser:
      - it anonymises every https:// URL, replacing it with {EXTERNAL_URL}
      - agent.py inspects the guardrail trace and substitutes a local
        corpus link (/corpus/PMCxxx) for any PMC article we have locally,
        or "[link removed]" for anything else.

    This demonstrates Bedrock Guardrails enforcing a "no external data
    egress" policy -- the model generates rich cited output but no actual
    external URLs reach the audience's browser.

    Returns:
        (guardrail_id, version) -- paste both into config.py.
    """
    br = boto3.client("bedrock", region_name=cfg.REGION)
    resp = br.create_guardrail(
        name="inside-the-lines-url-guard",
        description="Intercept external URLs in model output for the Inside the Lines demo.",
        sensitiveInformationPolicyConfig={
            "regexesConfig": [
                {
                    "name": "EXTERNAL_URL",
                    "description": "Matches any http/https URL in model output",
                    "pattern": r"https?://[^\s\)\]\"']+",
                    # ANONYMIZE replaces the matched text with {EXTERNAL_URL}
                    # in the model response.  agent.py re-processes those tokens.
                    "action": "ANONYMIZE",
                }
            ]
        },
        # These messages appear only if a guardrail *blocks* a request outright.
        # For this demo we only ANONYMIZE (not block), so these are fallbacks.
        blockedInputMessaging="Input blocked by guardrail.",
        blockedOutputsMessaging="Output blocked by guardrail.",
    )
    guardrail_id = resp["guardrailId"]
    version = resp.get("version", "DRAFT")
    print(f"  guardrail: {guardrail_id}  version: {version}")
    return guardrail_id, version


# ----------------------------------------------------------------------
# 6. AgentCore Gateway with Cedar policy engine
# ----------------------------------------------------------------------
def create_gateway() -> dict:
    """Create an AgentCore Gateway with a Cedar policy engine attached.

    The Gateway is the access-control layer for tool calls.  In this demo
    it intercepts Q4's attempt to call web_fetch and denies it, then the
    agent falls back to the knowledge base.  This demonstrates Cedar
    policy-based guardrails at the tool level.

    Steps:
      a) IAM role for the gateway (service principal bedrock-agentcore.amazonaws.com)
      b) PolicyEngine -- the Cedar evaluation service (waits for ACTIVE)
      c) Gateway -- MCP protocol, no user-level authoriser (waits for READY)
      d) Cedar policies -- created after gateway so the real ARN can be validated
      e) Attach policy engine to gateway in ENFORCE mode (waits for READY)

    Cedar policy quirks (verified 2026-05-21):
      - Tool actions use the format "{target-name}___{tool-name}" (three underscores).
        Our gateway target is named "web-tools", so the Cedar action for web_fetch
        is AgentCore::Action::"web-tools___web_fetch".
      - A Cedar policy denial comes back as HTTP 200 with a JSON-RPC error body
        ("Tool Execution Denied: ..."), NOT as HTTP 403.  query_gateway() in aws.py
        checks for that pattern.
      - The update_gateway() call that attaches the engine must repeat all the
        original create_gateway() fields (name, roleArn, authorizerType) -- it is
        a full replacement, not a patch.

    Returns:
        dict with gateway_id, gateway_url, engine_id -- paste into config.py.
    """
    br_ctrl = boto3.client("bedrock-agentcore-control", region_name=cfg.REGION)

    ROLE_NAME = "inside-the-lines-gateway-role"
    ROLE_ARN = f"arn:aws:iam::{cfg.ACCOUNT_ID}:role/{ROLE_NAME}"

    # a. IAM role -- lets the Gateway service assume permissions to call the
    #    Cedar evaluation endpoint and read policies.
    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    try:
        iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="Inside the Lines demo -- AgentCore Gateway role",
        )
        print(f"  IAM role created: {ROLE_ARN}")
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"  IAM role exists: {ROLE_ARN}")

    # The gateway role only needs read access to its own policy engine and policies.
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock-agentcore:GetPolicyEngine",
                    "bedrock-agentcore:GetPolicy",
                    "bedrock-agentcore:ListPolicies",
                    "bedrock-agentcore:GetGatewayTarget",
                    "bedrock-agentcore:ListGatewayTargets",
                ],
                "Resource": "*",
            }
        ],
    }
    iam.put_role_policy(
        RoleName=ROLE_NAME, PolicyName="gateway-access", PolicyDocument=json.dumps(policy)
    )
    # Wait for IAM propagation before creating the gateway.
    time.sleep(8)

    # b. PolicyEngine -- the Cedar evaluation service instance.
    #    Wait up to 60 seconds for it to reach ACTIVE.
    pe = br_ctrl.create_policy_engine(name="InsideTheLinesEngine")
    engine_id = pe["policyEngineId"]
    engine_arn = pe["policyEngineArn"]
    print(f"  policy engine: {engine_id}")
    for _ in range(20):
        if br_ctrl.get_policy_engine(policyEngineId=engine_id)["status"] == "ACTIVE":
            break
        time.sleep(3)

    # c. Gateway -- MCP protocol, no custom authoriser.
    #    authorizerType="NONE" means any caller with the right IAM perms can invoke.
    #    Cedar policies handle the fine-grained tool-level control.
    gw = br_ctrl.create_gateway(
        name=cfg.GATEWAY_NAME, roleArn=ROLE_ARN, protocolType="MCP", authorizerType="NONE"
    )
    gateway_id = gw["gatewayId"]
    gateway_url = gw["gatewayUrl"]
    gateway_arn = f"arn:aws:bedrock-agentcore:{cfg.REGION}:{cfg.ACCOUNT_ID}:gateway/{gateway_id}"
    print(f"  gateway: {gateway_id}")
    for _ in range(20):
        if br_ctrl.get_gateway(gatewayIdentifier=gateway_id)["status"] == "READY":
            break
        time.sleep(5)

    # d. Cedar policies -- created after the gateway exists so the real gateway
    #    ARN can be embedded in the policy resource principal.
    #    We use a short timestamp suffix to avoid "already exists" errors if
    #    this script is run again without a full teardown.
    ts = str(int(time.time()))[-6:]

    # PermitAll: baseline allow for all principals, actions, and resources.
    permit_cedar = f'permit(principal, action, resource == AgentCore::Gateway::"{gateway_arn}");'

    # ForbidWeb: deny the web_fetch tool call specifically.
    # toolName in the Cedar context is the tool name WITHOUT the target prefix --
    # i.e., "web_fetch", not "web-tools___web_fetch".
    forbid_cedar = (
        f'forbid(principal, action == AgentCore::Action::"InvokeTool", '
        f'resource == AgentCore::Gateway::"{gateway_arn}") '
        f'when {{ context.toolName == "web_fetch" }};'
    )

    for name, stmt in [(f"PermitAll{ts}", permit_cedar), (f"ForbidWeb{ts}", forbid_cedar)]:
        br_ctrl.create_policy(
            name=name,
            policyEngineId=engine_id,
            definition={"cedar": {"statement": stmt}},
            # IGNORE_ALL_FINDINGS: skip Cedar schema validation.
            # Use WARN or ERROR in production for stricter policy checking.
            validationMode="IGNORE_ALL_FINDINGS",
        )
        print(f"  policy: {name}")

    # e. Attach the policy engine to the gateway in ENFORCE mode.
    #    ENFORCE means Cedar denials block the call.  MONITOR would log
    #    without blocking (useful for testing new policies).
    #    update_gateway() requires ALL original create_gateway() fields --
    #    it replaces the whole resource, not just the policyEngineConfiguration.
    br_ctrl.update_gateway(
        gatewayIdentifier=gateway_id,
        name=cfg.GATEWAY_NAME,
        roleArn=ROLE_ARN,
        authorizerType="NONE",
        policyEngineConfiguration={"arn": engine_arn, "mode": "ENFORCE"},
    )
    for _ in range(20):
        if br_ctrl.get_gateway(gatewayIdentifier=gateway_id)["status"] == "READY":
            break
        time.sleep(5)
    print("  gateway READY with Cedar policy engine in ENFORCE mode")

    return {"gateway_id": gateway_id, "gateway_url": gateway_url, "engine_id": engine_id}


def estimate_ingestion_cost() -> None:
    """Print an ingestion cost estimate based on the local corpus directory.

    The Bedrock ingestion API reports document counts but NOT token counts,
    so we cannot use the API response to compute cost.  Instead we walk the
    local corpus/ directory, sum raw character counts, and approximate the
    token count at 4 chars/token (a rough but consistent heuristic for
    English biomedical text).

    The printed value should be pasted into config.py as
    INGESTION_COST_ESTIMATE so the UI can display it alongside run costs.

    If corpus/ is not present (e.g. you ran this on a different machine
    than the one that ran corpus_fetch.py), the estimate is skipped.
    """
    import os

    corpus_dir = "corpus"
    if not os.path.isdir(corpus_dir):
        print("  (corpus/ not found -- skipping ingestion cost estimate)")
        return

    total_chars = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _dirs, files in os.walk(corpus_dir)
        for f in files
    )
    approx_tokens = total_chars / 4  # 4 characters per token is a common approximation
    usd = (approx_tokens / 1_000_000) * cfg.EMBED_USD_PER_1M_TOKENS
    print(f"\n  Ingestion cost estimate ({total_chars:,} chars ≈ {approx_tokens:,.0f} tokens):")
    print(f"    INGESTION_COST_ESTIMATE = {usd:.4f}  # USD, paste into config.py")


if __name__ == "__main__":
    print("1/6  IAM role")
    role_arn = create_kb_role()

    print("2/6  S3 Vectors bucket + index")
    index_arn = create_vector_store()

    print("3/6  knowledge base + data source")
    kb_id, ds_id = create_kb(role_arn, index_arn)

    print("4/6  ingestion")
    ingest(kb_id, ds_id)
    estimate_ingestion_cost()

    print("5/6  guardrail")
    guardrail_id, guardrail_version = create_guardrail()

    print("6/6  AgentCore Gateway + Cedar policy engine")
    gw = create_gateway()

    print("\nDone. Paste these into config.py:")
    print(f'  KB_ID = "{kb_id}"')
    print(f'  DATA_SOURCE_ID = "{ds_id}"')
    print(f'  GUARDRAIL_ID = "{guardrail_id}"')
    print(f'  GUARDRAIL_VERSION = "{guardrail_version}"')
    print(f'  GATEWAY_ID = "{gw["gateway_id"]}"')
    print(f'  GATEWAY_URL = "{gw["gateway_url"]}"')
    print(f'  GATEWAY_ENGINE_ID = "{gw["engine_id"]}"')
    print("\nRun teardown.py after the talk.")
