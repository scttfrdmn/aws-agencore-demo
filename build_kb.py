#!/usr/bin/env python3
"""
build_kb.py  --  one-time, fully scripted Knowledge Base provisioning.

Uses Amazon S3 Vectors as the vector store: serverless, pay-per-use, no hourly
floor, no OpenSearch collection to manage. Steps:

  1. IAM role the knowledge base assumes
  2. an S3 Vectors bucket + vector index
  3. the Bedrock Knowledge Base + an S3 data source
  4. an ingestion job that embeds and indexes every paper

Run once, before the talk. Prints the KB id + data-source id for config.py.
Tear it down afterwards with teardown.py.

>>> VERIFY BEFORE RUNNING <<<
S3 Vectors is a recent service. The `s3vectors` client method names and the
`create_knowledge_base` storageConfiguration shape for S3 Vectors are the
newest APIs in this repo. Check current boto3 docs and adjust the two spots
marked  # VERIFY  below. Everything else (IAM, bedrock-agent KB/data-source/
ingestion) is stable.

Requires: boto3
"""

import json
import time

import boto3

import config as cfg

iam = boto3.client("iam")
agent = boto3.client("bedrock-agent", region_name=cfg.REGION)
s3v = boto3.client("s3vectors", region_name=cfg.REGION)  # VERIFY: service name


# ----------------------------------------------------------------------
# 1. IAM role for the knowledge base
# ----------------------------------------------------------------------
def create_kb_role() -> str:
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
        arn = iam.get_role(RoleName=cfg.KB_ROLE_NAME)["Role"]["Arn"]

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:ListBucket"],
                "Resource": [f"arn:aws:s3:::{cfg.BUCKET}", f"arn:aws:s3:::{cfg.BUCKET}/*"],
            },
            {
                "Effect": "Allow",
                "Action": "bedrock:InvokeModel",
                "Resource": f"arn:aws:bedrock:{cfg.REGION}::foundation-model/{cfg.EMBED_MODEL_ID}",
            },
            # VERIFY: scope s3vectors actions per current S3 Vectors IAM reference.
            {"Effect": "Allow", "Action": "s3vectors:*", "Resource": "*"},
        ],
    }
    iam.put_role_policy(
        RoleName=cfg.KB_ROLE_NAME, PolicyName="kb-access", PolicyDocument=json.dumps(policy)
    )
    print(f"  IAM role: {arn}")
    time.sleep(10)  # let the role propagate
    return arn


# ----------------------------------------------------------------------
# 2. S3 Vectors bucket + index
# ----------------------------------------------------------------------
def create_vector_store() -> str:
    # VERIFY: confirm create_vector_bucket / create_index signatures.
    try:
        s3v.create_vector_bucket(vectorBucketName=cfg.VECTOR_BUCKET_NAME)
        print(f"  vector bucket: {cfg.VECTOR_BUCKET_NAME}")
    except Exception as e:  # noqa: BLE001 -- already-exists is fine
        print(f"  vector bucket: {type(e).__name__} (continuing)")
    # Mark all Bedrock KB metadata keys as non-filterable to stay under S3
    # Vectors' 2048-byte filterable metadata limit (verified 2026-05-20).
    _NON_FILTERABLE = [
        "x-amz-bedrock-kb-source-uri",
        "x-amz-bedrock-kb-chunk-id",
        "x-amz-bedrock-kb-data-source-id",
        "x-amz-bedrock-kb-document-id",
        "x-amz-bedrock-kb-index-id",
        "x-amz-bedrock-kb-knowledge-base-id",
        "AMAZON_BEDROCK_TEXT",
        "AMAZON_BEDROCK_METADATA",
    ]
    try:
        s3v.create_index(
            vectorBucketName=cfg.VECTOR_BUCKET_NAME,
            indexName=cfg.VECTOR_INDEX_NAME,
            dataType="float32",
            dimension=cfg.EMBED_DIM,
            distanceMetric="cosine",
            metadataConfiguration={"nonFilterableMetadataKeys": _NON_FILTERABLE},
        )
        print(f"  vector index: {cfg.VECTOR_INDEX_NAME}")
    except Exception as e:  # noqa: BLE001
        print(f"  vector index: {type(e).__name__} (continuing)")
    return (
        f"arn:aws:s3vectors:{cfg.REGION}:{cfg.ACCOUNT_ID}:"
        f"bucket/{cfg.VECTOR_BUCKET_NAME}/index/{cfg.VECTOR_INDEX_NAME}"
    )


# ----------------------------------------------------------------------
# 3. Knowledge base + data source
# ----------------------------------------------------------------------
def create_kb(role_arn: str, index_arn: str) -> tuple[str, str]:
    embed_arn = f"arn:aws:bedrock:{cfg.REGION}::foundation-model/{cfg.EMBED_MODEL_ID}"
    # VERIFY: storageConfiguration shape for S3 Vectors against current boto3.
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
                "fixedSizeChunkingConfiguration": {"maxTokens": 512, "overlapPercentage": 15},
            }
        },
    )
    return kb_id, ds["dataSource"]["dataSourceId"]


# ----------------------------------------------------------------------
# 4. Ingestion
# ----------------------------------------------------------------------
def ingest(kb_id: str, ds_id: str) -> None:
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
    """Create a Bedrock Guardrail that anonymizes external URLs in model output.

    Returns (guardrail_id, version).
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
                    "action": "ANONYMIZE",
                }
            ]
        },
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

    Steps:
      a) IAM role for the gateway (service principal bedrock-agentcore.amazonaws.com)
      b) PolicyEngine (waits for ACTIVE)
      c) Gateway — protocolType MCP, authorizerType NONE (waits for READY)
      d) Cedar policies (created after gateway so the real ARN can be validated)
      e) Attach policy engine to gateway in ENFORCE mode (waits for READY)

    Returns dict with gateway_id, gateway_url, engine_id — paste these into config.py.
    """
    br_ctrl = boto3.client("bedrock-agentcore-control", region_name=cfg.REGION)

    ROLE_NAME = "inside-the-lines-gateway-role"
    ROLE_ARN = f"arn:aws:iam::{cfg.ACCOUNT_ID}:role/{ROLE_NAME}"

    # a. IAM role
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
    time.sleep(8)  # let the role propagate

    # b. PolicyEngine
    pe = br_ctrl.create_policy_engine(name="InsideTheLinesEngine")
    engine_id = pe["policyEngineId"]
    engine_arn = pe["policyEngineArn"]
    print(f"  policy engine: {engine_id}")
    for _ in range(20):
        if br_ctrl.get_policy_engine(policyEngineId=engine_id)["status"] == "ACTIVE":
            break
        time.sleep(3)

    # c. Gateway (create first so we have the real ARN for Cedar validation)
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

    # d. Cedar policies (after gateway exists so ARN can be validated)
    # Use a short timestamp suffix to avoid "already exists" on re-runs
    ts = str(int(time.time()))[-6:]
    permit_cedar = f'permit(principal, action, resource == AgentCore::Gateway::"{gateway_arn}");'
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
            validationMode="IGNORE_ALL_FINDINGS",
        )
        print(f"  policy: {name}")

    # e. Attach policy engine to gateway (must supply all original create params)
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
    """Print an ingestion cost estimate based on the local corpus.

    Walks the corpus/ directory, sums raw character counts, approximates
    token count at 4 chars/token, then prices at EMBED_USD_PER_1M_TOKENS.
    Paste INGESTION_COST_ESTIMATE into config.py with the printed value.
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
    approx_tokens = total_chars / 4
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
