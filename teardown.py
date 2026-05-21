#!/usr/bin/env python3
"""
teardown.py  --  delete everything build_kb.py created.

S3 Vectors is pay-per-use, so there is no hourly meter to race -- but tear it
down anyway to keep the account clean. Removes: the knowledge base, the S3
Vectors index + bucket, the IAM roles, the AgentCore Gateway, the Cedar policy
engine, and optionally the S3 corpus bucket.

WARNING: deleting the S3 corpus bucket (cfg.BUCKET) removes the paper corpus.
"""

import boto3

import config as cfg

agent = boto3.client("bedrock-agent", region_name=cfg.REGION)
br = boto3.client("bedrock", region_name=cfg.REGION)
br_ctrl = boto3.client("bedrock-agentcore-control", region_name=cfg.REGION)
s3 = boto3.client("s3")
s3v = boto3.client("s3vectors", region_name=cfg.REGION)
iam = boto3.client("iam")


def _try(label, fn):
    try:
        fn()
        print(f"  deleted: {label}")
    except Exception as e:  # noqa: BLE001 -- teardown is best-effort
        print(f"  skip ({label}): {type(e).__name__}")


def _delete_bucket(bucket: str) -> None:
    """Delete all objects in a bucket, then delete the bucket itself."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            s3.delete_object(Bucket=bucket, Key=obj["Key"])
    s3.delete_bucket(Bucket=bucket)


if __name__ == "__main__":
    # --- Gateway and Cedar policy engine ----------------------------------
    if getattr(cfg, "GATEWAY_ID", ""):
        _try(
            f"gateway {cfg.GATEWAY_ID}",
            lambda: br_ctrl.delete_gateway(gatewayIdentifier=cfg.GATEWAY_ID),
        )

    if getattr(cfg, "GATEWAY_ENGINE_ID", ""):
        _try(
            f"policy engine {cfg.GATEWAY_ENGINE_ID}",
            lambda: br_ctrl.delete_policy_engine(policyEngineId=cfg.GATEWAY_ENGINE_ID),
        )

    # Lambda web-tool function
    lam = boto3.client("lambda", region_name=cfg.REGION)
    _try(
        "Lambda inside-the-lines-web-tool",
        lambda: lam.delete_function(FunctionName="inside-the-lines-web-tool"),
    )

    def drop_lambda_role():
        role = "inside-the-lines-lambda-role"
        try:
            lam_iam = boto3.client("iam")
            attached = lam_iam.list_attached_role_policies(RoleName=role)["AttachedPolicies"]
            for p in attached:
                lam_iam.detach_role_policy(RoleName=role, PolicyArn=p["PolicyArn"])
            for p in lam_iam.list_role_policies(RoleName=role)["PolicyNames"]:
                lam_iam.delete_role_policy(RoleName=role, PolicyName=p)
            lam_iam.delete_role(RoleName=role)
        except Exception as e:  # noqa: BLE001
            raise e

    _try("IAM role inside-the-lines-lambda-role", drop_lambda_role)

    def drop_gateway_role():
        role = "inside-the-lines-gateway-role"
        for p in iam.list_role_policies(RoleName=role)["PolicyNames"]:
            iam.delete_role_policy(RoleName=role, PolicyName=p)
        iam.delete_role(RoleName=role)

    _try("IAM role inside-the-lines-gateway-role", drop_gateway_role)

    # --- Guardrail --------------------------------------------------------
    if getattr(cfg, "GUARDRAIL_ID", ""):
        _try(
            f"guardrail {cfg.GUARDRAIL_ID}",
            lambda: br.delete_guardrail(guardrailIdentifier=cfg.GUARDRAIL_ID),
        )

    # --- Knowledge base ---------------------------------------------------
    if cfg.KB_ID:
        _try(
            f"knowledge base {cfg.KB_ID}",
            lambda: agent.delete_knowledge_base(knowledgeBaseId=cfg.KB_ID),
        )

    _try(
        f"vector index {cfg.VECTOR_INDEX_NAME}",
        lambda: s3v.delete_index(
            vectorBucketName=cfg.VECTOR_BUCKET_NAME, indexName=cfg.VECTOR_INDEX_NAME
        ),
    )
    _try(
        f"vector bucket {cfg.VECTOR_BUCKET_NAME}",
        lambda: s3v.delete_vector_bucket(vectorBucketName=cfg.VECTOR_BUCKET_NAME),
    )

    def drop_kb_role():
        for p in iam.list_role_policies(RoleName=cfg.KB_ROLE_NAME)["PolicyNames"]:
            iam.delete_role_policy(RoleName=cfg.KB_ROLE_NAME, PolicyName=p)
        iam.delete_role(RoleName=cfg.KB_ROLE_NAME)

    _try(f"IAM role {cfg.KB_ROLE_NAME}", drop_kb_role)

    # --- S3 corpus bucket (deletes all papers — re-run corpus_fetch.py to rebuild) ---
    _try(
        f"S3 corpus bucket {cfg.BUCKET}",
        lambda: _delete_bucket(cfg.BUCKET),
    )

    print("\nDone.")
