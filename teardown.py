#!/usr/bin/env python3
"""
teardown.py  --  delete everything build_kb.py created.

Run this after the talk to clean up billable resources and keep your
AWS account tidy.

What this script deletes (and approximate ongoing costs if left running):
  - AgentCore Gateway        -- $0 idle, but pay-per-invocation + Cedar eval fees
  - Cedar PolicyEngine       -- $0 idle
  - Lambda web-tool function -- $0 for ≤1M requests/month (free tier), but clutter
  - IAM roles (gateway, lambda, KB)  -- no cost, but best practice to remove
  - Bedrock Guardrail        -- ~$0.75 per million text units if left in use
  - Bedrock Knowledge Base   -- $0 idle, but the S3 Vectors index below has storage
  - S3 Vectors index         -- ~$0.05/GB-month for stored vectors (~$0.002/month
                                for a 650-paper demo corpus; negligible but real)
  - S3 Vectors bucket        -- $0 if empty, deleted after index is gone
  - S3 corpus bucket         -- ~$0.023/GB-month for standard storage
                                (~$0.023 × 0.01 GB ≈ negligible, but accumulates)

What this script does NOT delete:
  - The local corpus/ directory  -- it is on your laptop, not in AWS.
    Delete it manually if you want to free disk space.

How it works:
  Each deletion is wrapped in _try() so that a single failure (e.g. a
  resource that was already deleted) does not stop the rest.  Every step
  prints either "deleted: ..." or "skip (...): ExceptionType".

Re-running safely:
  Idempotent -- if a resource is already gone, _try() catches the exception
  and continues.  You can run this multiple times safely.
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
    """Call fn(); print success or skip message.  Never raises.

    Teardown is inherently best-effort: if one resource is already gone
    or was never created, we still want to proceed to the next one.
    """
    try:
        fn()
        print(f"  deleted: {label}")
    except Exception as e:  # noqa: BLE001 -- teardown is best-effort
        print(f"  skip ({label}): {type(e).__name__}")


def _delete_bucket(bucket: str) -> None:
    """Delete all objects in an S3 bucket, then delete the bucket itself.

    S3 requires a bucket to be empty before it can be deleted.  This
    function paginates through all objects and deletes them one by one,
    then calls delete_bucket.

    Args:
        bucket: the S3 bucket name (not ARN).
    """
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            s3.delete_object(Bucket=bucket, Key=obj["Key"])
    s3.delete_bucket(Bucket=bucket)


if __name__ == "__main__":
    # --- Gateway and Cedar policy engine ----------------------------------
    # Delete the Gateway first; the PolicyEngine can only be deleted after
    # all gateways that reference it are gone.
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

    # Delete the Lambda web-tool function and its IAM role.
    # These are created as part of Gateway target setup; if you never created
    # a Lambda target, these deletes will just print "skip" and move on.
    lam = boto3.client("lambda", region_name=cfg.REGION)
    _try(
        "Lambda inside-the-lines-web-tool",
        lambda: lam.delete_function(FunctionName="inside-the-lines-web-tool"),
    )

    def drop_lambda_role():
        """Detach managed policies, delete inline policies, then delete the role.

        IAM requires all policies to be removed before a role can be deleted.
        """
        role = "inside-the-lines-lambda-role"
        try:
            lam_iam = boto3.client("iam")
            # Detach any AWS-managed or customer-managed policies first.
            attached = lam_iam.list_attached_role_policies(RoleName=role)["AttachedPolicies"]
            for p in attached:
                lam_iam.detach_role_policy(RoleName=role, PolicyArn=p["PolicyArn"])
            # Delete inline policies.
            for p in lam_iam.list_role_policies(RoleName=role)["PolicyNames"]:
                lam_iam.delete_role_policy(RoleName=role, PolicyName=p)
            lam_iam.delete_role(RoleName=role)
        except Exception as e:  # noqa: BLE001
            raise e

    _try("IAM role inside-the-lines-lambda-role", drop_lambda_role)

    def drop_gateway_role():
        """Delete inline policies then delete the gateway IAM role."""
        role = "inside-the-lines-gateway-role"
        for p in iam.list_role_policies(RoleName=role)["PolicyNames"]:
            iam.delete_role_policy(RoleName=role, PolicyName=p)
        iam.delete_role(RoleName=role)

    _try("IAM role inside-the-lines-gateway-role", drop_gateway_role)

    # --- Guardrail --------------------------------------------------------
    # Bedrock Guardrails are billed per text unit processed, not per month.
    # No ongoing cost if unused, but deleting keeps the account clean.
    if getattr(cfg, "GUARDRAIL_ID", ""):
        _try(
            f"guardrail {cfg.GUARDRAIL_ID}",
            lambda: br.delete_guardrail(guardrailIdentifier=cfg.GUARDRAIL_ID),
        )

    # --- Knowledge base ---------------------------------------------------
    # The KB itself has no storage cost -- the vectors live in S3 Vectors.
    # Deleting the KB removes the metadata and retrieval endpoint.
    if cfg.KB_ID:
        _try(
            f"knowledge base {cfg.KB_ID}",
            lambda: agent.delete_knowledge_base(knowledgeBaseId=cfg.KB_ID),
        )

    # Delete the S3 Vectors index first (must be empty or the bucket delete fails).
    # Cost while running: ~$0.05/GB-month for the vector data.
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
        """Delete inline policies then delete the KB IAM role."""
        for p in iam.list_role_policies(RoleName=cfg.KB_ROLE_NAME)["PolicyNames"]:
            iam.delete_role_policy(RoleName=cfg.KB_ROLE_NAME, PolicyName=p)
        iam.delete_role(RoleName=cfg.KB_ROLE_NAME)

    _try(f"IAM role {cfg.KB_ROLE_NAME}", drop_kb_role)

    # --- S3 corpus bucket -------------------------------------------------
    # WARNING: this deletes all the papers you downloaded with corpus_fetch.py.
    # The local corpus/ directory on your laptop is NOT deleted.
    # Re-run corpus_fetch.py + aws s3 sync to rebuild if needed.
    _try(
        f"S3 corpus bucket {cfg.BUCKET}",
        lambda: _delete_bucket(cfg.BUCKET),
    )

    print("\nDone.")
    print("Note: local corpus/ directory was NOT deleted -- remove it manually if needed.")
