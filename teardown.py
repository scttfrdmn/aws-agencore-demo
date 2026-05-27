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

What this script deletes locally:
  - corpus/  -- the downloaded papers on your laptop (re-run corpus_fetch.py to rebuild)

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
    import time

    # --- Gateway and Cedar policy engine ----------------------------------
    # Order matters: targets → gateway → policy engine.
    # The PolicyEngine cannot be deleted while any gateway references it.
    if getattr(cfg, "GATEWAY_ID", ""):
        # Delete all Gateway targets first (the web-tools Lambda target).
        # The Gateway cannot be deleted while targets exist.
        try:
            targets = br_ctrl.list_gateway_targets(gatewayIdentifier=cfg.GATEWAY_ID).get(
                "items", []
            )
            for t in targets:
                _try(
                    f"gateway target {t['name']} ({t['targetId']})",
                    lambda tid=t["targetId"]: br_ctrl.delete_gateway_target(
                        gatewayIdentifier=cfg.GATEWAY_ID, targetId=tid
                    ),
                )
        except Exception as e:  # noqa: BLE001
            print(f"  skip (list gateway targets): {type(e).__name__}")

        # Detach the policy engine before deleting the gateway.
        # delete_gateway() returns ValidationException if a policy engine is attached.
        try:
            gw_detail = br_ctrl.get_gateway(gatewayIdentifier=cfg.GATEWAY_ID)
            if gw_detail.get("policyEngineConfiguration"):
                br_ctrl.update_gateway(
                    gatewayIdentifier=cfg.GATEWAY_ID,
                    name=gw_detail["name"],
                    roleArn=gw_detail["roleArn"],
                    authorizerType=gw_detail["authorizerType"],
                    # Empty policyEngineConfiguration detaches it
                )
                # Wait for update to settle
                for _ in range(10):
                    if br_ctrl.get_gateway(gatewayIdentifier=cfg.GATEWAY_ID)["status"] == "READY":
                        break
                    time.sleep(3)
        except Exception as e:  # noqa: BLE001
            print(f"  skip (detach policy engine from gateway): {type(e).__name__}")

        _try(
            f"gateway {cfg.GATEWAY_ID}",
            lambda: br_ctrl.delete_gateway(gatewayIdentifier=cfg.GATEWAY_ID),
        )

        # Wait for the gateway to finish deleting before touching the policy engine.
        # The API returns immediately but the resource lingers in DELETING state;
        # deleting the engine while the gateway still exists returns ConflictException.
        for _ in range(30):
            try:
                g = br_ctrl.get_gateway(gatewayIdentifier=cfg.GATEWAY_ID)
                if g["status"] not in ("DELETING", "DELETE_UNSUCCESSFUL"):
                    break
            except Exception:
                break  # gateway is gone
            time.sleep(5)

    if getattr(cfg, "GATEWAY_ENGINE_ID", ""):
        # Delete all Cedar policies in the engine first.
        # The engine cannot be deleted while it contains policies.
        # IMPORTANT: list_policies() always returns empty (API bug as of 2026-05-21).
        # list_policy_summaries() is the working alternative.
        try:
            policies = br_ctrl.list_policy_summaries(policyEngineId=cfg.GATEWAY_ENGINE_ID).get(
                "policies", []
            )
            for pol in policies:
                _try(
                    f"Cedar policy {pol['name']} ({pol['policyId']})",
                    lambda pid=pol["policyId"]: br_ctrl.delete_policy(
                        policyEngineId=cfg.GATEWAY_ENGINE_ID, policyId=pid
                    ),
                )
            if policies:
                # Policy deletes are async; wait for them to settle before
                # deleting the engine (which requires zero policies remaining).
                for _ in range(20):
                    remaining = br_ctrl.list_policy_summaries(
                        policyEngineId=cfg.GATEWAY_ENGINE_ID
                    ).get("policies", [])
                    if not remaining:
                        break
                    time.sleep(3)
        except Exception as e:  # noqa: BLE001
            print(f"  skip (list Cedar policies): {type(e).__name__}")

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
    # CRITICAL ORDERING: Delete the KB BEFORE the vector store.
    #
    # When delete_knowledge_base() is called, Bedrock attempts to clean up
    # all data from the underlying vector store. If the vector store is already
    # gone, the KB gets stuck in DELETE_UNSUCCESSFUL status.
    #
    # Correct order:
    #   1. Delete KB (triggers async cleanup of vector data)
    #   2. Wait for KB deletion to complete
    #   3. Delete S3 Vectors index
    #   4. Delete S3 Vectors bucket
    if cfg.KB_ID:
        _try(
            f"knowledge base {cfg.KB_ID}",
            lambda: agent.delete_knowledge_base(knowledgeBaseId=cfg.KB_ID),
        )

        # Wait for KB deletion to complete before touching the vector store.
        # If we delete the vector store while the KB is still cleaning up,
        # the KB will fail and get stuck in DELETE_UNSUCCESSFUL.
        print("  waiting for KB deletion to complete...")
        for _ in range(60):  # up to 5 minutes
            try:
                kb_detail = agent.get_knowledge_base(knowledgeBaseId=cfg.KB_ID)
                kb_status = kb_detail["knowledgeBase"]["status"]
                if kb_status == "DELETING":
                    time.sleep(5)
                    continue
                if kb_status == "DELETE_UNSUCCESSFUL":
                    print("  warning: KB stuck in DELETE_UNSUCCESSFUL (safe to continue)")
                    break
                # Unexpected status, but continue anyway
                break
            except Exception as e:
                # KB not found = successfully deleted
                if "ResourceNotFoundException" in str(type(e).__name__):
                    print("  KB deletion confirmed")
                break

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
    # Re-run corpus_fetch.py + aws s3 sync to rebuild if needed.
    _try(
        f"S3 corpus bucket {cfg.BUCKET}",
        lambda: _delete_bucket(cfg.BUCKET),
    )

    # Delete the local corpus/ directory if it exists.
    import os
    import shutil

    corpus_dir = "corpus"
    if os.path.isdir(corpus_dir):
        shutil.rmtree(corpus_dir)
        print(f"  deleted: local {corpus_dir}/ directory")
    else:
        print(f"  skip (local {corpus_dir}/): not present")

    print("\nDone.")
