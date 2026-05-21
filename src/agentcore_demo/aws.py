"""
aws.py  --  the AWS backend for the demo.

One class, AwsBackend, wrapping the things the agent needs:
  retrieve()             -> Bedrock Knowledge Bases
  converse()             -> Bedrock model invocation (Claude + Nova)
  code_interpreter_run() -> AgentCore Code Interpreter
  kb_setup_costs()       -> KB panel costs (computed from corpus + vector count)
  kb_is_ready()          -> check whether KB has indexed documents
  kb_ingest()            -> start/poll an ingestion job with progress callback
  kb_flush()             -> reset local state to force a re-ingest

Verified 2026-05-20:
  - Bedrock ingestion job statistics: only document counts, NO token counts.
    Ingestion cost is computed from corpus character count (total_chars / 4
    tokens × embed rate).  Labelled "computed from corpus size", not metered.
  - S3 Vectors GetVectorBucket: no sizeBytes field.
    Storage cost is derived from ListVectors count × per-vector bytes (Titan
    Embed V2: 1024 float32 = 4096 bytes, plus ~512 bytes metadata overhead).
    Labelled "from vector count", not metered.
  - temperature is deprecated for Claude Opus 4.7; inferenceConfig uses only
    maxTokens.

API shapes here were verified against current AWS docs -- do not "fix" them
without checking.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

import boto3

from agentcore_demo.backend import Backend

__all__ = ["AwsBackend", "Backend"]

_EXPECTED_CORPUS_SIZE = 657  # actual corpus size (CC0/CC BY PCSK9 papers from PMC)


def _format_source(uri: str) -> str:
    """Convert an S3 URI to a readable PMC citation.

    s3://inside-the-lines-corpus/corpus/PMC13156736.txt  →  PMC13156736
    Falls back to the raw URI if the pattern doesn't match.
    """
    import re  # noqa: PLC0415

    m = re.search(r"(PMC\d+)", uri)
    return m.group(1) if m else uri


# Titan Embed V2 produces 1024-float32 vectors (4 096 B) + ~512 B metadata overhead.
_BYTES_PER_VECTOR = 4096 + 512


class AwsBackend:
    """Real AWS implementation of Backend."""

    def __init__(
        self,
        region: str,
        kb_id: str,
        ds_id: str,
        models: dict[str, str],
        rates: dict[str, float] | None = None,
        vector_bucket_name: str = "",
        guardrail_id: str = "",
        guardrail_version: str = "DRAFT",
        gateway_url: str = "",
    ):
        self.region = region
        self.kb_id = kb_id
        self.ds_id = ds_id
        self.models = models
        self.rates = rates or {}
        self.vector_bucket_name = vector_bucket_name
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
        self.gateway_url = gateway_url
        self._kb = boto3.client("bedrock-agent-runtime", region_name=region)
        self._llm = boto3.client("bedrock-runtime", region_name=region)
        self._agent = boto3.client("bedrock-agent", region_name=region)
        self._ingestion_stats: dict | None = None  # populated after kb_ingest()
        self._force_reingest: bool = False

    def retrieve(self, query: str, n: int = 12) -> list[dict]:
        """Semantic search over the Bedrock Knowledge Base."""
        resp = self._kb.retrieve(
            knowledgeBaseId=self.kb_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": n}},
        )
        return [
            {
                "text": item["content"]["text"],
                "source": _format_source(
                    item.get("location", {}).get("s3Location", {}).get("uri", "?")
                ),
                "score": item.get("score", 0.0),
            }
            for item in resp["retrievalResults"]
        ]

    def converse(
        self, tier: str, system: str, prompt: str, max_tokens: int = 1600
    ) -> tuple[str, dict, list[dict]]:
        """One Bedrock converse call. Works uniformly for Claude and Nova.

        Returns (text, usage, matches) where matches is a list of guardrail hits:
        [{"name": str, "match": str, "action": str}], empty if no guardrail or no hits.
        """
        kwargs: dict = {
            "modelId": self.models[tier],
            "system": [{"text": system}],
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if self.guardrail_id:
            kwargs["guardrailConfig"] = {
                "guardrailIdentifier": self.guardrail_id,
                "guardrailVersion": self.guardrail_version,
                "trace": "enabled",
            }
        resp = self._llm.converse(**kwargs)
        text = resp["output"]["message"]["content"][0]["text"]

        matches: list[dict] = []
        if self.guardrail_id:
            assessments = resp.get("trace", {}).get("guardrail", {}).get("outputAssessments", {})
            for assessment_list in assessments.values():
                for assessment in assessment_list if isinstance(assessment_list, list) else []:
                    regexes = assessment.get("sensitiveInformationPolicy", {}).get("regexes", [])
                    for r in regexes:
                        if r.get("detected"):
                            matches.append(
                                {"name": r["name"], "match": r["match"], "action": r["action"]}
                            )

        return text, resp["usage"], matches

    def code_interpreter_run(self, code: str) -> tuple[str, float]:
        """Execute code in an isolated AgentCore Code Interpreter microVM."""
        from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

        ci = CodeInterpreter(self.region)
        ci.start()
        t0 = time.time()
        try:
            resp = ci.invoke("executeCode", {"language": "python", "code": code})
            out: list[str] = []
            for event in resp["stream"]:
                for block in event.get("result", {}).get("content", []):
                    if block.get("type") == "text":
                        out.append(block["text"])
            return "\n".join(out), time.time() - t0
        finally:
            ci.stop()

    def kb_is_ready(self) -> bool:
        """Return True if the KB has at least one completed ingestion job with indexed docs."""
        if self._force_reingest:
            return False
        try:
            jobs = self._agent.list_ingestion_jobs(
                knowledgeBaseId=self.kb_id,
                dataSourceId=self.ds_id,
            )
            for job in jobs.get("ingestionJobSummaries", []):
                if job.get("status") == "COMPLETE":
                    stats = job.get("statistics", {})
                    if stats.get("numberOfNewDocumentsIndexed", 0) > 0:
                        return True
        except Exception:
            pass
        return False

    def kb_ingest(self, progress_cb: Callable[[int, int], None]) -> dict:
        """Start an ingestion job and poll until complete, reporting progress.

        Returns a costs dict suitable for the setup_cost event.
        """
        progress_cb(0, _EXPECTED_CORPUS_SIZE)
        job_id = self._agent.start_ingestion_job(
            knowledgeBaseId=self.kb_id,
            dataSourceId=self.ds_id,
        )["ingestionJob"]["ingestionJobId"]

        stats: dict = {}
        while True:
            st = self._agent.get_ingestion_job(
                knowledgeBaseId=self.kb_id,
                dataSourceId=self.ds_id,
                ingestionJobId=job_id,
            )["ingestionJob"]
            stats = st.get("statistics", {})
            indexed = stats.get("numberOfNewDocumentsIndexed", 0) + stats.get(
                "numberOfModifiedDocumentsIndexed", 0
            )
            progress_cb(indexed, _EXPECTED_CORPUS_SIZE)
            if st["status"] in ("COMPLETE", "FAILED"):
                break
            time.sleep(3)

        # Ingestion cost: computed from corpus char count (total_chars / 4 tokens).
        # The ingestion API reports document counts only, not token counts.
        ingestion_usd = self._compute_ingestion_cost_from_corpus()
        storage_usd = self._compute_storage_cost_from_vector_count()

        self._ingestion_stats = {
            "ingestion_usd": round(ingestion_usd, 4),
            "storage_usd_per_month": round(storage_usd, 4),
            "corpus_storage_usd_per_month": round(self._compute_corpus_s3_storage_cost(), 6),
        }
        self._force_reingest = False
        return dict(self._ingestion_stats)

    def kb_flush(self) -> None:
        """Reset local state so the next kb_ingest() triggers a new ingestion job."""
        self._force_reingest = True
        self._ingestion_stats = None

    def query_gateway(self, tool_name: str, arguments: dict) -> dict:
        """Invoke a tool on the AgentCore Gateway via the MCP HTTP endpoint.

        POSTs a JSON-RPC 2.0 tools/call request to ``{gateway_url}/mcp``.
        Returns ``{"result": body}`` on success, or
        ``{"denied": True, "reason": ...}`` on a Cedar policy denial (HTTP 403)
        or when no gateway URL is configured.
        """
        import json  # noqa: PLC0415
        import ssl  # noqa: PLC0415
        import urllib.error  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        if not self.gateway_url:
            return {"denied": True, "reason": "No gateway configured"}

        # The Gateway MCP protocol prefixes tool calls with "{target-name}___"
        # Our target is named "web-tools", so web_fetch becomes "web-tools___web_fetch"
        mcp_tool_name = f"web-tools___{tool_name}"

        payload = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": mcp_tool_name, "arguments": arguments},
            }
        ).encode()

        req = urllib.request.Request(
            self.gateway_url.rstrip("/") + "/mcp",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        ctx = ssl.create_default_context()
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
                body = json.loads(resp.read())
                # Cedar policy denials come as HTTP 200 with a JSON-RPC error body
                if "error" in body:
                    msg = body["error"].get("message", "")
                    if "Denied" in msg or "policy" in msg.lower() or "not allowed" in msg.lower():
                        return {"denied": True, "reason": f"Cedar policy denied: {msg}"}
                return {"result": body}
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code == 403:
                return {"denied": True, "reason": f"Cedar policy denied: {body[:200]}"}
            return {"denied": True, "reason": f"HTTP {e.code}: {body[:200]}"}
        except Exception as ex:  # noqa: BLE001
            return {"denied": True, "reason": str(ex)}

    def kb_setup_costs(self) -> dict:
        """Return KB setup costs + quantitative context for the UI panel.

        Ingestion:       computed from corpus char count (total_chars / 4 × embed rate).
        Vector storage:  derived from S3 Vectors ListVectors count × per-vector bytes.
        Corpus storage:  corpus dir size × S3 standard rate (first 50 TB tier).
        None are metered charges.  NOT included in the live run total.
        """
        # Quantitative context (always fresh — cheap local stats)
        vector_count, vector_size_mb = self._measure_vector_stats()
        corpus_files, corpus_size_mb = self._measure_corpus_stats()

        base = (
            self._ingestion_stats
            if self._ingestion_stats is not None
            else {
                "ingestion_usd": round(self._compute_ingestion_cost_from_corpus(), 4),
                "storage_usd_per_month": round(self._compute_storage_cost_from_vector_count(), 4),
                "corpus_storage_usd_per_month": round(self._compute_corpus_s3_storage_cost(), 6),
            }
        )
        return {
            **base,
            "vector_count": vector_count,
            "vector_size_mb": round(vector_size_mb, 1),
            "corpus_files": corpus_files,
            "corpus_size_mb": round(corpus_size_mb, 1),
        }

    # -- private helpers --------------------------------------------------

    def _measure_vector_stats(self) -> tuple[int, float]:
        """Return (vector_count, size_mb) from S3 Vectors ListVectors."""
        try:
            import boto3  # noqa: PLC0415

            s3v = boto3.client("s3vectors", region_name=self.region)
            import config  # type: ignore[import]  # noqa: PLC0415

            n = 0
            for page in s3v.get_paginator("list_vectors").paginate(
                vectorBucketName=self.vector_bucket_name or config.VECTOR_BUCKET_NAME,
                indexName=config.VECTOR_INDEX_NAME,
            ):
                n += len(page.get("vectors", []))
            size_mb = n * _BYTES_PER_VECTOR / (1024 * 1024)
            return n, size_mb
        except Exception:
            return 0, 0.0

    def _measure_corpus_stats(self) -> tuple[int, float]:
        """Return (file_count, size_mb) from the local corpus directory."""
        corpus_dir = "corpus"
        if not os.path.isdir(corpus_dir):
            return 0, 0.0
        files = list(os.listdir(corpus_dir))
        total_bytes = sum(os.path.getsize(os.path.join(corpus_dir, f)) for f in files)
        return len(files), total_bytes / (1024 * 1024)

    def _compute_ingestion_cost_from_corpus(self) -> float:
        """Compute embedding cost from local corpus character count.

        Bedrock ingestion does not expose a token count.  We walk the corpus/
        directory that was synced to S3, sum raw character counts, and
        approximate tokens at 4 chars/token.  Returns 0.0 if corpus/ is absent
        or the embed rate is unknown.
        """
        embed_rate = self.rates.get("embed_usd_per_1m_tokens", 0.0)
        if embed_rate <= 0:
            return 0.0
        corpus_dir = "corpus"
        if not os.path.isdir(corpus_dir):
            return 0.0
        total_chars = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _dirs, files in os.walk(corpus_dir)
            for f in files
        )
        approx_tokens = total_chars / 4
        return (approx_tokens / 1_000_000) * embed_rate

    def _compute_storage_cost_from_vector_count(self) -> float:
        """Derive monthly storage cost from the live vector count in S3 Vectors.

        S3 Vectors GetVectorBucket exposes no sizeBytes field.  We count vectors
        via ListVectors and estimate bytes as n_vectors × _BYTES_PER_VECTOR
        (Titan Embed V2: 4096 B float32 data + 512 B metadata overhead).
        Returns 0.0 if the bucket name is unknown or the rate is not set.
        """
        storage_rate = self.rates.get("s3v_storage_usd_per_gb_month", 0.0)
        bucket = self.vector_bucket_name
        if not storage_rate or not bucket:
            return 0.0
        try:
            s3v = boto3.client("s3vectors", region_name=self.region)
            import config  # type: ignore[import]  # noqa: PLC0415

            index_name = config.VECTOR_INDEX_NAME
            n_vectors = 0
            paginator = s3v.get_paginator("list_vectors")
            for page in paginator.paginate(
                vectorBucketName=bucket,
                indexName=index_name,
                # include only keys, not data, for speed
            ):
                n_vectors += len(page.get("vectors", []))
            size_bytes = n_vectors * _BYTES_PER_VECTOR
            size_gb = size_bytes / (1024**3)
            return size_gb * storage_rate
        except Exception:
            return 0.0

    def _compute_corpus_s3_storage_cost(self) -> float:
        """Compute monthly S3 storage cost for the local corpus directory.

        Uses S3 standard rate (first 50 TB tier, us-west-2: $0.023/GB-month).
        Falls back to a hard-coded rate if the Price List call wasn't made.
        """
        s3_rate = self.rates.get("s3_standard_usd_per_gb_month", 0.023)
        corpus_dir = "corpus"
        if not os.path.isdir(corpus_dir):
            return 0.0
        total_bytes = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _dirs, files in os.walk(corpus_dir)
            for f in files
        )
        size_gb = total_bytes / (1024**3)
        return size_gb * s3_rate
