"""
backend.py  --  the Backend interface.

The agent depends on this Protocol, not on AWS. agentcore_demo.aws.AwsBackend
implements it for real; tests provide a fake with the same methods.  Keeping the
interface here (no boto3 import) means the agent and the whole test suite import
cleanly with no AWS dependency.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class Backend(Protocol):
    """What agent.py needs from the world. Real impl + test fakes satisfy it."""

    def retrieve(self, query: str, n: int = 12) -> list[dict]:
        """Semantic search; returns dicts with text / source / score."""
        ...

    def converse(
        self, tier: str, system: str, prompt: str, max_tokens: int = 1600
    ) -> tuple[str, dict, list[dict]]:
        """One model call; returns (text, usage, matches).

        usage has *Tokens fields.
        matches is a list of guardrail hits: [{"name": str, "match": str, "action": str}],
        empty if no guardrail is configured or no matches were found.
        """
        ...

    def code_interpreter_run(self, code: str) -> tuple[str, float]:
        """Run code in a sandbox; returns (stdout, wall_clock_seconds)."""
        ...

    def kb_setup_costs(self) -> dict:
        """Return KB panel costs: {ingestion_usd, storage_usd_per_month}.

        Ingestion: computed from corpus character count (total_chars / 4 tokens
        × embed rate) — the Bedrock ingestion API does not expose token counts.
        Storage: derived from S3 Vectors ListVectors count × per-vector bytes —
        GetVectorBucket has no sizeBytes field.
        Neither value is metered. NOT included in the live run total.
        """
        ...

    def kb_is_ready(self) -> bool:
        """Return True if the KB has indexed documents and is ready to query."""
        ...

    def kb_ingest(self, progress_cb: Callable[[int, int], None]) -> dict:
        """Run (or resume) ingestion, calling progress_cb(indexed, total) as it proceeds.

        Returns {ingestion_usd, storage_usd_per_month} when complete.
        """
        ...

    def kb_flush(self) -> None:
        """Mark the KB for re-ingestion.

        On next kb_ingest() call a fresh ingestion job will run.  The AWS-side
        KB stays indexed; this only resets local state so the ingestion view
        will appear again (rehearsal tool).
        """
        ...

    def query_gateway(self, tool_name: str, arguments: dict) -> dict:
        """Invoke a tool on the AgentCore Gateway.

        Returns {"result": ...} on success, or {"denied": True, "reason": ...}
        when a Cedar policy denies the request (HTTP 403) or no gateway is configured.
        """
        ...
