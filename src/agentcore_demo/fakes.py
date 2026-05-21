"""
fakes.py  --  FakeBackend and test pricing for the demo.

Lives in the package (not in tests/) so both the app (DEMO_FAKE=1) and the
test suite can import it from the same place. No AWS deps, no network.

Environment variables that control fake behaviour:
  DEMO_FAKE=1           use FakeBackend (required)
  DEMO_FAKE_READY=0     start with KB not ready (default: ready)
  DEMO_FAKE_INGEST_DELAY=0.3  seconds between fake progress steps (default: 0)
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable

from agentcore_demo.cost import CostMeter

# Realistic-ish placeholder pricing for deterministic cost assertions.
TEST_PRICING: dict[str, tuple[float, float]] = {
    "haiku": (0.80, 4.00),
    "sonnet": (3.00, 15.00),
    "opus": (15.00, 75.00),
    "nova": (0.80, 3.20),
}

# Live KB rate used by CostMeter for retrieval pricing.
TEST_KB_RATES = {
    "kb_query_usd_per_1k": 0.40,  # $0.40 per 1,000 retrieval queries
}

# Canned KB setup costs returned by FakeBackend.kb_setup_costs().
FAKE_KB_SETUP_COSTS = {
    "ingestion_usd": 0.10,
    "storage_usd_per_month": 0.025,
    "corpus_storage_usd_per_month": 0.0002,
    "vector_count": 5508,
    "vector_size_mb": 24.2,
    "corpus_files": 200,
    "corpus_size_mb": 9.8,
}

# A valid 1x1 PNG, base64 -- enough to exercise the chart path.
_PNG_1x1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgAAIAAAUAAXpeqz8AAAAASUVORK5CYII="
)

# Fake ingestion steps: (indexed, total) tuples streamed to progress_cb.
_INGEST_STEPS = [(0, 1000), (250, 1000), (500, 1000), (750, 1000), (1000, 1000)]


class FakeBackend:
    """Canned implementation of the Backend protocol -- no AWS, no network."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        # KB readiness: controlled by DEMO_FAKE_READY env var (default ready).
        self._ready: bool = os.environ.get("DEMO_FAKE_READY", "1") != "0"
        # Delay between fake ingestion progress steps (0 = instant, for tests).
        self._ingest_delay: float = float(os.environ.get("DEMO_FAKE_INGEST_DELAY", "0"))

    def retrieve(self, query: str, n: int = 12) -> list[dict]:
        self.calls.append(f"retrieve:{query[:20]}")
        return [
            {
                "text": f"Passage {i} about PCSK9 and LDL.",
                "source": f"s3://demo/PMC{1000 + i}.txt",
                "score": 0.9 - i * 0.01,
            }
            for i in range(min(n, 6))
        ]

    def converse(
        self, tier: str, system: str, prompt: str, max_tokens: int = 1600
    ) -> tuple[str, dict, list[dict]]:
        self.calls.append(f"converse:{tier}")
        usage = {"inputTokens": 12_000, "outputTokens": 1_000}
        if tier == "sonnet" and "self-contained Python" in system:
            code = f"import io, base64\nprint('CHART_B64:{_PNG_1x1}')"
            return code, usage, []
        # Routing call: classify the question by scanning for keywords.
        if "Reply with exactly one word" in system:
            p = prompt.lower()
            if any(k in p for k in ["chart", "compare", "trial", "analys", "quantif", "visuali"]):
                return "ANALYSIS", {"inputTokens": 50, "outputTokens": 1}, []
            if any(k in p for k in ["disagree", "controvers", "adverse", "debate", "next"]):
                return "DEBATE", {"inputTokens": 50, "outputTokens": 1}, []
            return "SYNTHESIS", {"inputTokens": 50, "outputTokens": 1}, []
        return f"[{tier}] answer to: {prompt[:40]}", usage, []

    def code_interpreter_run(self, code: str) -> tuple[str, float]:
        """Simulate running code: echo the CHART_B64 line the script would print."""
        self.calls.append("code_interpreter")
        m = re.search(r"CHART_B64:([A-Za-z0-9+/=]+)", code)
        return (f"CHART_B64:{m.group(1)}" if m else "ran ok"), 2.5

    def kb_setup_costs(self) -> dict:
        """Return canned measured KB costs (no AWS)."""
        self.calls.append("kb_setup_costs")
        return dict(FAKE_KB_SETUP_COSTS)

    def kb_is_ready(self) -> bool:
        self.calls.append("kb_is_ready")
        return self._ready

    def kb_ingest(self, progress_cb: Callable[[int, int], None]) -> dict:
        """Simulate ingestion with fake progress steps."""
        self.calls.append("kb_ingest")
        for indexed, total in _INGEST_STEPS:
            progress_cb(indexed, total)
            if self._ingest_delay > 0:
                time.sleep(self._ingest_delay)
        self._ready = True
        return dict(FAKE_KB_SETUP_COSTS)

    def kb_flush(self) -> None:
        """Reset ready state so the next kb_ingest() will run."""
        self.calls.append("kb_flush")
        self._ready = False

    def query_gateway(self, tool_name: str, arguments: dict) -> dict:
        """Fake gateway: deny web_fetch (mimics Cedar policy), allow everything else."""
        self.calls.append(f"query_gateway:{tool_name}")
        if tool_name == "web_fetch":
            return {
                "denied": True,
                "reason": ("Cedar policy denied: web_fetch is not permitted in this environment"),
            }
        return {"result": {"content": [{"type": "text", "text": f"[fake result for {tool_name}]"}]}}


def make_fake_backend_and_meter() -> tuple[FakeBackend, CostMeter]:
    """Convenience factory: FakeBackend + CostMeter with test pricing + KB rates."""
    return FakeBackend(), CostMeter(pricing=TEST_PRICING, ci_per_second=0.0, **TEST_KB_RATES)
