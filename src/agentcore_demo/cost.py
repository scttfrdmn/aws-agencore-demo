"""
cost.py  --  the cost meter.

Pure logic: real token usage in, dollars out. No AWS, no I/O -- so it is
trivially unit-testable (see tests/test_cost.py).

All rows in the live run are MEASURED costs:
  - LLM rows: actual token counts from converse() × per-token rates.
  - Compute rows: Code Interpreter wall-clock seconds × per-second rate.
  - Retrieval rows: per-query API calls × live rate from the Price List.

KB setup costs (ingestion + storage) are returned by backend.kb_setup_costs()
and emitted as a separate `setup_cost` event.  They are NEVER added to .total.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class CostRow:
    """One line on the receipt."""

    step: str
    label: str
    in_tokens: int
    out_tokens: int
    usd: float


@dataclass
class CostMeter:
    """Accumulates per-step cost across a demo run."""

    pricing: dict[str, tuple[float, float]]  # tier -> (usd_per_1M_in, usd_per_1M_out)
    ci_per_second: float = 0.0
    kb_query_usd_per_1k: float = 0.0  # live rate from Price List for retrieval API calls
    rows: list[CostRow] = field(default_factory=list)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def add_llm(self, step: str, tier: str, label: str, usage: dict) -> float:
        """Record one model call. Thread-safe."""
        per_in, per_out = self.pricing[tier]
        i = int(usage.get("inputTokens", 0))
        o = int(usage.get("outputTokens", 0))
        usd = (i / 1_000_000) * per_in + (o / 1_000_000) * per_out
        with self._lock:
            self.rows.append(CostRow(step, label, i, o, usd))
        return usd

    def add_compute(self, step: str, seconds: float) -> float:
        """Record one Code Interpreter session by wall-clock seconds."""
        usd = seconds * self.ci_per_second
        with self._lock:
            self.rows.append(CostRow(step, "Code Interpreter", 0, 0, usd))
        return usd

    def add_retrieval(self, step: str, n_queries: int = 1) -> float:
        """Record KB retrieval API calls. Cost computed from the live Price List rate."""
        usd = (n_queries / 1_000) * self.kb_query_usd_per_1k
        with self._lock:
            self.rows.append(CostRow(step, "KB retrieval", 0, 0, usd))
        return usd

    @property
    def total(self) -> float:
        return sum(r.usd for r in self.rows)

    def receipt(self) -> dict:
        """Serialisable receipt for the UI / a printed summary."""
        return {
            "rows": [
                {
                    "step": r.step,
                    "label": r.label,
                    "in_tokens": r.in_tokens,
                    "out_tokens": r.out_tokens,
                    "usd": round(r.usd, 6),
                }
                for r in self.rows
            ],
            "total": round(self.total, 6),
        }
