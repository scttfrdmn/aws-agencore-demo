"""
agent.py  --  the demo orchestration.

Runs the three locked questions against a Backend, and reports progress by
calling an emit(event: dict) callback. The web app (app.py) provides an emit
that pushes each event over a WebSocket; a test or CLI can provide an emit
that just collects them.

This module is the source of truth for the event protocol -- see CLAUDE.md
for the table, and EVENT TYPES below.

Free-form questions are routed by a single Haiku call (ROUTING_SYSTEM) to one
of three paths: SYNTHESIS (Q1 path), ANALYSIS (Q2 path), or DEBATE (Q3 path).
The route event carries the resolved path so the UI can display it.
"""

from __future__ import annotations

import base64
import os
import re
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

from agentcore_demo import questions as Q
from agentcore_demo.backend import Backend
from agentcore_demo.cost import CostMeter

Emit = Callable[[dict], None]

# EVENT TYPES (every event is a dict with a "type" key):
#   question      {n, text}                  a new question started
#   phase         {label}                    status line
#   retrieval     {count}                    N passages retrieved
#   model         {tier, label, state, ...}  state="start" | "done" (+usage,cost)
#   answer        {title, text}              a synthesis / adjudication result
#   code          {text}                     generated analysis code
#   chart         {data}                     base64 PNG, for an inline <img>
#   cost          {total}                    running cost-meter total
#   route         {path, label}              which path was chosen for a free-form Q
#                                            path: "SYNTHESIS"|"ANALYSIS"|"DEBATE"
#   setup_cost    {ingestion_usd,            measured KB one-time + recurring costs
#                  storage_usd_per_month}    (NOT included in the live run total)
#   guardrail     {actions}                  URL matches intercepted by guardrail
#   policy_denied {tool, reason}             Cedar Gateway policy denied a tool call
#   receipt       {rows, total}              the final itemised receipt
#   done          {}                         run complete

# Maps routing labels to human-readable path descriptions for the UI.
ROUTE_LABELS = {
    "SYNTHESIS": "retrieval + synthesis · Claude Haiku",
    "ANALYSIS": "code generation + chart · Claude Sonnet + Code Interpreter",
    "DEBATE": "dual review + adjudication · Opus + Nova + Sonnet",
}


class Agent:
    """Drives the demo and emits events as it goes."""

    def __init__(self, backend: Backend, meter: CostMeter, emit: Emit):
        self.backend = backend
        self.meter = meter
        self.emit = emit

    # -- helpers ---------------------------------------------------------
    def _context(self, chunks: list[dict]) -> str:
        # Pass plain PMC IDs as source labels; the system prompt asks the model
        # to expand them to full NCBI URLs, which the guardrail then intercepts.
        parts = []
        for c in chunks:
            parts.append(f"[source: {c['source']}]\n{c['text']}")
        return "\n\n".join(parts)

    def _retrieve(self, step: str, query: str, n: int = 12) -> list[dict]:
        """Retrieve passages, record cost, and emit updated total."""
        chunks = self.backend.retrieve(query, n)
        self.meter.add_retrieval(step, n_queries=1)
        self.emit({"type": "cost", "total": round(self.meter.total, 6)})
        return chunks

    def _model(
        self, step: str, tier: str, label: str, system: str, prompt: str, max_tokens: int = 4096
    ) -> str:
        """Run one model call, emitting start/done events and metering cost."""
        self.emit({"type": "model", "tier": tier, "label": label, "state": "start"})
        t0 = time.monotonic()
        text, usage, matches = self.backend.converse(tier, system, prompt, max_tokens)
        elapsed = round(time.monotonic() - t0, 1)
        cost = self.meter.add_llm(step, tier, label, usage)
        self.emit(
            {
                "type": "model",
                "tier": tier,
                "label": label,
                "state": "done",
                "elapsed_s": elapsed,
                "usage": {
                    "inputTokens": usage.get("inputTokens", 0),
                    "outputTokens": usage.get("outputTokens", 0),
                },
                "cost": round(cost, 6),
            }
        )
        if matches:
            actions: list[dict] = []
            for match in matches:
                pmcid_m = re.search(r"PMC\d+", match["match"])
                if pmcid_m:
                    pmcid = pmcid_m.group(0)
                    if os.path.exists(f"corpus/{pmcid}.txt"):
                        actions.append(
                            {
                                "original": match["match"],
                                "local": pmcid,
                                "reason": "redirected to local corpus",
                            }
                        )
                        # Replace the guardrail placeholder with a local Markdown link
                        text = text.replace("{EXTERNAL_URL}", f"[{pmcid}](/corpus/{pmcid})", 1)
                    else:
                        actions.append(
                            {
                                "original": match["match"],
                                "local": None,
                                "reason": "PMC article not in local corpus",
                            }
                        )
                        text = text.replace("{EXTERNAL_URL}", f"[{pmcid}]", 1)
                else:
                    actions.append(
                        {
                            "original": match["match"],
                            "local": None,
                            "reason": "external link — no local copy",
                        }
                    )
                    text = text.replace("{EXTERNAL_URL}", "[link removed]", 1)
            # Clean up any remaining unreplaced placeholders
            text = text.replace("{EXTERNAL_URL}", "[link removed]")
            self.emit({"type": "guardrail", "actions": actions})
        self.emit({"type": "cost", "total": round(self.meter.total, 6)})
        return text

    # -- routing ---------------------------------------------------------
    def route(self, text: str) -> str:
        """Classify a free-form question via a single Haiku call.

        Returns one of "SYNTHESIS", "ANALYSIS", or "DEBATE".
        The Haiku call is metered (routing cost is included in the receipt)
        but not emitted as a model event — it is infrastructure, not an answer.
        """
        raw, usage, _ = self.backend.converse("haiku", Q.ROUTING_SYSTEM, text, max_tokens=5)
        self.meter.add_llm("routing", "haiku", "Claude Haiku (routing)", usage)
        self.emit({"type": "cost", "total": round(self.meter.total, 6)})
        word = raw.strip().upper().split()[0] if raw.strip() else "SYNTHESIS"
        if word not in ROUTE_LABELS:
            word = "SYNTHESIS"
        return word

    # -- Q1: friction gone -- Haiku reads and cites ----------------------
    def question_1(self, text: str = Q.QUESTIONS[0]) -> None:
        self.emit({"type": "question", "n": 1, "text": text})
        self.emit({"type": "phase", "label": "retrieving from the knowledge base"})
        chunks = self._retrieve("Q1  retrieval", text)
        self.emit({"type": "retrieval", "count": len(chunks)})
        result = self._model(
            "Q1  synthesis",
            "haiku",
            "Claude Haiku",
            Q.SYNTHESIS_SYSTEM,
            f"Passages:\n{self._context(chunks)}\n\nQuestion: {text}",
        )
        self.emit({"type": "answer", "title": "Cited synthesis  ·  Claude Haiku", "text": result})

    # -- Q2: real work -- Sonnet writes code, Code Interpreter runs it ---
    def question_2(self, text: str = Q.QUESTIONS[1]) -> None:
        self.emit({"type": "question", "n": 2, "text": text})
        self.emit({"type": "phase", "label": "retrieving trial data"})
        chunks = self._retrieve("Q2  retrieval", text, n=16)
        code = self._model(
            "Q2  code generation",
            "sonnet",
            "Claude Sonnet",
            Q.CODEGEN_SYSTEM,
            f"Passages:\n{self._context(chunks)}",
            max_tokens=4096,
        )
        code = re.sub(r"^```[a-z]*\n?|```$", "", code.strip(), flags=re.M)
        self.emit({"type": "code", "text": code})
        self.emit({"type": "phase", "label": "running in AgentCore Code Interpreter"})
        stdout, seconds = self.backend.code_interpreter_run(code)
        self.meter.add_compute("Q2  analysis run", seconds)
        self.emit({"type": "cost", "total": round(self.meter.total, 6)})
        chart, clean = self._extract_chart(stdout)
        if chart:
            self.emit({"type": "chart", "data": chart})
        # Only emit an answer if there is non-chart text output to show
        if clean:
            self.emit(
                {"type": "answer", "title": "Code Interpreter output  ·  microVM", "text": clean}
            )

    # -- Q3: the hard call -- Opus AND Nova run in parallel, then adjudicate
    def question_3(self, text: str = Q.QUESTIONS[2]) -> None:
        self.emit({"type": "question", "n": 3, "text": text})
        self.emit({"type": "phase", "label": "retrieving"})
        chunks = self._retrieve("Q3  retrieval", text, n=16)
        prompt = f"Passages:\n{self._context(chunks)}\n\n{text}"

        # Opus and Nova read the evidence independently and in parallel.
        emit_lock = threading.Lock()

        def safe_emit(event: dict) -> None:
            with emit_lock:
                self.emit(event)

        results: dict[str, str] = {}

        def run_review(step: str, tier: str, label: str, max_tok: int) -> tuple[str, str]:
            safe_emit({"type": "model", "tier": tier, "label": label, "state": "start"})
            t0 = time.monotonic()
            txt, usage, _ = self.backend.converse(tier, Q.REVIEW_SYSTEM, prompt, max_tok)
            elapsed = round(time.monotonic() - t0, 1)
            cost = self.meter.add_llm(step, tier, label, usage)
            safe_emit(
                {
                    "type": "model",
                    "tier": tier,
                    "label": label,
                    "state": "done",
                    "elapsed_s": elapsed,
                    "usage": {
                        "inputTokens": usage.get("inputTokens", 0),
                        "outputTokens": usage.get("outputTokens", 0),
                    },
                    "cost": round(cost, 6),
                }
            )
            safe_emit({"type": "cost", "total": round(self.meter.total, 6)})
            return tier, txt

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(run_review, "Q3  reading", "opus", "Claude Opus", 8192): "opus",
                pool.submit(run_review, "Q3  reading", "nova", "Amazon Nova Pro", 4096): "nova",
            }
            for fut in as_completed(futures):
                tier, txt = fut.result()
                results[tier] = txt

        adjudication = self._model(
            "Q3  adjudication",
            "sonnet",
            "Claude Sonnet",
            Q.ADJUDICATE_SYSTEM,
            "REVIEW A (Claude Opus):\n"
            f"{results['opus']}\n\n"
            f"REVIEW B (Amazon Nova Pro):\n{results['nova']}",
            4096,
        )
        self.emit(
            {"type": "answer", "title": "Two model families, cross-checked", "text": adjudication}
        )

    # -- Q4: Cedar Gateway demo -- web_fetch blocked, fallback to KB -----
    def question_4(self) -> None:
        self.emit({"type": "question", "n": 4, "text": Q.QUESTIONS[3]})
        self.emit({"type": "phase", "label": "querying AgentCore Gateway for clinical trial data"})

        # Attempt the web_fetch tool call through the Gateway.
        result = self.backend.query_gateway(
            "web_fetch",
            {
                "url": (
                    "https://clinicaltrials.gov/api/query/full_studies"
                    "?expr=PCSK9&min_rnk=1&max_rnk=5&fmt=json"
                )
            },
        )

        if result.get("denied"):
            # Cedar policy blocked the request — emit the denial event.
            self.emit({"type": "policy_denied", "tool": "web_fetch", "reason": result["reason"]})
            self.emit(
                {
                    "type": "phase",
                    "label": "web access denied by Cedar policy — answering from knowledge base",
                }
            )
            # Fall back to KB retrieval.
            chunks = self._retrieve("Q4  retrieval", Q.QUESTIONS[3], n=12)
            self.emit({"type": "retrieval", "count": len(chunks)})
            text = self._model(
                "Q4  synthesis",
                "haiku",
                "Claude Haiku",
                Q.Q4_GATEWAY_SYSTEM,
                (
                    f"Note: web_fetch was denied by policy.\n\n"
                    f"Passages:\n{self._context(chunks)}\n\n"
                    f"Question: {Q.QUESTIONS[3]}"
                ),
            )
        else:
            # Web fetch succeeded (unlikely with ENFORCE policy, but handle gracefully).
            self.emit({"type": "phase", "label": "processing clinical trial data"})
            trial_data = str(result.get("result", ""))[:2000]
            chunks = self._retrieve("Q4  retrieval", Q.QUESTIONS[3], n=8)
            text = self._model(
                "Q4  synthesis",
                "haiku",
                "Claude Haiku",
                Q.Q4_GATEWAY_SYSTEM,
                (
                    f"Trial data:\n{trial_data}\n\n"
                    f"Passages:\n{self._context(chunks)}\n\n"
                    f"Question: {Q.QUESTIONS[3]}"
                ),
            )

        self.emit({"type": "answer", "title": "Clinical trials  ·  Claude Haiku", "text": text})

    # -- free-form: route then dispatch ----------------------------------
    def run_freeform(self, text: str) -> None:
        """Route a free-form question via Haiku, then run the appropriate path."""
        self.emit({"type": "phase", "label": "classifying question…"})
        path = self.route(text)
        self.emit({"type": "route", "path": path, "label": ROUTE_LABELS[path]})
        dispatch = {
            "SYNTHESIS": self.question_1,
            "ANALYSIS": self.question_2,
            "DEBATE": self.question_3,
        }
        dispatch[path](text)
        self.emit({"type": "receipt", **self.meter.receipt()})
        self.emit({"type": "done"})

    # -- run (canned questions) -----------------------------------------
    def run(self, which: Sequence[int] = (1, 2, 3)) -> None:
        # Emit measured KB costs as the first event.
        self.emit({"type": "setup_cost", **self.backend.kb_setup_costs()})

        steps = {
            1: self.question_1,
            2: self.question_2,
            3: self.question_3,
            4: self.question_4,
        }
        for n in which:
            steps[n]()
        self.emit({"type": "receipt", **self.meter.receipt()})
        self.emit({"type": "done"})

    @staticmethod
    def _extract_chart(stdout: str) -> tuple[str | None, str]:
        """Pull a base64 PNG printed by the generated code; return (data, rest)."""
        m = re.search(r"CHART_B64:([A-Za-z0-9+/=]+)", stdout)
        if not m:
            return None, stdout.strip()
        base64.b64decode(m.group(1))  # validate it decodes
        return m.group(1), stdout.replace(m.group(0), "").strip()
