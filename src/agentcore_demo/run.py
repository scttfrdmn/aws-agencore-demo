"""
run.py  --  headless terminal runner for the Inside the Lines demo.

Builds the real AwsBackend + CostMeter from config.py (same seam as app.py),
runs the three questions through Agent, and prints a live transcript to stdout.
It is a terminal consumer of the same emit() event stream the WebSocket uses —
no new cost logic, no new question text.

Usage:
    python -m agentcore_demo.run                # all three questions
    python -m agentcore_demo.run --questions 1  # Q1 only (cheap iteration)
    python -m agentcore_demo.run --questions 1,3

    DEMO_FAKE=1 python -m agentcore_demo.run    # fake backend (no AWS)
"""

from __future__ import annotations

import argparse
import datetime
import sys
import textwrap

from agentcore_demo.agent import Agent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Headless terminal run of the Inside the Lines demo agent."
    )
    p.add_argument(
        "--questions",
        default="1,2,3",
        metavar="N[,N]",
        help="Comma-separated question numbers to run (default: 1,2,3).",
    )
    return p.parse_args()


def _build() -> tuple:
    """Build backend + meter using the same factory logic as app.py."""
    import os

    if os.environ.get("DEMO_FAKE") == "1":
        from agentcore_demo.fakes import make_fake_backend_and_meter  # noqa: PLC0415

        return make_fake_backend_and_meter()

    # Real AWS path — identical to app._build_backend / app._build_meter
    import importlib.util  # noqa: PLC0415

    if importlib.util.find_spec("config") is None:
        sys.exit(
            "config.py not found — copy config.example.py to config.py and fill it in,\n"
            "or set DEMO_FAKE=1 to run against the fake backend."
        )

    import config  # type: ignore[import]  # noqa: PLC0415
    from agentcore_demo.aws import AwsBackend  # noqa: PLC0415
    from agentcore_demo.cost import CostMeter  # noqa: PLC0415
    from agentcore_demo.pricing import fetch_rates  # noqa: PLC0415

    rates = fetch_rates(config.REGION)
    backend = AwsBackend(
        config.REGION,
        config.KB_ID,
        config.DATA_SOURCE_ID,
        config.MODELS,
        rates=rates,
        vector_bucket_name=getattr(config, "VECTOR_BUCKET_NAME", ""),
    )
    meter = CostMeter(
        pricing=config.PRICING,
        ci_per_second=config.CODE_INTERPRETER_PER_SECOND,
        kb_query_usd_per_1k=rates.get("s3v_query_usd_per_1k", 0.0),
    )
    return backend, meter


# ── terminal renderer ─────────────────────────────────────────────────────────

_WIDTH = 78
_HR = "─" * _WIDTH


def _hr(label: str = "") -> None:
    if label:
        pad = _WIDTH - len(label) - 4
        print(f"── {label} {'─' * max(0, pad)}")
    else:
        print(_HR)


def _emit_to_terminal(ev: dict) -> None:  # noqa: C901  (switch-like, intentional)
    t = ev["type"]

    if t == "setup_cost":
        print(f"\n{'KB SETUP COSTS (not in run total)':^{_WIDTH}}")
        print(_HR)
        print(f"  Corpus ingestion (computed from corpus size):  ${ev['ingestion_usd']:.4f}")
        mo = ev["storage_usd_per_month"]
        print(f"  S3 Vectors storage (from vector count):        ${mo:.4f}/mo")
        print()

    elif t == "question":
        print(f"\n{'':=<{_WIDTH}}")
        print(f"  QUESTION {ev['n']}")
        for line in textwrap.wrap(ev["text"], _WIDTH - 4):
            print(f"  {line}")
        print(f"{'':=<{_WIDTH}}")

    elif t == "phase":
        print(f"  ▸ {ev['label']}")

    elif t == "retrieval":
        print(f"  ▸ retrieved {ev['count']} passages")

    elif t == "model":
        if ev["state"] == "start":
            print(f"  ○ {ev['label']} …", end="", flush=True)
        else:
            u = ev.get("usage", {})
            cost = ev.get("cost", 0.0)
            print(
                f"\r  ● {ev['label']:<22}"
                f"  in={u.get('inputTokens', 0):>6,}  out={u.get('outputTokens', 0):>5,}"
                f"  ${cost:.6f}"
            )

    elif t == "code":
        print("\n  ── Generated code ──")
        for line in ev["text"].splitlines()[:12]:
            print(f"    {line}")
        extra = ev["text"].count("\n") - 12
        if extra > 0:
            print(f"    … ({extra} more lines)")
        print()

    elif t == "chart":
        kb = len(ev["data"]) * 3 // 4 // 1024
        print(f"  ▸ chart: {kb} KB PNG (base64, not displayed in terminal)")

    elif t == "answer":
        print(f"\n  ┌── {ev['title']}")
        wrapped = textwrap.wrap(ev["text"], _WIDTH - 6)
        for line in wrapped[:20]:
            print(f"  │  {line}")
        if len(wrapped) > 20:
            print(f"  │  … ({len(wrapped) - 20} more lines)")
        print(f"  └{'─' * (_WIDTH - 3)}")

    elif t == "cost":
        pass  # running total printed per-model; suppress intermediate updates

    elif t == "receipt":
        print(f"\n{'RECEIPT':^{_WIDTH}}")
        print(_HR)
        print(f"  {'Step':<26} {'Model':<22} {'In':>7} {'Out':>6} {'Cost':>11}")
        print(f"  {'-' * 26} {'-' * 22} {'-' * 7} {'-' * 6} {'-' * 11}")
        for row in ev["rows"]:
            in_t = f"{row['in_tokens']:,}" if row["in_tokens"] else "—"
            out_t = f"{row['out_tokens']:,}" if row["out_tokens"] else "—"
            print(f"  {row['step']:<26} {row['label']:<22} {in_t:>7} {out_t:>6}  ${row['usd']:.6f}")
        print(_HR)
        print(f"  {'TOTAL':>57}  ${ev['total']:.6f}")

    elif t == "done":
        print(f"\n{'✓ Run complete':^{_WIDTH}}")
        print(_HR)


def main() -> None:
    args = _parse_args()
    try:
        which = tuple(int(n.strip()) for n in args.questions.split(","))
    except ValueError:
        sys.exit(f"--questions: expected comma-separated integers, got {args.questions!r}")

    print(_HR)
    print(f"{'Inside the Lines — PCSK9':^{_WIDTH}}")
    print(f"{'Bedrock Knowledge Base  ·  Claude + Nova':^{_WIDTH}}")
    print(_HR)
    print(f"  Started:   {datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Questions: {list(which)}")
    print()

    backend, meter = _build()
    Agent(backend, meter, _emit_to_terminal).run(which=which)

    print(f"\n  Finished:  {datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(_HR)


if __name__ == "__main__":
    main()
