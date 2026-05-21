"""Tests for agentcore_demo.cost.CostMeter -- pure logic, no AWS."""

import pytest

from agentcore_demo.cost import CostMeter

PRICING = {"haiku": (0.80, 4.00), "opus": (15.00, 75.00)}


def test_add_llm_computes_cost_from_usage():
    m = CostMeter(pricing=PRICING)
    cost = m.add_llm(
        "Q1", "haiku", "Claude Haiku", {"inputTokens": 1_000_000, "outputTokens": 1_000_000}
    )
    # 1M in @ $0.80 + 1M out @ $4.00
    assert cost == 4.80
    assert m.total == 4.80


def test_total_sums_all_rows():
    m = CostMeter(pricing=PRICING)
    m.add_llm("a", "haiku", "Haiku", {"inputTokens": 500_000, "outputTokens": 0})
    m.add_llm("b", "opus", "Opus", {"inputTokens": 0, "outputTokens": 100_000})
    # 0.4 + 7.5
    assert round(m.total, 2) == 7.90
    assert len(m.rows) == 2


def test_add_compute_uses_per_second_rate():
    m = CostMeter(pricing=PRICING, ci_per_second=0.01)
    m.add_compute("ci", seconds=10)
    assert m.total == 0.10


def test_receipt_is_serialisable_and_consistent():
    m = CostMeter(pricing=PRICING)
    m.add_llm("Q1", "haiku", "Claude Haiku", {"inputTokens": 12_000, "outputTokens": 1_000})
    receipt = m.receipt()
    assert set(receipt) == {"rows", "total"}
    assert receipt["rows"][0]["label"] == "Claude Haiku"
    assert receipt["total"] == round(sum(r["usd"] for r in receipt["rows"]), 4)


# ── add_retrieval ────────────────────────────────────────────────────────────


def test_add_retrieval_computes_cost_from_live_rate():
    m = CostMeter(pricing=PRICING, kb_query_usd_per_1k=0.40)
    cost = m.add_retrieval("Q1  retrieval", n_queries=1)
    # 1 query / 1,000 * $0.40 = $0.0004
    assert cost == pytest.approx(0.0004)
    assert m.total == pytest.approx(0.0004)


def test_add_retrieval_included_in_total():
    m = CostMeter(pricing=PRICING, kb_query_usd_per_1k=0.40)
    llm_cost = m.add_llm(
        "Q1", "haiku", "Claude Haiku", {"inputTokens": 1_000_000, "outputTokens": 0}
    )
    ret_cost = m.add_retrieval("Q1  retrieval")
    assert m.total == pytest.approx(llm_cost + ret_cost)


def test_receipt_has_no_estimated_field():
    """Cost rows no longer carry an 'estimated' flag -- all costs are measured."""
    m = CostMeter(pricing=PRICING, kb_query_usd_per_1k=0.40)
    m.add_llm("Q1", "haiku", "Claude Haiku", {"inputTokens": 1_000, "outputTokens": 100})
    m.add_retrieval("Q1  retrieval")
    rows = m.receipt()["rows"]
    for row in rows:
        assert "estimated" not in row
