"""
conftest.py  --  shared test fixtures.

FakeBackend, TEST_PRICING, and TEST_KB_RATES live in agentcore_demo.fakes (so
both the app and the tests share exactly one implementation). This file
re-exports them and provides pytest fixtures.
"""

import pytest

from agentcore_demo.cost import CostMeter
from agentcore_demo.fakes import TEST_KB_RATES, TEST_PRICING, FakeBackend

__all__ = ["TEST_KB_RATES", "TEST_PRICING", "FakeBackend"]


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def meter() -> CostMeter:
    return CostMeter(pricing=TEST_PRICING, ci_per_second=0.0, **TEST_KB_RATES)


@pytest.fixture
def fake_env(monkeypatch):
    """Set DEMO_FAKE=1 so _make_backend_and_meter() returns the fake backend."""
    monkeypatch.setenv("DEMO_FAKE", "1")
