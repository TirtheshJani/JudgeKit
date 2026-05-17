"""Tests for BudgetCircuitBreaker. Phase 1."""

from __future__ import annotations

import math

import pytest

from judgekit.budget.circuit_breaker import BudgetCircuitBreaker
from judgekit.budget.tracker import BudgetTracker
from judgekit.clients.base import BudgetExceededError


def test_circuit_trips_before_http_call() -> None:
    """Key test: tracker at $24.99, call with est. cost ~$0.03 → BudgetExceededError raised."""
    tracker = BudgetTracker(cap_usd=25.0)
    # Pre-load tracker to $24.99 by recording that cost
    tracker.record("anthropic", input_tokens=0, output_tokens=0, cost_usd=24.99)
    assert math.isclose(tracker.total_usd(), 24.99, abs_tol=1e-9)

    breaker = BudgetCircuitBreaker(tracker)

    # Anthropic 3333 input + 1333 output:
    # cost = 3333/1000*0.003 + 1333/1000*0.015 = 0.009999 + 0.019995 = 0.029994
    # 24.99 + 0.029994 = 25.019994 > 25.0 → should raise
    with pytest.raises(BudgetExceededError):
        breaker.check("anthropic", "claude-sonnet-4-5", prompt_tokens=3333, max_tokens=1333)


def test_call_within_budget_no_exception() -> None:
    """Tracker at $0.00, Anthropic 1000 input + 100 output ($0.0045) → no exception."""
    tracker = BudgetTracker(cap_usd=25.0)
    breaker = BudgetCircuitBreaker(tracker)
    # Should not raise
    breaker.check("anthropic", "claude-sonnet-4-5", prompt_tokens=1000, max_tokens=100)


def test_free_tier_vendor_never_trips() -> None:
    """Free-tier vendor with any token count never trips the circuit breaker."""
    tracker = BudgetTracker(cap_usd=25.0)
    # Even if tracker is nearly at cap, a free-tier vendor should never trip
    tracker.record("groq", input_tokens=0, output_tokens=0, cost_usd=24.9999)
    breaker = BudgetCircuitBreaker(tracker)
    # 1M tokens for a free-tier vendor should not raise
    breaker.check("groq", "llama-3.3-70b-versatile", prompt_tokens=1_000_000, max_tokens=1_000_000)


def test_exact_boundary_below_cap_no_exception() -> None:
    """Tracker at (cap - epsilon), call that costs epsilon → NO exception raised."""
    cap = 25.0
    epsilon = 1e-9
    tracker = BudgetTracker(cap_usd=cap)
    # Pre-set tracker to cap - epsilon
    tracker.record("groq", input_tokens=0, output_tokens=0, cost_usd=cap - epsilon)

    breaker = BudgetCircuitBreaker(tracker)
    # A free-tier call has zero estimated cost → 0 + (cap - epsilon) == cap - epsilon <= cap
    breaker.check("groq", "llama-3.3-70b-versatile", prompt_tokens=1000, max_tokens=100)


def test_tracker_at_cap_raises_immediately() -> None:
    """Tracker at exactly cap → any call with estimated cost > 0 raises immediately."""
    cap = 25.0
    tracker = BudgetTracker(cap_usd=cap)
    tracker.record("anthropic", input_tokens=0, output_tokens=0, cost_usd=cap)
    assert math.isclose(tracker.total_usd(), cap, abs_tol=1e-12)

    breaker = BudgetCircuitBreaker(tracker)
    # Any Anthropic call (non-zero cost) should raise
    with pytest.raises(BudgetExceededError):
        breaker.check("anthropic", "claude-sonnet-4-5", prompt_tokens=1, max_tokens=1)
