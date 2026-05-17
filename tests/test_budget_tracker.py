"""Tests for BudgetTracker. Phase 1."""

from __future__ import annotations

import math
import threading

import pytest

from judgekit.budget.tracker import BudgetTracker


def test_fresh_tracker_starts_at_zero() -> None:
    """A new tracker with a cap starts with total_usd() == 0.0."""
    tracker = BudgetTracker(cap_usd=10.0)
    assert tracker.total_usd() == 0.0


def test_record_single_call_updates_total() -> None:
    """After recording one Anthropic call, total_usd() reflects the cost."""
    tracker = BudgetTracker(cap_usd=10.0)
    # 1000 input + 100 output for anthropic = $0.0045
    tracker.record("anthropic", input_tokens=1000, output_tokens=100, cost_usd=0.0045)
    assert math.isclose(tracker.total_usd(), 0.0045, abs_tol=1e-12)


def test_ten_calls_accumulate_correctly() -> None:
    """10 calls at $0.0045 each → total_usd() == $0.045."""
    tracker = BudgetTracker(cap_usd=1.0)
    for _ in range(10):
        tracker.record("anthropic", input_tokens=1000, output_tokens=100, cost_usd=0.0045)
    assert math.isclose(tracker.total_usd(), 0.045, abs_tol=1e-9)


def test_env_var_sets_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """JUDGEKIT_ANTHROPIC_BUDGET_USD env var sets cap_usd when no explicit cap given."""
    monkeypatch.setenv("JUDGEKIT_ANTHROPIC_BUDGET_USD", "5.0")
    tracker = BudgetTracker()
    assert tracker.cap_usd == 5.0


def test_thread_safety() -> None:
    """10 threads each record 100 calls at $0.001 → total == $1.0."""
    tracker = BudgetTracker(cap_usd=100.0)
    num_threads = 10
    calls_per_thread = 100
    cost_per_call = 0.001
    expected_total = num_threads * calls_per_thread * cost_per_call  # 1.0

    def worker() -> None:
        for _ in range(calls_per_thread):
            tracker.record("groq", input_tokens=100, output_tokens=50, cost_usd=cost_per_call)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert math.isclose(tracker.total_usd(), expected_total, abs_tol=1e-9)


def test_vendor_summary_returns_per_vendor_breakdown() -> None:
    """vendor_summary() returns per-vendor breakdown with correct token counts."""
    tracker = BudgetTracker(cap_usd=50.0)
    tracker.record("anthropic", input_tokens=1000, output_tokens=100, cost_usd=0.0045)
    tracker.record("anthropic", input_tokens=500, output_tokens=50, cost_usd=0.00225)
    tracker.record("groq", input_tokens=2000, output_tokens=200, cost_usd=0.0)

    summary = tracker.vendor_summary()

    assert "anthropic" in summary
    assert "groq" in summary

    anthropic = summary["anthropic"]
    assert anthropic.input_tokens == 1500
    assert anthropic.output_tokens == 150
    assert math.isclose(anthropic.cost_usd, 0.00675, abs_tol=1e-12)

    groq = summary["groq"]
    assert groq.input_tokens == 2000
    assert groq.output_tokens == 200
    assert groq.cost_usd == 0.0
