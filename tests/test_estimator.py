"""Tests for budget estimator module. Phase 4."""

from __future__ import annotations

from judgekit.budget.estimator import count_tokens, estimate_config
from judgekit.config import BenchmarkConfig, EvalConfig, JudgeConfig

# ---------------------------------------------------------------------------
# count_tokens tests
# ---------------------------------------------------------------------------


def test_count_tokens_empty_string() -> None:
    """count_tokens returns 0 for an empty string."""
    assert count_tokens("") == 0


def test_count_tokens_hello_world() -> None:
    """count_tokens returns a positive count for 'Hello world'."""
    assert count_tokens("Hello world") >= 1


def test_count_tokens_is_deterministic() -> None:
    """count_tokens returns the same result when called twice."""
    text = "The quick brown fox jumps over the lazy dog."
    assert count_tokens(text) == count_tokens(text)


def test_count_tokens_scales_with_length() -> None:
    """count_tokens returns more tokens for longer text."""
    short = "Hello"
    long = "Hello " * 100
    assert count_tokens(long) > count_tokens(short)


# ---------------------------------------------------------------------------
# estimate_config tests
# ---------------------------------------------------------------------------


def _make_config(judges: list[JudgeConfig], benchmarks: list[BenchmarkConfig]) -> EvalConfig:
    return EvalConfig(
        run_id="test_run",
        judges=judges,
        benchmarks=benchmarks,
        seed=42,
        budget_usd=25.0,
    )


def test_estimate_config_returns_one_row_per_judge() -> None:
    """estimate_config returns exactly one dict per judge in the config."""
    judges = [
        JudgeConfig(vendor="groq", model="llama-3.3-70b-versatile"),
        JudgeConfig(vendor="cerebras", model="llama-3.3-70b"),
    ]
    benchmarks = [BenchmarkConfig(name="pubmedqa", n=10)]
    cfg = _make_config(judges, benchmarks)
    rows = estimate_config(cfg)
    assert len(rows) == 2


def test_estimate_config_free_tier_cost_zero() -> None:
    """estimate_config returns est_cost_usd == 0.0 for a free-tier judge (groq)."""
    judges = [JudgeConfig(vendor="groq", model="llama-3.3-70b-versatile")]
    benchmarks = [BenchmarkConfig(name="pubmedqa", n=10)]
    cfg = _make_config(judges, benchmarks)
    rows = estimate_config(cfg)
    assert len(rows) == 1
    assert rows[0]["est_cost_usd"] == 0.0


def test_estimate_config_anthropic_cost_positive() -> None:
    """estimate_config returns est_cost_usd > 0 for anthropic judge with n_items > 0."""
    judges = [JudgeConfig(vendor="anthropic", model="claude-sonnet-4-5")]
    benchmarks = [BenchmarkConfig(name="pubmedqa", n=5)]
    cfg = _make_config(judges, benchmarks)
    rows = estimate_config(cfg)
    assert len(rows) == 1
    assert rows[0]["est_cost_usd"] > 0.0


def test_estimate_config_n_items() -> None:
    """estimate_config returns n_items == 7 when benchmark config sets n=7."""
    judges = [JudgeConfig(vendor="groq", model="llama-3.3-70b-versatile")]
    benchmarks = [BenchmarkConfig(name="pubmedqa", n=7)]
    cfg = _make_config(judges, benchmarks)
    rows = estimate_config(cfg)
    assert len(rows) == 1
    assert rows[0]["n_items"] == 7
