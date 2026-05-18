"""Tests for budget pricing table. Phase 1."""

from __future__ import annotations

import math

from judgekit.budget.pricing import ModelPricing, lookup


def test_lookup_anthropic_claude_sonnet_4_5() -> None:
    """lookup returns correct pricing for Anthropic Claude Sonnet 4.5."""
    result = lookup("anthropic", "claude-sonnet-4-5")
    assert result == ModelPricing(input_per_1k=0.003, output_per_1k=0.015)


def test_lookup_groq_llama_free_tier() -> None:
    """lookup returns zero pricing for Groq free tier."""
    result = lookup("groq", "llama-3.3-70b-versatile")
    assert result == ModelPricing(input_per_1k=0.0, output_per_1k=0.0)


def test_lookup_unknown_vendor_model_fallback() -> None:
    """lookup returns zero pricing for unknown vendor/model combinations."""
    result = lookup("unknown_vendor", "unknown_model")
    assert result == ModelPricing(input_per_1k=0.0, output_per_1k=0.0)


def test_cost_calculation_anthropic() -> None:
    """Cost for Anthropic: 1000 input + 100 output tokens = $0.0045."""
    pricing = lookup("anthropic", "claude-sonnet-4-5")
    input_cost = 1000 / 1000 * pricing.input_per_1k  # 0.003
    output_cost = 100 / 1000 * pricing.output_per_1k  # 0.0015
    total = input_cost + output_cost
    expected = 0.003 + 0.0015  # 0.0045
    assert math.isclose(total, expected, rel_tol=1e-9, abs_tol=1e-12)
