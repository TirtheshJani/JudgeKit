"""Token-accurate pre-flight cost estimator. Phase 4."""

from __future__ import annotations

from typing import Any

import tiktoken

from judgekit.budget.pricing import lookup
from judgekit.config import EvalConfig

# Representative prompt size approximation (same as CLI Phase 2).
_APPROX_INPUT_TOKENS = 600

# Lazy-initialised encoding. None until first call to count_tokens.
_ENCODING: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding | None:
    """Load cl100k_base encoding, returning None if the BPE file is unavailable."""
    global _ENCODING  # noqa: PLW0603
    if _ENCODING is not None:
        return _ENCODING
    try:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
        return _ENCODING
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken cl100k_base encoding.

    Falls back to a whitespace-split word count when the BPE data file cannot
    be loaded (e.g. in network-restricted environments).  The fallback is ~10%
    accurate but keeps the module importable offline.
    """
    enc = _get_encoding()
    if enc is not None:
        return len(enc.encode(text))
    # Offline fallback: words ≈ tokens for most English text.
    return len(text.split()) if text else 0


def estimate_config(config: EvalConfig, max_tokens: int = 256) -> list[dict[str, Any]]:
    """Estimate token cost for an eval config without loading real datasets.

    Returns a list of dicts, one per judge:
    {
        "vendor": str,
        "model": str,
        "n_items": int,      # sum of bc.n or 100 for each benchmark
        "est_input_tokens": int,   # n_items * approx_tokens_per_prompt
        "est_output_tokens": int,  # n_items * max_tokens
        "est_cost_usd": float,
    }

    Uses a fixed 600-token prompt approximation (same as CLI, but now returning
    structured data rather than printing).
    """
    n_items = sum(bc.n if bc.n is not None else 100 for bc in config.benchmarks)
    rows: list[dict[str, Any]] = []
    for jc in config.judges:
        pricing = lookup(jc.vendor, jc.model)
        est_input = n_items * _APPROX_INPUT_TOKENS
        est_output = n_items * max_tokens
        est_cost = (
            est_input / 1000 * pricing.input_per_1k
            + est_output / 1000 * pricing.output_per_1k
        )
        rows.append(
            {
                "vendor": jc.vendor,
                "model": jc.model,
                "n_items": n_items,
                "est_input_tokens": est_input,
                "est_output_tokens": est_output,
                "est_cost_usd": est_cost,
            }
        )
    return rows
