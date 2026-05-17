"""Tests for the eval runner. Phase 2."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from judgekit.benchmarks.base import BenchmarkItem
from judgekit.clients.base import BudgetExceededError, JudgeResponse
from judgekit.config import BenchmarkConfig, EvalConfig, JudgeConfig
from judgekit.runner import _parse_label, run_eval, validate_run

# ---------------------------------------------------------------------------
# Step 1: Label parser tests
# ---------------------------------------------------------------------------


def test_parse_label_extracts_last_label_line() -> None:
    assert _parse_label("Some reasoning.\n\nCORRECT") == "CORRECT"
    assert _parse_label("INCORRECT") == "INCORRECT"
    assert _parse_label("No label here") == "UNPARSABLE"
    # DeepSeek-style: reasoning block then label
    assert _parse_label("<think>lots of thinking</think>\nCORRECT") == "CORRECT"
    assert _parse_label("label: UNCERTAIN\nmore text\nCORRECT") == "CORRECT"


def test_parse_label_returns_unparsable_for_empty() -> None:
    assert _parse_label("") == "UNPARSABLE"


def test_parse_label_case_insensitive() -> None:
    # The function uppercases lines before comparing
    assert _parse_label("correct") == "CORRECT"
    assert _parse_label("Incorrect") == "INCORRECT"
    assert _parse_label("uncertain") == "UNCERTAIN"


def test_parse_label_last_wins() -> None:
    # When multiple labels appear, last one wins
    assert _parse_label("CORRECT\nINCORRECT") == "INCORRECT"
    assert _parse_label("INCORRECT\nCORRECT") == "CORRECT"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mock_judge(text: str = "CORRECT") -> MagicMock:
    judge = MagicMock()
    judge.judge.return_value = JudgeResponse(
        text=text,
        prompt_tokens=100,
        completion_tokens=20,
        vendor="groq",
        model="llama-3.3-70b-versatile",
        latency_ms=150.0,
    )
    return judge


def make_mock_adapter(n: int = 3) -> MagicMock:
    adapter = MagicMock()
    adapter.iter_items.return_value = iter(
        [
            BenchmarkItem(
                id=str(i),
                question=f"Question {i}",
                reference_answer="yes",
                metadata={"candidate_answer": f"Answer {i}"},
            )
            for i in range(n)
        ]
    )
    return adapter


def make_config(
    n_judges: int = 1,
    n_items: int = 3,
    budget_usd: float = 1.0,
    benchmark: str = "pubmedqa",
) -> EvalConfig:
    judges = [
        JudgeConfig(vendor="groq", model="llama-3.3-70b-versatile", max_tokens=256)
        for _ in range(n_judges)
    ]
    return EvalConfig(
        run_id="test_run",
        judges=judges,
        benchmarks=[BenchmarkConfig(name=benchmark, n=n_items)],
        budget_usd=budget_usd,
    )


# ---------------------------------------------------------------------------
# Step 2: Core runner test — n=3, 1 judge, all mocked
# ---------------------------------------------------------------------------


def test_run_eval_writes_3_jsonl_records(tmp_path: Path) -> None:
    config = make_config(n_judges=1, n_items=3)
    mock_adapter = make_mock_adapter(3)
    mock_judge = make_mock_judge("CORRECT")

    with (
        patch("judgekit.runner.build_judge", return_value=mock_judge),
        patch("judgekit.runner._build_adapter", return_value=mock_adapter),
    ):
        out = run_eval(config, tmp_path)

    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) == 3

    r = records[0]
    assert r["run_id"] == "test_run"
    assert r["benchmark_id"] == "pubmedqa"
    assert r["judge_id"] == "groq/llama-3.3-70b-versatile"
    assert r["label"] == "CORRECT"
    assert r["prompt_tokens"] == 100
    assert r["completion_tokens"] == 20
    assert r["latency_ms"] == 150.0
    assert r["error"] is None


# ---------------------------------------------------------------------------
# Step 3a: validate_run
# ---------------------------------------------------------------------------


def test_validate_run_returns_true_for_valid_jsonl(tmp_path: Path) -> None:
    config = make_config(n_judges=1, n_items=2)
    mock_adapter = make_mock_adapter(2)
    mock_judge = make_mock_judge("CORRECT")

    with (
        patch("judgekit.runner.build_judge", return_value=mock_judge),
        patch("judgekit.runner._build_adapter", return_value=mock_adapter),
    ):
        out = run_eval(config, tmp_path)

    assert validate_run(out) is True


def test_validate_run_returns_false_for_invalid_jsonl(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text('{"run_id": "x", "missing_keys": true}\n')
    assert validate_run(bad_file) is False


def test_validate_run_returns_false_for_missing_file(tmp_path: Path) -> None:
    assert validate_run(tmp_path / "nonexistent.jsonl") is False


def test_validate_run_returns_false_for_malformed_json(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text("not json at all\n")
    assert validate_run(bad_file) is False


# ---------------------------------------------------------------------------
# Step 3b: 2 judges × 3 items = 6 records
# ---------------------------------------------------------------------------


def test_run_eval_2_judges_3_items_yields_6_records(tmp_path: Path) -> None:
    config = EvalConfig(
        run_id="multi_judge_run",
        judges=[
            JudgeConfig(vendor="groq", model="llama-3.3-70b-versatile", max_tokens=256),
            JudgeConfig(vendor="cerebras", model="llama-3.3-70b", max_tokens=256),
        ],
        benchmarks=[BenchmarkConfig(name="pubmedqa", n=3)],
        budget_usd=1.0,
    )
    mock_adapter = make_mock_adapter(3)
    mock_judge_groq = make_mock_judge("CORRECT")
    mock_judge_cerebras = make_mock_judge("INCORRECT")

    # build_judge called twice (once per judge), return different judges
    call_count = [0]

    def side_effect_build_judge(vendor: str, model: str, base_url: object = None) -> MagicMock:
        call_count[0] += 1
        if vendor == "groq":
            return mock_judge_groq
        return mock_judge_cerebras

    with (
        patch("judgekit.runner.build_judge", side_effect=side_effect_build_judge),
        patch("judgekit.runner._build_adapter", return_value=mock_adapter),
    ):
        out = run_eval(config, tmp_path)

    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) == 6

    judge_ids = {r["judge_id"] for r in records}
    assert "groq/llama-3.3-70b-versatile" in judge_ids
    assert "cerebras/llama-3.3-70b" in judge_ids


# ---------------------------------------------------------------------------
# Step 3c: Judge raises exception → label=ERROR, run continues
# ---------------------------------------------------------------------------


def test_run_eval_judge_exception_records_error_and_continues(tmp_path: Path) -> None:
    config = make_config(n_judges=1, n_items=3)
    mock_adapter = make_mock_adapter(3)
    mock_judge = MagicMock()

    call_responses = [
        RuntimeError("API timeout"),
        JudgeResponse(
            text="CORRECT",
            prompt_tokens=100,
            completion_tokens=20,
            vendor="groq",
            model="llama-3.3-70b-versatile",
            latency_ms=150.0,
        ),
        JudgeResponse(
            text="INCORRECT",
            prompt_tokens=80,
            completion_tokens=15,
            vendor="groq",
            model="llama-3.3-70b-versatile",
            latency_ms=120.0,
        ),
    ]

    def side_effect(*args: object, **kwargs: object) -> JudgeResponse:
        result = call_responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    mock_judge.judge.side_effect = side_effect

    with (
        patch("judgekit.runner.build_judge", return_value=mock_judge),
        patch("judgekit.runner._build_adapter", return_value=mock_adapter),
    ):
        out = run_eval(config, tmp_path)

    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) == 3

    error_records = [r for r in records if r["label"] == "ERROR"]
    assert len(error_records) == 1
    assert error_records[0]["error"] is not None
    assert "API timeout" in error_records[0]["error"]

    # Other records must have succeeded
    ok_records = [r for r in records if r["label"] != "ERROR"]
    assert len(ok_records) == 2


# ---------------------------------------------------------------------------
# Step 3d: dry_run=True returns path without calling judge
# ---------------------------------------------------------------------------


def test_run_eval_dry_run_returns_path_without_api_calls(tmp_path: Path) -> None:
    config = make_config(n_judges=1, n_items=3)
    mock_adapter = make_mock_adapter(3)
    mock_judge = make_mock_judge("CORRECT")

    with (
        patch("judgekit.runner.build_judge", return_value=mock_judge),
        patch("judgekit.runner._build_adapter", return_value=mock_adapter),
    ):
        out = run_eval(config, tmp_path, dry_run=True)

    # Should return a path
    assert isinstance(out, Path)
    # Judge should NOT have been called
    mock_judge.judge.assert_not_called()
    # File should not exist (dry run skips actual writing)
    assert not out.exists()


# ---------------------------------------------------------------------------
# Step 3e: BudgetExceededError propagates out of run_eval
# ---------------------------------------------------------------------------


def test_run_eval_budget_exceeded_propagates(tmp_path: Path) -> None:
    # Set budget so low that any call would exceed it
    config = EvalConfig(
        run_id="budget_test",
        judges=[
            JudgeConfig(
                vendor="anthropic",
                model="claude-sonnet-4-5",
                max_tokens=512,
            )
        ],
        benchmarks=[BenchmarkConfig(name="pubmedqa", n=1)],
        budget_usd=0.0,  # zero budget — any call exceeds
    )
    mock_adapter = make_mock_adapter(1)
    mock_judge = make_mock_judge("CORRECT")

    with (
        patch("judgekit.runner.build_judge", return_value=mock_judge),
        patch("judgekit.runner._build_adapter", return_value=mock_adapter),
    ):
        with pytest.raises(BudgetExceededError):
            run_eval(config, tmp_path)
