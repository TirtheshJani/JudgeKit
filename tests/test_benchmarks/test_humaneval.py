"""Tests for HumanEvalAdapter. Phase 5."""

from __future__ import annotations

from unittest.mock import patch

from judgekit.benchmarks.humaneval import HumanEvalAdapter

FAKE_ITEMS = [
    {
        "task_id": "HumanEval/0",
        "prompt": "def add(a: int, b: int) -> int:\n    \"\"\"Add two numbers.\"\"\"\n",
        "canonical_solution": "    return a + b\n",
        "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
        "entry_point": "add",
    },
    {
        "task_id": "HumanEval/1",
        "prompt": "def mul(a: int, b: int) -> int:\n    \"\"\"Multiply.\"\"\"\n",
        "canonical_solution": "    return a * b\n",
        "test": "def check(candidate):\n    assert candidate(2, 3) == 6\n",
        "entry_point": "mul",
    },
]


def test_humaneval_adapter_returns_correct_shape():
    """Items from HumanEvalAdapter have required fields populated correctly."""
    with patch("judgekit.benchmarks.humaneval.load_dataset", return_value=FAKE_ITEMS):
        adapter = HumanEvalAdapter(n=2)
        items = list(adapter.iter_items())
    assert len(items) == 2
    item = items[0]
    assert item.id == "HumanEval/0"
    assert "def add" in item.question
    assert "return" in item.reference_answer
    assert item.metadata["entry_point"] == "add"


def test_humaneval_n_limits_to_one_item():
    """n=1 yields exactly 1 item."""
    with patch("judgekit.benchmarks.humaneval.load_dataset", return_value=FAKE_ITEMS):
        adapter = HumanEvalAdapter(n=1)
        items = list(adapter.iter_items())
    assert len(items) == 1


def test_humaneval_n_none_yields_all_items():
    """n=None yields all items from the dataset."""
    with patch("judgekit.benchmarks.humaneval.load_dataset", return_value=FAKE_ITEMS):
        adapter = HumanEvalAdapter(n=None)
        items = list(adapter.iter_items())
    assert len(items) == 2


def test_humaneval_id_is_always_string():
    """id field is always a string."""
    with patch("judgekit.benchmarks.humaneval.load_dataset", return_value=FAKE_ITEMS):
        adapter = HumanEvalAdapter(n=2)
        items = list(adapter.iter_items())
    for item in items:
        assert isinstance(item.id, str)


def test_humaneval_reference_answer_is_canonical_solution():
    """reference_answer equals the canonical_solution field."""
    with patch("judgekit.benchmarks.humaneval.load_dataset", return_value=FAKE_ITEMS):
        adapter = HumanEvalAdapter(n=2)
        items = list(adapter.iter_items())
    assert items[0].reference_answer == "    return a + b\n"


def test_humaneval_metadata_has_entry_point_and_test():
    """metadata dict contains both 'entry_point' and 'test' keys."""
    with patch("judgekit.benchmarks.humaneval.load_dataset", return_value=FAKE_ITEMS):
        adapter = HumanEvalAdapter(n=2)
        items = list(adapter.iter_items())
    item = items[0]
    assert "entry_point" in item.metadata
    assert "test" in item.metadata
    assert item.metadata["test"] == "def check(candidate):\n    assert candidate(1, 2) == 3\n"
