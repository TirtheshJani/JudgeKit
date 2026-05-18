"""Tests for MBPPAdapter. Phase 5."""

from __future__ import annotations

from unittest.mock import patch

from judgekit.benchmarks.mbpp import MBPPAdapter

FAKE_ITEMS = [
    {
        "task_id": 11,
        "text": "Write a python function to remove duplicates from a list.",
        "code": "def remove_dups(lst):\n    return list(set(lst))",
        "test_list": ["assert remove_dups([1,2,2,3]) == [1,2,3]"],
        "test_setup_code": "",
        "challenge_test_list": [],
        "source_file": "",
    },
    {
        "task_id": 12,
        "text": "Write a function to add two numbers.",
        "code": "def add(a, b):\n    return a + b",
        "test_list": ["assert add(1, 2) == 3"],
        "test_setup_code": "",
        "challenge_test_list": [],
        "source_file": "",
    },
]


def test_mbpp_adapter_returns_correct_shape():
    """Items from MBPPAdapter have required fields populated."""
    with patch("judgekit.benchmarks.mbpp.load_dataset", return_value=FAKE_ITEMS):
        adapter = MBPPAdapter(n=2)
        items = list(adapter.iter_items())
    assert len(items) == 2
    item = items[0]
    assert item.id == "11"
    assert "remove duplicates" in item.question
    assert "def remove_dups" in item.reference_answer
    assert isinstance(item.metadata["test_list"], list)


def test_mbpp_n_limits_to_one_item():
    """n=1 yields exactly 1 item."""
    with patch("judgekit.benchmarks.mbpp.load_dataset", return_value=FAKE_ITEMS):
        adapter = MBPPAdapter(n=1)
        items = list(adapter.iter_items())
    assert len(items) == 1


def test_mbpp_n_none_yields_all_items():
    """n=None yields all items from the dataset."""
    with patch("judgekit.benchmarks.mbpp.load_dataset", return_value=FAKE_ITEMS):
        adapter = MBPPAdapter(n=None)
        items = list(adapter.iter_items())
    assert len(items) == 2


def test_mbpp_id_is_always_string():
    """id field is always a string even if task_id is an int."""
    with patch("judgekit.benchmarks.mbpp.load_dataset", return_value=FAKE_ITEMS):
        adapter = MBPPAdapter(n=2)
        items = list(adapter.iter_items())
    for item in items:
        assert isinstance(item.id, str)


def test_mbpp_reference_answer_is_code():
    """reference_answer is the code field from the dataset record."""
    with patch("judgekit.benchmarks.mbpp.load_dataset", return_value=FAKE_ITEMS):
        adapter = MBPPAdapter(n=2)
        items = list(adapter.iter_items())
    assert items[0].reference_answer == FAKE_ITEMS[0]["code"]
    assert items[1].reference_answer == FAKE_ITEMS[1]["code"]


def test_mbpp_passes_split_to_load_dataset():
    """iter_items passes the configured split to load_dataset."""
    with patch("judgekit.benchmarks.mbpp.load_dataset", return_value=FAKE_ITEMS) as mock_load:
        adapter = MBPPAdapter(n=1, split="train")
        list(adapter.iter_items())
    mock_load.assert_called_once_with("mbpp", split="train")
