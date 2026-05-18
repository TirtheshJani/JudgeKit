"""Tests for PubMedQAAdapter. Phase 2."""

from __future__ import annotations

from unittest.mock import patch

from judgekit.benchmarks.pubmedqa import PubMedQAAdapter

FAKE_ITEMS = [
    {
        "pubid": 12345,
        "question": "Does X affect Y?",
        "final_decision": "yes",
        "long_answer": "Evidence suggests X does affect Y.",
        "context": {"contexts": ["Study 1 found...", "Study 2 found..."]},
    },
    {
        "pubid": 99999,
        "question": "Is Z safe?",
        "final_decision": "no",
        "long_answer": "Z is not safe because...",
        "context": {"contexts": ["Report A...", "Report B..."]},
    },
]


def test_pubmedqa_adapter_returns_benchmark_item_with_correct_shape():
    """Items from PubMedQAAdapter have required fields populated."""
    with patch("judgekit.benchmarks.pubmedqa.load_dataset", return_value=FAKE_ITEMS):
        adapter = PubMedQAAdapter(n=2)
        items = list(adapter.iter_items())
    assert len(items) == 2
    item = items[0]
    assert item.id == "12345"
    assert item.question == "Does X affect Y?"
    assert item.reference_answer == "yes"
    assert item.metadata["candidate_answer"] == "Evidence suggests X does affect Y."
    assert item.metadata["context"] == ["Study 1 found...", "Study 2 found..."]


def test_pubmedqa_n_limits_to_one_item():
    """n=1 yields exactly 1 item."""
    with patch("judgekit.benchmarks.pubmedqa.load_dataset", return_value=FAKE_ITEMS):
        adapter = PubMedQAAdapter(n=1)
        items = list(adapter.iter_items())
    assert len(items) == 1


def test_pubmedqa_n_none_yields_all_items():
    """n=None yields all items from the dataset."""
    with patch("judgekit.benchmarks.pubmedqa.load_dataset", return_value=FAKE_ITEMS):
        adapter = PubMedQAAdapter(n=None)
        items = list(adapter.iter_items())
    assert len(items) == 2


def test_pubmedqa_id_is_always_string():
    """id field is always a string even if pubid is an int."""
    with patch("judgekit.benchmarks.pubmedqa.load_dataset", return_value=FAKE_ITEMS):
        adapter = PubMedQAAdapter(n=2)
        items = list(adapter.iter_items())
    for item in items:
        assert isinstance(item.id, str)


def test_pubmedqa_reference_answer_is_final_decision():
    """reference_answer is the final_decision value."""
    with patch("judgekit.benchmarks.pubmedqa.load_dataset", return_value=FAKE_ITEMS):
        adapter = PubMedQAAdapter(n=2)
        items = list(adapter.iter_items())
    assert items[0].reference_answer == "yes"
    assert items[1].reference_answer == "no"


def test_pubmedqa_passes_split_to_load_dataset():
    """iter_items passes the configured split to load_dataset."""
    with patch("judgekit.benchmarks.pubmedqa.load_dataset", return_value=FAKE_ITEMS) as mock_load:
        adapter = PubMedQAAdapter(n=1, split="test")
        list(adapter.iter_items())
    mock_load.assert_called_once_with("pubmed_qa", "pqa_labeled", split="test")
