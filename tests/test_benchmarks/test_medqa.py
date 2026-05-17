"""Tests for MedQAAdapter. Phase 5."""

from __future__ import annotations

from unittest.mock import patch

from judgekit.benchmarks.medqa import MedQAAdapter

FAKE_ITEMS = [
    {
        "question": "A 45-year-old man presents with...",
        "answer": "Metformin",
        "answer_idx": "B",
        "options": {
            "A": "Insulin",
            "B": "Metformin",
            "C": "Glipizide",
            "D": "Acarbose",
        },
        "meta_info": "endocrine",
    },
    {
        "question": "A 30-year-old woman presents with chest pain...",
        "answer": "Aspirin",
        "answer_idx": "A",
        "options": {
            "A": "Aspirin",
            "B": "Ibuprofen",
            "C": "Acetaminophen",
            "D": "Naproxen",
        },
        "meta_info": "cardiology",
    },
]


def test_medqa_adapter_returns_benchmark_item_with_correct_shape():
    """Items from MedQAAdapter have required fields populated."""
    with patch("judgekit.benchmarks.medqa.load_dataset", return_value=FAKE_ITEMS):
        adapter = MedQAAdapter(n=2)
        items = list(adapter.iter_items())
    assert len(items) == 2
    item = items[0]
    assert item.id == "0"
    assert item.question == "A 45-year-old man presents with..."
    assert item.reference_answer == "B"
    assert item.metadata["candidate_answer"] == "Metformin"
    assert item.metadata["options"] == {
        "A": "Insulin",
        "B": "Metformin",
        "C": "Glipizide",
        "D": "Acarbose",
    }


def test_medqa_n_limits_to_one_item():
    """n=1 yields exactly 1 item."""
    with patch("judgekit.benchmarks.medqa.load_dataset", return_value=FAKE_ITEMS):
        adapter = MedQAAdapter(n=1)
        items = list(adapter.iter_items())
    assert len(items) == 1


def test_medqa_n_none_yields_all_items():
    """n=None yields all items from the dataset."""
    with patch("judgekit.benchmarks.medqa.load_dataset", return_value=FAKE_ITEMS):
        adapter = MedQAAdapter(n=None)
        items = list(adapter.iter_items())
    assert len(items) == 2


def test_medqa_id_is_always_string():
    """id field is always a string (enumerate index coerced to str)."""
    with patch("judgekit.benchmarks.medqa.load_dataset", return_value=FAKE_ITEMS):
        adapter = MedQAAdapter(n=2)
        items = list(adapter.iter_items())
    for item in items:
        assert isinstance(item.id, str)


def test_medqa_reference_answer_is_answer_idx():
    """reference_answer is the answer_idx value (letter), not the answer text."""
    with patch("judgekit.benchmarks.medqa.load_dataset", return_value=FAKE_ITEMS):
        adapter = MedQAAdapter(n=2)
        items = list(adapter.iter_items())
    assert items[0].reference_answer == "B"
    assert items[1].reference_answer == "A"


def test_medqa_passes_split_to_load_dataset():
    """iter_items passes the configured split to load_dataset."""
    with patch("judgekit.benchmarks.medqa.load_dataset", return_value=FAKE_ITEMS) as mock_load:
        adapter = MedQAAdapter(n=1, split="validation")
        list(adapter.iter_items())
    mock_load.assert_called_once_with("GBaker/MedQA-USMLE-4-options", split="validation")
