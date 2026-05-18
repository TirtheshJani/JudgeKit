"""Tests for MMLUClinicalAdapter. Phase 5."""

from __future__ import annotations

from unittest.mock import patch

from judgekit.benchmarks.mmlu_clinical import MMLUClinicalAdapter

FAKE_ROW = {
    "question": "Which structure is responsible for X?",
    "choices": ["Brain", "Heart", "Liver", "Kidney"],
    "answer": 2,  # "C" = "Liver"
}

FAKE_DATASET = [FAKE_ROW, FAKE_ROW]  # 2 records per subject call


def test_mmlu_clinical_returns_correct_shape():
    """Items have the required fields: id, question, reference_answer, metadata."""
    with patch(
        "judgekit.benchmarks.mmlu_clinical.load_dataset", return_value=FAKE_DATASET
    ):
        adapter = MMLUClinicalAdapter(subjects=["anatomy"], n=1)
        items = list(adapter.iter_items())

    assert len(items) == 1
    item = items[0]
    assert item.id == "anatomy_0"
    assert item.question == "Which structure is responsible for X?"
    assert item.reference_answer == "C"
    assert item.metadata["choices"] == ["Brain", "Heart", "Liver", "Kidney"]
    assert item.metadata["subject"] == "anatomy"


def test_mmlu_clinical_n_limits_items():
    """n=1 yields exactly 1 item total across all subjects."""
    with patch(
        "judgekit.benchmarks.mmlu_clinical.load_dataset", return_value=FAKE_DATASET
    ):
        adapter = MMLUClinicalAdapter(n=1)
        items = list(adapter.iter_items())

    assert len(items) == 1


def test_mmlu_clinical_n_none_yields_all_items():
    """n=None yields all items: 2 per subject × 4 subjects = 8 total."""
    with patch(
        "judgekit.benchmarks.mmlu_clinical.load_dataset", return_value=FAKE_DATASET
    ):
        adapter = MMLUClinicalAdapter(n=None)
        items = list(adapter.iter_items())

    assert len(items) == 8


def test_mmlu_clinical_id_format():
    """id contains subject name and underscore-separated index."""
    with patch(
        "judgekit.benchmarks.mmlu_clinical.load_dataset", return_value=FAKE_DATASET
    ):
        adapter = MMLUClinicalAdapter(subjects=["anatomy"], n=2)
        items = list(adapter.iter_items())

    assert items[0].id == "anatomy_0"
    assert items[1].id == "anatomy_1"
    assert "_" in items[0].id
    assert "anatomy" in items[0].id


def test_mmlu_clinical_reference_answer_is_letter():
    """answer int 2 maps to letter 'C'."""
    with patch(
        "judgekit.benchmarks.mmlu_clinical.load_dataset", return_value=FAKE_DATASET
    ):
        adapter = MMLUClinicalAdapter(subjects=["anatomy"], n=1)
        items = list(adapter.iter_items())

    assert items[0].reference_answer == "C"


def test_mmlu_clinical_uses_configured_subjects():
    """Passing subjects=['anatomy'] only calls load_dataset once with 'anatomy'."""
    with patch(
        "judgekit.benchmarks.mmlu_clinical.load_dataset", return_value=FAKE_DATASET
    ) as mock_load:
        adapter = MMLUClinicalAdapter(subjects=["anatomy"], n=None)
        list(adapter.iter_items())

    assert mock_load.call_count == 1
    mock_load.assert_called_once_with("cais/mmlu", "anatomy", split="test")
