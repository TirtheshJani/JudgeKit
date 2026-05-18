"""Tests for PromptLoader and templates. Phase 2."""

from __future__ import annotations

import pytest
from jinja2 import UndefinedError

from judgekit.prompts.loader import PromptLoader


def test_medical_reference_v1_renders_question_and_answers():
    """medical_reference_v1.j2 renders with all three required variables."""
    loader = PromptLoader()
    rendered = loader.render(
        "medical_reference_v1.j2",
        question="Does aspirin reduce fever?",
        reference_answer="yes",
        candidate_answer="Aspirin is an antipyretic and reduces fever.",
    )
    assert "Does aspirin reduce fever?" in rendered
    assert "yes" in rendered
    assert "Aspirin is an antipyretic" in rendered
    assert "CORRECT" in rendered
    assert "INCORRECT" in rendered
    assert "UNCERTAIN" in rendered


def test_code_reference_v1_renders_question_and_answers():
    """code_reference_v1.j2 renders with question, reference_answer, candidate_answer."""
    loader = PromptLoader()
    rendered = loader.render(
        "code_reference_v1.j2",
        question="Write a function that returns the sum of a list.",
        reference_answer="def total(lst): return sum(lst)",
        candidate_answer="def total(lst): return sum(lst)",
    )
    assert "Write a function that returns the sum of a list." in rendered
    assert "def total(lst): return sum(lst)" in rendered
    assert "CORRECT" in rendered
    assert "INCORRECT" in rendered
    assert "UNCERTAIN" in rendered


def test_prompt_loader_raises_for_missing_variable():
    """Rendering a template with a missing variable raises UndefinedError (StrictUndefined)."""
    loader = PromptLoader()
    with pytest.raises(UndefinedError):
        loader.render(
            "medical_reference_v1.j2",
            question="Does aspirin reduce fever?",
            # reference_answer and candidate_answer intentionally omitted
        )


def test_prompt_loader_raises_for_unknown_template():
    """Requesting a non-existent template raises TemplateNotFound."""
    from jinja2 import TemplateNotFound

    loader = PromptLoader()
    with pytest.raises(TemplateNotFound):
        loader.render("nonexistent_template.j2", question="x")
