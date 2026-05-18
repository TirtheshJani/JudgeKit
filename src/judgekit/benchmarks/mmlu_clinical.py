from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

from datasets import load_dataset

from judgekit.benchmarks.base import BenchmarkAdapter, BenchmarkItem

CLINICAL_SUBJECTS = [
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "professional_medicine",
]

LETTERS = ["A", "B", "C", "D"]


class MMLUClinicalAdapter(BenchmarkAdapter):
    """Loads MMLU clinical subsets from Hugging Face datasets."""

    def __init__(
        self,
        subjects: list[str] | None = None,
        n: int | None = None,
        split: str = "test",
    ) -> None:
        self._subjects = subjects or CLINICAL_SUBJECTS
        self._n = n
        self._split = split

    def iter_items(self) -> Iterator[BenchmarkItem]:
        all_items: Iterator[Any] = itertools.chain.from_iterable(
            (
                (subject, idx, row)
                for idx, row in enumerate(
                    load_dataset("cais/mmlu", subject, split=self._split)
                )
            )
            for subject in self._subjects
        )
        if self._n is not None:
            all_items = itertools.islice(all_items, self._n)
        for subject, idx, item in all_items:
            yield BenchmarkItem(
                id=f"{subject}_{idx}",
                question=str(item["question"]),
                reference_answer=LETTERS[item["answer"]],
                metadata={
                    "choices": item["choices"],
                    "subject": subject,
                },
            )
