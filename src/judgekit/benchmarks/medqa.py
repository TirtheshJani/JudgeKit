from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

from datasets import load_dataset

from judgekit.benchmarks.base import BenchmarkAdapter, BenchmarkItem


class MedQAAdapter(BenchmarkAdapter):
    """Loads MedQA-USMLE 4-option from Hugging Face datasets."""

    def __init__(self, n: int | None = None, split: str = "test") -> None:
        self._n = n
        self._split = split

    def iter_items(self) -> Iterator[BenchmarkItem]:
        dataset: Any = load_dataset("GBaker/MedQA-USMLE-4-options", split=self._split)
        items: Iterator[Any] = iter(dataset)
        if self._n is not None:
            items = itertools.islice(items, self._n)
        for idx, item in enumerate(items):
            yield BenchmarkItem(
                id=str(idx),
                question=str(item["question"]),
                reference_answer=str(item["answer_idx"]),
                metadata={
                    "options": item["options"],
                    "candidate_answer": str(item["answer"]),
                },
            )
