from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

from datasets import load_dataset

from judgekit.benchmarks.base import BenchmarkAdapter, BenchmarkItem


class HumanEvalAdapter(BenchmarkAdapter):
    """Loads HumanEval from Hugging Face datasets. Judges rate CORRECT/INCORRECT."""

    def __init__(self, n: int | None = None, split: str = "test") -> None:
        self._n = n
        self._split = split

    def iter_items(self) -> Iterator[BenchmarkItem]:
        dataset: Any = load_dataset("openai_humaneval", split=self._split)
        items: Iterator[Any] = iter(dataset)
        if self._n is not None:
            items = itertools.islice(items, self._n)
        for item in items:
            yield BenchmarkItem(
                id=str(item["task_id"]),
                question=str(item["prompt"]),
                reference_answer=str(item["canonical_solution"]),
                metadata={
                    "entry_point": str(item["entry_point"]),
                    "test": str(item["test"]),
                },
            )
