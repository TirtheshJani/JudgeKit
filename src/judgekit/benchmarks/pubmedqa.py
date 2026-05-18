from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

from datasets import load_dataset

from judgekit.benchmarks.base import BenchmarkAdapter, BenchmarkItem


class PubMedQAAdapter(BenchmarkAdapter):
    """Loads PubMedQA (pqa_labeled) from Hugging Face datasets."""

    def __init__(self, n: int | None = None, split: str = "train") -> None:
        self._n = n
        self._split = split

    def iter_items(self) -> Iterator[BenchmarkItem]:
        dataset: Any = load_dataset("pubmed_qa", "pqa_labeled", split=self._split)
        items: Iterator[Any] = iter(dataset)
        if self._n is not None:
            items = itertools.islice(items, self._n)
        for item in items:
            yield BenchmarkItem(
                id=str(item["pubid"]),
                question=str(item["question"]),
                reference_answer=str(item["final_decision"]),
                metadata={
                    "context": item["context"]["contexts"],
                    "candidate_answer": str(item["long_answer"]),
                },
            )
