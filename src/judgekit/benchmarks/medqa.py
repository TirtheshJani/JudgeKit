from __future__ import annotations

from collections.abc import Iterator

from judgekit.benchmarks.base import BenchmarkAdapter, BenchmarkItem


class MedQAAdapter(BenchmarkAdapter):
    """Loads MedQA-USMLE 4-option from Hugging Face datasets."""

    def __init__(self, n: int | None = None, split: str = "test") -> None:
        self._n = n
        self._split = split

    def iter_items(self) -> Iterator[BenchmarkItem]:
        raise NotImplementedError
