from __future__ import annotations

from collections.abc import Iterator

from judgekit.benchmarks.base import BenchmarkAdapter, BenchmarkItem


class HumanEvalAdapter(BenchmarkAdapter):
    """Loads HumanEval from Hugging Face datasets. Judges rate CORRECT/INCORRECT."""

    def __init__(self, n: int | None = None) -> None:
        self._n = n

    def iter_items(self) -> Iterator[BenchmarkItem]:
        raise NotImplementedError
