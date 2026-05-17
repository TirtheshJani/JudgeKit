from __future__ import annotations

from collections.abc import Iterator

from judgekit.benchmarks.base import BenchmarkAdapter, BenchmarkItem

CLINICAL_SUBJECTS = [
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "professional_medicine",
]


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
        raise NotImplementedError
