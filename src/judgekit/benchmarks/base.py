from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkItem:
    id: str
    question: str
    reference_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkAdapter(ABC):
    @abstractmethod
    def iter_items(self) -> Iterator[BenchmarkItem]: ...
