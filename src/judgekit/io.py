from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class JudgmentRecord:
    run_id: str
    benchmark_id: str
    item_id: str
    judge_id: str
    label: str
    raw_text: str
    prompt_tokens: int
    completion_tokens: int
    est_cost_usd: float
    latency_ms: float
    error: str | None = None


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._tmp = Path(str(path) + ".tmp")
        self._fh = self._tmp.open("a", encoding="utf-8")

    def write(self, record: JudgmentRecord) -> None:
        self._fh.write(json.dumps(asdict(record)) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
        os.replace(self._tmp, self._path)

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
