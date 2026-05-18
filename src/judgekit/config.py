from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vendor: str
    model: str
    base_url: str | None = None
    max_tokens: int = 512


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    n: int | None = None
    split: str = "test"


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    judges: list[JudgeConfig]
    benchmarks: list[BenchmarkConfig]
    seed: int = 42
    budget_usd: float = 25.0


def load_config(path: Path) -> EvalConfig:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return EvalConfig.model_validate(data)
