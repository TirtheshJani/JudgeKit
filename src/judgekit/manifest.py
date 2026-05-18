"""Run manifest: sibling JSON to judgments.jsonl describing a JudgeKit run."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from judgekit.config import EvalConfig

MANIFEST_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "run_id",
        "judgekit_version",
        "started_at",
        "ended_at",
        "config",
        "judges",
        "benchmarks",
        "vendor_spend_usd",
        "total_spend_usd",
        "n_records_written",
        "per_benchmark_counts",
        "per_judge_counts",
        "per_judge_error_counts",
    }
)


class ManifestSchemaError(ValueError):
    """Raised when a manifest file fails schema validation."""


@dataclass
class Manifest:
    run_id: str
    judgekit_version: str
    started_at: str
    ended_at: str
    config: dict[str, Any]
    judges: list[dict[str, Any]]
    benchmarks: list[dict[str, Any]]
    vendor_spend_usd: dict[str, float]
    total_spend_usd: float
    n_records_written: int
    per_benchmark_counts: dict[str, int]
    per_judge_counts: dict[str, int]
    per_judge_error_counts: dict[str, int] = field(default_factory=dict)


def write_manifest(
    path: Path,
    *,
    config: EvalConfig,
    vendor_spend: dict[str, float],
    n_records_written: int,
    per_benchmark_counts: dict[str, int],
    per_judge_counts: dict[str, int],
    per_judge_error_counts: dict[str, int],
    started_at_iso: str,
    ended_at_iso: str,
    judgekit_version: str,
) -> None:
    """Write a manifest JSON file describing a completed run."""
    payload: dict[str, Any] = {
        "run_id": config.run_id,
        "judgekit_version": judgekit_version,
        "started_at": started_at_iso,
        "ended_at": ended_at_iso,
        "config": config.model_dump(mode="json"),
        "judges": [jc.model_dump(mode="json") for jc in config.judges],
        "benchmarks": [bc.model_dump(mode="json") for bc in config.benchmarks],
        "vendor_spend_usd": {k: float(v) for k, v in sorted(vendor_spend.items())},
        "total_spend_usd": float(sum(vendor_spend.values())),
        "n_records_written": int(n_records_written),
        "per_benchmark_counts": {k: int(v) for k, v in sorted(per_benchmark_counts.items())},
        "per_judge_counts": {k: int(v) for k, v in sorted(per_judge_counts.items())},
        "per_judge_error_counts": {k: int(v) for k, v in sorted(per_judge_error_counts.items())},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def validate_manifest(path: Path) -> Manifest:
    """Load and validate a manifest. Raises ManifestSchemaError on failure."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestSchemaError(f"cannot read manifest at {path}: {exc}") from exc

    missing = MANIFEST_REQUIRED_KEYS - payload.keys()
    if missing:
        raise ManifestSchemaError(f"manifest at {path} missing keys: {sorted(missing)}")

    if not isinstance(payload["total_spend_usd"], (int, float)) or payload["total_spend_usd"] < 0:
        raise ManifestSchemaError("total_spend_usd must be a non-negative number")
    if not isinstance(payload["n_records_written"], int) or payload["n_records_written"] < 0:
        raise ManifestSchemaError("n_records_written must be a non-negative int")
    for k, v in payload["vendor_spend_usd"].items():
        if not isinstance(v, (int, float)) or v < 0:
            raise ManifestSchemaError(f"vendor_spend_usd[{k}] must be non-negative")

    return Manifest(
        run_id=payload["run_id"],
        judgekit_version=payload["judgekit_version"],
        started_at=payload["started_at"],
        ended_at=payload["ended_at"],
        config=payload["config"],
        judges=payload["judges"],
        benchmarks=payload["benchmarks"],
        vendor_spend_usd=payload["vendor_spend_usd"],
        total_spend_usd=payload["total_spend_usd"],
        n_records_written=payload["n_records_written"],
        per_benchmark_counts=payload["per_benchmark_counts"],
        per_judge_counts=payload["per_judge_counts"],
        per_judge_error_counts=payload["per_judge_error_counts"],
    )
