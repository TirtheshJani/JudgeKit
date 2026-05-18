"""Tests for manifest writer/validator. Phase 6."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from judgekit.config import EvalConfig, load_config
from judgekit.manifest import (
    Manifest,
    ManifestSchemaError,
    validate_manifest,
    write_manifest,
)


def _smoke_cfg(tmp_path: Path) -> EvalConfig:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
run_id: test_manifest
seed: 42
budget_usd: 1.0
judges:
  - vendor: groq
    model: llama-3.3-70b-versatile
    max_tokens: 256
benchmarks:
  - name: pubmedqa
    n: 3
"""
    )
    return load_config(cfg_path)


def test_write_manifest_creates_file_with_required_keys(tmp_path):
    cfg = _smoke_cfg(tmp_path)
    out = tmp_path / "manifest.json"

    write_manifest(
        out,
        config=cfg,
        vendor_spend={"groq": 0.0},
        n_records_written=3,
        per_benchmark_counts={"pubmedqa": 3},
        per_judge_counts={"groq/llama-3.3-70b-versatile": 3},
        per_judge_error_counts={"groq/llama-3.3-70b-versatile": 0},
        started_at_iso="2026-05-18T10:00:00+00:00",
        ended_at_iso="2026-05-18T10:00:05+00:00",
        judgekit_version="0.1.0",
    )

    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    required = {
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
    assert required.issubset(payload.keys()), f"missing keys: {required - payload.keys()}"


def test_total_spend_sums_per_vendor(tmp_path):
    cfg = _smoke_cfg(tmp_path)
    out = tmp_path / "m.json"

    write_manifest(
        out,
        config=cfg,
        vendor_spend={"anthropic": 12.3456, "groq": 0.0},
        n_records_written=10,
        per_benchmark_counts={"pubmedqa": 10},
        per_judge_counts={"anthropic/claude-sonnet-4-5": 5, "groq/llama-3.3-70b-versatile": 5},
        per_judge_error_counts={
            "anthropic/claude-sonnet-4-5": 0,
            "groq/llama-3.3-70b-versatile": 0,
        },
        started_at_iso="2026-05-18T10:00:00+00:00",
        ended_at_iso="2026-05-18T10:05:00+00:00",
        judgekit_version="0.1.0",
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["total_spend_usd"] == pytest.approx(12.3456, abs=1e-9)


def test_validate_manifest_accepts_well_formed(tmp_path):
    cfg = _smoke_cfg(tmp_path)
    out = tmp_path / "m.json"
    write_manifest(
        out,
        config=cfg,
        vendor_spend={"groq": 0.0},
        n_records_written=3,
        per_benchmark_counts={"pubmedqa": 3},
        per_judge_counts={"groq/llama-3.3-70b-versatile": 3},
        per_judge_error_counts={"groq/llama-3.3-70b-versatile": 0},
        started_at_iso="2026-05-18T10:00:00+00:00",
        ended_at_iso="2026-05-18T10:00:05+00:00",
        judgekit_version="0.1.0",
    )
    parsed = validate_manifest(out)
    assert isinstance(parsed, Manifest)
    assert parsed.run_id == "test_manifest"


def test_validate_manifest_rejects_missing_key(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"run_id": "x"}), encoding="utf-8")
    with pytest.raises(ManifestSchemaError):
        validate_manifest(bad)


def test_validate_manifest_rejects_negative_spend(tmp_path):
    cfg = _smoke_cfg(tmp_path)
    out = tmp_path / "m.json"
    write_manifest(
        out,
        config=cfg,
        vendor_spend={"groq": 0.0},
        n_records_written=0,
        per_benchmark_counts={},
        per_judge_counts={},
        per_judge_error_counts={},
        started_at_iso="2026-05-18T10:00:00+00:00",
        ended_at_iso="2026-05-18T10:00:00+00:00",
        judgekit_version="0.1.0",
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    payload["total_spend_usd"] = -1.0
    out.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestSchemaError):
        validate_manifest(out)
