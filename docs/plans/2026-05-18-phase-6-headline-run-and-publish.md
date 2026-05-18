# Phase 6: Headline Run, Analysis, Tech Note, PyPI Implementation Plan

> **For agentic workers:** Use `.claude/skills/executing-plans` to execute task-by-task. Each task ends with a commit; do not batch. Steps use checkbox (`- [ ]`) syntax for tracking. TDD per `.claude/skills/test-driven-development`: failing test first, minimal implementation, green, commit.

**Goal:** Land all non-spend Phase 6 deliverables (manifest writer, enhanced report CLI with golden CSV, paper tech-note skeleton, README budget + repro sections) so the only thing left between `main` and a tagged `v0.1.0` release is the live 5-judge headline run, which is gated on user authorization and the four missing API keys (ANTHROPIC, CEREBRAS, SAMBANOVA, OPENROUTER).

**Architecture:** Manifest is a sibling JSON file next to `judgments.jsonl` describing the run (config snapshot, judges, benchmarks, totals, vendor spend, timestamps). The runner writes it on close. The CLI `report` command grows a `--out` CSV that bundles the existing AgreementReport plus per-vendor spend and per-benchmark counts — a golden test pins the byte layout against a deterministic fixture. The paper is a LaTeX skeleton with placeholders that the headline-run task fills in. The README gets the user-facing surface: install, 5-judge table, budget plan, repro recipe, dataset licenses.

**Tech Stack:** Python 3.11, pydantic, typer, pytest, jinja2 (existing), LaTeX (paper only), GitHub Actions (publish.yml already in place).

**Out of scope this session:** the live headline run (Task 5 below) and `v0.1.0` tag/push. Both require credentials and explicit authorization.

---

## Task 1: Manifest writer + runner integration (TDD)

**Files:**
- Create: `src/judgekit/manifest.py`
- Create: `tests/test_manifest.py`
- Modify: `src/judgekit/runner.py:72-142` (write manifest on close)

- [ ] **Step 1.1: Write the failing schema test**

Create `tests/test_manifest.py`:

```python
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
        per_judge_error_counts={"anthropic/claude-sonnet-4-5": 0, "groq/llama-3.3-70b-versatile": 0},
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
```

- [ ] **Step 1.2: Run tests, expect import failure**

Run: `uv run pytest tests/test_manifest.py -v`
Expected: 5 failures with `ModuleNotFoundError: No module named 'judgekit.manifest'`.

- [ ] **Step 1.3: Implement `src/judgekit/manifest.py`**

```python
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
```

- [ ] **Step 1.4: Run manifest tests, expect green**

Run: `uv run pytest tests/test_manifest.py -v`
Expected: 5 passed.

- [ ] **Step 1.5: Wire manifest into runner**

Modify `src/judgekit/runner.py`. At the top, add imports:

```python
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as _pkg_version

from judgekit.manifest import write_manifest
```

In `run_eval`, replace the `with JsonlWriter(...)` block with a version that tracks counters and writes the manifest after the JSONL is closed:

```python
def run_eval(config: EvalConfig, output_dir: Path, dry_run: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "judgments.jsonl"
    manifest_path = run_dir / "manifest.json"

    if dry_run:
        return jsonl_path

    tracker = BudgetTracker(cap_usd=config.budget_usd)
    breaker = BudgetCircuitBreaker(tracker)
    loader = PromptLoader()

    judges = [build_judge(jc.vendor, jc.model, jc.base_url) for jc in config.judges]
    adapters = [_build_adapter(bc) for bc in config.benchmarks]

    per_benchmark_counts: Counter[str] = Counter()
    per_judge_counts: Counter[str] = Counter()
    per_judge_error_counts: Counter[str] = Counter()
    n_records_written = 0

    started_at = datetime.now(timezone.utc).isoformat()

    with JsonlWriter(jsonl_path) as writer:
        for bc, adapter in zip(config.benchmarks, adapters, strict=True):
            template = BENCHMARK_TEMPLATE[bc.name]
            for item in adapter.iter_items():
                candidate = item.metadata.get("candidate_answer", "")
                for jc, judge in zip(config.judges, judges, strict=True):
                    pricing = lookup(jc.vendor, jc.model)
                    prompt = loader.render(
                        template,
                        question=item.question,
                        reference_answer=item.reference_answer,
                        candidate_answer=candidate,
                    )
                    est_prompt_tokens = len(prompt) // 4
                    breaker.check(jc.vendor, jc.model, est_prompt_tokens, jc.max_tokens)
                    judge_id = f"{jc.vendor}/{jc.model}"
                    try:
                        resp = judge.judge(prompt, max_tokens=jc.max_tokens)
                        label = _parse_label(resp.text)
                        error = None
                        est_cost = (
                            resp.prompt_tokens / 1000 * pricing.input_per_1k
                            + resp.completion_tokens / 1000 * pricing.output_per_1k
                        )
                        tracker.record(
                            jc.vendor, resp.prompt_tokens, resp.completion_tokens, est_cost
                        )
                    except Exception as exc:
                        label = "ERROR"
                        error = str(exc)
                        est_cost = 0.0
                        resp = None
                        per_judge_error_counts[judge_id] += 1

                    writer.write(
                        JudgmentRecord(
                            run_id=config.run_id,
                            benchmark_id=bc.name,
                            item_id=item.id,
                            judge_id=judge_id,
                            label=label,
                            raw_text=resp.text if resp else "",
                            prompt_tokens=resp.prompt_tokens if resp else 0,
                            completion_tokens=resp.completion_tokens if resp else 0,
                            est_cost_usd=est_cost,
                            latency_ms=resp.latency_ms if resp else 0.0,
                            error=error,
                        )
                    )
                    n_records_written += 1
                    per_benchmark_counts[bc.name] += 1
                    per_judge_counts[judge_id] += 1

    ended_at = datetime.now(timezone.utc).isoformat()

    try:
        jk_version = _pkg_version("judgekit")
    except PackageNotFoundError:
        jk_version = "0.0.0+dev"

    write_manifest(
        manifest_path,
        config=config,
        vendor_spend=dict(tracker.per_vendor_spend()),
        n_records_written=n_records_written,
        per_benchmark_counts=dict(per_benchmark_counts),
        per_judge_counts=dict(per_judge_counts),
        per_judge_error_counts=dict(per_judge_error_counts),
        started_at_iso=started_at,
        ended_at_iso=ended_at,
        judgekit_version=jk_version,
    )

    return jsonl_path
```

- [ ] **Step 1.6: Verify BudgetTracker exposes `per_vendor_spend()`**

Run: `uv run python -c "from judgekit.budget.tracker import BudgetTracker; t = BudgetTracker(cap_usd=1.0); t.record('groq', 100, 20, 0.0); print(t.per_vendor_spend())"`
Expected: a dict with `'groq': 0.0` (or similar). If the method is missing, add it to `BudgetTracker` returning a `dict[str, float]` keyed by vendor (sum of `est_cost_usd` recorded per vendor). Use the existing internal state — do not duplicate accounting. Add `tests/test_budget_tracker.py::test_per_vendor_spend_returns_dict` first if you change the public surface (TDD).

- [ ] **Step 1.7: Run full suite to verify no regression**

Run: `uv run pytest --tb=short`
Expected: all tests pass, including `test_manifest.py` (5 new tests).

- [ ] **Step 1.8: Commit**

```bash
git add src/judgekit/manifest.py tests/test_manifest.py src/judgekit/runner.py
# include budget/tracker.py only if step 1.6 modified it
git commit -m "feat: run manifest writer + runner integration (Phase 6)"
```

---

## Task 2: Enhanced report CLI with golden CSV (TDD)

**Files:**
- Modify: `src/judgekit/cli.py:60-101` (rewrite `report`)
- Modify: `tests/test_cli.py:87-119` (replace minimal report tests, add golden CSV test)
- Create: `tests/fixtures/golden_report_v1.csv`

- [ ] **Step 2.1: Write the failing golden-CSV test**

Append to `tests/test_cli.py`:

```python
def _write_golden_jsonl(path):
    """Deterministic 3-judge × 6-item fixture. Stable order; pinned label distribution."""
    judges = ["anthropic/claude-sonnet-4-5", "groq/llama-3.3-70b-versatile", "sambanova/DeepSeek-R1"]
    labels_by_judge = {
        judges[0]: ["CORRECT", "CORRECT", "INCORRECT", "CORRECT", "UNCERTAIN", "CORRECT"],
        judges[1]: ["CORRECT", "INCORRECT", "INCORRECT", "CORRECT", "UNCERTAIN", "CORRECT"],
        judges[2]: ["CORRECT", "CORRECT", "INCORRECT", "INCORRECT", "UNCERTAIN", "CORRECT"],
    }
    costs_by_judge = {judges[0]: 0.004, judges[1]: 0.0, judges[2]: 0.0}
    benchmarks = ["pubmedqa", "pubmedqa", "medqa", "medqa", "humaneval", "humaneval"]
    records = []
    for jg in judges:
        for i, lbl in enumerate(labels_by_judge[jg]):
            records.append(
                {
                    "run_id": "golden",
                    "benchmark_id": benchmarks[i],
                    "item_id": str(i),
                    "judge_id": jg,
                    "label": lbl,
                    "raw_text": lbl,
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "est_cost_usd": costs_by_judge[jg],
                    "latency_ms": 100.0,
                    "error": None,
                }
            )
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def test_report_golden_csv(tmp_path):
    from pathlib import Path

    run_dir = tmp_path / "golden"
    run_dir.mkdir()
    _write_golden_jsonl(run_dir / "judgments.jsonl")

    out = tmp_path / "report.csv"
    result = cli_runner.invoke(
        app, ["report", "golden", "--results-dir", str(tmp_path), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()

    golden = Path("tests/fixtures/golden_report_v1.csv").read_text(encoding="utf-8")
    actual = out.read_text(encoding="utf-8")
    assert actual == golden, (
        "report.csv drifted from golden fixture. "
        "If the change is intentional, update tests/fixtures/golden_report_v1.csv."
    )
```

- [ ] **Step 2.2: Run test, expect golden file not found**

Run: `uv run pytest tests/test_cli.py::test_report_golden_csv -v`
Expected: FAIL — fixture `tests/fixtures/golden_report_v1.csv` does not exist.

- [ ] **Step 2.3: Rewrite `judgekit report` to emit structured CSV**

In `src/judgekit/cli.py`, replace the existing `report` function with:

```python
@app.command()
def report(
    run_id: str = typer.Argument(..., help="Run ID to generate report for."),
    results_dir: Path = typer.Option(Path("results"), "--results-dir"),
    out: Path | None = typer.Option(None, "--out", help="Output CSV path."),
) -> None:
    """Generate agreement statistics + spend report from a completed run."""
    from judgekit.agreement import compute_all  # local import to keep CLI startup fast

    jsonl_path = results_dir / run_id / "judgments.jsonl"
    if not jsonl_path.exists():
        typer.echo(f"Error: {jsonl_path} not found.", err=True)
        raise typer.Exit(1)

    records: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))

    total = len(records)
    per_judge: Counter[str] = Counter(str(r["judge_id"]) for r in records)
    label_dist: Counter[str] = Counter(str(r["label"]) for r in records)
    per_benchmark: Counter[str] = Counter(str(r["benchmark_id"]) for r in records)
    vendor_spend: dict[str, float] = {}
    for r in records:
        vendor = str(r["judge_id"]).split("/", 1)[0]
        vendor_spend[vendor] = vendor_spend.get(vendor, 0.0) + float(r.get("est_cost_usd", 0.0))

    agreement = compute_all(jsonl_path)

    typer.echo(f"Run ID:        {run_id}")
    typer.echo(f"Total records: {total}")
    typer.echo(f"Items:         {agreement.n_items}")
    typer.echo(f"Judges:        {len(agreement.judge_ids)}")
    typer.echo("")
    typer.echo(f"Fleiss kappa:        {agreement.fleiss_kappa_score:.4f}")
    typer.echo(f"Krippendorff alpha:  {agreement.krippendorff_alpha_score:.4f}")
    typer.echo("")
    typer.echo("Records per judge:")
    for judge, count in sorted(per_judge.items()):
        typer.echo(f"  {judge}: {count}")
    typer.echo("")
    typer.echo("Label distribution:")
    for label, count in sorted(label_dist.items()):
        typer.echo(f"  {label}: {count}")

    if out is None:
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as csv_fh:
        writer = csv.writer(csv_fh)
        writer.writerow(["section", "key", "value"])
        writer.writerow(["meta", "run_id", run_id])
        writer.writerow(["meta", "total_records", total])
        writer.writerow(["meta", "n_items", agreement.n_items])
        writer.writerow(["meta", "n_judges", len(agreement.judge_ids)])
        writer.writerow(["agreement", "fleiss_kappa", f"{agreement.fleiss_kappa_score:.6f}"])
        writer.writerow(
            ["agreement", "krippendorff_alpha", f"{agreement.krippendorff_alpha_score:.6f}"]
        )
        for (a, b), kappa in sorted(agreement.pairwise_kappa.items()):
            writer.writerow(["pairwise_kappa", f"{a}|{b}", f"{kappa:.6f}"])
        for label, count in sorted(label_dist.items()):
            writer.writerow(["label_distribution", label, count])
        for judge, count in sorted(per_judge.items()):
            writer.writerow(["per_judge_count", judge, count])
        for bench, count in sorted(per_benchmark.items()):
            writer.writerow(["per_benchmark_count", bench, count])
        for vendor, spend in sorted(vendor_spend.items()):
            writer.writerow(["vendor_spend_usd", vendor, f"{spend:.6f}"])
        writer.writerow(["vendor_spend_usd", "total", f"{sum(vendor_spend.values()):.6f}"])
    typer.echo(f"\nReport written to {out}")
```

- [ ] **Step 2.4: Generate the golden fixture from the new code**

Run:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from typer.testing import CliRunner
from judgekit.cli import app
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    run_dir = tmp_path / "golden"
    run_dir.mkdir()
    judges = ["anthropic/claude-sonnet-4-5", "groq/llama-3.3-70b-versatile", "sambanova/DeepSeek-R1"]
    labels_by_judge = {
        judges[0]: ["CORRECT", "CORRECT", "INCORRECT", "CORRECT", "UNCERTAIN", "CORRECT"],
        judges[1]: ["CORRECT", "INCORRECT", "INCORRECT", "CORRECT", "UNCERTAIN", "CORRECT"],
        judges[2]: ["CORRECT", "CORRECT", "INCORRECT", "INCORRECT", "UNCERTAIN", "CORRECT"],
    }
    costs_by_judge = {judges[0]: 0.004, judges[1]: 0.0, judges[2]: 0.0}
    benchmarks = ["pubmedqa", "pubmedqa", "medqa", "medqa", "humaneval", "humaneval"]
    records = []
    for jg in judges:
        for i, lbl in enumerate(labels_by_judge[jg]):
            records.append({"run_id": "golden","benchmark_id": benchmarks[i],"item_id": str(i),"judge_id": jg,"label": lbl,"raw_text": lbl,"prompt_tokens": 100,"completion_tokens": 20,"est_cost_usd": costs_by_judge[jg],"latency_ms": 100.0,"error": None,})
    (run_dir / "judgments.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    out = tmp_path / "report.csv"
    result = CliRunner().invoke(app, ["report", "golden", "--results-dir", str(tmp_path), "--out", str(out)])
    assert result.exit_code == 0, result.output
    target = Path("tests/fixtures/golden_report_v1.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"wrote {target}, {target.stat().st_size} bytes")
PY
```

Expected: file `tests/fixtures/golden_report_v1.csv` is created.

Open the file, read it top to bottom, and confirm it is sane: pairwise_kappa rows present, vendor_spend_usd total = 0.024, label_distribution sums to 18 (3 judges × 6 items). If anything is wrong, the implementation is wrong — fix the implementation, not the fixture.

- [ ] **Step 2.5: Run the golden test, expect green**

Run: `uv run pytest tests/test_cli.py::test_report_golden_csv -v`
Expected: PASS.

- [ ] **Step 2.6: Run full CLI test suite, fix regressions**

Run: `uv run pytest tests/test_cli.py -v`
Expected: all CLI tests pass. The pre-existing `test_report_command_prints_label_distribution` may need a minor adjustment since `report` now imports `compute_all` at call time — it should still pass without change because the fixture has 4 records and one judge. Verify.

- [ ] **Step 2.7: Commit**

```bash
git add src/judgekit/cli.py tests/test_cli.py tests/fixtures/golden_report_v1.csv
git commit -m "feat: report CLI emits pairwise kappa, alpha, per-vendor spend with golden test"
```

---

## Task 3: Paper tech-note skeleton

**Files:**
- Create: `paper/tech_note.tex`
- Create: `paper/refs.bib`
- Create: `paper/figures/.gitkeep`
- Create: `paper/README.md`

- [ ] **Step 3.1: Create directory**

Run: `mkdir -p paper/figures`

- [ ] **Step 3.2: Write `paper/tech_note.tex`**

Create `paper/tech_note.tex`:

```latex
\documentclass[11pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{microtype}
\usepackage[numbers,sort&compress]{natbib}

\title{JudgeKit: Cross-Vendor LLM-as-Judge Agreement on Medical and Code Benchmarks}
\author{Tirthesh Jani \\ \texttt{tirtheshjani@gmail.com}}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
We measure inter-judge agreement across five LLM judges from four vendor families
on five public benchmarks (PubMedQA, MedQA, MMLU clinical subjects, HumanEval, MBPP),
under a fixed prompt and a hard \$25 USD spend cap. We report pairwise Cohen's
$\kappa$, Fleiss' $\kappa$, and Krippendorff's $\alpha$, and characterise the
recurring modes of disagreement. Code, prompts, and per-judgment outputs are released as
the \texttt{judgekit} Python package.
\end{abstract}

\section{Motivation}
LLM-as-judge is now a default substitute for human raters across alignment, code, and
medical benchmarks. Single-judge protocols inherit one model's idiosyncrasies; multi-judge
protocols are rarely instrumented with cost accounting or with structured disagreement
analysis. JudgeKit fixes both: a uniform interface across five vendor families and a
disagreement-mode report that complements scalar agreement statistics.

\section{Setup}
\paragraph{Judges.} See Table~\ref{tab:judges}.
\paragraph{Benchmarks.} See Table~\ref{tab:benchmarks}. Three medical, two code.
\paragraph{Prompt.} Two pinned templates (\texttt{medical\_reference\_v1.j2},
\texttt{code\_reference\_v1.j2}) ask the judge to label
\textsc{correct} / \textsc{incorrect} / \textsc{uncertain} against a reference
answer. Final-line parsing tolerates DeepSeek-style reasoning prefaces.

\begin{table}[ht]
\centering
\caption{The five judges (free-tier vendors except Anthropic). Model IDs pinned in
\texttt{configs/all\_five\_judges.yaml}.}
\label{tab:judges}
\begin{tabular}{llll}
\toprule
Vendor & Model & API style & Tier \\
\midrule
Groq        & Llama 3.3 70B Versatile & OpenAI-compat & Free \\
Cerebras    & Llama 3.3 70B           & OpenAI-compat & Free \\
SambaNova   & DeepSeek R1             & OpenAI-compat & Free \\
OpenRouter  & Llama 3.3 70B Instruct  & OpenAI-compat & Free \\
Anthropic   & Claude Sonnet 4.5       & Anthropic SDK & Paid \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[ht]
\centering
\caption{The five benchmarks.}
\label{tab:benchmarks}
\begin{tabular}{llll}
\toprule
Benchmark & Domain & Source & License \\
\midrule
PubMedQA      & Medical QA   & \texttt{pubmed\_qa} (pqa\_labeled)            & MIT \\
MedQA-USMLE   & Medical QA   & \texttt{GBaker/MedQA-USMLE-4-options}         & MIT \\
MMLU clinical & Medical MC   & \texttt{cais/mmlu} (4 clinical subjects)      & MIT \\
HumanEval     & Code         & \texttt{openai\_humaneval}                    & MIT \\
MBPP          & Code         & \texttt{mbpp} (sanitized)                     & CC-BY-4.0 \\
\bottomrule
\end{tabular}
\end{table}

\section{Agreement results}
% Filled by the report CSV after the headline run.
Fleiss' $\kappa$ and Krippendorff's $\alpha$ on the full 5-judge $\times$ 5-benchmark
matrix are reported in Section~\ref{sec:numbers}. Pairwise $\kappa$ heatmap in
Figure~\ref{fig:kappa-heatmap}.

\begin{figure}[ht]
\centering
% \includegraphics[width=0.8\linewidth]{figures/pairwise_kappa_heatmap.pdf}
\caption{Pairwise Cohen's $\kappa$ across the five judges. Placeholder.}
\label{fig:kappa-heatmap}
\end{figure}

\section{Disagreement modes}
\texttt{judgekit.agreement.disagreement.cluster\_disagreements} groups per-question
disagreement vectors with KMeans (elbow over $k \in [3,8]$). For each cluster we report
the dominant per-judge label pattern and the three most representative items. The full
table is in the released report CSV.

\section{Cost accounting}
\label{sec:numbers}
Total Anthropic spend on the headline run was \$\textbf{TBD}, against a \$25.00 cap
enforced by \texttt{judgekit.budget.circuit\_breaker}. Per-vendor breakdown:

\begin{table}[ht]
\centering
\caption{Spend per vendor. Estimated (from \texttt{judgekit estimate}) vs.\
realised (from the run manifest).}
\label{tab:spend}
\begin{tabular}{lrr}
\toprule
Vendor & Estimated (\$) & Actual (\$) \\
\midrule
Anthropic   & TBD & TBD \\
Groq        & 0.00 & 0.00 \\
Cerebras    & 0.00 & 0.00 \\
SambaNova   & 0.00 & 0.00 \\
OpenRouter  & 0.00 & 0.00 \\
\midrule
Total       & TBD & TBD \\
\bottomrule
\end{tabular}
\end{table}

\section{Limitations}
Five judges across four vendor families is wide but shallow: 70B-parameter Llama and
8B-parameter Qwen sample a narrow slice of architecture space. The judge-as-rater protocol
does not execute candidate code; for HumanEval and MBPP the judge sees a reference
solution and the question only. Calibrated Bayesian uncertainty over agreement is not
modelled; bootstrap CIs on $\kappa$ are an obvious next step.

\bibliographystyle{abbrvnat}
\bibliography{refs}

\end{document}
```

- [ ] **Step 3.3: Write `paper/refs.bib`**

```bibtex
@article{cohen1960kappa,
  author  = {Cohen, J.},
  title   = {A coefficient of agreement for nominal scales},
  journal = {Educational and Psychological Measurement},
  year    = {1960},
  volume  = {20},
  number  = {1},
  pages   = {37--46}
}

@article{fleiss1971kappa,
  author  = {Fleiss, J. L.},
  title   = {Measuring nominal scale agreement among many raters},
  journal = {Psychological Bulletin},
  year    = {1971},
  volume  = {76},
  number  = {5},
  pages   = {378--382}
}

@book{krippendorff2004content,
  author    = {Krippendorff, K.},
  title     = {Content Analysis: An Introduction to Its Methodology},
  publisher = {Sage Publications},
  year      = {2004},
  edition   = {2nd}
}

@inproceedings{jin2019pubmedqa,
  author    = {Jin, Q. and Dhingra, B. and Liu, Z. and Cohen, W. W. and Lu, X.},
  title     = {{PubMedQA}: A Dataset for Biomedical Research Question Answering},
  booktitle = {EMNLP-IJCNLP},
  year      = {2019}
}

@article{jin2020medqa,
  author  = {Jin, D. and Pan, E. and Oufattole, N. and Weng, W.-H. and Fang, H. and Szolovits, P.},
  title   = {What Disease Does This Patient Have? {A} Large-scale Open Domain Question Answering Dataset from Medical Exams},
  journal = {Applied Sciences},
  year    = {2021},
  volume  = {11},
  number  = {14}
}

@inproceedings{hendrycks2021mmlu,
  author    = {Hendrycks, D. and Burns, C. and Basart, S. and Zou, A. and Mazeika, M. and Song, D. and Steinhardt, J.},
  title     = {Measuring Massive Multitask Language Understanding},
  booktitle = {ICLR},
  year      = {2021}
}

@article{chen2021humaneval,
  author = {Chen, M. and others},
  title  = {Evaluating Large Language Models Trained on Code},
  journal = {arXiv preprint arXiv:2107.03374},
  year   = {2021}
}

@article{austin2021mbpp,
  author = {Austin, J. and Odena, A. and Nye, M. and others},
  title  = {Program Synthesis with Large Language Models},
  journal = {arXiv preprint arXiv:2108.07732},
  year   = {2021}
}
```

- [ ] **Step 3.4: Write `paper/README.md`**

```markdown
# JudgeKit tech note (paper/)

Build:

```
cd paper
latexmk -pdf tech_note.tex
```

Numbers (Fleiss kappa, Krippendorff alpha, per-vendor spend, pairwise kappa heatmap)
are filled from the headline run report. After running
`judgekit report headline_v01 --out paper/figures/report.csv`, regenerate the
heatmap from the `pairwise_kappa` rows.
```

- [ ] **Step 3.5: Create `paper/figures/.gitkeep`**

Run: `touch paper/figures/.gitkeep`

- [ ] **Step 3.6: Commit**

```bash
git add paper/
git commit -m "docs: paper tech-note skeleton with placeholders pending headline run"
```

---

## Task 4: README budget table + repro section

**Files:**
- Modify: `README.md`

- [ ] **Step 4.1: Replace `README.md` with the full user-facing surface**

Overwrite `README.md`:

````markdown
# JudgeKit

Cross-vendor LLM-as-judge evaluation. Runs the same prompt set across five judges from four vendor families, reports cross-judge agreement (Cohen's kappa, Fleiss' kappa, Krippendorff's alpha) and disagreement clusters, with visible per-vendor cost accounting against a hard budget cap.

## Install

```bash
uv pip install judgekit
# or, from source:
git clone https://github.com/TirtheshJani/JudgeKit
cd JudgeKit
uv sync --group dev
```

## The five judges

| Vendor | Model | API style | Tier |
| --- | --- | --- | --- |
| Groq | `llama-3.3-70b-versatile` | OpenAI-compat | Free |
| Cerebras | `llama-3.3-70b` | OpenAI-compat | Free |
| SambaNova | `DeepSeek-R1` | OpenAI-compat | Free |
| OpenRouter | `meta-llama/llama-3.3-70b-instruct:free` | OpenAI-compat | Free |
| Anthropic | `claude-sonnet-4-5` | Anthropic SDK | Paid |

Optional sixth judge: local Qwen3-8B INT4 via vLLM (`scripts/serve_qwen_vllm.sh`).

## Benchmarks

| Benchmark | Source | License |
| --- | --- | --- |
| PubMedQA | `pubmed_qa` (pqa_labeled split) | MIT |
| MedQA-USMLE | `GBaker/MedQA-USMLE-4-options` | MIT |
| MMLU clinical | `cais/mmlu` (anatomy, clinical_knowledge, college_medicine, professional_medicine) | MIT |
| HumanEval | `openai_humaneval` | MIT |
| MBPP | `mbpp` (sanitized) | CC-BY-4.0 |

Datasets are downloaded on first use via `datasets` and cached under `~/.cache/judgekit/datasets/`.

## Budget plan

Anthropic pricing (verified Phase 4): ~$3 / 1M input tokens, ~$15 / 1M output tokens. Per-judgment back-of-envelope cost:

| Judgment shape | Per-call cost |
| --- | --- |
| 600 input + 150 output tokens (rubric + short justification) | ~$0.004 |

| Vendor | Estimated spend (headline run) | Actual spend |
| --- | --- | --- |
| Anthropic | ~$22.00 | _filled by manifest_ |
| Groq | $0.00 | $0.00 |
| Cerebras | $0.00 | $0.00 |
| SambaNova | $0.00 | $0.00 |
| OpenRouter | $0.00 | $0.00 |
| **Total** | **~$22.00 (cap $25.00)** | _filled by manifest_ |

The budget cap is enforced by `judgekit.budget.circuit_breaker.BudgetCircuitBreaker`, which blocks any call whose estimated cost would push past `JUDGEKIT_ANTHROPIC_BUDGET_USD` (default $25.00).

## Reproduce

```bash
export ANTHROPIC_API_KEY=...
export GROQ_API_KEY=...
export CEREBRAS_API_KEY=...
export SAMBANOVA_API_KEY=...
export OPENROUTER_API_KEY=...
export JUDGEKIT_ANTHROPIC_BUDGET_USD=25

# 1. Inspect estimated spend before paying:
judgekit estimate --config configs/all_five_judges.yaml

# 2. Smoke run (no cost, 1 vendor, n=10):
judgekit run --config configs/smoke_test.yaml --run-id smoke

# 3. Full headline run:
judgekit run --config configs/all_five_judges.yaml --run-id headline_v01

# 4. Report (writes CSV + per-judge / per-vendor breakdown):
judgekit report headline_v01 --out results/headline_v01/report.csv
```

Each run writes `results/<run_id>/judgments.jsonl` plus `results/<run_id>/manifest.json` (config snapshot, judges, vendor spend, totals, timestamps).

## Environment variables

| Variable | Purpose |
| --- | --- |
| `GROQ_API_KEY` | Groq judge |
| `CEREBRAS_API_KEY` | Cerebras judge |
| `SAMBANOVA_API_KEY` | SambaNova judge |
| `OPENROUTER_API_KEY` | OpenRouter judge |
| `ANTHROPIC_API_KEY` | Anthropic judge |
| `JUDGEKIT_ANTHROPIC_BUDGET_USD` | Hard spend cap (default 25.0) |

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run mypy src/judgekit
```

CI (GitHub Actions) runs the same four commands on every push. The CI environment has **no** API keys; the test suite is `pytest-httpx`-strict and fails on any unmocked HTTP call.

## License

MIT.
````

- [ ] **Step 4.2: Sanity-check the rendered README**

Run: `uv run python -c "import pathlib; print(len(pathlib.Path('README.md').read_text()))"`
Expected: a number above 2000.

- [ ] **Step 4.3: Commit**

```bash
git add README.md
git commit -m "docs: README with judge table, budget plan, repro recipe, license table"
```

---

## Task 5 (GATED): Live headline run

> **STOP.** Do not execute without:
> 1. `ANTHROPIC_API_KEY`, `CEREBRAS_API_KEY`, `SAMBANOVA_API_KEY`, `OPENROUTER_API_KEY` all present in the environment.
> 2. Explicit user authorization in chat.
> 3. A fresh `judgekit estimate` showing total < $23 (leaving headroom under the $25 cap).

**Files:**
- Modify: `configs/all_five_judges.yaml` (n per benchmark, after sizing)
- Generate: `results/headline_v01/judgments.jsonl`
- Generate: `results/headline_v01/manifest.json`
- Generate: `results/headline_v01/report.csv`
- Generate: `paper/figures/pairwise_kappa_heatmap.pdf`
- Modify: `paper/tech_note.tex` (replace TBD numbers with actuals)
- Modify: `README.md` (fill the Actual column in the spend table)

- [ ] **Step 5.1: Sizing iteration**

Run `judgekit estimate --config configs/all_five_judges.yaml` and adjust `n` per benchmark in the YAML until the Anthropic line is in $20-$23. Document the chosen size in a `decisions.md` entry.

- [ ] **Step 5.2: Smoke before headline**

Run `judgekit run --config configs/smoke_test.yaml --run-id smoke_pre_headline` against Groq only. Confirm `manifest.json` exists, schema validates, JSONL is well-formed.

- [ ] **Step 5.3: Execute the headline run**

```bash
judgekit run --config configs/all_five_judges.yaml --run-id headline_v01
```

Monitor wall-clock and budget tracker. If the circuit breaker fires, the run halts cleanly — that is the contract, not a bug.

- [ ] **Step 5.4: Cross-check tracker against Anthropic console**

After the first 100 Claude calls, pause and compare the tracker's running Anthropic total to the Anthropic console. Within 5% is acceptable; outside, abort and investigate pricing constants.

- [ ] **Step 5.5: Generate the report CSV**

```bash
judgekit report headline_v01 --out results/headline_v01/report.csv
```

- [ ] **Step 5.6: Pairwise kappa heatmap**

Write `paper/figures/make_heatmap.py` that reads the `pairwise_kappa` rows from the report CSV and emits `pairwise_kappa_heatmap.pdf`. (One-off analysis script; not part of the package.)

- [ ] **Step 5.7: Fill paper and README**

Replace every `TBD` in `paper/tech_note.tex` and the Actual column in `README.md`'s spend table with values from `results/headline_v01/manifest.json` and `report.csv`.

- [ ] **Step 5.8: Compile the paper**

```bash
cd paper && latexmk -pdf tech_note.tex
```

Expected: `tech_note.pdf` is produced, 3-4 pages.

- [ ] **Step 5.9: Tag and publish**

```bash
git tag v0.1.0
git push origin v0.1.0
```

`publish.yml` triggers on the tag and publishes to PyPI via OIDC trusted-publishing.

- [ ] **Step 5.10: Verify install in a fresh venv**

```bash
uv venv /tmp/jk-verify --python 3.11
/tmp/jk-verify/bin/python -m pip install judgekit==0.1.0
/tmp/jk-verify/bin/judgekit --help
```

Expected: package installs from PyPI, CLI prints help.

---

## Self-review

**Spec coverage (against PLAN.md §3.7 Phase 6 gate):**
- `results/<run_id>/manifest.json` validates against a schema → Task 1 (`validate_manifest`).
- `judgekit report --run-id <id>` produces CSV that reproduces the table in the tech note → Task 2 (golden CSV with pairwise kappa, alpha, spend, label distribution).
- Full run JSONL exists; Anthropic spend < $25 → Task 5 (gated).
- Tech note PDF compiles → Task 3 (skeleton) + Task 5.8 (compile with filled numbers).
- `pip install judgekit==0.1.0` works → Task 5.10.
- README repro command executes → Task 4 covers content; Task 5.10 verifies post-publish.

**Placeholder scan:** All code blocks contain real code. Steps 5.1-5.10 use placeholder words (`TBD`, the `n` to be chosen) because they cannot be known without the live run; that is the intended gate, not a plan defect.

**Type consistency:** `Manifest`, `ManifestSchemaError`, `write_manifest`, `validate_manifest` used consistently in Task 1 tests and implementation. `compute_all` from Task 2 matches the existing `AgreementReport` signature in `src/judgekit/agreement/__init__.py:114`. `BudgetTracker.per_vendor_spend()` is asserted in Step 1.6 — if missing, that step adds it under TDD.

**Risk:** Step 1.6 assumes `BudgetTracker.per_vendor_spend()` exists. If it doesn't, the runner integration in Step 1.5 must be amended to compute vendor spend from a local accumulator that sums `est_cost` per call (mirroring the per-judge counters). Both paths produce identical manifests; pick whichever is shorter when implementing.

---

## Execution

Inline execution via `.claude/skills/executing-plans` (single Claude session, batch with checkpoints at each task's commit step). Subagent-per-task is overkill for four small, mostly-independent files.
