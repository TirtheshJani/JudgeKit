"""End-to-end eval runner for JudgeKit.

Orchestrates benchmark × judge × item evaluation, writes JSONL output,
and accounts token costs against a configurable budget cap.
"""

from __future__ import annotations

import json
from pathlib import Path

from judgekit.benchmarks.base import BenchmarkAdapter
from judgekit.benchmarks.humaneval import HumanEvalAdapter
from judgekit.benchmarks.mbpp import MBPPAdapter
from judgekit.benchmarks.medqa import MedQAAdapter
from judgekit.benchmarks.mmlu_clinical import MMLUClinicalAdapter
from judgekit.benchmarks.pubmedqa import PubMedQAAdapter
from judgekit.budget.circuit_breaker import BudgetCircuitBreaker
from judgekit.budget.pricing import lookup
from judgekit.budget.tracker import BudgetTracker
from judgekit.clients.registry import build_judge
from judgekit.config import BenchmarkConfig, EvalConfig
from judgekit.io import JsonlWriter, JudgmentRecord
from judgekit.prompts.loader import PromptLoader

BENCHMARK_REGISTRY: dict[str, type[BenchmarkAdapter]] = {
    "pubmedqa": PubMedQAAdapter,
    "medqa": MedQAAdapter,
    "mmlu_clinical": MMLUClinicalAdapter,
    "humaneval": HumanEvalAdapter,
    "mbpp": MBPPAdapter,
}

BENCHMARK_TEMPLATE: dict[str, str] = {
    "pubmedqa": "medical_reference_v1.j2",
    "medqa": "medical_reference_v1.j2",
    "mmlu_clinical": "medical_reference_v1.j2",
    "humaneval": "code_reference_v1.j2",
    "mbpp": "code_reference_v1.j2",
}

LABELS = {"CORRECT", "INCORRECT", "UNCERTAIN"}


def _parse_label(text: str) -> str:
    """Extract the last line matching a known label.

    Scans lines in reverse order so that the final label wins when
    multiple labels appear (e.g. reasoning text followed by a verdict).
    Returns 'UNPARSABLE' if no recognised label is found.
    """
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip().upper()
        if stripped in LABELS:
            return stripped
    return "UNPARSABLE"


def _build_adapter(bc: BenchmarkConfig) -> BenchmarkAdapter:
    """Instantiate the adapter for a benchmark config.

    Passes only ``n`` to the constructor; each adapter manages its own
    default split.  Adapters that accept ``split`` (PubMedQA, MedQA, MBPP,
    MMLUClinical) will use their own defaults; HumanEval accepts only ``n``.
    """
    cls = BENCHMARK_REGISTRY.get(bc.name)
    if cls is None:
        raise ValueError(f"Unknown benchmark: {bc.name!r}")
    return cls(n=bc.n)  # type: ignore[call-arg]


def run_eval(config: EvalConfig, output_dir: Path, dry_run: bool = False) -> Path:
    """Orchestrate a full eval run: iterate (benchmark × judge × item), write JSONL.

    Returns the path to the written JSONL file.
    If dry_run=True, returns the expected output path without making any API calls.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "judgments.jsonl"

    if dry_run:
        return jsonl_path

    tracker = BudgetTracker(cap_usd=config.budget_usd)
    breaker = BudgetCircuitBreaker(tracker)
    loader = PromptLoader()

    judges = [build_judge(jc.vendor, jc.model, jc.base_url) for jc in config.judges]
    adapters = [_build_adapter(bc) for bc in config.benchmarks]

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
                    # Approximate prompt tokens: chars / 4
                    est_prompt_tokens = len(prompt) // 4
                    breaker.check(jc.vendor, jc.model, est_prompt_tokens, jc.max_tokens)
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

                    writer.write(
                        JudgmentRecord(
                            run_id=config.run_id,
                            benchmark_id=bc.name,
                            item_id=item.id,
                            judge_id=f"{jc.vendor}/{jc.model}",
                            label=label,
                            raw_text=resp.text if resp else "",
                            prompt_tokens=resp.prompt_tokens if resp else 0,
                            completion_tokens=resp.completion_tokens if resp else 0,
                            est_cost_usd=est_cost,
                            latency_ms=resp.latency_ms if resp else 0.0,
                            error=error,
                        )
                    )

    return jsonl_path


def validate_run(jsonl_path: Path) -> bool:
    """Verify every record in a run JSONL matches the expected schema."""
    required = {
        "run_id",
        "benchmark_id",
        "item_id",
        "judge_id",
        "label",
        "raw_text",
        "prompt_tokens",
        "completion_tokens",
        "est_cost_usd",
        "latency_ms",
    }
    try:
        with jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                if not required.issubset(record.keys()):
                    return False
        return True
    except Exception:
        return False
