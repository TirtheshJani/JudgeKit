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

Datasets are downloaded on first use via the `datasets` library and cached under `~/.cache/judgekit/datasets/`.

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
