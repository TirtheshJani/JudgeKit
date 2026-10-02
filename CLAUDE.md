# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What JudgeKit is

A Python package (`src/judgekit`) that runs the same prompt set across five LLM judges from different vendor families, then reports cross-judge agreement statistics and disagreement clustering. The product framing is deliberate: a multi-judge framework with explicit cost accounting, not a "GPT-4 as judge" wrapper.

The package is implemented: CLI (`judgekit run | estimate | report`), vendor clients, five benchmark adapters, agreement statistics, disagreement clustering, budget tracking, run manifests, eval configs, an offline test suite, CI and a PyPI publish workflow. The headline five-judge run has **not** been executed yet, so there are no real results; `paper/` is a skeleton with TBD placeholders. Never invent numbers to fill them.

The five judges are fixed by design (see `configs/all_five_judges.yaml`). Do not silently swap or drop one without flagging it:

| Judge | Vendor | API style |
| --- | --- | --- |
| Llama 3.3 70B (`llama-3.3-70b-versatile`) | Groq | OpenAI-compatible |
| Llama 3.3 70B (`llama-3.3-70b`) | Cerebras | OpenAI-compatible |
| DeepSeek R1 (`DeepSeek-R1`) | SambaNova | OpenAI-compatible |
| Llama 3.3 70B free (`meta-llama/llama-3.3-70b-instruct:free`) | OpenRouter | OpenAI-compatible |
| Claude Sonnet 4.5 (`claude-sonnet-4-5`) | Anthropic | Anthropic SDK (wrapped to the same `Judge` interface) |

Optional sixth judge: Qwen3-8B INT4 on local vLLM (`vendor: vllm`), not in the headline config.

PyPI note: the distribution name `judgekit` is already taken on PyPI by an unrelated project. The publish workflow will fail (or publish nothing useful) until a distinct name is chosen. Flag this to the user before tagging a release.

The paid Anthropic judge is load-bearing for the project's positioning; it is the "deliberate cost-discipline tradeoff" framing. Total experiment cost target: **under $25 USD**. Token-budget accounting must be visible in outputs/reports.

## Architecture intent

- **One client class** with a base-URL switch handles the four OpenAI-compatible vendors (Groq, Cerebras, SambaNova, OpenRouter). A thin Claude wrapper exposes the same interface. Avoid building four parallel client classes.
- **Rate-limit-aware retry decorator** wraps every judge call. Different vendors have different rate-limit response shapes; the decorator must handle them uniformly.
- **Eval configs are YAML** and must be reproducible (pinned model IDs, seeds, prompt templates, dataset slices).
- **Local Qwen3-8B** is served via vLLM with `--gpu-memory-utilization 0.85 --max-model-len 4096` (target hardware: RTX 4080). Treat vLLM as just another OpenAI-compatible endpoint.
- **Agreement statistics**: Cohen's kappa (pairwise), Krippendorff's alpha (overall), plus disagreement clustering. Don't report only mean agreement; disagreement *modes* are a stated deliverable.
- **Benchmarks**: PubMedQA, MedQA, MMLU clinical subsets, HumanEval, MBPP. Medical + code is intentional; the cross-domain split is part of the story.

### Environment variables

Expected API-key env vars (used by the single client based on which vendor is configured):

- `GROQ_API_KEY`
- `CEREBRAS_API_KEY`
- `SAMBANOVA_API_KEY`
- `OPENROUTER_API_KEY`
- `ANTHROPIC_API_KEY`

The local vLLM endpoint needs no key; default to `http://localhost:8000/v1` unless overridden in the eval YAML.

## Commands

```bash
uv sync --group dev                       # install (dev deps are a [dependency-groups] group, not an extra)
uv run ruff check .                       # lint (CI)
uv run ruff format --check .              # format check (CI)
uv run mypy src/judgekit                  # strict type check (CI)
uv run pytest --tb=short                  # full offline test suite (CI)
uv run pytest tests/test_cli.py -k golden # single test

uv run python examples/agreement_demo.py                     # offline demo, synthetic data
uv run judgekit estimate --config configs/all_five_judges.yaml   # pre-flight cost, no network
uv run judgekit run --config configs/smoke_test.yaml --run-id smoke   # needs GROQ_API_KEY
uv run judgekit report <run_id> --out results/<run_id>/report.csv

bash scripts/serve_qwen_vllm.sh           # local Qwen3-8B judge (needs CUDA GPU)
```

`tests/test_cli.py::test_report_golden_csv` compares report output byte-for-byte to `tests/fixtures/golden_report_v1.csv`; update the fixture only for intentional format changes.

## Deliverables

1. PyPI package `judgekit`.
2. GitHub repo with reproducible eval YAML configs.
3. A 3 to 4 page tech-note for arXiv documenting cross-judge agreement findings.
4. Optional: Streamlit dashboard (free hosting) with live judge-agreement curves.
5. Optional: PR to EleutherAI's `lm-evaluation-harness` adding a multi-judge backend.

## Working in this repo

- **Branch**: develop on a feature branch (e.g. `claude/<topic>`) and open a PR to `main`. Never push to `main` without explicit permission.
- Keep `ruff format --check .` clean; CI fails on unformatted files.
- Keep cost accounting first-class: any judge-call code path should account tokens against a per-run budget the user can see.

## Skills

Project-level skills are installed under `.claude/skills/`. Invoke by name when the task matches:

- `writing-plans` and `executing-plans` for per-phase implementation plans (see `PLAN.md` section 1a).
- `test-driven-development` is the default for all code-writing tasks (no training loops in JudgeKit, so no exemptions).
- `dispatching-parallel-agents` when building independent components (vendor clients, benchmark adapters).
- `using-git-worktrees` whenever parallel agents are dispatched. Worktrees live under `.worktrees/` (gitignored).
- `systematic-debugging` for every bug investigation. Iron law: root cause before fix.
- `karpathy-guidelines` for behavior baseline.
