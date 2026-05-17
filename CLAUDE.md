# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What JudgeKit is

A Python package (`judgekit`, to be published to PyPI) that runs the same prompt set across five LLM judges from different vendor families, then reports cross-judge agreement statistics and disagreement clustering. The product framing is deliberate: a multi-judge framework with explicit cost accounting, not a "GPT-4 as judge" wrapper.

The repo is currently a stub (LICENSE, one-line README, this file). No source, tests, or config yet. The sections below describe the **intended design** so the first scaffolding session stays consistent with the project's framing.

The five judges are fixed by design. Do not silently swap or drop one without flagging it:

| Judge | Vendor | API style |
| --- | --- | --- |
| Llama 3.3 70B | Groq | OpenAI-compatible |
| Llama 3.3 70B | Cerebras | OpenAI-compatible |
| DeepSeek R1 | SambaNova | OpenAI-compatible |
| Qwen3-8B INT4 | local vLLM | OpenAI-compatible |
| Claude Sonnet 4.5 (`claude-sonnet-4-5`) | Anthropic | Anthropic SDK (wrapped to look OpenAI-compatible) |

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

The only command fixed at this point is the local judge server. Add `pytest` / lint / build commands here once `pyproject.toml` exists.

```bash
# Serve Qwen3-8B INT4 as the local judge (RTX 4080)
vllm serve Qwen/Qwen3-8B --quantization awq --gpu-memory-utilization 0.85 --max-model-len 4096
```

## Deliverables

1. PyPI package `judgekit`.
2. GitHub repo with reproducible eval YAML configs.
3. A 3 to 4 page tech-note for arXiv documenting cross-judge agreement findings.
4. Optional: Streamlit dashboard (free hosting) with live judge-agreement curves.
5. Optional: PR to EleutherAI's `lm-evaluation-harness` adding a multi-judge backend.

## Working in this repo

- **Branch**: develop on `claude/multi-judge-evals-xvHDQ` (the current branch). Do not push to other branches without explicit permission.
- **No build/test/lint commands exist yet.** Add them to this file once the package is scaffolded (pyproject.toml, pytest config, etc.).
- When scaffolding, prefer `pyproject.toml` + `src/judgekit/` layout so the PyPI package is the primary artifact from day one.
- Keep cost accounting first-class: any judge-call code path should account tokens against a per-run budget the user can see.

## Skills

Project-level skills are installed under `.claude/skills/`. Invoke by name when the task matches:

- `writing-plans` and `executing-plans` for per-phase implementation plans (see `PLAN.md` section 1a).
- `test-driven-development` is the default for all code-writing tasks (no training loops in JudgeKit, so no exemptions).
- `dispatching-parallel-agents` when building independent components (vendor clients, benchmark adapters).
- `using-git-worktrees` whenever parallel agents are dispatched. Worktrees live under `.worktrees/` (gitignored).
- `systematic-debugging` for every bug investigation. Iron law: root cause before fix.
- `karpathy-guidelines` for behavior baseline.
