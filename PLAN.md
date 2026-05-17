# JudgeKit Implementation Plan

Wall-clock target: 1 to 1.5 weeks (8 working days). No external spend until Phase 4. Total Anthropic cap $25 USD.

## 1. Scope recap

Five judges, five benchmarks, three agreement statistics, one budget cap. The five judges are fixed by design (Groq Llama 3.3 70B, Cerebras Llama 3.3 70B, SambaNova DeepSeek R1, OpenRouter free-tier, Anthropic Claude Sonnet 4.5). Local Qwen3-8B INT4 via vLLM is an optional sixth judge. The framing asset is the paid Anthropic judge plus visible cost accounting; do not silently drop it to save money.

## 1a. Methodology (skills used)

This plan is governed by five skills installed at `.claude/skills/`. Future Claude sessions in this repo should invoke them by name:

- **writing-plans** governs the format of per-phase plans. PLAN.md is the *roadmap*; before each phase, write a detailed bite-sized TDD plan into `docs/plans/YYYY-MM-DD-phase-N-<name>.md` using this skill, then execute.
- **executing-plans** governs how those per-phase plans are run (load, review critically, execute task-by-task, stop on blockers).
- **test-driven-development** is the default for every code-writing task in this repo. No production code without a failing test first. The only exemption pre-agreed with the user is training loops, and JudgeKit has none, so TDD is universal here.
- **dispatching-parallel-agents** applies in Phases 3 and 5 where multiple independent vendor clients or benchmark adapters can be built concurrently.
- **karpathy-guidelines** is the behavior baseline: surface assumptions, simplicity first, surgical changes, goal-driven execution.

The `writing-plans` skill suggests `docs/superpowers/plans/` as the plan directory. This repo uses `docs/plans/` instead (no "superpowers" rebrand inside a public package).

## 2. File tree (target end-of-Phase-6 layout)

```
JudgeKit/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── PLAN.md
├── LICENSE
├── .gitignore                       # results/, .venv, __pycache__, *.egg-info
├── .pre-commit-config.yaml          # ruff, ruff-format, mypy
├── .github/
│   └── workflows/
│       ├── ci.yml                   # ubuntu-latest: ruff, mypy, pytest
│       └── publish.yml              # PyPI on tag v*
├── src/
│   └── judgekit/
│       ├── __init__.py
│       ├── cli.py                   # `judgekit run|estimate|report`
│       ├── config.py                # pydantic YAML eval-config models
│       ├── runner.py                # orchestrates a full eval run
│       ├── io.py                    # JSONL writer for per-judgment records
│       ├── retry.py                 # rate-limit-aware retry decorator
│       ├── clients/
│       │   ├── __init__.py
│       │   ├── base.py              # Judge ABC: judge(prompt) -> JudgeResponse
│       │   ├── openai_compat.py     # one class, base-URL switch
│       │   ├── anthropic_client.py  # SDK wrapper exposing the same shape
│       │   └── registry.py          # vendor key -> client factory
│       ├── benchmarks/
│       │   ├── __init__.py
│       │   ├── base.py              # BenchmarkAdapter ABC
│       │   ├── pubmedqa.py
│       │   ├── medqa.py
│       │   ├── mmlu_clinical.py     # anatomy, clinical_knowledge,
│       │   │                        # college_medicine, professional_medicine
│       │   ├── humaneval.py
│       │   └── mbpp.py
│       ├── prompts/
│       │   ├── __init__.py
│       │   ├── loader.py            # jinja2 loader with version pinning
│       │   └── templates/
│       │       ├── medical_reference_v1.j2
│       │       ├── medical_reference_free_v1.j2
│       │       ├── code_reference_v1.j2
│       │       └── code_reference_free_v1.j2
│       ├── agreement/
│       │   ├── __init__.py
│       │   ├── cohens_kappa.py
│       │   ├── fleiss_kappa.py
│       │   ├── krippendorff.py
│       │   └── disagreement.py      # KMeans on disagreement vectors
│       └── budget/
│           ├── __init__.py
│           ├── pricing.py           # $/1k tokens per vendor+model
│           ├── tracker.py           # real-time spend tally
│           └── circuit_breaker.py   # raises BudgetExceededError
├── configs/
│   ├── smoke_test.yaml              # 10 questions, 1 judge, 1 benchmark
│   ├── pubmedqa_full.yaml
│   ├── medical_full.yaml            # PubMedQA + MedQA + MMLU clinical
│   ├── code_full.yaml               # HumanEval + MBPP
│   └── all_five_judges.yaml         # the headline reproducible run
├── results/                         # gitignored; JSONL per run
├── paper/
│   ├── tech_note.tex
│   ├── refs.bib
│   └── figures/
├── streamlit_app/                   # Phase 7, optional
│   └── app.py
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   ├── mock_api_responses.py    # canned 200/429/5xx per vendor
│   │   └── synthetic_judgments.py   # known-kappa toy data
│   ├── test_openai_compat_client.py
│   ├── test_anthropic_client.py
│   ├── test_retry.py
│   ├── test_budget_tracker.py
│   ├── test_circuit_breaker.py
│   ├── test_pricing.py
│   ├── test_benchmarks/
│   │   ├── test_pubmedqa.py
│   │   ├── test_medqa.py
│   │   ├── test_mmlu_clinical.py
│   │   ├── test_humaneval.py
│   │   └── test_mbpp.py
│   ├── test_prompts.py
│   ├── test_agreement_stats.py
│   ├── test_disagreement_cluster.py
│   ├── test_runner.py
│   └── test_cli.py
└── scripts/
    ├── estimate_budget.py           # pre-flight cost estimator
    └── serve_qwen_vllm.sh           # local judge launcher
```

## 3. Phased plan with gates

Each phase has a single objective, a verification gate, and a list of artifacts. **Do not advance to the next phase until the gate passes.** Gates are testable, not aspirational.

### Phase 0: Scaffolding (0.5 day)

**Objective.** Get the repo to a state where `uv run pytest` and `uv run ruff check .` execute on green CI.

**Work.**
- `uv init` with Python 3.11; pyproject.toml using src/ layout; pin minimum deps (httpx, anthropic, pydantic, pyyaml, jinja2).
- Dev deps: ruff, mypy, pytest, pytest-cov, pytest-httpx, respx.
- `.pre-commit-config.yaml` with ruff, ruff-format, mypy.
- `.github/workflows/ci.yml`: ubuntu-latest, matrix on Python 3.11 only initially.
- `.gitignore` for results/, .venv, .ruff_cache, dist/, build/.
- Empty package skeleton (every module above as a stub with `pass` or a typed-but-unimplemented signature).

**Gate.**
- `uv run pytest` returns "no tests ran" without import errors.
- `uv run ruff check .` and `uv run mypy src/judgekit` both pass.
- CI workflow green on first push.

**Dependencies.** None.

### Phase 1: Client, retry, budget, fully mocked (1 day)

**Objective.** Every judge call path works against mocked HTTP responses, including 429s, 5xx, malformed bodies, and budget overflow.

**TDD:** Red-green-refactor per `.claude/skills/test-driven-development`. Each task in this phase's per-phase plan file is a bite-sized step (write failing test, run it, implement, run again, commit).

**Work.**
- `clients/base.py`: `Judge` ABC with `judge(prompt: str, *, max_tokens: int) -> JudgeResponse`. JudgeResponse carries `text`, `prompt_tokens`, `completion_tokens`, `vendor`, `model`, `latency_ms`.
- `clients/openai_compat.py`: one concrete class, constructor takes (base_url, api_key, model). Used for Groq, Cerebras, SambaNova, OpenRouter, vLLM. Validates response shape, raises typed errors.
- `clients/anthropic_client.py`: thin wrapper over `anthropic.Anthropic`, returns the same `JudgeResponse`. Token counts come from `usage`.
- `clients/registry.py`: vendor key (`"groq"`, `"cerebras"`, `"sambanova"`, `"openrouter"`, `"anthropic"`, `"vllm"`) maps to a factory that reads env vars.
- `retry.py`: decorator that catches `httpx.HTTPStatusError` for 429/5xx, honors `Retry-After` when present, exponential backoff with jitter, max 5 tries. Per-vendor backoff floors (Groq 30 RPM is the tightest).
- `budget/pricing.py`: dict literal of $/1k input + $/1k output per vendor+model. Free-tier vendors set to 0.0.
- `budget/tracker.py`: thread-safe running tally per vendor. Reads `JUDGEKIT_ANTHROPIC_BUDGET_USD` env var (default 25.0).
- `budget/circuit_breaker.py`: raises `BudgetExceededError` before any call whose *estimated* cost would push past the cap. Estimate uses prompt token count plus `max_tokens` ceiling for the response.

**Tests (TDD applies here).** Use `pytest-httpx` to canned-respond.
- Successful 200 round trip yields correct `JudgeResponse` shape per vendor.
- 429 with `Retry-After: 1` triggers exactly one sleep then succeeds.
- 429 storm exhausts retries and raises `RateLimitExhausted`.
- 5xx triggers retry with exponential backoff.
- Malformed JSON raises `JudgeResponseError`.
- Budget tracker: 10 calls at known cost equal the expected total within float epsilon.
- Circuit breaker: estimated overflow raises before the HTTP call is made.

**Gate.**
- All client + retry + budget tests pass.
- Coverage on `clients/`, `retry.py`, `budget/` at 90% or higher.
- Zero real network calls in the test suite (verified by `pytest-httpx` strict mode).

**Dependencies.** Phase 0.

### Phase 2: Single benchmark, single judge, end-to-end (1 day)

**Objective.** `judgekit run --config configs/smoke_test.yaml` against Groq + PubMedQA n=10 produces a well-formed JSONL file. Zero spend.

**TDD:** Red-green-refactor per `.claude/skills/test-driven-development`. Each task is a bite-sized step (write failing test, run it, implement, run again, commit).

**Work.**
- `benchmarks/base.py`: `BenchmarkAdapter` ABC with `iter_items() -> Iterator[BenchmarkItem]`. `BenchmarkItem` has `id`, `question`, `reference_answer`, `metadata`.
- `benchmarks/pubmedqa.py`: loads from HF `datasets` (`pubmed_qa`, `pqa_labeled` split). Caches locally under `~/.cache/judgekit/datasets/`.
- `prompts/loader.py`: jinja2 environment, templates pinned by filename version. Rendered prompt is reproducible (no nondeterministic context).
- `prompts/templates/medical_reference_v1.j2`: prompts the judge to label CORRECT / INCORRECT / UNCERTAIN given a candidate answer and reference.
- `runner.py`: takes a parsed config, iterates (benchmark, judge, item), calls `judge.judge()`, writes JSONL with one record per (run_id, benchmark_id, item_id, judge_id, label, raw_text, prompt_tokens, completion_tokens, est_cost_usd, latency_ms, error).
- `io.py`: append-only JSONL writer with atomic rename on close.
- `config.py`: pydantic model for the YAML schema; rejects unknown keys.
- `cli.py`: Typer-based CLI with `run`, `estimate`, `report` subcommands.
- `configs/smoke_test.yaml`: 1 benchmark, 1 judge, n=10.

**Tests.**
- PubMedQA adapter: known record at a fixed index has the expected question and label.
- Prompt template: rendering a fixed item produces a byte-exact expected string (golden file).
- Runner: with all client calls mocked, an n=3 smoke run produces 3 JSONL records of the correct shape.
- CLI: `judgekit run --config <smoke> --dry-run` prints estimated cost and call count without making calls.

**Gate.**
- Live n=10 run on Groq PubMedQA completes without error.
- Output JSONL passes schema validation (a `validate_run` helper to be written).
- Total spend logged is $0.00.

**Dependencies.** Phase 1.

### Phase 3: Four more free judges + pairwise agreement (1.5 days)

**Objective.** A 4-judge live run on PubMedQA n=50 produces a Cohen's kappa pairwise matrix, a Fleiss kappa scalar, and a Krippendorff alpha scalar, all validated against reference implementations.

**TDD:** Red-green-refactor per `.claude/skills/test-driven-development` for both client wiring and agreement statistics.

**Parallel dispatch:** Cerebras, SambaNova, and OpenRouter client wiring are independent of each other (each is a base-URL change plus tests). Per `.claude/skills/dispatching-parallel-agents`, dispatch one agent per vendor, each with the focused scope "wire vendor X through `openai_compat`, add tests for it, do not touch other vendors." Reviewer integrates.

**Work.**
- Wire Cerebras, SambaNova, OpenRouter through `openai_compat` with their respective base URLs and free-tier model IDs. OpenRouter free-tier model choice: default to `meta-llama/llama-3.3-70b-instruct:free` (subject to availability, see risk R5).
- Optional: vLLM client config for the local Qwen3-8B judge.
- `agreement/cohens_kappa.py`: pairwise kappa for the categorical label set {CORRECT, INCORRECT, UNCERTAIN}. Implementation by hand, cross-checked against `sklearn.metrics.cohen_kappa_score`.
- `agreement/fleiss_kappa.py`: multi-rater categorical kappa.
- `agreement/krippendorff.py`: nominal alpha, missing-data tolerant. Cross-check against `krippendorff` PyPI lib in tests but do not depend on it at runtime.
- `agreement/__init__.py`: `compute_all(jsonl_path) -> AgreementReport` with pairwise matrix, multi-rater scalars, per-judge marginal counts.
- `cli.py report`: reads a results JSONL and prints + writes CSV report.

**Tests.**
- Synthetic toy data with known kappa (textbook examples) within 1e-6 of reference.
- Krippendorff alpha matches the `krippendorff` PyPI lib on three test fixtures.
- Fleiss kappa matches `statsmodels.stats.inter_rater.fleiss_kappa` on three fixtures.
- All-judges-agree fixture yields kappa = alpha = 1.0.
- All-judges-disagree-uniformly fixture yields kappa near 0.

**Gate.**
- Live n=50 PubMedQA × 4 free judges run completes; spend logged $0.00.
- Agreement report produced with all three statistics.
- Per-judge missing-record rate (calls that errored or returned UNPARSABLE) below 5%.

**Dependencies.** Phase 2.

### Phase 4: Anthropic judge with hard budget cap (1 day)

**Objective.** Anthropic Claude judge integrated. Pre-flight estimator predicts spend within 10% of actual on the smoke run. Circuit breaker provably blocks overflow.

**TDD:** Red-green-refactor per `.claude/skills/test-driven-development`. The circuit-breaker test (pre-set tracker to overflow, attempt call, assert raise *before* HTTP layer) is the highest-value test in this phase; write it first.

**Work.**
- `clients/anthropic_client.py` finalized; reads `ANTHROPIC_API_KEY`; model id pinned `claude-sonnet-4-5`.
- `scripts/estimate_budget.py`: reads a config, walks the benchmark item list, renders prompts, counts tokens via `anthropic.Anthropic.messages.count_tokens` for Anthropic and `tiktoken` (cl100k) for everything else (acknowledge the approximation in the README), multiplies by pricing table, prints a per-vendor breakdown.
- CLI `judgekit estimate --config <path>` exposes the same logic.
- Smoke run: Claude on PubMedQA n=50, then n=200, log actual spend per item.

**Tests.**
- Estimator on a fixed config returns a deterministic number.
- Circuit breaker test: pre-set tracker to $24.99, attempt a call whose estimated cost is $0.02, assert `BudgetExceededError` raised before HTTP layer is touched.
- Anthropic client: mocked SDK response yields correct `JudgeResponse` shape with token counts from `usage`.

**Gate.**
- n=50 Claude PubMedQA smoke spend matches estimator to within 10%.
- Spend logged accurately to within $0.01 of Anthropic console (manual check).
- `JUDGEKIT_ANTHROPIC_BUDGET_USD=0.10` actively blocks a 1k-token run.

**Dependencies.** Phase 1 (budget) + Phase 3 (eval scaffold).

### Phase 5: Remaining benchmarks (1.5 days)

**Objective.** MedQA, MMLU clinical, HumanEval, MBPP all running end-to-end against all five judges on small slices.

**TDD:** Red-green-refactor per `.claude/skills/test-driven-development`. Each adapter starts with a "known item at fixed index has expected fields" test.

**Parallel dispatch:** MedQA, MMLU clinical, HumanEval, and MBPP adapters are independent. Per `.claude/skills/dispatching-parallel-agents`, dispatch one agent per benchmark with scope "implement adapter X with TDD against the HF dataset, do not touch other benchmarks or the runner."

**Work.**
- `benchmarks/medqa.py`: HF `bigbio/med_qa` or `GBaker/MedQA-USMLE-4-options`; license check.
- `benchmarks/mmlu_clinical.py`: HF `cais/mmlu` filtered to four clinical subjects.
- `benchmarks/humaneval.py`: HF `openai_humaneval`. Code-judge prompt asks for CORRECT/INCORRECT against a reference solution and visible tests (no execution; this is judge-as-rater, not execution-based scoring).
- `benchmarks/mbpp.py`: HF `mbpp` sanitized split.
- `prompts/templates/code_reference_v1.j2`: code-specific rubric.
- Per-benchmark license note added to README (verify each is permissively licensed for redistribution of references; ship pointers not content where the license is ambiguous, see risk R7).

**Tests.**
- One adapter test per benchmark: known item at a fixed index matches expected fields.
- Code prompt template: golden render for one HumanEval and one MBPP item.

**Gate.**
- Smoke (n=10 per benchmark) on all five judges completes within $1 total spend.
- All five benchmark adapter tests pass.
- Each benchmark's license is documented in README.

**Dependencies.** Phase 2 (adapter pattern) + Phase 4 (Anthropic for the full set).

### Phase 6: Full run, analysis, tech note, PyPI (2 days)

**Objective.** The headline reproducible run completes under $25, disagreement analysis is interpretable, tech note compiles, package is on PyPI.

**TDD:** TDD applies to `agreement/disagreement.py` and the report generator. The tech note and README are prose, not under TDD.

**Work.**
- Pre-flight estimate on `configs/all_five_judges.yaml` (5 benchmarks × 5 judges × n per benchmark sized to hit ~$22 on Claude). Iterate on n until estimate is comfortably under $25.
- Full run. Save JSONL under `results/<run_id>/`.
- `agreement/disagreement.py`: per-question disagreement vector (5-dim, one entry per judge label), KMeans clustering (k chosen by elbow, k in 3 to 8), top-N representative questions per cluster for qualitative reading.
- `paper/tech_note.tex`: 3 to 4 pages. Sections: Motivation, Setup (judges + benchmarks), Agreement results (kappa table + alpha), Disagreement modes (cluster summaries with examples), Cost accounting, Limitations.
- README: budget table (estimated vs actual per vendor), one-command repro (`uv pip install judgekit && judgekit run --config <url>`).
- `publish.yml`: PyPI trusted-publishing on tag `v*`.
- Tag `v0.1.0`, publish.

**Tests.**
- `results/<run_id>/manifest.json` validates against a schema.
- `judgekit report --run-id <id>` produces a CSV that reproduces the table in the tech note (golden test).

**Gate.**
- Full run JSONL exists; total Anthropic spend below $25 as logged by tracker.
- Tech note PDF compiles via `latexmk`.
- `pip install judgekit==0.1.0` works in a fresh venv.
- README repro command executes (with a smoke config) without modification.

**Dependencies.** Phase 5.

### Phase 7 (optional, post 1.5w): harness PR + Streamlit (2 days)

**Objective.** A live demo and an upstream contribution.

**Work.**
- `streamlit_app/app.py`: reads a published results JSONL from GitHub Releases, plots pairwise kappa heatmap, Krippendorff alpha vs n, per-cluster disagreement examples. Deploy to HF Spaces (CPU Basic).
- `lm-evaluation-harness` PR: a `MultiJudgeRunner` adapter that wraps JudgeKit as a backend. Open a draft PR for discussion before polishing.

**Gate.**
- HF Spaces URL loads, plots render.
- Draft PR opened upstream with a maintainer acknowledging or commenting.

**Dependencies.** Phase 6.

## 3a. Per-phase plan files

PLAN.md is the roadmap. Before starting each phase, write a detailed task-by-task plan using the `writing-plans` skill, save it to:

```
docs/plans/YYYY-MM-DD-phase-N-<short-name>.md
```

For example: `docs/plans/2026-05-18-phase-1-client-retry-budget.md`.

Each per-phase plan follows the `writing-plans` header format (Goal, Architecture, Tech Stack), decomposes into Tasks with exact file paths, and breaks each task into 2-5 minute steps. After writing it, run the plan-document reviewer prompt from `.claude/skills/writing-plans/plan-document-reviewer-prompt.md` as a subagent dispatch before executing.

## 4. Phase dependency graph

```
P0
└── P1
    ├── P2
    │   ├── P3
    │   │   └── P4
    │   │       └── P5
    │   │           └── P6
    │   │               └── P7
    │   └── (P3 also enables P5 directly)
```

P4 needs both P1 (budget) and P3 (the eval scaffold to exercise Claude through). P5 needs P2 (adapter pattern) and benefits from P4 being done (Claude available for the full benchmark sweep). P6 is the convergence point.

## 5. Budget plan (Anthropic Sonnet 4.5 only)

Assumed pricing (verify in Phase 4 against current rates): roughly $3 / 1M input, $15 / 1M output. Treat as $0.000003 / input token, $0.000015 / output token.

Per-judgment back-of-envelope:
- Prompt: question + reference + rubric ≈ 600 input tokens.
- Output: short rubric label + 1-2 sentence justification ≈ 150 output tokens.
- Cost per judgment: 600 × 3e-6 + 150 × 15e-6 = $0.00180 + $0.00225 = **~$0.004**.

At $0.004 per Claude judgment and a $22 working budget (leaving $3 headroom):
- Total Claude judgments affordable: ~5500.
- Distribution across 5 benchmarks: ~1100 per benchmark.

This sets the headline run size. The other four judges run on the same item set at $0 marginal cost. Total judgments across 5 judges: ~27,500. **Run the estimator before the headline run; do not trust this paragraph.**

## 6. Risk register

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|-----------|
| R1 | Anthropic spend overruns $25 cap | Med | High | Circuit breaker in `budget/circuit_breaker.py` blocks calls whose estimated cost would exceed cap. Pre-flight estimator required before any non-smoke run. Manual check against Anthropic console after first 100 Claude calls. |
| R2 | Free-tier rate limits cause cascading failures | High | Med | Per-vendor token-bucket inside `retry.py`. Sequential queueing on the tightest-limited vendor (Groq 30 RPM on 70B). Long runs scheduled overnight. |
| R3 | Free-tier endpoint disappears or changes model availability | Med | High | Vendor registry takes a `fallback_model_id`. Document which judges were live as of run date in the manifest. If a vendor dies before Phase 6, swap to an equivalent free judge from OpenRouter and note the substitution in the tech note. |
| R4 | DeepSeek R1 reasoning output (with `<think>` tags or long chain-of-thought) breaks the label parser | Med | Med | Prompt template explicitly instructs final-line labeling. Parser tolerates leading reasoning blocks and extracts the last line matching the label rubric. Test fixture with a verbose DeepSeek-style response. |
| R5 | OpenRouter free-tier model availability changes mid-run | Med | Med | Pin the exact OpenRouter slug in the YAML config. If unavailable at runtime, fail loudly rather than silently swapping. |
| R6 | Token-count estimation diverges between providers (no shared tokenizer) | Med | Low | Estimator uses `anthropic.messages.count_tokens` for Anthropic, `tiktoken` cl100k as an approximation for the OpenAI-compat vendors. Document this in README. The estimator is for budget planning, not billing. |
| R7 | Benchmark answer-key licensing ambiguous for redistribution | Med | Med | Verify each dataset license before Phase 5. PubMedQA (MIT), MMLU (MIT), HumanEval (MIT), MBPP (CC-BY-4.0) are clean. MedQA: check license; if ambiguous, do not redistribute, load from HF at runtime only and document that the user pulls it themselves. |
| R8 | Reproducibility: providers do not honor `temperature=0` strictly | High | Med | Set `temperature=0`, `top_p=1`, document that exact-reproduction is not guaranteed across providers. The reproducibility claim is about the *recipe* (configs, prompts, model IDs, seed where supported), not byte-identical outputs. |
| R9 | vLLM OOM on 4080 16GB serving Qwen3-8B INT4 at `--max-model-len 4096` | Low | Low | Already constrained by `--gpu-memory-utilization 0.85`. If OOM, drop to `--max-model-len 2048`. Local judge is optional; not on the critical path. |
| R10 | Disagreement clusters not interpretable (KMeans on a sparse 5-dim categorical disagreement vector may produce uninformative clusters) | Med | Med | Have a fallback: if elbow gives k=1 or clusters are unbalanced past 90/10, switch to qualitative grouping by (benchmark × dominant-disagreement-pattern). Tech note frames this as exploratory either way. |
| R11 | PyPI name `judgekit` is taken | Low | Med | Check before Phase 0. If taken, fall back to `judgekit-evals`. |
| R12 | CI minutes blown on accidental live API calls in tests | Low | Med | `pytest-httpx` configured in strict mode; tests fail if any unmocked HTTP call is attempted. CI does not have API key env vars. |
| R13 | A phase is executed without a per-phase plan file (drift from methodology) | Med | Med | Each phase's gate now requires the per-phase plan file to exist at `docs/plans/...` before any code is written. Phase 0 acceptance includes adding this rule to `CLAUDE.md`. |
| R14 | TDD violated under time pressure ("I'll add tests after") | High | Med | The TDD skill's iron law applies: code without a preceding failing test is deleted and re-done. Self-review at end of each phase verifies test-first by inspecting commit order (test commit precedes implementation commit). |

## 7. Open questions for the user

These do not block writing the plan but should be resolved before Phase 3 or Phase 5:

1. **OpenRouter judge model.** Default proposed: `meta-llama/llama-3.3-70b-instruct:free`. Confirm or substitute.
2. **MedQA dataset variant.** Four-option USMLE (`GBaker/MedQA-USMLE-4-options`) or full BigBio? Four-option is cleaner for the judge rubric.
3. **vLLM/Qwen as the sixth judge.** In or out of the headline run? Treating it as optional in this plan; including it changes the agreement-matrix dimension to 6.
4. **Tech note venue.** arXiv only (cs.CL or cs.LG), or also a Hugging Face blog post mirror?
5. **Streamlit dashboard.** HF Spaces (this plan's assumption) or Oracle Cloud ARM? HF is simpler; Oracle is the stated default for persistent services. JudgeKit's data is static per release, so HF Spaces is the better fit.

## 8. End-of-plan verification (Phase 6 acceptance)

End-to-end acceptance, run by hand:

```bash
uv pip install judgekit==0.1.0
export ANTHROPIC_API_KEY=...
export GROQ_API_KEY=...
export CEREBRAS_API_KEY=...
export SAMBANOVA_API_KEY=...
export OPENROUTER_API_KEY=...
export JUDGEKIT_ANTHROPIC_BUDGET_USD=25

judgekit estimate --config configs/all_five_judges.yaml
# Inspect: total estimate < $25, no surprises.

judgekit run --config configs/all_five_judges.yaml --run-id headline_v01
# Wall clock: many hours. Should complete or trip the circuit breaker cleanly.

judgekit report --run-id headline_v01 --out results/headline_v01/report.csv
# Verify: pairwise kappa matrix, Fleiss kappa, Krippendorff alpha, per-vendor spend.

(cd paper && latexmk -pdf tech_note.tex)
# Verify: PDF compiles, contains the same numbers as report.csv.
```

If all four commands succeed and total spend is logged under $25, ship.
