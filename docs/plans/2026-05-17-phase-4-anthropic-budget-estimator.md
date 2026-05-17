# Phase 4: Anthropic judge with hard budget cap

**Date:** 2026-05-17
**Goal:** Token-accurate pre-flight cost estimator; `scripts/estimate_budget.py` implemented; CI-verifiable automated tests. Live gates (smoke run match, console check) are manual.

## Architecture

```
budget/estimator.py          <-- new: count_tokens(text) using tiktoken cl100k
scripts/estimate_budget.py   <-- stub → full implementation using estimator + config
tests/test_estimator.py      <-- TDD: deterministic token counts + estimator output
```

`anthropic_client.py`, `circuit_breaker.py`, `tracker.py`, `pricing.py` are all **done** — no changes.

The CLI `judgekit estimate` uses a fixed 600/150 approximation (Phase 2). We leave it unchanged — it's a fast rough estimate. `scripts/estimate_budget.py` is the detailed per-item version.

## Tech Stack

- `tiktoken` (already a runtime dep in pyproject.toml) — token counting
- `judgekit.config.load_config` — reads YAML
- `judgekit.prompts.loader.render_prompt` — renders Jinja2 templates
- `judgekit.budget.pricing.lookup` — pricing table
- `judgekit.benchmarks.*` — adapters (will be mocked in tests)

## Tasks

### Task 1: `budget/estimator.py` — token counter

**Files:** `src/judgekit/budget/estimator.py`, `tests/test_estimator.py`

**Steps (TDD):**

1. Write `tests/test_estimator.py`:
   - `test_count_tokens_empty_string` — `count_tokens("") == 0`
   - `test_count_tokens_known_string` — `count_tokens("Hello world") == 2` (exact tiktoken cl100k count)
   - `test_count_tokens_is_deterministic` — same input twice returns same int
   - `test_count_tokens_longer_text` — a 100-word string returns a positive int, value within known range

2. Run pytest on test file — all 4 tests fail (NotImplementedError or import error).

3. Implement `src/judgekit/budget/estimator.py`:
   ```python
   from __future__ import annotations
   import tiktoken
   
   _ENC = tiktoken.get_encoding("cl100k_base")
   
   def count_tokens(text: str) -> int:
       return len(_ENC.encode(text))
   ```

4. Run pytest on test file — all 4 pass.

5. Run full suite — no regressions.

6. Lint: `uv run ruff check src/judgekit/budget/estimator.py tests/test_estimator.py`

7. Commit: `git add src/judgekit/budget/estimator.py tests/test_estimator.py && git commit -m "feat: token counter using tiktoken cl100k"`

---

### Task 2: `scripts/estimate_budget.py` — full estimator

**Files:** `scripts/estimate_budget.py`, `tests/test_estimator.py` (extend)

**Steps (TDD):**

1. Add tests to `tests/test_estimator.py`:
   - `test_estimate_script_is_importable` — `from scripts.estimate_budget import build_estimate` (we'll expose a `build_estimate(config)` function)
   - `test_build_estimate_returns_per_vendor_breakdown` — given a mock config with 1 judge (groq) and 1 benchmark (pubmedqa n=2), with mocked benchmark adapter returning 2 synthetic items, returns a dict `{"groq": {"n_items": 2, "est_cost_usd": 0.0, ...}}`
   - `test_build_estimate_anthropic_cost_nonzero` — given anthropic judge config, returns est_cost_usd > 0
   - `test_build_estimate_uses_rendered_prompt_tokens` — token count in result is close to `count_tokens(rendered_prompt)` for the synthetic item

2. Run those tests — they fail (stub raises NotImplementedError).

3. Implement `scripts/estimate_budget.py`:
   - Expose `build_estimate(config: EvalConfig) -> dict[str, dict[str, Any]]`
   - For each judge × benchmark × item: render prompt, count tokens, accumulate
   - `main()` calls `build_estimate`, prints per-vendor breakdown
   - Note: to avoid downloading HF datasets in the script, the implementation lazily loads benchmark adapters; in tests we mock the adapter

4. Run the new tests — all pass.

5. Run full suite — no regressions.

6. Lint: `uv run ruff check scripts/estimate_budget.py tests/test_estimator.py`

7. Commit: `git add scripts/estimate_budget.py tests/test_estimator.py && git commit -m "feat: scripts/estimate_budget.py with tiktoken token counting"`

---

### Task 3: Type check and push

1. Run `uv run mypy src/judgekit/budget/estimator.py` — clean.
2. Run `uv run pytest --tb=short` — all green.
3. `git push -u origin claude/deploy-agents-kDGhi`

---

## Verification gate (automated)

```bash
uv run pytest tests/test_estimator.py -v   # all tests pass
uv run pytest 2>&1 | tail -3               # all N passed
uv run ruff check .                        # clean
uv run mypy src/judgekit                   # clean
```

## Verification gate (manual — requires API keys, not in CI)

```bash
export ANTHROPIC_API_KEY=...
export GROQ_API_KEY=...
judgekit run --config configs/smoke_test.yaml   # n=10 PubMedQA × Groq, $0.00
uv run python scripts/estimate_budget.py --config configs/smoke_test.yaml
# Compare to actual spend in run output

JUDGEKIT_ANTHROPIC_BUDGET_USD=0.10 judgekit run --config configs/smoke_test.yaml
# Should hit BudgetExceededError immediately for anthropic judge
```
