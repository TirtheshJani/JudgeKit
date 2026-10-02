# Contributing to JudgeKit

Thanks for your interest. Issues and pull requests are welcome.

## Setup

```bash
uv sync --group dev
```

## Before opening a PR

Run the same checks CI runs:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/judgekit
uv run pytest --tb=short
```

## Ground rules

- **Tests first.** New behaviour starts with a failing test. The suite must stay offline: mock HTTP with `pytest-httpx` or `respx`, never call a real vendor API from tests.
- **Cost accounting stays visible.** Any code path that calls a judge must record tokens and estimated cost and go through the budget circuit breaker.
- **Reproducible configs.** Pin model IDs, splits and prompt template versions in YAML. Changing a prompt template means adding a new version (for example `_v2.j2`), not editing `_v1` in place.
- **No secrets in commits.** Keys live in environment variables (see `.env.example`). Stage files explicitly rather than with `git add -A`.
- **Report results honestly.** Do not add agreement or cost numbers to docs unless they come from a real run's `judgekit report` output.
