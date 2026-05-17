from __future__ import annotations

from pathlib import Path

from judgekit.config import EvalConfig


def run_eval(config: EvalConfig, output_dir: Path, dry_run: bool = False) -> Path:
    """Orchestrate a full eval run: iterate (benchmark × judge × item), write JSONL."""
    raise NotImplementedError


def validate_run(jsonl_path: Path) -> bool:
    """Verify every record in a run JSONL matches the expected schema."""
    raise NotImplementedError
