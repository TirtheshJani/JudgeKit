from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_TEMPLATES_DIR = Path(__file__).parent / "templates"


class PromptLoader:
    def __init__(self, templates_dir: Path | None = None) -> None:
        tdir = templates_dir or _TEMPLATES_DIR
        self._env = Environment(
            loader=FileSystemLoader(str(tdir)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )

    def render(self, template_name: str, **context: Any) -> str:
        tmpl = self._env.get_template(template_name)
        return tmpl.render(**context)
