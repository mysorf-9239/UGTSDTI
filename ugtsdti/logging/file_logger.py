"""Offline-capable file logger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ugtsdti.logging.base import Logger


class FileLogger(Logger):
    """Append metrics and text logs to local files."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self._root / "metrics.jsonl"

    def log_metrics(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        payload = {"step": step, "metrics": metrics}
        with self._metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def log_text(self, name: str, text: str) -> None:
        (self._root / f"{name}.txt").write_text(text, encoding="utf-8")

    def close(self) -> None:
        return None
