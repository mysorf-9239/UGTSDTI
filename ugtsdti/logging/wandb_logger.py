"""Optional Weights & Biases logger with safe fallback."""

from __future__ import annotations

from typing import Any

from ugtsdti.logging.base import Logger


class WandbLogger(Logger):
    """Thin adapter around wandb that degrades gracefully when unavailable."""

    def __init__(self, *, project: str, enabled: bool = True) -> None:
        self._enabled = enabled
        self._run = None
        if not enabled:
            return
        try:
            import wandb

            self._run = wandb.init(project=project, mode="offline")
        except Exception:
            self._enabled = False
            self._run = None

    def log_metrics(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        if self._enabled and self._run is not None:
            self._run.log(metrics, step=step)

    def log_text(self, name: str, text: str) -> None:
        if self._enabled and self._run is not None:
            self._run.summary[name] = text

    def close(self) -> None:
        if self._enabled and self._run is not None:
            self._run.finish()
