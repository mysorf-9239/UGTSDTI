"""Fan-out logger that broadcasts to multiple backends."""

from __future__ import annotations

from typing import Any

from ugtsdti.logging.base import Logger


class CompositeLogger(Logger):
    """Dispatch log calls to multiple child loggers."""

    def __init__(self, loggers: list[Logger]) -> None:
        self._loggers = list(loggers)

    def log_metrics(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        for logger in self._loggers:
            logger.log_metrics(metrics, step=step)

    def log_text(self, name: str, text: str) -> None:
        for logger in self._loggers:
            logger.log_text(name, text)

    def close(self) -> None:
        for logger in self._loggers:
            logger.close()
