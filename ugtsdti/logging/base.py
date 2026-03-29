"""Logger abstraction."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Logger(ABC):
    """Abstract logging interface for runtime/train/eval events."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        """Log scalar-like metrics."""

    @abstractmethod
    def log_text(self, name: str, text: str) -> None:
        """Log textual artifacts."""

    @abstractmethod
    def close(self) -> None:
        """Flush and close resources."""
