"""Base interface for the decision stage.

REQ-DEC-001
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.state import State


class DecisionModule(ABC):
    """Abstract base for decision-stage modules."""

    @abstractmethod
    def forward(
        self,
        state: State,
        context: ExecutionContext,
    ) -> dict[str, Any]:
        """Produce final decision outputs, including canonical `logits`."""
