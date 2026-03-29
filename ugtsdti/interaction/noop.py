"""Built-in no-op interaction used by the minimal baseline."""
from __future__ import annotations

from typing import Any

from ugtsdti.core.context import ExecutionContext
from ugtsdti.interaction.base import InteractionRuntime


class NoOpInteraction(InteractionRuntime):
    """No-op interaction that preserves the stage boundary without outputs."""

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled

    def forward(
        self,
        inputs: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        del inputs, context
        if not self._enabled:
            return {}
        return {}
