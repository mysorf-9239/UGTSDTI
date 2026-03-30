"""Base contracts for the interaction stage.

REQ-INT-001
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from ugtsdti.core.context import ExecutionContext


@dataclass(frozen=True)
class InteractionPluginSpec:
    """Static metadata for planning an interaction module."""

    type_key: str
    output_keys_fn: Callable[[dict[str, Any]], list[str]]


@dataclass(frozen=True)
class InteractionDefinition:
    """Configured interaction instance for one experiment."""

    name: str
    type_key: str
    inputs: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class InteractionPlan:
    """Validated, ordered execution plan for interaction modules."""

    definitions: list[InteractionDefinition]
    order: list[str]
    producers: dict[str, str]
    produced_keys: dict[str, list[str]]

    def definition_map(self) -> dict[str, InteractionDefinition]:
        return {definition.name: definition for definition in self.definitions}


class InteractionRuntime(ABC):
    """Abstract base for interaction modules."""

    @abstractmethod
    def forward(
        self,
        inputs: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        """Produce interaction outputs from declared inputs."""
