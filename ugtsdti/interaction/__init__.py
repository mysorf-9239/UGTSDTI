"""Interaction stage foundations."""

from ugtsdti.interaction.base import (
    InteractionDefinition,
    InteractionPlan,
    InteractionPluginSpec,
    InteractionRuntime,
)
from ugtsdti.interaction.noop import NoOpInteraction
from ugtsdti.interaction.registry import InteractionPlanner, InteractionRegistry

__all__ = [
    "InteractionPluginSpec",
    "InteractionDefinition",
    "InteractionPlan",
    "InteractionRuntime",
    "InteractionRegistry",
    "InteractionPlanner",
    "NoOpInteraction",
]
