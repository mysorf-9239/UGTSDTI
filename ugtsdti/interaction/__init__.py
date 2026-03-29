"""Interaction stage foundations."""

from ugtsdti.interaction.base import (
    InteractionDefinition,
    InteractionPlan,
    InteractionPluginSpec,
    InteractionRuntime,
)
from ugtsdti.interaction.diagnostics import DiagnosticsInteraction, diagnostics_output_keys
from ugtsdti.interaction.engine import InteractionEngine
from ugtsdti.interaction.kd import KDInteraction, binary_logits_to_dist, kd_output_keys
from ugtsdti.interaction.noop import NoOpInteraction
from ugtsdti.interaction.registry import InteractionPlanner, InteractionRegistry
from ugtsdti.interaction.uncertainty import UncertaintyInteraction, uncertainty_output_keys

__all__ = [
    "InteractionPluginSpec",
    "InteractionDefinition",
    "InteractionPlan",
    "InteractionRuntime",
    "InteractionEngine",
    "KDInteraction",
    "binary_logits_to_dist",
    "kd_output_keys",
    "UncertaintyInteraction",
    "uncertainty_output_keys",
    "DiagnosticsInteraction",
    "diagnostics_output_keys",
    "InteractionRegistry",
    "InteractionPlanner",
    "NoOpInteraction",
]
