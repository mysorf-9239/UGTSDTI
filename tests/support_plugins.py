"""Test-only runtime plugin registrars."""
from __future__ import annotations

from typing import Any

from ugtsdti.graph.specs import NodePluginSpec
from ugtsdti.interaction.base import InteractionPluginSpec
from ugtsdti.nodes.base import NodeRuntime


class EchoEncoderRuntime(NodeRuntime):
    """Return the first provided input as an embedding."""

    def forward(self, inputs: dict[str, Any], context: Any) -> dict[str, Any]:
        del context
        return {"embedding": next(iter(inputs.values()))}


def register_test_runtime_plugins(graph_registry: Any, interaction_registry: Any) -> None:
    """Register a tiny encoder and a no-op interaction alias for CLI tests."""
    from ugtsdti.interaction.noop import NoOpInteraction

    graph_registry.register(
        NodePluginSpec(type_key="encoder.echo", output_attrs=["embedding"], input_kinds=["drug_seq"]),
        EchoEncoderRuntime,
    )
    interaction_registry.register(
        InteractionPluginSpec(type_key="noop.echo", output_keys_fn=lambda params: []),
        NoOpInteraction,
    )
