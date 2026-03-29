"""NodeRegistry — maps plugin type keys to specs and runtime classes.

REQ-GRAPH-001
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ugtsdti.core.errors import MissingDependencyError
from ugtsdti.graph.specs import NodeDefinition, NodePluginSpec

if TYPE_CHECKING:
    from ugtsdti.nodes.base import NodeRuntime


class NodeRegistry:
    """Central registry for node plugin specs and runtime classes.

    Usage::

        registry = NodeRegistry()
        registry.register(spec, MyNodeRuntime)
        spec = registry.get_spec("encoder.drug_seq")
        runtime = registry.build_runtime(definition)
    """

    def __init__(self) -> None:
        self._specs: dict[str, NodePluginSpec] = {}
        self._runtime_classes: dict[str, type["NodeRuntime"]] = {}

    def register(
        self,
        spec: NodePluginSpec,
        runtime_cls: type["NodeRuntime"],
    ) -> None:
        """Register a plugin spec together with its runtime class.

        Args:
            spec:        Static capability metadata for the plugin type.
            runtime_cls: Class (not instance) that implements NodeRuntime.forward.
        """
        self._specs[spec.type_key] = spec
        self._runtime_classes[spec.type_key] = runtime_cls

    def get_spec(self, type_key: str) -> NodePluginSpec:
        """Return the NodePluginSpec for *type_key*.

        Raises:
            MissingDependencyError: If *type_key* is not registered.
        """
        if type_key not in self._specs:
            raise MissingDependencyError(
                f"Node plugin type {type_key!r} is not registered.",
                stage="graph",
                component="NodeRegistry",
                key=type_key,
            )
        return self._specs[type_key]

    def build_runtime(self, definition: NodeDefinition) -> "NodeRuntime":
        """Instantiate a NodeRuntime for *definition*.

        Args:
            definition: NodeDefinition from the graph config.

        Returns:
            A new NodeRuntime instance constructed with definition.params.

        Raises:
            MissingDependencyError: If the type_key is not registered.
        """
        if definition.type_key not in self._runtime_classes:
            raise MissingDependencyError(
                f"Node plugin type {definition.type_key!r} is not registered.",
                stage="graph",
                component="NodeRegistry",
                key=definition.type_key,
            )
        cls = self._runtime_classes[definition.type_key]
        return cls(**definition.params)

    def registered_type_keys(self) -> list[str]:
        """Return sorted list of all registered type keys."""
        return sorted(self._specs.keys())
