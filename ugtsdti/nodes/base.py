"""NodeRuntime — abstract base class for all graph node implementations.

REQ-GRAPH-001
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ugtsdti.core.context import ExecutionContext


class NodeRuntime(ABC):
    """Abstract base class for runtime graph nodes.

    Nodes receive only their declared inputs (materialized from State by
    GraphEngine) and must NOT access State directly.

    The return dict keys are *attribute names* (e.g. ``"embedding"``), NOT
    full State keys.  GraphEngine qualifies them to ``<node_name>.<attr>``
    before committing to State.
    """

    @abstractmethod
    def forward(
        self,
        inputs: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        """Run the node's computation.

        Args:
            inputs:  Materialized values for all declared inputs of this node.
                     Keys are the State keys listed in NodeDefinition.inputs.
            context: Execution context (mode, device, seed, …).

        Returns:
            Dict mapping output attribute name -> value.
            Keys must match the ``output_attrs`` declared in NodePluginSpec.
        """
