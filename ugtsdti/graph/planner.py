"""GraphPlanner — topological sort and dry-run validation.

REQ-GRAPH-002, REQ-GRAPH-003, REQ-QUAL-004
"""

from __future__ import annotations

from collections import deque
from typing import Union

from ugtsdti.core.errors import InvalidInteractionGraphError
from ugtsdti.graph.registry import NodeRegistry
from ugtsdti.graph.specs import GraphPlan, NodeDefinition


class GraphPlanner:
    """Produces a deterministic execution order for a GraphPlan.

    Responsibilities:
    - Topological sort with stable (alphabetical) tie-breaking.
    - Dry-run validation using NodePluginSpec (no weights loaded).
    - O(V + E) DAG validation via Kahn's algorithm.
    - Clear reporting of unresolved dependencies.

    Usage::

        planner = GraphPlanner(registry)
        plan = planner.plan(unordered_plan)
    """

    def __init__(self, registry: NodeRegistry) -> None:
        self._registry = registry

    def plan(self, graph_plan: GraphPlan) -> GraphPlan:
        """Compute topological order and return an updated GraphPlan.

        The returned plan is a new GraphPlan with ``order`` filled in.
        All other fields are preserved from *graph_plan*.

        Args:
            graph_plan: A GraphPlan produced by GraphBuilder (order may be empty).

        Returns:
            GraphPlan with ``order`` set to a deterministic topological sort.

        Raises:
            InvalidInteractionGraphError: If a cycle is detected.
            MissingDependencyError:       If unresolved dependencies are found.
        """
        nodes = [d.name for d in graph_plan.node_definitions]
        edges = graph_plan.edges  # node -> list of nodes it depends on

        # Dry-run validation: ensure all type_keys are registered
        self._dry_run_validate(graph_plan)

        # Kahn's algorithm with stable tie-breaking (alphabetical sort)
        order = _topological_sort(nodes, edges, graph_plan.node_definitions)

        return GraphPlan(
            node_definitions=graph_plan.node_definitions,
            order=order,
            produced_keys=graph_plan.produced_keys,
            producers=graph_plan.producers,
            edges=graph_plan.edges,
        )

    def dump_plan(self, graph_plan: GraphPlan) -> None:
        """Debug output of plan order for testing determinism."""
        print("=== GraphPlan Debug ===")
        print(f"Node count: {len(graph_plan.node_definitions)}")
        print(f"Execution order: {graph_plan.order}")
        print("Node definitions:")
        for i, defn in enumerate(graph_plan.node_definitions):
            config_order = getattr(defn, "_config_order", "N/A")
            print(f"  {i}: {defn.name} (config_order: {config_order})")
        print("========================")

    def _dry_run_validate(self, graph_plan: GraphPlan) -> None:
        """Validate all node type_keys are registered (no weights loaded).

        Raises:
            MissingDependencyError: If a type_key is not in the registry.
        """
        for defn in graph_plan.node_definitions:
            # This calls get_spec which raises MissingDependencyError if absent
            self._registry.get_spec(defn.type_key)


# ---------------------------------------------------------------------------
# Topological sort (Kahn's algorithm, O(V+E), stable tie-break)
# ---------------------------------------------------------------------------


def _stable_sort_key(node: str, node_definitions: list[NodeDefinition]) -> str:
    """Return stable sort key for deterministic node ordering.

    Priority order:
    1. Config insertion order (if available)
    2. Node name (lexicographic)
    """
    # Try to get config order from node definitions
    for defn in node_definitions:
        if defn.name == node and hasattr(defn, "_config_order"):
            return str(defn._config_order)

    # Fallback to lexicographic
    return node.lower()  # Use lowercase for consistent sorting


def _topological_sort(
    nodes: list[str],
    edges: dict[str, list[str]],
    node_definitions: list[NodeDefinition],
) -> list[str]:
    """Return a deterministic topological order of *nodes*.

    *edges[n]* is the list of nodes that *n* depends on (predecessors).

    Uses Kahn's algorithm with a sorted queue for stable tie-breaking.

    Raises:
        InvalidInteractionGraphError: If a cycle is detected (remaining nodes
                                      after Kahn's pass).
    """
    # Build in-degree and reverse adjacency (successor map)
    in_degree: dict[str, int] = {n: 0 for n in nodes}
    successors: dict[str, list[str]] = {n: [] for n in nodes}

    for node in nodes:
        for dep in edges.get(node, []):
            in_degree[node] += 1
            successors[dep].append(node)

    # Create a mapping from node name to config order
    config_order_map = {}
    for defn in node_definitions:
        if hasattr(defn, "_config_order"):
            config_order_map[defn.name] = defn._config_order

    def stable_sort_key(node_name: str) -> tuple[Union[int, float], str]:
        """Return stable sort key: (config_order, node_name)."""
        config_order = config_order_map.get(node_name, None)
        if config_order is None:
            # No config order means no config_order, use large number to put after config-ordered nodes
            return (float("inf"), node_name.lower())
        else:
            return (config_order, node_name.lower())

    # Initialize queue with zero-in-degree nodes, sorted for determinism
    queue: deque[str] = deque(sorted((n for n in nodes if in_degree[n] == 0), key=stable_sort_key))
    order: list[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        # Reduce in-degree of successors; add newly zero-in-degree ones
        newly_ready: list[str] = []
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                newly_ready.append(succ)

        # CRITICAL FIX: Re-sort entire queue for global determinism
        # Append new nodes to queue, then re-sort entire queue
        queue.extend(newly_ready)
        queue = deque(sorted(queue, key=stable_sort_key))

    if len(order) != len(nodes):
        remaining = sorted(set(nodes) - set(order))
        raise InvalidInteractionGraphError(
            f"Cycle detected in graph; nodes involved: {remaining}.",
            stage="graph",
            component="GraphPlanner",
        )

    return order
