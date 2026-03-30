"""GraphPlanner — topological sort and dry-run validation.

REQ-GRAPH-002, REQ-GRAPH-003, REQ-QUAL-004
"""

from __future__ import annotations

from collections import deque

from ugtsdti.core.errors import InvalidInteractionGraphError
from ugtsdti.graph.registry import NodeRegistry
from ugtsdti.graph.specs import GraphPlan


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
        order = _topological_sort(nodes, edges)

        return GraphPlan(
            node_definitions=graph_plan.node_definitions,
            order=order,
            produced_keys=graph_plan.produced_keys,
            producers=graph_plan.producers,
            edges=graph_plan.edges,
        )

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


def _topological_sort(
    nodes: list[str],
    edges: dict[str, list[str]],
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

    # Initialize queue with zero-in-degree nodes, sorted for determinism
    queue: deque[str] = deque(sorted(n for n in nodes if in_degree[n] == 0))
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
        # Sort for stable tie-breaking before extending queue
        for n in sorted(newly_ready):
            queue.append(n)

    if len(order) != len(nodes):
        remaining = sorted(set(nodes) - set(order))
        raise InvalidInteractionGraphError(
            f"Cycle detected in graph; nodes involved: {remaining}.",
            stage="graph",
            component="GraphPlanner",
        )

    return order
