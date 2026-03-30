"""GraphBuilder — parse config, validate, and produce a GraphPlan.

REQ-GRAPH-002, REQ-ARCH-004
"""

from __future__ import annotations

from typing import Any

from ugtsdti.core.errors import (
    InvalidConfigError,
    InvalidInteractionGraphError,
    KeyCollisionError,
    MissingDependencyError,
)
from ugtsdti.core.schema import validate_graph_key
from ugtsdti.graph.registry import NodeRegistry
from ugtsdti.graph.specs import GraphPlan, NodeDefinition


class GraphBuilder:
    """Builds a validated GraphPlan from config and a NodeRegistry.

    Works entirely on NodeDefinition + NodePluginSpec — does NOT instantiate
    runtime nodes or load weights.

    Usage::

        builder = GraphBuilder(registry)
        plan = builder.build(graph_cfg)
    """

    def __init__(self, registry: NodeRegistry) -> None:
        self._registry = registry

    def build(self, graph_cfg: dict[str, Any]) -> GraphPlan:
        """Parse *graph_cfg* and return a validated GraphPlan.

        Args:
            graph_cfg: The ``graph`` section of a normalized config dict.
                       Expected shape::

                           {
                               "nodes": [
                                   {
                                       "name": "student_encoder",
                                       "type_key": "encoder.drug_seq",
                                       "inputs": ["drug_seq"],
                                       "params": {}
                                   },
                                   ...
                               ]
                           }

        Returns:
            A validated GraphPlan ready for GraphPlanner.

        Raises:
            InvalidConfigError:          Duplicate node names or invalid key naming.
            MissingDependencyError:      Unresolved input dependency.
            KeyCollisionError:           Two nodes produce the same output key.
            InvalidInteractionGraphError: Cycle detected in the dependency graph.
        """
        raw_nodes: list[dict[str, Any]] = graph_cfg.get("nodes", [])

        # --- 1. Parse NodeDefinitions ----------------------------------------
        definitions: list[NodeDefinition] = []
        seen_names: set[str] = set()
        for raw in raw_nodes:
            name = raw.get("name", "")
            if not name:
                raise InvalidConfigError(
                    "Graph node is missing a 'name' field.",
                    stage="graph",
                    component="GraphBuilder",
                )
            if name in seen_names:
                raise InvalidConfigError(
                    f"Duplicate node name {name!r} in graph config.",
                    stage="graph",
                    component="GraphBuilder",
                    key=name,
                )
            seen_names.add(name)
            type_key = raw.get("type_key", raw.get("type", ""))
            if not type_key:
                raise InvalidConfigError(
                    f"Node {name!r} is missing a 'type_key' field.",
                    stage="graph",
                    component="GraphBuilder",
                    key=name,
                )
            definitions.append(
                NodeDefinition(
                    name=name,
                    type_key=type_key,
                    inputs=list(raw.get("inputs", [])),
                    params=dict(raw.get("params", {})),
                )
            )

        # --- 2. Resolve output keys and build producer map -------------------
        produced_keys: dict[str, list[str]] = {}  # node_name -> [state_key, ...]
        producers: dict[str, str] = {}  # state_key -> node_name

        for defn in definitions:
            spec = self._registry.get_spec(defn.type_key)
            node_output_keys: list[str] = []
            for attr in spec.output_attrs:
                key = f"{defn.name}.{attr}"
                # Validate naming convention
                validate_graph_key(key)
                # Single-producer enforcement
                if key in producers:
                    raise KeyCollisionError(
                        f"Output key {key!r} is produced by both {producers[key]!r} and {defn.name!r}.",
                        stage="graph",
                        component="GraphBuilder",
                        key=key,
                    )
                producers[key] = defn.name
                node_output_keys.append(key)
            produced_keys[defn.name] = node_output_keys

        # --- 3. Build adjacency (edges) and validate explicit deps -----------
        # All keys available before graph execution (batch keys + prior nodes)
        # We track which keys are produced by which node to resolve deps.
        # Batch keys are treated as "available from the start" — they are not
        # produced by any graph node, so we don't validate them here.
        # We only validate that if an input key looks like a graph key
        # (<node>.<attr>), the producing node must exist.

        edges: dict[str, list[str]] = {defn.name: [] for defn in definitions}
        name_set = {defn.name for defn in definitions}

        for defn in definitions:
            for input_key in defn.inputs:
                # If the input key matches graph key pattern, it must be
                # produced by another node in this graph.
                if "." in input_key and not _is_reserved_namespace(input_key):
                    if input_key not in producers:
                        raise MissingDependencyError(
                            f"Node {defn.name!r} declares input {input_key!r} "
                            f"which is not produced by any node in the graph.",
                            stage="graph",
                            component="GraphBuilder",
                            key=input_key,
                        )
                    dep_node = producers[input_key]
                    if dep_node not in edges[defn.name]:
                        edges[defn.name].append(dep_node)

        # --- 4. Cycle detection (Kahn's algorithm, O(V+E)) -------------------
        _detect_cycles(edges, name_set)

        return GraphPlan(
            node_definitions=definitions,
            order=[],  # filled by GraphPlanner
            produced_keys=produced_keys,
            producers=producers,
            edges=edges,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RESERVED_PREFIXES = ("loss.", "metrics.", "diagnostics.", "gate.")


def _is_reserved_namespace(key: str) -> bool:
    """Return True if *key* belongs to a reserved non-graph namespace."""
    return any(key.startswith(p) for p in _RESERVED_PREFIXES)


def _detect_cycles(
    edges: dict[str, list[str]],
    nodes: set[str],
) -> None:
    """Raise InvalidInteractionGraphError if *edges* contains a cycle.

    Uses iterative DFS with three-color marking (white/gray/black).
    O(V + E).
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in nodes}

    def dfs(start: str) -> None:
        stack: list[tuple[str, bool]] = [(start, False)]
        while stack:
            node, returning = stack.pop()
            if returning:
                color[node] = BLACK
                continue
            if color[node] == GRAY:
                raise InvalidInteractionGraphError(
                    f"Cycle detected in graph involving node {node!r}.",
                    stage="graph",
                    component="GraphBuilder",
                    key=node,
                )
            if color[node] == BLACK:
                continue
            color[node] = GRAY
            stack.append((node, True))  # schedule blackening on return
            for dep in edges.get(node, []):
                if color[dep] != BLACK:
                    stack.append((dep, False))

    for node in sorted(nodes):  # sorted for determinism
        if color[node] == WHITE:
            dfs(node)
