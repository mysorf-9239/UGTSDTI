"""Static data models for graph planning.

REQ-GRAPH-001, REQ-GRAPH-002
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NodePluginSpec:
    """Static capability metadata for a node plugin type.

    Used during dry-run validation and planning — does NOT require loading
    model weights or external resources.

    Attributes:
        type_key:     Unique identifier for this plugin type (e.g. "encoder.drug_seq").
        output_attrs: List of attribute names this node produces (e.g. ["embedding"]).
                      Output keys are resolved as ``<node_name>.<attr>``.
        capabilities: Arbitrary metadata for config-level validation (e.g. modality info).
        input_kinds:  Batch-level input kinds this node requires (e.g. ["drug_seq"]).
    """

    type_key: str
    output_attrs: list[str]
    capabilities: dict[str, Any] = field(default_factory=dict)
    input_kinds: list[str] = field(default_factory=list)


@dataclass
class NodeDefinition:
    """Instance-level definition of a graph node from config.

    Attributes:
        name:     Unique node name within the graph (e.g. "student_encoder").
        type_key: Plugin type key — must be registered in NodeRegistry.
        inputs:   Explicit list of State keys this node reads (e.g. ["drug_seq"]).
        params:   Node-specific hyperparameters passed to the runtime constructor.
    """

    name: str
    type_key: str
    inputs: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphPlan:
    """Validated, ordered execution plan for the graph stage.

    Produced by GraphBuilder + GraphPlanner; consumed by GraphEngine.

    Attributes:
        node_definitions: All NodeDefinition objects in the graph.
        order:            Topologically sorted list of node names (execution order).
        produced_keys:    Mapping from node name -> list of output State keys it produces.
        producers:        Mapping from output State key -> node name that produces it.
        edges:            Adjacency list: node name -> list of node names it depends on.
    """

    node_definitions: list[NodeDefinition]
    order: list[str]
    produced_keys: dict[str, list[str]]
    producers: dict[str, str]
    edges: dict[str, list[str]]

    def definition_map(self) -> dict[str, NodeDefinition]:
        """Return a mapping from node name to NodeDefinition."""
        return {d.name: d for d in self.node_definitions}
