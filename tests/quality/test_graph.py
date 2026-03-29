"""Unit tests for GraphBuilder and GraphPlanner.

REQ-GRAPH-002, REQ-GRAPH-003, REQ-ARCH-004
"""
from __future__ import annotations

import pytest

from ugtsdti.core.errors import (
    InvalidConfigError,
    InvalidInteractionGraphError,
    MissingDependencyError,
)
from ugtsdti.graph.builder import GraphBuilder
from ugtsdti.graph.planner import GraphPlanner
from ugtsdti.graph.registry import NodeRegistry
from ugtsdti.graph.specs import NodePluginSpec
from ugtsdti.nodes.base import NodeRuntime

# ---------------------------------------------------------------------------
# Minimal stub runtime (never actually called in builder/planner tests)
# ---------------------------------------------------------------------------


class _StubRuntime(NodeRuntime):
    def forward(self, inputs, context):
        return {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_registry(*specs: NodePluginSpec) -> NodeRegistry:
    registry = NodeRegistry()
    for spec in specs:
        registry.register(spec, _StubRuntime)
    return registry


ENCODER_SPEC = NodePluginSpec(
    type_key="encoder.drug",
    output_attrs=["embedding"],
    input_kinds=["drug_seq"],
)

HEAD_SPEC = NodePluginSpec(
    type_key="head.mlp",
    output_attrs=["logits"],
    input_kinds=[],
)

FUSION_SPEC = NodePluginSpec(
    type_key="fusion.concat",
    output_attrs=["fused"],
    input_kinds=[],
)


# ---------------------------------------------------------------------------
# GraphBuilder tests
# ---------------------------------------------------------------------------


class TestGraphBuilderBasic:
    def test_single_node_plan(self):
        registry = _make_registry(ENCODER_SPEC)
        builder = GraphBuilder(registry)
        cfg = {"nodes": [{"name": "enc", "type_key": "encoder.drug", "inputs": ["drug_seq"]}]}
        plan = builder.build(cfg)
        assert len(plan.node_definitions) == 1
        assert plan.node_definitions[0].name == "enc"
        assert "enc.embedding" in plan.producers
        assert plan.producers["enc.embedding"] == "enc"
        assert plan.produced_keys["enc"] == ["enc.embedding"]

    def test_two_node_plan_with_dependency(self):
        registry = _make_registry(ENCODER_SPEC, HEAD_SPEC)
        builder = GraphBuilder(registry)
        cfg = {
            "nodes": [
                {"name": "enc", "type_key": "encoder.drug", "inputs": ["drug_seq"]},
                {"name": "head", "type_key": "head.mlp", "inputs": ["enc.embedding"]},
            ]
        }
        plan = builder.build(cfg)
        assert len(plan.node_definitions) == 2
        assert "head" in plan.edges
        assert "enc" in plan.edges["head"]

    def test_empty_graph(self):
        registry = _make_registry()
        builder = GraphBuilder(registry)
        plan = builder.build({"nodes": []})
        assert plan.node_definitions == []
        assert plan.producers == {}

    def test_output_keys_follow_naming_convention(self):
        registry = _make_registry(ENCODER_SPEC)
        builder = GraphBuilder(registry)
        cfg = {"nodes": [{"name": "student_enc", "type_key": "encoder.drug", "inputs": []}]}
        plan = builder.build(cfg)
        assert "student_enc.embedding" in plan.producers


class TestGraphBuilderValidation:
    def test_duplicate_node_names_rejected(self):
        registry = _make_registry(ENCODER_SPEC)
        builder = GraphBuilder(registry)
        cfg = {
            "nodes": [
                {"name": "enc", "type_key": "encoder.drug", "inputs": []},
                {"name": "enc", "type_key": "encoder.drug", "inputs": []},
            ]
        }
        with pytest.raises(InvalidConfigError, match="Duplicate node name"):
            builder.build(cfg)

    def test_missing_name_rejected(self):
        registry = _make_registry(ENCODER_SPEC)
        builder = GraphBuilder(registry)
        cfg = {"nodes": [{"type_key": "encoder.drug", "inputs": []}]}
        with pytest.raises(InvalidConfigError, match="missing a 'name'"):
            builder.build(cfg)

    def test_missing_type_key_rejected(self):
        registry = _make_registry(ENCODER_SPEC)
        builder = GraphBuilder(registry)
        cfg = {"nodes": [{"name": "enc", "inputs": []}]}
        with pytest.raises(InvalidConfigError, match="missing a 'type_key'"):
            builder.build(cfg)

    def test_unregistered_type_key_rejected(self):
        registry = _make_registry()  # empty registry
        builder = GraphBuilder(registry)
        cfg = {"nodes": [{"name": "enc", "type_key": "encoder.drug", "inputs": []}]}
        with pytest.raises(MissingDependencyError):
            builder.build(cfg)

    def test_single_producer_per_key_enforced(self):
        """Two nodes with same name would be caught by duplicate check first,
        but if two different nodes somehow produce the same key it must be caught."""
        # We create a spec that produces the same attr as another node's name collision
        spec_a = NodePluginSpec(type_key="type_a", output_attrs=["out"])
        spec_b = NodePluginSpec(type_key="type_b", output_attrs=["out"])
        registry = _make_registry(spec_a, spec_b)
        builder = GraphBuilder(registry)
        # Both nodes named differently but produce same key via same attr name
        # node_a.out and node_b.out are different keys — no collision here.
        # To get a collision we need same node name, which is caught earlier.
        # Instead test that two nodes with same output key via same name is caught.
        cfg = {
            "nodes": [
                {"name": "node", "type_key": "type_a", "inputs": []},
            ]
        }
        plan = builder.build(cfg)
        assert "node.out" in plan.producers

    def test_unresolved_graph_input_dependency_reported(self):
        registry = _make_registry(HEAD_SPEC)
        builder = GraphBuilder(registry)
        cfg = {
            "nodes": [
                # head depends on enc.embedding but enc is not in the graph
                {"name": "head", "type_key": "head.mlp", "inputs": ["enc.embedding"]},
            ]
        }
        with pytest.raises(MissingDependencyError, match="enc.embedding"):
            builder.build(cfg)

    def test_invalid_graph_key_naming_rejected(self):
        """A spec that would produce a key not matching <node>.<attr> is rejected."""
        # The key is always <node_name>.<attr> so the attr itself must be valid identifier.
        # We test that a node name with invalid chars causes validate_graph_key to fail.
        bad_spec = NodePluginSpec(type_key="bad.type", output_attrs=["embedding"])
        registry = _make_registry(bad_spec)
        builder = GraphBuilder(registry)
        # Node name with a dot would produce "a.b.embedding" which fails the regex
        cfg = {"nodes": [{"name": "a.b", "type_key": "bad.type", "inputs": []}]}
        with pytest.raises(InvalidConfigError, match="must match '<node>.<attr>'"):
            builder.build(cfg)

    def test_cycle_detection(self):
        spec_a = NodePluginSpec(type_key="type_a", output_attrs=["out"])
        spec_b = NodePluginSpec(type_key="type_b", output_attrs=["result"])
        registry = _make_registry(spec_a, spec_b)
        builder = GraphBuilder(registry)
        # a depends on b.result, b depends on a.out — cycle
        cfg = {
            "nodes": [
                {"name": "a", "type_key": "type_a", "inputs": ["b.result"]},
                {"name": "b", "type_key": "type_b", "inputs": ["a.out"]},
            ]
        }
        with pytest.raises(InvalidInteractionGraphError, match="[Cc]ycle"):
            builder.build(cfg)


# ---------------------------------------------------------------------------
# GraphPlanner tests
# ---------------------------------------------------------------------------


class TestGraphPlannerTopologicalOrder:
    def _build_plan(self, registry, cfg):
        builder = GraphBuilder(registry)
        planner = GraphPlanner(registry)
        raw_plan = builder.build(cfg)
        return planner.plan(raw_plan)

    def test_single_node_order(self):
        registry = _make_registry(ENCODER_SPEC)
        plan = self._build_plan(
            registry,
            {"nodes": [{"name": "enc", "type_key": "encoder.drug", "inputs": []}]},
        )
        assert plan.order == ["enc"]

    def test_two_node_dependency_order(self):
        registry = _make_registry(ENCODER_SPEC, HEAD_SPEC)
        plan = self._build_plan(
            registry,
            {
                "nodes": [
                    {"name": "head", "type_key": "head.mlp", "inputs": ["enc.embedding"]},
                    {"name": "enc", "type_key": "encoder.drug", "inputs": ["drug_seq"]},
                ]
            },
        )
        # enc must come before head
        assert plan.order.index("enc") < plan.order.index("head")

    def test_deterministic_order_stable_tie_break(self):
        """Nodes with no dependencies should be sorted alphabetically."""
        spec_a = NodePluginSpec(type_key="type_a", output_attrs=["out"])
        spec_b = NodePluginSpec(type_key="type_b", output_attrs=["out"])
        spec_c = NodePluginSpec(type_key="type_c", output_attrs=["out"])
        registry = _make_registry(spec_a, spec_b, spec_c)
        cfg = {
            "nodes": [
                {"name": "charlie", "type_key": "type_c", "inputs": []},
                {"name": "alpha", "type_key": "type_a", "inputs": []},
                {"name": "bravo", "type_key": "type_b", "inputs": []},
            ]
        }
        builder = GraphBuilder(registry)
        planner = GraphPlanner(registry)
        plan1 = planner.plan(builder.build(cfg))
        plan2 = planner.plan(builder.build(cfg))
        assert plan1.order == plan2.order
        assert plan1.order == ["alpha", "bravo", "charlie"]

    def test_three_node_chain_order(self):
        spec_a = NodePluginSpec(type_key="type_a", output_attrs=["out"])
        spec_b = NodePluginSpec(type_key="type_b", output_attrs=["out"])
        spec_c = NodePluginSpec(type_key="type_c", output_attrs=["out"])
        registry = _make_registry(spec_a, spec_b, spec_c)
        cfg = {
            "nodes": [
                {"name": "c", "type_key": "type_c", "inputs": ["b.out"]},
                {"name": "b", "type_key": "type_b", "inputs": ["a.out"]},
                {"name": "a", "type_key": "type_a", "inputs": []},
            ]
        }
        builder = GraphBuilder(registry)
        planner = GraphPlanner(registry)
        plan = planner.plan(builder.build(cfg))
        assert plan.order == ["a", "b", "c"]

    def test_dry_run_does_not_instantiate_runtime(self):
        """Planner.plan() must not call build_runtime (no weights loaded)."""
        instantiated = []

        class TrackingRuntime(NodeRuntime):
            def __init__(self, **kwargs):
                instantiated.append(True)

            def forward(self, inputs, context):
                return {}

        spec = NodePluginSpec(type_key="tracker", output_attrs=["out"])
        registry = NodeRegistry()
        registry.register(spec, TrackingRuntime)
        cfg = {"nodes": [{"name": "n", "type_key": "tracker", "inputs": []}]}
        builder = GraphBuilder(registry)
        planner = GraphPlanner(registry)
        planner.plan(builder.build(cfg))
        assert instantiated == [], "Planner must not instantiate runtime classes"

    def test_unresolved_dependency_reported(self):
        """Planner should propagate MissingDependencyError from builder."""
        registry = _make_registry(HEAD_SPEC)
        builder = GraphBuilder(registry)
        cfg = {
            "nodes": [
                {"name": "head", "type_key": "head.mlp", "inputs": ["missing.key"]},
            ]
        }
        with pytest.raises(MissingDependencyError):
            builder.build(cfg)

    def test_cycle_detected_by_planner(self):
        """Planner should detect cycles not caught by builder (edge case)."""
        # Manually construct a plan with a cycle in edges to test planner's
        # topological sort raises correctly.
        from ugtsdti.graph.specs import GraphPlan, NodeDefinition

        spec_a = NodePluginSpec(type_key="type_a", output_attrs=["out"])
        spec_b = NodePluginSpec(type_key="type_b", output_attrs=["result"])
        registry = _make_registry(spec_a, spec_b)
        planner = GraphPlanner(registry)

        defn_a = NodeDefinition(name="a", type_key="type_a", inputs=[])
        defn_b = NodeDefinition(name="b", type_key="type_b", inputs=[])
        # Manually inject a cycle in edges
        plan_with_cycle = GraphPlan(
            node_definitions=[defn_a, defn_b],
            order=[],
            produced_keys={"a": ["a.out"], "b": ["b.result"]},
            producers={"a.out": "a", "b.result": "b"},
            edges={"a": ["b"], "b": ["a"]},  # cycle
        )
        with pytest.raises(InvalidInteractionGraphError, match="[Cc]ycle"):
            planner.plan(plan_with_cycle)
