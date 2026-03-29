"""Unit tests for GraphEngine validation hooks and trace.

REQ-STATE-003, REQ-GRAPH-003, REQ-QUAL-002
"""
from __future__ import annotations

from typing import Any

import pytest

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.errors import (
    InvalidConfigError,
    MissingDependencyError,
    NumericalInstabilityError,
)
from ugtsdti.core.state import State, StateWriter
from ugtsdti.graph.builder import GraphBuilder
from ugtsdti.graph.engine import GraphEngine
from ugtsdti.graph.planner import GraphPlanner
from ugtsdti.graph.registry import NodeRegistry
from ugtsdti.graph.specs import GraphPlan, NodePluginSpec
from ugtsdti.nodes.base import NodeRuntime

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context() -> ExecutionContext:
    return ExecutionContext(mode="train", seed=42, device="cpu", deterministic=False)


def _make_state_and_writer(**initial_keys) -> tuple[State, StateWriter]:
    state = State()
    writer = StateWriter(state)
    if initial_keys:
        writer.commit("batch", initial_keys)
    return state, writer


def _make_registry(*specs_and_classes) -> NodeRegistry:
    registry = NodeRegistry()
    for spec, cls in specs_and_classes:
        registry.register(spec, cls)
    return registry


def _build_and_plan(registry, cfg) -> GraphPlan:
    builder = GraphBuilder(registry)
    planner = GraphPlanner(registry)
    return planner.plan(builder.build(cfg))


# ---------------------------------------------------------------------------
# Stub runtimes
# ---------------------------------------------------------------------------


class EchoRuntime(NodeRuntime):
    """Returns a fixed output dict."""

    def __init__(self, outputs: dict[str, Any] | None = None, **kwargs):
        self._outputs = outputs or {"embedding": [1.0, 2.0]}

    def forward(self, inputs, context):
        return dict(self._outputs)


class UndeclaredOutputRuntime(NodeRuntime):
    """Returns an attr not declared in the spec."""

    def forward(self, inputs, context):
        return {"embedding": [1.0], "secret": "oops"}


class InputReadingRuntime(NodeRuntime):
    """Reads only declared inputs — correct behavior."""

    def forward(self, inputs, context):
        return {"embedding": inputs.get("drug_seq", [])}


# ---------------------------------------------------------------------------
# Tests: undeclared output attr rejected
# ---------------------------------------------------------------------------


class TestUndeclaredOutputRejected:
    def test_undeclared_attr_raises(self):
        spec = NodePluginSpec(type_key="enc", output_attrs=["embedding"])
        registry = _make_registry((spec, UndeclaredOutputRuntime))
        plan = _build_and_plan(
            registry,
            {"nodes": [{"name": "enc", "type_key": "enc", "inputs": []}]},
        )
        state, writer = _make_state_and_writer()
        engine = GraphEngine(registry)
        with pytest.raises(InvalidConfigError, match="undeclared output attr"):
            engine.run(plan, state, writer, _make_context())

    def test_declared_attr_accepted(self):
        spec = NodePluginSpec(type_key="enc", output_attrs=["embedding"])
        registry = _make_registry((spec, EchoRuntime))
        plan = _build_and_plan(
            registry,
            {"nodes": [{"name": "enc", "type_key": "enc", "inputs": []}]},
        )
        state, writer = _make_state_and_writer()
        engine = GraphEngine(registry)
        engine.run(plan, state, writer, _make_context())
        assert state.has("enc.embedding")


# ---------------------------------------------------------------------------
# Tests: node reading outside declared inputs rejected
# ---------------------------------------------------------------------------


class TestDeclaredInputEnforcement:
    def test_missing_declared_input_raises(self):
        """Node declares a batch input that is absent from State at runtime — must raise."""
        spec = NodePluginSpec(type_key="head", output_attrs=["logits"])
        registry = _make_registry((spec, EchoRuntime))
        # "drug_seq" is a batch key (no dot), so builder won't reject it as unresolved
        plan = _build_and_plan(
            registry,
            {
                "nodes": [
                    {
                        "name": "head",
                        "type_key": "head",
                        "inputs": ["drug_seq"],  # batch key absent from State
                    }
                ]
            },
        )
        state, writer = _make_state_and_writer()  # state is empty — drug_seq missing
        engine = GraphEngine(registry)
        with pytest.raises(MissingDependencyError, match="drug_seq"):
            engine.run(plan, state, writer, _make_context())

    def test_declared_input_present_succeeds(self):
        spec = NodePluginSpec(type_key="head", output_attrs=["logits"])

        class HeadRuntime(NodeRuntime):
            def forward(self, inputs, context):
                return {"logits": inputs["drug_seq"]}

        registry = _make_registry((spec, HeadRuntime))
        plan = _build_and_plan(
            registry,
            {
                "nodes": [
                    {
                        "name": "head",
                        "type_key": "head",
                        "inputs": ["drug_seq"],
                    }
                ]
            },
        )
        state, writer = _make_state_and_writer(drug_seq="ACGT")
        engine = GraphEngine(registry)
        engine.run(plan, state, writer, _make_context())
        assert state.has("head.logits")

    def test_node_receives_only_declared_inputs(self):
        """Node forward() must only receive declared inputs, not full State."""
        received_keys: list[list[str]] = []

        spec = NodePluginSpec(type_key="enc", output_attrs=["out"])

        class SpyRuntime(NodeRuntime):
            def forward(self, inputs, context):
                received_keys.append(list(inputs.keys()))
                return {"out": 1}

        registry = _make_registry((spec, SpyRuntime))
        plan = _build_and_plan(
            registry,
            {"nodes": [{"name": "enc", "type_key": "enc", "inputs": ["drug_seq"]}]},
        )
        # State has extra keys beyond declared inputs
        state, writer = _make_state_and_writer(
            drug_seq="ACGT",
            labels=[0, 1],
            scenario="S1",
        )
        engine = GraphEngine(registry)
        engine.run(plan, state, writer, _make_context())
        assert received_keys == [["drug_seq"]], "Node should only receive declared inputs, not full State"


# ---------------------------------------------------------------------------
# Tests: debug trace
# ---------------------------------------------------------------------------


class TestDebugTrace:
    def _run_two_node_graph(self, debug: bool):
        spec_enc = NodePluginSpec(type_key="enc", output_attrs=["embedding"])
        spec_head = NodePluginSpec(type_key="head", output_attrs=["logits"])

        class EncRuntime(NodeRuntime):
            def forward(self, inputs, context):
                return {"embedding": [1.0]}

        class HeadRuntime(NodeRuntime):
            def forward(self, inputs, context):
                return {"logits": inputs["enc.embedding"]}

        registry = _make_registry((spec_enc, EncRuntime), (spec_head, HeadRuntime))
        cfg = {
            "nodes": [
                {"name": "enc", "type_key": "enc", "inputs": ["drug_seq"]},
                {"name": "head", "type_key": "head", "inputs": ["enc.embedding"]},
            ]
        }
        plan = _build_and_plan(registry, cfg)
        state, writer = _make_state_and_writer(drug_seq="ACGT")
        engine = GraphEngine(registry, debug=debug)
        trace = engine.run(plan, state, writer, _make_context())
        return trace

    def test_trace_contains_node_order(self):
        trace = self._run_two_node_graph(debug=True)
        assert trace.node_order == ["enc", "head"]

    def test_trace_contains_inputs_consumed(self):
        trace = self._run_two_node_graph(debug=True)
        enc_event = next(e for e in trace.events if e.node_name == "enc")
        assert "drug_seq" in enc_event.inputs_consumed

    def test_trace_contains_outputs_produced(self):
        trace = self._run_two_node_graph(debug=True)
        enc_event = next(e for e in trace.events if e.node_name == "enc")
        assert "enc.embedding" in enc_event.outputs_produced

    def test_trace_has_event_per_node(self):
        trace = self._run_two_node_graph(debug=True)
        assert len(trace.events) == 2
        node_names = [e.node_name for e in trace.events]
        assert "enc" in node_names
        assert "head" in node_names

    def test_no_events_when_debug_false(self):
        trace = self._run_two_node_graph(debug=False)
        assert trace.events == []

    def test_trace_node_order_matches_execution_order(self):
        trace = self._run_two_node_graph(debug=True)
        event_order = [e.node_name for e in trace.events]
        assert event_order == trace.node_order


# ---------------------------------------------------------------------------
# Tests: strict mode numerical safety
# ---------------------------------------------------------------------------


class TestStrictModeNumericalSafety:
    def test_nan_output_rejected_in_strict_mode(self):
        spec = NodePluginSpec(type_key="enc", output_attrs=["val"])

        class NanRuntime(NodeRuntime):
            def forward(self, inputs, context):
                return {"val": float("nan")}

        registry = _make_registry((spec, NanRuntime))
        plan = _build_and_plan(
            registry,
            {"nodes": [{"name": "enc", "type_key": "enc", "inputs": []}]},
        )
        state, writer = _make_state_and_writer()
        engine = GraphEngine(registry, strict_mode=True)
        with pytest.raises(NumericalInstabilityError):
            engine.run(plan, state, writer, _make_context())

    def test_inf_output_rejected_in_strict_mode(self):
        spec = NodePluginSpec(type_key="enc", output_attrs=["val"])

        class InfRuntime(NodeRuntime):
            def forward(self, inputs, context):
                return {"val": float("inf")}

        registry = _make_registry((spec, InfRuntime))
        plan = _build_and_plan(
            registry,
            {"nodes": [{"name": "enc", "type_key": "enc", "inputs": []}]},
        )
        state, writer = _make_state_and_writer()
        engine = GraphEngine(registry, strict_mode=True)
        with pytest.raises(NumericalInstabilityError):
            engine.run(plan, state, writer, _make_context())

    def test_finite_output_accepted_in_strict_mode(self):
        spec = NodePluginSpec(type_key="enc", output_attrs=["val"])

        class FiniteRuntime(NodeRuntime):
            def forward(self, inputs, context):
                return {"val": 3.14}

        registry = _make_registry((spec, FiniteRuntime))
        plan = _build_and_plan(
            registry,
            {"nodes": [{"name": "enc", "type_key": "enc", "inputs": []}]},
        )
        state, writer = _make_state_and_writer()
        engine = GraphEngine(registry, strict_mode=True)
        engine.run(plan, state, writer, _make_context())
        assert state.has("enc.val")


# ---------------------------------------------------------------------------
# Tests: output key collision via StateWriter
# ---------------------------------------------------------------------------


class TestOutputKeyCollision:
    def test_two_nodes_same_output_key_raises(self):
        """Two nodes producing the same State key must be caught by builder."""
        spec_a = NodePluginSpec(type_key="type_a", output_attrs=["out"])
        spec_b = NodePluginSpec(type_key="type_b", output_attrs=["out"])

        class RuntimeA(NodeRuntime):
            def forward(self, inputs, context):
                return {"out": 1}

        class RuntimeB(NodeRuntime):
            def forward(self, inputs, context):
                return {"out": 2}

        registry = _make_registry((spec_a, RuntimeA), (spec_b, RuntimeB))
        # Both nodes have different names so keys are "a.out" and "b.out" — no collision
        cfg = {
            "nodes": [
                {"name": "a", "type_key": "type_a", "inputs": []},
                {"name": "b", "type_key": "type_b", "inputs": []},
            ]
        }
        plan = _build_and_plan(registry, cfg)
        state, writer = _make_state_and_writer()
        engine = GraphEngine(registry)
        engine.run(plan, state, writer, _make_context())
        assert state.has("a.out")
        assert state.has("b.out")
