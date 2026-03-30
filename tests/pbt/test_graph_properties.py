"""Property-based tests for graph planning invariants."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ugtsdti.core.errors import InvalidInteractionGraphError
from ugtsdti.graph.builder import GraphBuilder
from ugtsdti.graph.planner import GraphPlanner
from ugtsdti.graph.registry import NodeRegistry
from ugtsdti.graph.specs import NodePluginSpec
from ugtsdti.nodes.base import NodeRuntime


class _StubRuntime(NodeRuntime):
    def forward(self, inputs, context):
        return {}


def _make_registry(*specs: NodePluginSpec) -> NodeRegistry:
    registry = NodeRegistry()
    for spec in specs:
        registry.register(spec, _StubRuntime)
    return registry


@st.composite
def _acyclic_graph_cfgs(draw):
    names = draw(st.lists(st.sampled_from(["a", "b", "c", "d", "e"]), min_size=1, max_size=5, unique=True))
    nodes = []
    for index, name in enumerate(names):
        possible_deps = names[:index]
        deps = draw(st.lists(st.sampled_from(possible_deps), unique=True)) if possible_deps else []
        nodes.append(
            {
                "name": name,
                "type_key": "prop.node",
                "inputs": [f"{dep}.out" for dep in deps],
            }
        )
    return {"nodes": nodes}


@given(_acyclic_graph_cfgs())
def test_graph_builder_and_planner_preserve_topological_validity_for_acyclic_graphs(cfg):
    spec = NodePluginSpec(type_key="prop.node", output_attrs=["out"])
    registry = _make_registry(spec)
    builder = GraphBuilder(registry)
    planner = GraphPlanner(registry)

    plan = planner.plan(builder.build(cfg))
    positions = {name: index for index, name in enumerate(plan.order)}

    for node_name, deps in plan.edges.items():
        for dep in deps:
            assert positions[dep] < positions[node_name]


@given(st.lists(st.sampled_from(["a", "b", "c", "d"]), min_size=2, max_size=4, unique=True))
def test_graph_builder_rejects_cycles_for_generated_graphs(names):
    spec = NodePluginSpec(type_key="prop.node", output_attrs=["out"])
    registry = _make_registry(spec)
    builder = GraphBuilder(registry)

    nodes = []
    for index, name in enumerate(names):
        inputs = [f"{names[index - 1]}.out"] if index > 0 else []
        nodes.append({"name": name, "type_key": "prop.node", "inputs": inputs})
    nodes[0]["inputs"] = [f"{names[-1]}.out"]

    with pytest.raises(InvalidInteractionGraphError):
        builder.build({"nodes": nodes})
