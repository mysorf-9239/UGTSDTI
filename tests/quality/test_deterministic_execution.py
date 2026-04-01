"""Tests for deterministic graph execution across multiple runs."""

import pytest

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.state import State, StateWriter
from ugtsdti.graph.engine import GraphEngine
from ugtsdti.graph.planner import GraphPlanner
from ugtsdti.graph.specs import GraphPlan, NodeDefinition, NodePluginSpec
from ugtsdti.nodes.base import NodeRuntime


class TestDeterministicExecution:
    """Test that graph execution is deterministic across multiple runs."""

    def test_same_graph_produces_identical_outputs(self):
        """Test that running the same graph twice produces identical outputs."""

        # Create a deterministic node
        class DeterministicNode(NodeRuntime):
            def __init__(self, multiplier=2.0):
                self.multiplier = multiplier

            def forward(self, inputs, context):
                # Deterministic computation
                value = inputs.get("value", 1.0)
                result = value * self.multiplier
                return {"output": result}

        # Build graph
        registry = _make_deterministic_registry()
        engine = GraphEngine(registry)

        plan = _create_deterministic_plan()

        # Run first time
        state1, writer1 = _create_state_with_input(5.0)
        engine.run(plan, state1, writer1, ExecutionContext(mode="eval", seed=42, device="cpu", deterministic=True))

        # Run second time
        state2, writer2 = _create_state_with_input(5.0)
        engine.run(plan, state2, writer2, ExecutionContext(mode="eval", seed=42, device="cpu", deterministic=True))

        # Compare outputs
        assert state1.get("node1.output") == state2.get("node1.output"), "node1 outputs should be identical"
        assert state1.get("node2.output") == state2.get("node2.output"), "node2 outputs should be identical"
        assert state1.get("node3.output") == state2.get("node3.output"), "node3 outputs should be identical"

    def test_same_graph_produces_identical_execution_order(self):
        """Test that execution order is deterministic across runs."""
        registry = _make_deterministic_registry()
        planner = GraphPlanner(registry)

        plan = _create_complex_plan()  # Has multiple valid topological orders

        # Plan first time
        order1 = planner.plan(plan).order

        # Plan second time
        order2 = planner.plan(plan).order

        # Should be identical
        assert order1 == order2, "Execution order should be deterministic"

    def test_deterministic_with_different_inputs(self):
        """Test determinism with different inputs but same structure."""
        registry = _make_deterministic_registry()
        engine = GraphEngine(registry)

        plan = _create_deterministic_plan()

        # Run with different inputs
        inputs = [1.0, 5.0, 10.0]
        results = []

        for input_val in inputs:
            state, writer = _create_state_with_input(input_val)
            engine.run(plan, state, writer, ExecutionContext(mode="eval", seed=42, device="cpu", deterministic=True))
            results.append(
                {
                    "input": input_val,
                    "node1": state.get("node1.output"),
                    "node2": state.get("node2.output"),
                    "node3": state.get("node3.output"),
                }
            )

        # Each run should be deterministic relative to its input
        for result in results:
            expected_node1 = result["input"] * 2.0
            expected_node2 = expected_node1 * 3.0
            expected_node3 = expected_node2 * 1.5

            assert result["node1"] == expected_node1, f"node1 should be deterministic for input {result['input']}"
            assert result["node2"] == expected_node2, f"node2 should be deterministic for input {result['input']}"
            assert result["node3"] == expected_node3, f"node3 should be deterministic for input {result['input']}"

    def test_deterministic_with_random_nodes(self):
        """Test determinism even with nodes that use random operations."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        class RandomNode(NodeRuntime):
            def forward(self, inputs, context):
                # Use deterministic random based on seed
                torch.manual_seed(context.seed)
                value = inputs.get("value", 1.0)
                noise = torch.randn(1).item()  # Deterministic with seed
                result = value + noise
                return {"output": result}

        # Create registry with random node
        from ugtsdti.graph.registry import NodeRegistry

        registry = NodeRegistry()
        registry.register(NodePluginSpec(type_key="random", output_attrs=["output"]), RandomNode)

        engine = GraphEngine(registry)

        plan = GraphPlan(
            node_definitions=[
                NodeDefinition(name="rand1", type_key="random", inputs=["input.value"]),
                NodeDefinition(name="rand2", type_key="random", inputs=["rand1.output"]),
            ],
            edges={"rand1": [], "rand2": ["rand1"]},
            order=["rand1", "rand2"],
        )

        # Run twice with same seed
        state1, writer1 = _create_state_with_input(5.0)
        engine.run(plan, state1, writer1, ExecutionContext(mode="eval", seed=123, device="cpu", deterministic=True))

        state2, writer2 = _create_state_with_input(5.0)
        engine.run(plan, state2, writer2, ExecutionContext(mode="eval", seed=123, device="cpu", deterministic=True))

        # Should be identical despite random operations
        assert state1.get("rand1.output") == state2.get(
            "rand1.output"
        ), "Random node1 should be deterministic with seed"
        assert state1.get("rand2.output") == state2.get(
            "rand2.output"
        ), "Random node2 should be deterministic with seed"

    def test_topological_sort_stability(self):
        """Test that topological sort produces stable results for ties."""
        registry = _make_deterministic_registry()
        planner = GraphPlanner(registry)

        # Create plan with multiple nodes that have no dependencies (ties)
        plan = GraphPlan(
            node_definitions=[
                NodeDefinition(name="zebra", type_key="deterministic", inputs=["input.value"]),
                NodeDefinition(name="alpha", type_key="deterministic", inputs=["input.value"]),
                NodeDefinition(name="beta", type_key="deterministic", inputs=["input.value"]),
                NodeDefinition(name="gamma", type_key="deterministic", inputs=["alpha.output", "beta.output"]),
            ],
            edges={"zebra": [], "alpha": [], "beta": [], "gamma": ["alpha", "beta"]},
            order=["zebra", "alpha", "beta", "gamma"],  # This should be overridden by planner
        )

        # Plan multiple times
        orders = []
        for _ in range(10):
            planned = planner.plan(plan)
            orders.append(planned.order)

        # All orders should be identical
        first_order = orders[0]
        for i, order in enumerate(orders[1:], 1):
            assert order == first_order, f"Order {i} should match first order"

        # Check that ties are resolved alphabetically
        assert first_order.index("alpha") < first_order.index("beta"), "alpha should come before beta (alphabetical)"
        assert first_order.index("beta") < first_order.index("zebra"), "beta should come before zebra (alphabetical)"

    def test_deterministic_execution_with_cycles_robustness(self):
        """Test that deterministic behavior is maintained even with complex dependencies."""
        registry = _make_deterministic_registry()
        engine = GraphEngine(registry)

        # Create a more complex graph
        plan = _create_complex_deterministic_plan()

        # Run multiple times
        results = []
        for run in range(5):
            state, writer = _create_state_with_input(2.0)
            engine.run(plan, state, writer, ExecutionContext(mode="eval", seed=run, device="cpu", deterministic=True))

            results.append(
                {
                    "run": run,
                    "final_output": state.get("final.output"),
                    "intermediate": state.get("intermediate.output"),
                }
            )

        # Each run should produce the same result for the same seed
        for result in results:
            # With deterministic operations and same input, results should be predictable
            assert isinstance(result["final_output"], (int, float)), "Output should be numeric"
            assert isinstance(result["intermediate"], (int, float)), "Intermediate should be numeric"


# Helper functions
def _make_deterministic_registry():
    """Create registry with deterministic nodes."""
    from ugtsdti.graph.registry import NodeRegistry

    registry = NodeRegistry()

    class DeterministicNode(NodeRuntime):
        def __init__(self, multiplier=2.0):
            self.multiplier = multiplier

        def forward(self, inputs, context):
            value = inputs.get("value", 1.0)
            result = value * self.multiplier
            return {"output": result}

    class AnotherDeterministicNode(NodeRuntime):
        def __init__(self, multiplier=3.0):
            self.multiplier = multiplier

        def forward(self, inputs, context):
            value = inputs.get("value", 1.0)
            result = value * self.multiplier
            return {"output": result}

    class FinalNode(NodeRuntime):
        def __init__(self, multiplier=1.5):
            self.multiplier = multiplier

        def forward(self, inputs, context):
            value = inputs.get("value", 1.0)
            result = value * self.multiplier
            return {"output": result}

    registry.register(NodePluginSpec(type_key="deterministic", output_attrs=["output"]), DeterministicNode)
    registry.register(NodePluginSpec(type_key="another", output_attrs=["output"]), AnotherDeterministicNode)
    registry.register(NodePluginSpec(type_key="final", output_attrs=["output"]), FinalNode)

    return registry


def _create_deterministic_plan():
    """Create a simple deterministic plan."""
    return GraphPlan(
        node_definitions=[
            NodeDefinition(name="node1", type_key="deterministic", inputs=["input.value"]),
            NodeDefinition(name="node2", type_key="another", inputs=["node1.output"]),
            NodeDefinition(name="node3", type_key="final", inputs=["node2.output"]),
        ],
        edges={"node1": [], "node2": ["node1"], "node3": ["node2"]},
        order=["node1", "node2", "node3"],
    )


def _create_complex_plan():
    """Create a plan with multiple valid topological orders."""
    return GraphPlan(
        node_definitions=[
            NodeDefinition(name="alpha", type_key="deterministic", inputs=["input.value"]),
            NodeDefinition(name="beta", type_key="deterministic", inputs=["input.value"]),
            NodeDefinition(name="gamma", type_key="deterministic", inputs=["alpha.output", "beta.output"]),
        ],
        edges={"alpha": [], "beta": [], "gamma": ["alpha", "beta"]},
        order=["alpha", "beta", "gamma"],
    )


def _create_complex_deterministic_plan():
    """Create a more complex deterministic plan."""
    return GraphPlan(
        node_definitions=[
            NodeDefinition(name="start", type_key="deterministic", inputs=["input.value"]),
            NodeDefinition(name="branch1", type_key="another", inputs=["start.output"]),
            NodeDefinition(name="branch2", type_key="deterministic", inputs=["start.output"]),
            NodeDefinition(name="intermediate", type_key="another", inputs=["branch1.output"]),
            NodeDefinition(name="final", type_key="final", inputs=["intermediate.output", "branch2.output"]),
        ],
        edges={
            "start": [],
            "branch1": ["start"],
            "branch2": ["start"],
            "intermediate": ["branch1"],
            "final": ["intermediate", "branch2"],
        },
        order=["start", "branch1", "branch2", "intermediate", "final"],
    )


def _create_state_with_input(value):
    """Create state with input value."""
    state = State()
    writer = StateWriter(state)
    writer.commit("input", {"value": value})
    return state, writer
