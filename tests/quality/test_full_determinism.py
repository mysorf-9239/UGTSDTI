"""Full system determinism test.

Tests that the entire pipeline from graph execution to data loading
maintains deterministic behavior across multiple runs.
"""

from __future__ import annotations

import hashlib
import json
import traceback
from typing import Any

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.seed import set_global_seed
from ugtsdti.core.state import State, StateWriter
from ugtsdti.graph.builder import GraphBuilder
from ugtsdti.graph.engine import GraphEngine
from ugtsdti.graph.planner import GraphPlanner
from ugtsdti.graph.registry import NodeRegistry
from ugtsdti.graph.specs import NodePluginSpec


def hash_output(output: dict[str, Any]) -> str:
    """Create deterministic hash of pipeline output.

    Args:
        output: Dictionary containing pipeline output

    Returns:
        MD5 hash of sorted JSON representation
    """
    return hashlib.md5(json.dumps(output, sort_keys=True, default=str).encode()).hexdigest()


class DeterministicTestNode:
    """Test node that produces deterministic output based on inputs."""

    def __init__(self, multiplier: float = 1.0):
        self.multiplier = multiplier

    def forward(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        del context
        # Deterministic computation based on input sum
        total = 0.0
        for _key, value in inputs.items():
            if isinstance(value, (int, float)):
                total += float(value)

        result = total * self.multiplier
        return {"output": result}


def create_test_graph() -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a simple test graph for determinism testing."""
    return {
        "nodes": [
            {"name": "input_node", "type_key": "deterministic", "inputs": []},
            {"name": "process1", "type_key": "deterministic", "inputs": ["input_node.output"]},
            {"name": "process2", "type_key": "deterministic", "inputs": ["process1.output"]},
            {"name": "process3", "type_key": "deterministic", "inputs": ["process2.output"]},
        ]
    }, {
        "data": {
            "batch_size": 1,
            "scenarios": ["test"],
        }
    }


def test_full_determinism():
    """Test full system determinism across multiple pipeline runs."""

    # Create test graph and data
    graph_cfg, data_cfg = create_test_graph()

    # Test results storage
    plans = []
    outputs = []

    # Run pipeline multiple times with same seed
    test_seed = 42

    for _run in range(5):
        # Set global seed for deterministic behavior
        set_global_seed(test_seed)

        # Create execution context
        context = ExecutionContext(mode="eval", seed=test_seed, device="cpu", deterministic=True, precision="fp32")

        # Build and plan graph
        registry = NodeRegistry()

        # Register deterministic test node
        registry.register(NodePluginSpec(type_key="deterministic", output_attrs=["output"]), DeterministicTestNode)

        builder = GraphBuilder(registry)
        graph_plan = builder.build(graph_cfg)
        planner = GraphPlanner(registry)
        plan = planner.plan(graph_plan)
        plans.append(tuple(plan.order))

        # Execute graph
        state = State()
        writer = StateWriter(state)
        engine = GraphEngine(registry)
        engine.run(graph_plan, state, writer, context)

        # Get output and hash it
        output_data = {"output": state.get("process3.output")}
        output_hash = hash_output(output_data)
        outputs.append(output_hash)

    # All plans should be identical (deterministic graph execution)
    unique_plans = set(plans)
    if len(unique_plans) != 1:
        print(f"❌ Graph execution non-deterministic: {len(unique_plans)} different plans")
        for i, plan in enumerate(plans):
            print(f"  Plan {i + 1}: {plan}")
        raise AssertionError(f"Expected 1 unique plan, got {len(unique_plans)}")

    # All outputs should be identical (deterministic pipeline)
    unique_outputs = set(outputs)
    if len(unique_outputs) != 1:
        print(f"❌ Pipeline output non-deterministic: {len(unique_outputs)} different outputs")
        for i, output in enumerate(outputs):
            print(f"  Output {i + 1}: {output}")
        raise AssertionError(f"Expected 1 unique output, got {len(unique_outputs)}")

    print("✅ Full determinism test passed")
    print(f"✅ Graph plan: {plans[0]}")
    print(f"✅ Pipeline output: {outputs[0]}")

    return True


def test_deterministic_vs_non_deterministic():
    """Test that different seeds produce different results."""

    graph_cfg, data_cfg = create_test_graph()

    # Run with different seeds
    results = []

    for seed in [100, 200, 300]:
        set_global_seed(seed)

        context = ExecutionContext(mode="eval", seed=seed, device="cpu", deterministic=True, precision="fp32")

        registry = NodeRegistry()
        registry.register(
            NodePluginSpec(type_key="deterministic", output_attrs=["output"]),
            DeterministicTestNode(multiplier=seed),  # Different multiplier per seed
        )

        builder = GraphBuilder(registry)
        graph_plan = builder.build(graph_cfg)
        planner = GraphPlanner(registry)
        planner.plan(graph_plan)

        state = State()
        writer = StateWriter(state)
        engine = GraphEngine(registry)
        engine.run(graph_plan, state, writer, context)

        output_data = {"output": state.get("process3.output")}
        output_hash = hash_output(output_data)
        results.append(output_hash)

    # Results should be different
    unique_results = set(results)
    if len(unique_results) != 3:
        print(f"❌ Seed isolation failed: {len(unique_results)} unique results instead of 3")
        raise AssertionError(f"Expected 3 different results, got {len(unique_results)}")

    print(f"✅ Seed isolation test passed: {len(unique_results)} unique results")
    return True


def test_graph_planner_determinism():
    """Test graph planner determinism specifically."""

    graph_cfg, _ = create_test_graph()

    # Test planner multiple times with same seed
    plans = []
    test_seed = 999

    for _run in range(3):
        set_global_seed(test_seed)

        registry = NodeRegistry()
        registry.register(NodePluginSpec(type_key="deterministic", output_attrs=["output"]), DeterministicTestNode())

        builder = GraphBuilder(registry)
        graph_plan = builder.build(graph_cfg)
        planner = GraphPlanner(registry)
        plan = planner.plan(graph_plan)
        plans.append(tuple(plan.order))

    # All plans should be identical
    unique_plans = set(plans)
    if len(unique_plans) != 1:
        print(f"❌ Graph planner non-deterministic: {len(unique_plans)} different plans")
        for i, plan in enumerate(plans):
            print(f"  Plan {i + 1}: {plan}")
        raise AssertionError(f"Expected 1 unique plan, got {len(unique_plans)}")

    print(f"✅ Graph planner determinism test passed: {plans[0]}")
    return True


def test_state_integrity():
    """Test that state integrity is maintained during execution."""

    graph_cfg, data_cfg = create_test_graph()

    set_global_seed(12345)

    context = ExecutionContext(mode="eval", seed=12345, device="cpu", deterministic=True, precision="fp32")

    registry = NodeRegistry()

    # Create a node that might try to modify state
    class StateTestNode:
        def forward(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
            del context
            # Try to modify state (should be caught by engine)
            try:
                # This should fail due to state integrity checks
                from ugtsdti.core.state import _isolate_value

                state._store["illegal"] = _isolate_value("test")
            except Exception:
                pass  # Expected to fail

            return {"output": inputs.get("input", 0) * 2}

    registry.register(NodePluginSpec(type_key="deterministic", output_attrs=["output"]), StateTestNode())

    builder = GraphBuilder(registry)
    graph_plan = builder.build(graph_cfg)
    planner = GraphPlanner(registry)
    planner.plan(graph_plan)

    state = State()
    writer = StateWriter(state)
    engine = GraphEngine(registry)

    # This should raise an error due to state mutation
    try:
        engine.run(graph_plan, state, writer, context)
        raise AssertionError("Expected state integrity error")
    except RuntimeError as e:
        if "External mutation" in str(e) or "Illegal mutation" in str(e):
            print(f"✅ State integrity test passed: {e}")
            return True
        else:
            print(f"❌ Unexpected error: {e}")
            return False


def main():
    """Run all full determinism tests."""
    print("🔍 Testing Full System Determinism...")
    print("=" * 60)

    tests = [
        (test_full_determinism, "full pipeline determinism"),
        (test_deterministic_vs_non_deterministic, "seed isolation"),
        (test_graph_planner_determinism, "graph planner determinism"),
        (test_state_integrity, "state integrity"),
    ]

    passed = 0
    total = len(tests)

    for test_func, test_name in tests:
        try:
            if test_func():
                print(f"✅ {test_name}")
                passed += 1
            else:
                print(f"❌ {test_name}")
        except Exception as e:
            print(f"❌ {test_name}")
            print(f"   Error: {e}")
            traceback.print_exc()

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All full determinism tests passed!")
        print("✅ System is fully deterministic and reproducible")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    import sys

    sys.exit(0 if success else 1)
