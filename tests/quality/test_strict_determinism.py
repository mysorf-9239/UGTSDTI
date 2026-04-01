"""Strict multi-run determinism proof test.

Implements comprehensive verification of:
same(config, seed, batch) → same(outputs)
"""

from __future__ import annotations

import hashlib
import pickle
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
    """Create deterministic hash of pipeline output using pickle.

    Args:
        output: Dictionary containing pipeline output

    Returns:
        MD5 hash of pickled output
    """
    return hashlib.md5(pickle.dumps(output, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


def hash_state(state: State) -> str:
    """Create hash of state fingerprint for debugging."""
    return hashlib.md5(str(state.get_fingerprint()).encode()).hexdigest()


class DeterministicTestNode:
    """Test node with controlled output and potential nondeterminism detection."""

    def __init__(self, node_id: int = 0, add_noise: bool = False):
        self.node_id = node_id
        self.add_noise = add_noise

    def forward(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        del context

        # Deterministic computation
        total = 0.0
        for _key, value in inputs.items():
            if isinstance(value, (int, float)):
                total += float(value)

        result = total + self.node_id  # Node ID adds deterministic variation

        # Add controlled noise if requested (for testing)
        if self.add_noise:
            import random

            result += random.random() * 0.000001  # Tiny controlled noise

        return {"output": result, "node_id": self.node_id}


def create_test_graph() -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a test graph with multiple processing paths."""
    return {
        "nodes": [
            {"name": "input", "type_key": "deterministic", "inputs": []},
            {"name": "process_a", "type_key": "deterministic", "inputs": ["input.output"]},
            {"name": "process_b", "type_key": "deterministic", "inputs": ["input.output", "process_a.output"]},
            {"name": "process_c", "type_key": "deterministic", "inputs": ["process_a.output", "process_b.output"]},
            {"name": "final", "type_key": "deterministic", "inputs": ["process_c.output"]},
        ]
    }, {
        "data": {
            "batch_size": 1,
            "scenarios": ["test"],
        }
    }


def test_strict_multi_run_determinism():
    """Strict multi-run verification of end-to-end determinism."""

    graph_cfg, data_cfg = create_test_graph()

    # Storage for strict verification
    plans = []
    outputs = []
    execution_orders = []
    state_hashes = []
    node_outputs = {}

    test_seed = 99999
    num_runs = 10

    print(f"🔍 Running {num_runs} strict determinism tests...")
    print(f"📊 Test seed: {test_seed}")

    for run in range(num_runs):
        try:
            # Set global seed
            set_global_seed(test_seed)

            # Create execution context
            context = ExecutionContext(mode="eval", seed=test_seed, device="cpu", deterministic=True, precision="fp32")

            # Build and plan graph
            registry = NodeRegistry()
            registry.register(
                NodePluginSpec(type_key="deterministic", output_attrs=["output", "node_id"]),
                DeterministicTestNode(node_id=run),
            )

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

            # Get final output and hash it
            available_keys = list(state.keys())
            if "final.output" in available_keys:
                output_data = {"output": state.get("final.output")}
            else:
                print(f"❌ Missing 'final.output' key. Available keys: {available_keys}")
                return False

            output_hash = hash_output(output_data)
            outputs.append(output_hash)

            # Store execution order and state hash
            execution_orders.append(tuple(plan.order))
            state_hashes.append(hash_state(state))

            # Collect per-node outputs for debugging
            for node_name in plan.order:
                node_output = state.get(f"{node_name}.output")
                if node_output is not None:
                    if node_name not in node_outputs:
                        node_outputs[node_name] = []
                    node_outputs[node_name].append(node_output)

        except Exception as e:
            print(f"❌ Run {run + 1} failed: {e}")
            traceback.print_exc()
            return False

    print(f"✅ Completed {num_runs} runs")

    # STRICT VERIFICATION 1: Identical execution orders
    unique_plans = set(plans)
    if len(unique_plans) != 1:
        print("❌ EXECUTION ORDER NONDETERMINISTIC")
        print("Expected: 1 unique plan")
        print(f"Got: {len(unique_plans)} different plans")

        # Show differences
        plan_counts = {}
        for plan in plans:
            plan_str = str(plan)
            plan_counts[plan_str] = plan_counts.get(plan_str, 0) + 1

        for plan_str, count in plan_counts.items():
            if count > 1:
                print(f"  Plan (appears {count} times): {plan_str}")
        return False

    print(f"✅ EXECUTION ORDER DETERMINISTIC: {plans[0]}")

    # STRICT VERIFICATION 2: Identical outputs
    unique_outputs = set(outputs)
    if len(unique_outputs) != 1:
        print("❌ OUTPUT NONDETERMINISTIC")
        print("Expected: 1 unique output hash")
        print(f"Got: {len(unique_outputs)} different output hashes")

        # Show output differences
        for i, output_hash in enumerate(outputs):
            print(f"  Output {i + 1}: {output_hash}")
        return False

    print(f"✅ OUTPUT DETERMINISTIC: {outputs[0]}")

    # STRICT VERIFICATION 3: Identical state hashes
    unique_state_hashes = set(state_hashes)
    if len(unique_state_hashes) != 1:
        print("❌ STATE NONDETERMINISTIC")
        print("Expected: 1 unique state hash")
        print(f"Got: {len(unique_state_hashes)} different state hashes")

        # Show state differences
        for i, state_hash in enumerate(state_hashes):
            print(f"  State {i + 1}: {state_hash}")
        return False

    print(f"✅ STATE DETERMINISTIC: {state_hashes[0]}")

    # STRICT VERIFICATION 4: Node-level output consistency
    print("🔍 Analyzing node-level output consistency...")

    all_nodes_consistent = True
    for node_name in node_outputs:
        node_values = node_outputs[node_name]
        if len(set(node_values)) != 1:
            print(f"❌ NODE {node_name} NONDETERMINISTIC")
            print(f"  Expected: 1 unique value, got {len(set(node_values))}")
            all_nodes_consistent = False

    if all_nodes_consistent:
        print("✅ ALL NODES CONSISTENT")
    else:
        print("❌ SOME NODES INCONSISTENT")

    # FINAL VERDICT
    if all_nodes_consistent and len(unique_plans) == 1 and len(unique_outputs) == 1 and len(unique_state_hashes) == 1:
        print("🎉 STRICT DETERMINISM PROVEN!")
        print("✅ same(config, seed, batch) → same(execution_order) → same(outputs)")
        return True
    else:
        print("❌ STRICT DETERMINISM FAILED")
        return False


def test_determinism_breakdown():
    """Test individual components for debugging."""

    graph_cfg, data_cfg = create_test_graph()

    # Test each component separately
    test_seed = 77777

    print("🔍 Testing individual determinism components...")

    # Test 1: Graph planning determinism
    try:
        set_global_seed(test_seed)

        registry = NodeRegistry()
        registry.register(NodePluginSpec(type_key="deterministic", output_attrs=["output"]), DeterministicTestNode())

        builder = GraphBuilder(registry)
        graph_plan = builder.build(graph_cfg)
        planner = GraphPlanner(registry)
        plan1 = planner.plan(graph_plan)
        plan2 = planner.plan(graph_plan)

        if plan1.order != plan2.order:
            print("❌ Graph planning nondeterministic")
            return False
        else:
            print("✅ Graph planning deterministic")
    except Exception as e:
        print(f"❌ Graph planning test failed: {e}")
        return False

    # Test 2: State integrity
    try:
        set_global_seed(test_seed)

        registry = NodeRegistry()
        registry.register(NodePluginSpec(type_key="deterministic", output_attrs=["output"]), DeterministicTestNode())

        builder = GraphBuilder(registry)
        graph_plan = builder.build(graph_cfg)
        planner = GraphPlanner(registry)
        planner.plan(graph_plan)

        state1 = State()
        state2 = State()
        writer1 = StateWriter(state1)
        writer2 = StateWriter(state2)
        engine = GraphEngine(registry)

        # First execution
        engine.run(
            graph_plan,
            state1,
            writer1,
            ExecutionContext(mode="eval", seed=test_seed, device="cpu", deterministic=True, precision="fp32"),
        )

        # Try to mutate state2 (should fail)
        try:
            state2._store["illegal"] = "test"
            engine.run(
                graph_plan,
                state2,
                writer2,
                ExecutionContext(mode="eval", seed=test_seed, device="cpu", deterministic=True, precision="fp32"),
            )
            print("❌ State integrity test failed - mutation not detected")
            return False
        except RuntimeError:
            print("✅ State integrity working - mutation detected")
        else:
            print("❌ State integrity test failed - unexpected error")
            return False

    except Exception as e:
        print(f"❌ State integrity test failed: {e}")
        return False


def main():
    """Run strict determinism verification."""
    print("🔍 STRICT MULTI-RUN DETERMINISM PROOF")
    print("=" * 60)

    tests = [
        (test_strict_multi_run_determinism, "strict multi-run determinism"),
        (test_determinism_breakdown, "determinism component breakdown"),
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
        print("🎉 STRICT DETERMINISM PROVEN!")
        print("✅ same(config, seed, batch) → same(execution_order) → same(outputs)")
        return True
    else:
        print("❌ STRICT DETERMINISM FAILED")
        return False


if __name__ == "__main__":
    success = main()
    import sys

    sys.exit(0 if success else 1)
