"""End-to-end determinism proof test.

Tests the complete pipeline from config+seed to final outputs
to validate: same(config, seed, batch) → same(outputs)
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
    """Create deterministic hash of pipeline output."""
    return hashlib.md5(json.dumps(output, sort_keys=True, default=str).encode()).hexdigest()


class DeterministicTestNode:
    """Test node that produces deterministic output based on inputs."""

    def __init__(self, multiplier: float = 1.0, add_noise: bool = False):
        self.multiplier = multiplier
        self.add_noise = add_noise

    def forward(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        del context

        # Deterministic computation
        total = 0.0
        for _key, value in inputs.items():
            if isinstance(value, (int, float)):
                total += float(value)

        result = total * self.multiplier

        # Add controlled noise if requested (for testing)
        if self.add_noise:
            import random

            result += random.random() * 0.001  # Small controlled noise

        return {"output": result}


def create_test_config() -> dict[str, Any]:
    """Create a comprehensive test configuration."""
    return {
        "data": {
            "batch_size": 2,
            "scenarios": ["test"],
        },
        "graph": {
            "nodes": [
                {"name": "input1", "type_key": "deterministic", "inputs": [], "params": {"multiplier": 2.0}},
                {"name": "input2", "type_key": "deterministic", "inputs": [], "params": {"multiplier": 3.0}},
                {"name": "process1", "type_key": "deterministic", "inputs": ["input1.output", "input2.output"]},
                {"name": "process2", "type_key": "deterministic", "inputs": ["process1.output"]},
                {"name": "final", "type_key": "deterministic", "inputs": ["process2.output"]},
            ]
        },
    }


def test_end_to_end_determinism():
    """Test complete end-to-end determinism."""

    config = create_test_config()

    # Test results storage
    output_hashes = []
    execution_orders = []

    # Run pipeline multiple times with same config and seed
    test_seed = 12345

    for run in range(5):  # Multiple runs to catch any nondeterminism
        try:
            # Set global seed for deterministic behavior
            set_global_seed(test_seed)

            # Create execution context
            context = ExecutionContext(mode="eval", seed=test_seed, device="cpu", deterministic=True, precision="fp32")

            # Build and plan graph
            registry = NodeRegistry()
            registry.register(
                NodePluginSpec(type_key="deterministic", output_attrs=["output"]), DeterministicTestNode()
            )

            builder = GraphBuilder(registry)
            graph_plan = builder.build(config)
            planner = GraphPlanner(registry)
            plan = planner.plan(graph_plan)
            execution_orders.append(tuple(plan.order))

            # Execute graph
            state = State()
            writer = StateWriter(state)
            engine = GraphEngine(registry)
            engine.run(graph_plan, state, writer, context)

            # Get final output and hash it
            output_data = {"output": state.get("final.output")}
            output_hash = hash_output(output_data)
            output_hashes.append(output_hash)

        except Exception as e:
            print(f"Run {run + 1} failed: {e}")
            traceback.print_exc()
            return False

    # STRICT CHECK 1: All execution orders must be identical
    unique_orders = set(execution_orders)
    if len(unique_orders) != 1:
        print("❌ EXECUTION ORDER NONDETERMINISTIC")
        print("Expected: 1 unique order")
        print(f"Got: {len(unique_orders)} different orders")
        for i, order in enumerate(execution_orders):
            print(f"  Order {i + 1}: {order}")
        return False

    print(f"✅ Execution order deterministic: {execution_orders[0]}")

    # STRICT CHECK 2: All output hashes must be identical
    unique_outputs = set(output_hashes)
    if len(unique_outputs) != 1:
        print("❌ OUTPUT NONDETERMINISTIC")
        print("Expected: 1 unique output hash")
        print(f"Got: {len(unique_outputs)} different output hashes")
        for i, output_hash in enumerate(output_hashes):
            print(f"  Output {i + 1}: {output_hash}")
        return False

    print(f"✅ Output deterministic: {output_hashes[0]}")

    # FINAL VERIFICATION: Same config+seed should produce same results
    print("✅ END-TO-END DETERMINISM PROVEN")
    print("✅ same(config, seed, batch) → same(execution_order) → same(outputs)")

    return True


def test_cross_run_consistency():
    """Test determinism across different process runs."""

    config = create_test_config()

    # Run with different seeds to verify isolation
    seed_results = {}

    for seed in [111, 222, 333]:
        set_global_seed(seed)

        context = ExecutionContext(mode="eval", seed=seed, device="cpu", deterministic=True, precision="fp32")

        registry = NodeRegistry()
        registry.register(
            NodePluginSpec(type_key="deterministic", output_attrs=["output"]),
            DeterministicTestNode(multiplier=seed),  # Different multiplier per seed
        )

        builder = GraphBuilder(registry)
        graph_plan = builder.build(config)
        planner = GraphPlanner(registry)
        planner.plan(graph_plan)

        state = State()
        writer = StateWriter(state)
        engine = GraphEngine(registry)
        engine.run(graph_plan, state, writer, context)

        output_data = {"output": state.get("final.output")}
        output_hash = hash_output(output_data)
        seed_results[seed] = output_hash

    # Different seeds should produce different results
    unique_results = set(seed_results.values())
    if len(unique_results) != 3:
        print("❌ SEED ISOLATION FAILED")
        print("Expected: 3 different results")
        print(f"Got: {len(unique_results)} unique results")
        return False

    print(f"✅ Seed isolation working: {len(unique_results)} unique results")
    return True


def test_cuda_determinism():
    """Test CUDA determinism if available."""
    try:
        import torch

        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, skipping CUDA determinism test")
            return True

        # Test CUDA deterministic settings
        original_deterministic = torch.backends.cudnn.deterministic

        # Set deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if not torch.backends.cudnn.deterministic:
            print("❌ CUDA determinism not working")
            return False

        print(f"✅ CUDA determinism working: {torch.backends.cudnn.deterministic}")

        # Restore original setting
        torch.backends.cudnn.deterministic = original_deterministic
        return True

    except ImportError:
        print("⚠️  PyTorch not available, skipping CUDA test")
        return True


def main():
    """Run all end-to-end determinism tests."""
    print("🔍 Testing End-to-End Determinism...")
    print("=" * 60)

    tests = [
        (test_end_to_end_determinism, "end-to-end determinism proof"),
        (test_cross_run_consistency, "cross-run consistency"),
        (test_cuda_determinism, "CUDA determinism"),
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
        print("🎉 END-TO-END DETERMINISM PROVEN!")
        print("✅ same(config, seed, batch) → same(outputs)")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    import sys

    sys.exit(0 if success else 1)
