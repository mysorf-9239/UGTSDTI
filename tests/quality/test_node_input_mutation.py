"""Tests for node input mutation safety."""

import pytest

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.state import State, StateWriter
from ugtsdti.graph.engine import GraphEngine
from ugtsdti.graph.specs import GraphPlan, NodeDefinition, NodePluginSpec
from ugtsdti.nodes.base import NodeRuntime


class TestNodeInputMutation:
    """Test that node input mutations cannot affect State."""

    def test_node_tensor_input_mutation_isolated(self):
        """Test that in-place tensor mutation in node doesn't affect State."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        original = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        state = State()
        writer = StateWriter(state)
        writer.commit("source", {"source.input_tensor": original})

        # Create a node that mutates its input
        class MutatingNode(NodeRuntime):
            def forward(self, inputs, context):
                tensor = inputs["source.input_tensor"]
                # In-place mutation
                tensor.add_(1.0)
                tensor[0] = 999.0
                return {"output": tensor}

        # Create graph with mutating node
        registry = _make_registry_with_mutating_node()
        engine = GraphEngine(registry)

        plan = _create_plan_with_mutating_node()

        # Run the graph
        engine.run(plan, state, writer, ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False))

        # Original tensor should be unchanged
        assert torch.equal(original, torch.tensor([1.0, 2.0, 3.0])), "Original tensor should be unaffected"
        assert original.requires_grad, "Original should still require grad"

    def test_node_list_input_mutation_isolated(self):
        """Test that list mutation in node doesn't affect State."""
        original_list = [1, 2, 3]

        state = State()
        writer = StateWriter(state)
        writer.commit("source", {"input_list": original_list})

        # Create a node that mutates its input
        class MutatingNode(NodeRuntime):
            def forward(self, inputs, context):
                lst = inputs["input_list"]
                # In-place mutations
                lst.append(999)
                lst[0] = 777
                lst.sort()
                return {"output": lst}

        # Execute node
        node = MutatingNode()
        inputs = {"input_list": state.get_isolated("input_list")}
        node.forward(inputs, ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False))

        # Original list should be unchanged
        assert original_list == [1, 2, 3], "Original list should be unaffected"

    def test_node_dict_input_mutation_isolated(self):
        """Test that dict mutation in node doesn't affect State."""
        original_dict = {"nested": {"value": 42, "list": [1, 2, 3]}}

        state = State()
        writer = StateWriter(state)
        writer.commit("source", {"input_dict": original_dict})

        # Create a node that mutates its input
        class MutatingNode(NodeRuntime):
            def forward(self, inputs, context):
                d = inputs["input_dict"]
                # Deep mutations
                d["nested"]["value"] = 999
                d["nested"]["list"].append(888)
                d["new_key"] = "new_value"
                return {"output": d}

        # Execute node
        node = MutatingNode()
        inputs = {"input_dict": state.get_isolated("input_dict")}
        node.forward(inputs, ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False))

        # Original dict should be unchanged
        assert original_dict["nested"]["value"] == 42, "Original dict should be unaffected"
        assert original_dict["nested"]["list"] == [1, 2, 3], "Original nested list should be unaffected"
        assert "new_key" not in original_dict, "Original should not have new keys"

    def test_node_backward_propagation_isolation(self):
        """Test that get_isolated() provides memory isolation but NOT gradient isolation.

        Since _isolate_value uses clone() (not detach()), the returned tensor still
        participates in the gradient graph. If gradient isolation is needed, node code
        must call .detach() explicitly.
        """
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        original = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        state = State()
        writer = StateWriter(state)
        writer.commit("source", {"input_tensor": original})

        # Create a node that computes gradients
        class GradientNode(NodeRuntime):
            def forward(self, inputs, context):
                tensor = inputs["input_tensor"]
                # Compute something that will have gradients
                output = tensor * 2.0 + 1.0
                return {"output": output}

        # Execute node and compute gradients
        node = GradientNode()
        inputs = {"input_tensor": state.get_isolated("input_tensor")}
        result = node.forward(inputs, ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False))

        # Backward pass — clone() preserves grad graph, so original CAN receive gradients
        loss = result["output"].sum()
        loss.backward()

        # original.grad is populated because get_isolated() clones (not detaches)
        assert original.grad is not None, "Gradient flows to original via clone()"
        assert original.requires_grad, "Original should still require grad"

        # Memory isolation is still guaranteed: in-place mutation of the clone
        # does not affect the stored value
        clone = state.get_isolated("input_tensor")
        clone[0] = 999.0
        fresh = state.get_isolated("input_tensor")
        assert fresh[0] == 1.0, "Stored value should be unaffected by mutation of clone"

    def test_graph_engine_uses_isolated_inputs(self):
        """Test that GraphEngine provides isolated inputs to nodes."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        original = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        state = State()
        writer = StateWriter(state)
        writer.commit("source", {"source.tensor": original})

        # Track if node receives a clone (memory-isolated) tensor
        received_values = []

        class IsolationCheckNode(NodeRuntime):
            def forward(self, inputs, context):
                tensor = inputs["source.tensor"]
                # Clone preserves requires_grad — grad graph is intact
                received_values.append(tensor.requires_grad)
                # Mutate to test memory isolation
                tensor[0] = 999.0
                return {"output": tensor}

        registry = _make_registry_with_node(IsolationCheckNode, "check_node")
        engine = GraphEngine(registry)

        plan = GraphPlan(
            node_definitions=[NodeDefinition(name="check", type_key="check_node", inputs=["source.tensor"])],
            edges={"check": []},
            order=["check"],
            produced_keys={"check": ["check.output"]},
            producers={"check.output": "check"},
        )

        engine.run(plan, state, writer, ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False))

        # Verify node received a tensor with grad preserved (clone, not detach)
        assert received_values[0], "Node should receive tensor with requires_grad=True (clone preserves grad)"

        # Verify original unchanged (memory isolation via clone)
        assert original[0] == 1.0, "Original tensor should be unchanged"


# Helper functions
def _make_registry_with_mutating_node():
    """Create registry with a mutating node for testing."""
    from ugtsdti.graph.registry import NodeRegistry

    registry = NodeRegistry()

    class MutatingNode(NodeRuntime):
        def forward(self, inputs, context):
            tensor = inputs["source.input_tensor"]
            tensor.add_(1.0)
            return {"output": tensor}

    registry.register(NodePluginSpec(type_key="mutating", output_attrs=["output"]), MutatingNode)
    return registry


def _make_registry_with_node(node_class, type_key):
    """Create registry with specified node class."""
    from ugtsdti.graph.registry import NodeRegistry

    registry = NodeRegistry()
    registry.register(NodePluginSpec(type_key=type_key, output_attrs=["output"]), node_class)
    return registry


def _create_plan_with_mutating_node():
    """Create graph plan with mutating node."""
    return GraphPlan(
        node_definitions=[NodeDefinition(name="mutate", type_key="mutating", inputs=["source.input_tensor"])],
        edges={"mutate": []},
        order=["mutate"],
        produced_keys={"mutate": ["mutate.output"]},
        producers={"mutate.output": "mutate"},
    )
