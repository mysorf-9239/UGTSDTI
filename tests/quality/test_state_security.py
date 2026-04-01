"""Tests for State security vulnerabilities and bypass attempts."""

import pytest

from ugtsdti.core.errors import KeyCollisionError
from ugtsdti.core.state import State, StateWriter


class TestStateBypassProtection:
    """Test that State._store direct access is properly blocked."""

    def test_state_store_direct_assignment_blocked(self):
        """Test that direct assignment to _store raises AttributeError."""
        state = State()

        # After initialization, _store should be read-only
        with pytest.raises(AttributeError, match="State._store is read-only"):
            state._store = {}

    def test_state_store_mutation_blocked(self):
        """Test that modifying _store content raises AttributeError."""
        state = State()
        writer = StateWriter(state)

        # Normal write should work
        writer.commit("test", {"key": "value"})
        assert state.get("key") == "value"

        # Direct mutation should fail
        with pytest.raises(AttributeError, match="State._store is read-only"):
            state._store["new_key"] = "bypass"

    def test_state_store_deletion_blocked(self):
        """Test that deleting _store raises AttributeError."""
        state = State()

        with pytest.raises(AttributeError, match="State._store cannot be deleted"):
            del state._store

    def test_setattr_state_store_blocked(self):
        """Test that setattr cannot bypass _store protection."""
        state = State()

        with pytest.raises(AttributeError, match="State._store is read-only"):
            state._store = {}

    def test_state_store_access_during_init_allowed(self):
        """Test that _store can be set during initialization."""
        # This should work - _store protection only after __init__
        state = State()
        assert hasattr(state, "_store")
        assert isinstance(state._store, dict)

    def test_statewriter_internal_access_works(self):
        """Test that StateWriter can still access internal storage."""
        state = State()
        writer = StateWriter(state)

        # StateWriter should work normally
        writer.commit("producer", {"key": "value"})
        assert state.get("key") == "value"

        # Second write should fail (write-once enforcement)
        with pytest.raises(KeyCollisionError):
            writer.commit("producer2", {"key": "value2"})


class TestStateReferenceLeaks:
    """Test that all State reads return isolated copies."""

    def test_get_always_isolates_tensors(self):
        """Test that get() always isolates tensors, even non-sensitive keys."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        original = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        state = State()
        writer = StateWriter(state)
        writer.commit("test", {"data.tensor": original})

        # get() should isolate
        result = state.get("data.tensor")
        assert not result.requires_grad, "get() should break grad graph"
        assert result.grad_fn is None, "get() should detach from computation graph"

        # Mutation should not affect original
        result[0] = 999.0
        assert original[0] == 1.0, "Original should be unaffected"

    def test_get_always_isolates_mutable_objects(self):
        """Test that get() always isolates mutable objects."""
        original_list = [1, 2, 3]
        original_dict = {"nested": {"value": 42}}

        state = State()
        writer = StateWriter(state)
        writer.commit("test", {"list": original_list, "dict": original_dict})

        # Test list isolation
        result_list = state.get("list")
        result_list.append(999)
        assert original_list == [1, 2, 3], "Original list should be unaffected"

        # Test dict isolation
        result_dict = state.get("dict")
        result_dict["nested"]["value"] = 999
        assert original_dict["nested"]["value"] == 42, "Original dict should be unaffected"

    def test_get_isolated_compatibility(self):
        """Test that get_isolated() still works and returns same as get()."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        original = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        state = State()
        writer = StateWriter(state)
        writer.commit("test", {"tensor": original})

        get_result = state.get("tensor")
        get_isolated_result = state.get_isolated("tensor")

        # Both should be isolated
        assert not get_result.requires_grad
        assert not get_isolated_result.requires_grad

        # Should have same values
        assert torch.equal(get_result, get_isolated_result)

    def test_snapshot_always_isolated(self):
        """Test that snapshot() provides complete isolation."""
        original = {"nested": {"list": [1, 2, 3]}}

        state = State()
        writer = StateWriter(state)
        writer.commit("test", original)

        snapshot = state.snapshot()

        # Mutate snapshot deeply
        snapshot["test"]["nested"]["list"].append(999)
        snapshot["test"]["nested"]["list"][0] = 777

        # Original should be unchanged
        assert original["nested"]["list"] == [1, 2, 3], "Original should be unaffected"
