"""Tests for State security vulnerabilities and bypass attempts."""

import pytest

from ugtsdti.core.errors import KeyCollisionError
from ugtsdti.core.state import State, StateWriter


class TestStateBypassProtection:
    """Test the actual contract for State._store access.

    State._store is a plain dict attribute accessible to trusted internals.
    It is NOT protected by __setattr__ guards — the contract is that external
    code should use the public API (get/has/keys/snapshot) and only
    StateWriter.commit() should write to State.
    """

    def test_state_store_is_accessible(self):
        """Test that _store is a plain dict accessible to trusted internals."""
        state = State()
        assert hasattr(state, "_store")
        assert isinstance(state._store, dict)

    def test_state_store_can_be_read_directly(self):
        """Test that _store can be read directly (trusted internal access)."""
        state = State()
        writer = StateWriter(state)

        writer.commit("test", {"key": "value"})

        # Direct read is allowed — _store is not protected
        assert "key" in state._store

    def test_state_store_reassignment_is_possible(self):
        """Test that _store can be reassigned (no __setattr__ guard exists)."""
        state = State()
        # _store is a plain attribute — reassignment is possible
        state._store = {}
        assert isinstance(state._store, dict)

    def test_state_store_deletion_is_possible(self):
        """Test that _store can be deleted (no __delattr__ guard exists)."""
        state = State()
        del state._store
        assert not hasattr(state, "_store")

    def test_state_store_access_during_init_allowed(self):
        """Test that _store is set during initialization."""
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
        """Test that get() clones tensors to prevent shared memory, preserving grad graph."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        original = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        state = State()
        writer = StateWriter(state)
        writer.commit("test", {"data.tensor": original})

        # get() clones but does NOT detach — grad graph is preserved
        result = state.get("data.tensor")
        assert result.requires_grad, "get() should preserve requires_grad"

        # Memory isolation: mutation of result must not affect the stored value
        result[0] = 999.0
        stored_again = state.get("data.tensor")
        assert stored_again[0] == 1.0, "Stored value should be unaffected by mutation of returned clone"

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
        """Test that get_isolated() is equivalent to get() and both preserve grad."""
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

        # Both preserve requires_grad (clone, not detach)
        assert get_result.requires_grad
        assert get_isolated_result.requires_grad

        # Should have same values
        assert torch.equal(get_result, get_isolated_result)

    def test_snapshot_always_isolated(self):
        """Test that snapshot() provides complete isolation."""
        original_list = [1, 2, 3]

        state = State()
        writer = StateWriter(state)
        writer.commit("test", {"nested_list": original_list})

        snapshot = state.snapshot()

        # Mutate snapshot deeply
        snapshot["nested_list"].append(999)
        snapshot["nested_list"][0] = 777

        # Original should be unchanged
        assert original_list == [1, 2, 3], "Original should be unaffected"
