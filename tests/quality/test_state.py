"""Unit tests for State and StateWriter invariants.

Covers:
- write-once: committing the same key twice raises KeyCollisionError
- single-producer: same key from different producers raises KeyCollisionError
- monotonic growth: keys only added, never removed
- snapshot() returns a copy that cannot mutate internal state

REQ-STATE-001, REQ-STATE-002
"""
import pytest

from ugtsdti.core.errors import KeyCollisionError
from ugtsdti.core.state import State, StateWriter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_state_and_writer() -> tuple[State, StateWriter]:
    state = State()
    writer = StateWriter(state)
    return state, writer


# ---------------------------------------------------------------------------
# Write-once invariant
# ---------------------------------------------------------------------------


class TestWriteOnce:
    def test_commit_same_key_twice_same_producer_raises(self):
        state, writer = make_state_and_writer()
        writer.commit("node_a", {"embedding": [1, 2, 3]})
        with pytest.raises(KeyCollisionError) as exc_info:
            writer.commit("node_a", {"embedding": [4, 5, 6]})
        assert "embedding" in str(exc_info.value)

    def test_commit_different_keys_succeeds(self):
        state, writer = make_state_and_writer()
        writer.commit("node_a", {"key1": 1})
        writer.commit("node_a", {"key2": 2})
        assert state.has("key1")
        assert state.has("key2")

    def test_committed_value_is_retrievable(self):
        state, writer = make_state_and_writer()
        writer.commit("node_a", {"x": 42})
        assert state.get("x") == 42

    def test_missing_key_raises_key_error(self):
        state, _ = make_state_and_writer()
        with pytest.raises(KeyError):
            state.get("nonexistent")

    def test_failed_multi_key_commit_is_atomic(self):
        state, writer = make_state_and_writer()
        writer.commit("producer_a", {"existing": 1})
        with pytest.raises(KeyCollisionError):
            writer.commit("producer_b", {"new_key": 2, "existing": 3})
        assert state.keys() == ["existing"]
        assert not state.has("new_key")


# ---------------------------------------------------------------------------
# Single-producer invariant
# ---------------------------------------------------------------------------


class TestSingleProducer:
    def test_same_key_different_producers_raises(self):
        state, writer = make_state_and_writer()
        writer.commit("producer_a", {"shared_key": "value_a"})
        with pytest.raises(KeyCollisionError) as exc_info:
            writer.commit("producer_b", {"shared_key": "value_b"})
        err = exc_info.value
        assert err.key == "shared_key"
        assert "producer_a" in str(err) or "producer_b" in str(err)

    def test_error_contains_stage_and_component(self):
        state, writer = make_state_and_writer()
        writer.commit("node_x", {"out": 1})
        with pytest.raises(KeyCollisionError) as exc_info:
            writer.commit("node_y", {"out": 2})
        err = exc_info.value
        assert err.stage == "state"
        assert err.component == "node_y"
        assert err.key == "out"


# ---------------------------------------------------------------------------
# Monotonic growth invariant
# ---------------------------------------------------------------------------


class TestMonotonicGrowth:
    def test_keys_only_grow(self):
        state, writer = make_state_and_writer()
        assert state.keys() == []

        writer.commit("a", {"k1": 1})
        keys_after_first = state.keys()
        assert "k1" in keys_after_first

        writer.commit("b", {"k2": 2})
        keys_after_second = state.keys()
        # All previous keys still present
        assert set(keys_after_first).issubset(set(keys_after_second))
        assert "k2" in keys_after_second

    def test_has_returns_false_before_commit(self):
        state, _ = make_state_and_writer()
        assert not state.has("anything")

    def test_has_returns_true_after_commit(self):
        state, writer = make_state_and_writer()
        writer.commit("p", {"flag": True})
        assert state.has("flag")

    def test_keys_sorted(self):
        state, writer = make_state_and_writer()
        writer.commit("p", {"z_key": 1, "a_key": 2, "m_key": 3})
        assert state.keys() == sorted(state.keys())


# ---------------------------------------------------------------------------
# snapshot() isolation invariant
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_snapshot_returns_copy(self):
        state, writer = make_state_and_writer()
        writer.commit("p", {"val": [1, 2, 3]})
        snap = state.snapshot()
        assert snap == {"val": [1, 2, 3]}

    def test_mutating_snapshot_does_not_affect_state(self):
        state, writer = make_state_and_writer()
        writer.commit("p", {"val": 10})
        snap = state.snapshot()
        snap["val"] = 999
        snap["new_key"] = "injected"
        # Internal state must be unchanged
        assert state.get("val") == 10
        assert not state.has("new_key")

    def test_snapshot_reflects_current_state(self):
        state, writer = make_state_and_writer()
        writer.commit("p", {"a": 1})
        snap1 = state.snapshot()
        writer.commit("q", {"b": 2})
        snap2 = state.snapshot()
        assert "b" not in snap1
        assert "b" in snap2

    def test_empty_state_snapshot_is_empty_dict(self):
        state, _ = make_state_and_writer()
        assert state.snapshot() == {}
