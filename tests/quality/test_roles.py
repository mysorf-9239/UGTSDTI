"""Unit tests for RoleBinder validation and aggregation.

REQ-ROLE-001, REQ-ARCH-004
"""
from __future__ import annotations

import pytest

from ugtsdti.core.errors import InvalidRoleBindingError
from ugtsdti.core.state import State, StateWriter
from ugtsdti.roles.binder import RoleBinder, RoleBinding


def _make_state_and_writer(**initial_keys):
    state = State()
    writer = StateWriter(state)
    if initial_keys:
        writer.commit("graph", initial_keys)
    return state, writer


class TestRoleBinder:
    def test_first_aggregation_commits_canonical_role_logits(self):
        state, writer = _make_state_and_writer(**{"student_head.logits": 1.23})
        binder = RoleBinder([RoleBinding(role="student", outputs=["student_head.logits"], aggregation="first")])
        binder.bind(state, writer)
        assert state.get("student.logits") == 1.23

    def test_missing_graph_output_raises(self):
        state, writer = _make_state_and_writer()
        binder = RoleBinder([RoleBinding(role="student", outputs=["student_head.logits"])])
        with pytest.raises(InvalidRoleBindingError, match="missing graph output"):
            binder.bind(state, writer)

    def test_invalid_role_naming_raises(self):
        state, writer = _make_state_and_writer(**{"student_head.logits": 1.0})
        binder = RoleBinder([RoleBinding(role="student.bad", outputs=["student_head.logits"])])
        with pytest.raises(InvalidRoleBindingError, match="Invalid role binding target"):
            binder.bind(state, writer)

    def test_mean_aggregation_requires_shape_compatibility(self):
        torch = pytest.importorskip("torch")
        state, writer = _make_state_and_writer(
            **{
                "head_a.logits": torch.ones(2, 1),
                "head_b.logits": torch.ones(3, 1),
            }
        )
        binder = RoleBinder(
            [RoleBinding(role="student", outputs=["head_a.logits", "head_b.logits"], aggregation="mean")]
        )
        with pytest.raises(InvalidRoleBindingError, match="shape-compatible"):
            binder.bind(state, writer)

    def test_mean_aggregation_averages_tensors(self):
        torch = pytest.importorskip("torch")
        state, writer = _make_state_and_writer(
            **{
                "head_a.logits": torch.tensor([[1.0], [3.0]]),
                "head_b.logits": torch.tensor([[3.0], [5.0]]),
            }
        )
        binder = RoleBinder(
            [RoleBinding(role="student", outputs=["head_a.logits", "head_b.logits"], aggregation="mean")]
        )
        binder.bind(state, writer)
        expected = torch.tensor([[2.0], [4.0]])
        assert torch.equal(state.get("student.logits"), expected)
