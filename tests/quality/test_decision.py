"""Unit tests for advanced decision modules."""

from __future__ import annotations

import pytest

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.errors import InvalidAlphaRangeError, InvalidDecisionOutputError
from ugtsdti.core.state import State, StateWriter
from ugtsdti.decision import HardSelectionDecisionModule, SoftBlendingDecisionModule
from ugtsdti.decision.policy import DecisionPolicy
from ugtsdti.decision.trust import TrustEstimator


def _make_state(**entries):
    state = State()
    StateWriter(state).commit("test", entries)
    return state


def test_soft_blending_module_falls_back_to_student_when_teacher_absent():
    torch = pytest.importorskip("torch")
    state = _make_state(**{"student.logits": torch.tensor([[0.2], [0.6]])})

    outputs = SoftBlendingDecisionModule(fallback={"no_teacher": "student"}).forward(
        state,
        ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False),
    )

    assert torch.allclose(outputs["logits"], state.get("student.logits"))
    assert "gate.alpha" not in outputs


def test_soft_blending_module_uses_configured_no_uncertainty_fallback():
    torch = pytest.importorskip("torch")
    state = _make_state(
        **{
            "teacher.logits": torch.tensor([[0.8], [0.7]]),
            "student.logits": torch.tensor([[0.2], [0.3]]),
        }
    )
    module = SoftBlendingDecisionModule(
        trust_estimator=TrustEstimator(use_uncertainty=True),
        fallback={"no_uncertainty": "student"},
    )

    outputs = module.forward(
        state,
        ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False),
    )

    assert torch.allclose(outputs["logits"], state.get("student.logits"))


def test_decision_policy_rejects_invalid_alpha_range():
    torch = pytest.importorskip("torch")
    policy = DecisionPolicy()

    with pytest.raises(InvalidAlphaRangeError):
        policy.soft_blend(
            torch.tensor([[1.0]]),
            torch.tensor([[0.0]]),
            torch.tensor([[1.2]]),
        )


def test_soft_blending_module_rejects_non_finite_gate_outputs():
    torch = pytest.importorskip("torch")

    class BadTrustEstimator:
        def estimate(self, state):
            del state
            return {"gate.alpha": torch.tensor([[float("nan")]])}

    state = _make_state(
        **{
            "teacher.logits": torch.tensor([[1.0]]),
            "student.logits": torch.tensor([[0.0]]),
        }
    )
    module = SoftBlendingDecisionModule(trust_estimator=BadTrustEstimator())

    with pytest.raises(InvalidAlphaRangeError):
        module.forward(
            state,
            ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False),
        )


def test_hard_selection_module_emits_gate_outputs_and_logits():
    torch = pytest.importorskip("torch")
    state = _make_state(
        **{
            "teacher.logits": torch.tensor([[3.0], [-1.0]]),
            "student.logits": torch.tensor([[0.0], [0.5]]),
            "teacher.var": torch.tensor([[0.1], [0.2]]),
            "student.var": torch.tensor([[0.5], [0.4]]),
        }
    )
    module = HardSelectionDecisionModule(
        trust_estimator=TrustEstimator(use_uncertainty=True),
        fallback={"no_teacher": "student", "no_student": "teacher"},
    )

    outputs = module.forward(
        state,
        ExecutionContext(mode="infer", seed=0, device="cpu", deterministic=False),
    )

    assert "gate.alpha" in outputs
    assert outputs["gate.uncertainty_source"] == "none"
    assert outputs["logits"].shape == state.get("teacher.logits").shape


def test_hard_selection_module_without_any_branch_raises():
    state = _make_state()
    module = HardSelectionDecisionModule()

    with pytest.raises(InvalidDecisionOutputError):
        module.forward(
            state,
            ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False),
        )
