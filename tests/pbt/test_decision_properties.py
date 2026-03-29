"""Property-based tests for decision modules."""
from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.state import State, StateWriter
from ugtsdti.decision import SoftBlendingDecisionModule
from ugtsdti.decision.trust import TrustEstimator


def _make_state(**entries):
    state = State()
    StateWriter(state).commit("test", entries)
    return state


@given(
    teacher_logits=st.lists(
        st.floats(min_value=-20, max_value=20, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=6,
    ),
    student_logits=st.lists(
        st.floats(min_value=-20, max_value=20, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=6,
    ),
    teacher_var=st.lists(
        st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=6,
    ),
    student_var=st.lists(
        st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=6,
    ),
)
def test_soft_blending_alpha_property(teacher_logits, student_logits, teacher_var, student_var):
    torch = pytest.importorskip("torch")
    batch_size = min(len(teacher_logits), len(student_logits), len(teacher_var), len(student_var))
    state = _make_state(
        **{
            "teacher.logits": torch.tensor([[value] for value in teacher_logits[:batch_size]], dtype=torch.float32),
            "student.logits": torch.tensor([[value] for value in student_logits[:batch_size]], dtype=torch.float32),
            "teacher.var": torch.tensor([[value] for value in teacher_var[:batch_size]], dtype=torch.float32),
            "student.var": torch.tensor([[value] for value in student_var[:batch_size]], dtype=torch.float32),
        }
    )
    outputs = SoftBlendingDecisionModule(
        trust_estimator=TrustEstimator(use_uncertainty=True),
        fallback={"no_teacher": "student", "no_student": "teacher"},
    ).forward(
        state,
        ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False),
    )

    assert torch.all(torch.isfinite(outputs["gate.alpha"]))
    assert torch.all(outputs["gate.alpha"] >= 0)
    assert torch.all(outputs["gate.alpha"] <= 1)
