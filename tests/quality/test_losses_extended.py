"""Loss plugin tests: BCELoss, KDLoss.

Covers:
- Output contract: scalar, finite, gradient flows
- BCELoss: only uses "logits", ignores other keys
- KDLoss: alpha range, fallback to BCE, None-value keys (HybridDTIModel contract),
  alpha=0 equals BCE, alpha=1 equals MSE
"""

import pytest
import torch
import torch.nn as nn

from ugtsdti.losses.bce import BCELoss
from ugtsdti.losses.kd import KDLoss

# ---------------------------------------------------------------------------
# BCELoss
# ---------------------------------------------------------------------------


def test_bce_scalar_output():
    loss_fn = BCELoss()
    out = loss_fn({"logits": torch.tensor([1.0, -1.0, 0.5])}, torch.tensor([1.0, 0.0, 1.0]))
    assert out.dim() == 0
    assert not torch.isnan(out)


def test_bce_gradient_flows():
    loss_fn = BCELoss()
    logits = torch.tensor([1.0, -1.0], requires_grad=True)
    loss_fn({"logits": logits}, torch.tensor([1.0, 0.0])).backward()
    assert logits.grad is not None


def test_bce_ignores_extra_keys():
    """BCELoss must only use 'logits'; all other keys in the standard output dict are ignored."""
    loss_fn = BCELoss()
    model_outputs = {
        "logits": torch.tensor([0.5, -0.5]),
        "student_logits": torch.randn(2),
        "teacher_logits": torch.randn(2),
        "gate_alpha": torch.randn(2),
        "student_var": None,
        "teacher_var": None,
    }
    out = loss_fn(model_outputs, torch.tensor([1.0, 0.0]))
    assert out.dim() == 0
    assert not torch.isnan(out)


def test_bce_perfect_prediction_low_loss():
    loss_fn = BCELoss()
    loss = loss_fn({"logits": torch.tensor([10.0, -10.0])}, torch.tensor([1.0, 0.0]))
    assert loss.item() < 0.01


# ---------------------------------------------------------------------------
# KDLoss — basic contract
# ---------------------------------------------------------------------------


def test_kd_scalar_output():
    loss_fn = KDLoss(alpha=0.5)
    out = loss_fn(
        {
            "logits": torch.tensor([0.5, -0.5]),
            "student_logits": torch.tensor([0.5, -0.5]),
            "teacher_logits": torch.tensor([0.6, -0.4]),
        },
        torch.tensor([1.0, 0.0]),
    )
    assert out.dim() == 0
    assert not torch.isnan(out)


def test_kd_gradient_flows_to_student():
    loss_fn = KDLoss(alpha=0.5)
    s_logits = torch.randn(2, requires_grad=True)
    loss_fn(
        {"logits": s_logits, "student_logits": s_logits, "teacher_logits": torch.randn(2)},
        torch.randint(0, 2, (2,)).float(),
    ).backward()
    assert s_logits.grad is not None


# ---------------------------------------------------------------------------
# KDLoss — None-value keys (HybridDTIModel contract)
# HybridDTIModel._standardize_output always sets all 6 keys; values may be None.
# KDLoss must not crash when student_logits/teacher_logits keys exist but are None.
# ---------------------------------------------------------------------------


def test_kd_fallback_when_student_logits_is_none():
    """student_logits key present but None → fallback to pure BCE."""
    loss_fn = KDLoss(alpha=0.5)
    logits = torch.tensor([0.5, -0.5], requires_grad=True)
    model_outputs = {
        "logits": logits,
        "student_logits": None,
        "teacher_logits": torch.randn(2),
        "gate_alpha": None,
        "student_var": None,
        "teacher_var": None,
    }
    loss = loss_fn(model_outputs, torch.tensor([1.0, 0.0]))
    expected = nn.BCEWithLogitsLoss()(logits, torch.tensor([1.0, 0.0]))
    assert torch.allclose(loss, expected)


def test_kd_fallback_when_teacher_logits_is_none():
    """teacher_logits key present but None → fallback to pure BCE."""
    loss_fn = KDLoss(alpha=0.5)
    logits = torch.tensor([0.5, -0.5], requires_grad=True)
    model_outputs = {
        "logits": logits,
        "student_logits": torch.randn(2),
        "teacher_logits": None,
        "gate_alpha": None,
        "student_var": None,
        "teacher_var": None,
    }
    loss = loss_fn(model_outputs, torch.tensor([1.0, 0.0]))
    expected = nn.BCEWithLogitsLoss()(logits, torch.tensor([1.0, 0.0]))
    assert torch.allclose(loss, expected)


def test_kd_fallback_when_both_branch_logits_none():
    """Both branch logits None (teacher-only or student-only mode) → pure BCE."""
    loss_fn = KDLoss(alpha=0.5)
    logits = torch.tensor([0.5, -0.5], requires_grad=True)
    model_outputs = {
        "logits": logits,
        "student_logits": None,
        "teacher_logits": None,
        "gate_alpha": None,
        "student_var": None,
        "teacher_var": None,
    }
    loss = loss_fn(model_outputs, torch.tensor([1.0, 0.0]))
    expected = nn.BCEWithLogitsLoss()(logits, torch.tensor([1.0, 0.0]))
    assert torch.allclose(loss, expected)


def test_kd_fallback_no_branch_keys_at_all():
    """Dict with only 'logits' key → fallback to pure BCE."""
    loss_fn = KDLoss(alpha=0.5)
    logits = torch.tensor([0.5, -0.5], requires_grad=True)
    loss = loss_fn({"logits": logits}, torch.tensor([1.0, 0.0]))
    expected = nn.BCEWithLogitsLoss()(logits, torch.tensor([1.0, 0.0]))
    assert torch.allclose(loss, expected)


# ---------------------------------------------------------------------------
# KDLoss — alpha boundary values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.5, 0.7, 1.0])
def test_kd_alpha_range_finite(alpha):
    loss_fn = KDLoss(alpha=alpha)
    out = loss_fn(
        {"logits": torch.randn(4), "student_logits": torch.randn(4), "teacher_logits": torch.randn(4)},
        torch.randint(0, 2, (4,)).float(),
    )
    assert torch.isfinite(out)


def test_kd_alpha_zero_equals_bce():
    """alpha=0 → pure task loss (BCE)."""
    loss_fn = KDLoss(alpha=0.0)
    logits = torch.randn(4)
    labels = torch.randint(0, 2, (4,)).float()
    loss_kd = loss_fn({"logits": logits, "student_logits": torch.randn(4), "teacher_logits": torch.randn(4)}, labels)
    assert torch.allclose(loss_kd, nn.BCEWithLogitsLoss()(logits, labels))


def test_kd_alpha_one_equals_mse():
    """alpha=1 → pure distillation loss (MSE)."""
    loss_fn = KDLoss(alpha=1.0)
    logits = torch.randn(4)
    s_logits = torch.randn(4)
    t_logits = torch.randn(4)
    labels = torch.randint(0, 2, (4,)).float()
    loss_kd = loss_fn({"logits": logits, "student_logits": s_logits, "teacher_logits": t_logits}, labels)
    assert torch.allclose(loss_kd, nn.MSELoss()(s_logits, t_logits))
