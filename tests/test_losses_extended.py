"""Extended loss tests: BCEWithLogitsLossWrapper, KDDualLoss edge cases."""

import pytest
import torch

from ugtsdti.losses import BCEWithLogitsLossWrapper
from ugtsdti.losses.distillation import KDDualLoss

# ---------------------------------------------------------------------------
# BCEWithLogitsLossWrapper
# ---------------------------------------------------------------------------


def test_bce_wrapper_scalar_output():
    loss_fn = BCEWithLogitsLossWrapper()
    logits = torch.tensor([[1.0], [-1.0], [0.5]])
    labels = torch.tensor([[1.0], [0.0], [1.0]])
    out = loss_fn({"logits": logits}, labels)
    assert out.dim() == 0
    assert not torch.isnan(out)


def test_bce_wrapper_gradient_flows():
    loss_fn = BCEWithLogitsLossWrapper()
    logits = torch.tensor([[1.0], [-1.0]], requires_grad=True)
    labels = torch.tensor([[1.0], [0.0]])
    loss = loss_fn({"logits": logits}, labels)
    loss.backward()
    assert logits.grad is not None


def test_bce_wrapper_ignores_extra_keys():
    """BCEWrapper must only use 'logits' key, ignore student/teacher logits."""
    loss_fn = BCEWithLogitsLossWrapper()
    logits = torch.tensor([[0.5], [-0.5]])
    labels = torch.tensor([[1.0], [0.0]])
    model_outputs = {
        "logits": logits,
        "student_logits": torch.randn(2, 1),
        "teacher_logits": torch.randn(2, 1),
    }
    out = loss_fn(model_outputs, labels)
    assert out.dim() == 0
    assert not torch.isnan(out)


def test_bce_wrapper_perfect_prediction_low_loss():
    """Very confident correct predictions → loss near 0."""
    loss_fn = BCEWithLogitsLossWrapper()
    logits = torch.tensor([[10.0], [-10.0]])  # very confident
    labels = torch.tensor([[1.0], [0.0]])
    loss = loss_fn({"logits": logits}, labels)
    assert loss.item() < 0.01


# ---------------------------------------------------------------------------
# KDDualLoss
# ---------------------------------------------------------------------------


def test_kd_loss_scalar():
    loss_fn = KDDualLoss(alpha=0.5)
    out = {
        "logits": torch.tensor([[0.5], [-0.5]]),
        "student_logits": torch.tensor([[0.5], [-0.5]]),
        "teacher_logits": torch.tensor([[0.6], [-0.4]]),
    }
    labels = torch.tensor([[1.0], [0.0]])
    loss = loss_fn(out, labels)
    assert loss.dim() == 0
    assert not torch.isnan(loss)


def test_kd_loss_fallback_no_student_teacher():
    """Without student/teacher logits, KDDualLoss falls back to pure BCE."""
    loss_fn = KDDualLoss(alpha=0.5)
    logits = torch.tensor([[0.5], [-0.5]], requires_grad=True)
    labels = torch.tensor([[1.0], [0.0]])

    loss_kd = loss_fn({"logits": logits}, labels)

    # Compare with pure BCE
    import torch.nn as nn

    bce = nn.BCEWithLogitsLoss()(logits, labels)
    assert torch.allclose(loss_kd, bce)


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.5, 0.7, 1.0])
def test_kd_loss_alpha_range(alpha):
    """KDDualLoss must be finite for all valid alpha values."""
    loss_fn = KDDualLoss(alpha=alpha)
    out = {"logits": torch.randn(4, 1), "student_logits": torch.randn(4, 1), "teacher_logits": torch.randn(4, 1)}
    labels = torch.randint(0, 2, (4, 1)).float()
    loss = loss_fn(out, labels)
    assert torch.isfinite(loss)


def test_kd_loss_alpha_zero_equals_bce():
    """alpha=0 → pure task loss (BCE), distillation term has zero weight."""
    import torch.nn as nn

    loss_fn = KDDualLoss(alpha=0.0)
    logits = torch.randn(4, 1)
    s_logits = torch.randn(4, 1)
    t_logits = torch.randn(4, 1)
    labels = torch.randint(0, 2, (4, 1)).float()

    loss_kd = loss_fn({"logits": logits, "student_logits": s_logits, "teacher_logits": t_logits}, labels)
    loss_bce = nn.BCEWithLogitsLoss()(logits, labels)
    assert torch.allclose(loss_kd, loss_bce)


def test_kd_loss_alpha_one_equals_mse():
    """alpha=1 → pure distillation loss (MSE), task term has zero weight."""
    import torch.nn as nn

    loss_fn = KDDualLoss(alpha=1.0)
    logits = torch.randn(4, 1)
    s_logits = torch.randn(4, 1)
    t_logits = torch.randn(4, 1)
    labels = torch.randint(0, 2, (4, 1)).float()

    loss_kd = loss_fn({"logits": logits, "student_logits": s_logits, "teacher_logits": t_logits}, labels)
    loss_mse = nn.MSELoss()(s_logits, t_logits)
    assert torch.allclose(loss_kd, loss_mse)


def test_kd_loss_gradient_flows_to_student():
    loss_fn = KDDualLoss(alpha=0.5)
    s_logits = torch.randn(2, 1, requires_grad=True)
    t_logits = torch.randn(2, 1)
    fused = s_logits  # fused = student for simplicity
    labels = torch.randint(0, 2, (2, 1)).float()

    loss = loss_fn({"logits": fused, "student_logits": s_logits, "teacher_logits": t_logits}, labels)
    loss.backward()
    assert s_logits.grad is not None
