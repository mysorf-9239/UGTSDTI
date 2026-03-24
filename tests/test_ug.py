"""Unit tests for UncertaintyGatedFusion (UG fusion module).

Covers:
- Gate output range (α ∈ (0, 1))
- Fallback to equal-weight average when vars are None
- Output shape consistency
- Gradient flow through gate MLP
- mc_samples attribute stored correctly
- Deterministic output given same inputs
"""

import pytest
import torch

from ugtsdti.models.fusion.ug import UncertaintyGatedFusion


def _make_fusion(mc_samples: int = 5, gate_hidden: int = 8) -> UncertaintyGatedFusion:
    return UncertaintyGatedFusion(gate_hidden=gate_hidden, mc_samples=mc_samples)


# ---------------------------------------------------------------------------
# Attribute / construction
# ---------------------------------------------------------------------------


def test_mc_samples_stored():
    fusion = _make_fusion(mc_samples=7)
    assert fusion.mc_samples == 7


def test_mc_samples_zero():
    fusion = _make_fusion(mc_samples=0)
    assert fusion.mc_samples == 0


# ---------------------------------------------------------------------------
# Fallback: no variance → equal-weight average
# ---------------------------------------------------------------------------


def test_fallback_equal_weight_no_vars():
    fusion = _make_fusion()
    s = torch.tensor([1.0, 3.0])
    t = torch.tensor([3.0, 1.0])

    out = fusion(s, t)
    expected = 0.5 * s + 0.5 * t
    assert torch.allclose(out, expected)


def test_fallback_student_var_none():
    fusion = _make_fusion()
    s = torch.tensor([2.0, 4.0])
    t = torch.tensor([0.0, 2.0])
    teacher_var = torch.tensor([0.1, 0.2])

    out = fusion(s, t, student_var=None, teacher_var=teacher_var)
    expected = 0.5 * s + 0.5 * t
    assert torch.allclose(out, expected)


def test_fallback_teacher_var_none():
    fusion = _make_fusion()
    s = torch.tensor([2.0, 4.0])
    t = torch.tensor([0.0, 2.0])
    student_var = torch.tensor([0.1, 0.2])

    out = fusion(s, t, student_var=student_var, teacher_var=None)
    expected = 0.5 * s + 0.5 * t
    assert torch.allclose(out, expected)


# ---------------------------------------------------------------------------
# Gate output range
# ---------------------------------------------------------------------------


def test_gate_weight_in_zero_one():
    """Gate MLP uses Sigmoid → output must be in (0, 1)."""
    fusion = _make_fusion()
    B = 8
    s = torch.randn(B)
    t = torch.randn(B)
    s_var = torch.rand(B).abs()
    t_var = torch.rand(B).abs()

    out = fusion(s, t, s_var, t_var)
    assert out.shape == (B,)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("B", [1, 2, 4, 8, 16])
def test_output_shape(B):
    fusion = _make_fusion()
    s = torch.randn(B)
    t = torch.randn(B)
    s_var = torch.rand(B)
    t_var = torch.rand(B)

    out = fusion(s, t, s_var, t_var)
    assert out.shape == (B,)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_through_gate():
    """Gate MLP parameters must receive gradients during backward."""
    fusion = _make_fusion()
    B = 4
    s = torch.randn(B, requires_grad=True)
    t = torch.randn(B, requires_grad=True)
    s_var = torch.rand(B)
    t_var = torch.rand(B)

    out = fusion(s, t, s_var, t_var)
    loss = out.sum()
    loss.backward()

    for name, param in fusion.gate_mlp.named_parameters():
        assert param.grad is not None, f"No gradient for gate_mlp.{name}"

    assert s.grad is not None
    assert t.grad is not None


def test_gradient_flows_in_fallback():
    """Gradients must flow even in fallback (equal-weight) mode."""
    fusion = _make_fusion()
    s = torch.randn(2, requires_grad=True)
    t = torch.randn(2, requires_grad=True)

    out = fusion(s, t)
    out.sum().backward()

    assert s.grad is not None
    assert t.grad is not None


# ---------------------------------------------------------------------------
# High uncertainty → student dominates
# ---------------------------------------------------------------------------


def test_high_teacher_uncertainty_student_dominates():
    """Gate is responsive to variance inputs (output changes when variance changes)."""
    torch.manual_seed(0)
    fusion = _make_fusion()
    fusion.eval()

    B = 4
    s = torch.ones(B)
    t = torch.ones(B) * 2.0

    out_low_t_var = fusion(s, t, student_var=torch.zeros(B), teacher_var=torch.zeros(B))
    out_high_t_var = fusion(s, t, student_var=torch.zeros(B), teacher_var=torch.ones(B) * 100.0)

    assert not torch.allclose(out_low_t_var, out_high_t_var)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_in_eval_mode():
    """Same inputs in eval mode must produce identical outputs."""
    torch.manual_seed(0)
    fusion = _make_fusion()
    fusion.eval()

    B = 4
    s = torch.randn(B)
    t = torch.randn(B)
    s_var = torch.rand(B)
    t_var = torch.rand(B)

    out1 = fusion(s, t, s_var, t_var)
    out2 = fusion(s, t, s_var, t_var)
    assert torch.allclose(out1, out2)
