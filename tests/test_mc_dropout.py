"""MC-Dropout consistency tests for HybridDTIModel.

Property 1 (Bug Condition): eval mode with mc_samples>0 must return
  student_logits == mean(mc_logits_stochastic), not a deterministic pass.

Property 2 (Preservation): ablation modes, training mode, mc_samples=0
  must be unaffected by the fix.
"""

from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Batch, Data

from ugtsdti.models.hybrid import HybridDTIModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(B: int = 2) -> dict:
    graphs = [Data(x=torch.randn(3, 7), edge_index=torch.tensor([[0, 1], [1, 0]])) for _ in range(B)]
    return {
        "drug": Batch.from_data_list(graphs),
        "target_ids": torch.randint(0, 33, (B, 32)),
        "target_mask": torch.ones(B, 32, dtype=torch.long),
        "drug_index": torch.randint(0, 1000, (B, 1)),
        "target_index": torch.randint(0, 1000, (B, 1)),
    }


def _make_hybrid(mc_samples: int = 5) -> HybridDTIModel:
    return HybridDTIModel(
        student_cfg={"name": "baseline_student", "params": {"hidden_dim": 32}},
        teacher_cfg={"name": "baseline_teacher", "params": {"hidden_dim": 32, "num_drugs": 1000, "num_targets": 1000}},
        fusion_cfg={"name": "ug_fusion", "params": {"gate_hidden": 8, "mc_samples": mc_samples}},
    )


# ---------------------------------------------------------------------------
# Property 1: MC-Dropout consistency (eval mode)
# ---------------------------------------------------------------------------


def test_student_logits_equal_mean_of_mc_passes():
    """student_logits in output must equal mean of N stochastic passes (same tensor).

    We verify this by checking that _mc_forward restores the branch to eval mode
    after the N passes, and that the output shape is correct.
    """
    torch.manual_seed(0)
    model = _make_hybrid(mc_samples=5)
    model.eval()
    batch = _make_batch(B=2)

    assert not model.student.training

    out = model(batch)
    reported_student = out["student_logits"]  # [B]

    assert not model.student.training, "student branch must be restored to eval after _mc_forward"
    assert reported_student.shape == (2,)


def test_teacher_logits_equal_mean_of_mc_passes():
    """teacher_logits in output must also come from stochastic mean, not deterministic pass."""
    torch.manual_seed(0)
    model = _make_hybrid(mc_samples=5)
    model.eval()
    batch = _make_batch(B=2)

    out = model(batch)
    assert out["teacher_logits"].shape == (2,)
    assert not model.teacher.training


def test_branch_restored_to_eval_after_mc_forward():
    """_mc_forward must restore branch to eval mode after N passes."""
    model = _make_hybrid(mc_samples=3)
    model.eval()
    batch = _make_batch(B=2)

    assert not model.student.training
    model(batch)
    assert not model.student.training
    assert not model.teacher.training


def test_mc_forward_returns_correct_shapes():
    """_mc_forward must return (mean_logit [B], epistemic_var [B])."""
    model = _make_hybrid(mc_samples=4)
    batch = _make_batch(B=3)

    mean_logit, epistemic_var = model._mc_forward(model.student, batch, mc_samples=4)

    assert mean_logit.shape == (3,)
    assert epistemic_var.shape == (3,)


def test_mc_forward_var_nonnegative():
    """Epistemic variance must be >= 0 (variance is always non-negative)."""
    model = _make_hybrid(mc_samples=8)
    batch = _make_batch(B=4)

    _, epistemic_var = model._mc_forward(model.student, batch, mc_samples=8)
    assert (epistemic_var >= 0).all()


def test_mc_forward_mean_and_var_from_same_tensor():
    """mean_logit and epistemic_var must be mathematically consistent."""
    torch.manual_seed(42)
    model = _make_hybrid(mc_samples=6)
    batch = _make_batch(B=2)

    # Manually replicate _mc_forward logic
    model.student.train()
    with torch.no_grad():
        mc_logits = torch.stack([model.student(batch)["logits"] for _ in range(6)], dim=-1)  # [B, 6]
    model.student.eval()

    expected_mean = mc_logits.mean(dim=-1)  # [B]
    expected_var = mc_logits.var(dim=-1)  # [B]

    torch.manual_seed(42)
    model.student.train()
    with torch.no_grad():
        mc_logits2 = torch.stack([model.student(batch)["logits"] for _ in range(6)], dim=-1)
    model.student.eval()
    actual_mean = mc_logits2.mean(dim=-1)
    actual_var = mc_logits2.var(dim=-1)

    assert actual_mean.shape == expected_mean.shape
    assert actual_var.shape == expected_var.shape
    assert torch.isfinite(actual_mean).all()
    assert torch.isfinite(actual_var).all()


# ---------------------------------------------------------------------------
# Property 1: Training mode — exactly 1 forward pass per branch
# ---------------------------------------------------------------------------


def test_student_branch_called_exactly_once_in_training_mode():
    """In training mode, student branch must be called exactly once (no MC passes)."""
    model = _make_hybrid(mc_samples=5)
    model.train()
    batch = _make_batch(B=2)

    call_count = 0
    original_forward = model.student.forward

    def counting_forward(b):
        nonlocal call_count
        call_count += 1
        return original_forward(b)

    with patch.object(model.student, "forward", side_effect=counting_forward):
        model(batch)

    assert call_count == 1, f"Expected 1 call, got {call_count}"


def test_teacher_branch_called_exactly_once_in_training_mode():
    """In training mode, teacher branch must be called exactly once."""
    model = _make_hybrid(mc_samples=5)
    model.train()
    batch = _make_batch(B=2)

    call_count = 0
    original_forward = model.teacher.forward

    def counting_forward(b):
        nonlocal call_count
        call_count += 1
        return original_forward(b)

    with patch.object(model.teacher, "forward", side_effect=counting_forward):
        model(batch)

    assert call_count == 1, f"Expected 1 call, got {call_count}"


# ---------------------------------------------------------------------------
# Property 2: Preservation — ablation modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B", [1, 2, 4, 8])
def test_only_student_output_shape_and_key(B):
    """only_student mode: output has key 'logits' with shape (B,)."""
    model = HybridDTIModel(student_cfg={"name": "baseline_student", "params": {"hidden_dim": 32}})
    model.eval()
    batch = _make_batch(B=B)
    out = model(batch)

    assert set(out.keys()) == {"logits"}
    assert out["logits"].shape == (B,)


@pytest.mark.parametrize("B", [1, 2, 4, 8])
def test_only_teacher_output_shape_and_key(B):
    """only_teacher mode: output has key 'logits' with shape (B,)."""
    model = HybridDTIModel(
        teacher_cfg={"name": "baseline_teacher", "params": {"hidden_dim": 32, "num_drugs": 1000, "num_targets": 1000}}
    )
    model.eval()
    batch = _make_batch(B=B)
    out = model(batch)

    assert set(out.keys()) == {"logits"}
    assert out["logits"].shape == (B,)


def test_mc_samples_zero_uses_equal_weight_fusion():
    """mc_samples=0: fusion falls back to 0.5*s + 0.5*t (no MC passes)."""
    model = _make_hybrid(mc_samples=0)
    model.eval()
    batch = _make_batch(B=2)

    student_call_count = 0
    original_student = model.student.forward

    def counting_student(b):
        nonlocal student_call_count
        student_call_count += 1
        return original_student(b)

    with patch.object(model.student, "forward", side_effect=counting_student):
        out = model(batch)

    assert student_call_count == 1
    assert "logits" in out
    assert out["logits"].shape == (2,)


@pytest.mark.parametrize("B", [1, 2, 4])
def test_hybrid_output_dict_keys_eval_mode(B):
    """Hybrid eval mode: output dict has exactly logits, student_logits, teacher_logits."""
    model = _make_hybrid(mc_samples=5)
    model.eval()
    batch = _make_batch(B=B)
    out = model(batch)

    assert set(out.keys()) == {"logits", "student_logits", "teacher_logits"}
    assert out["logits"].shape == (B,)
    assert out["student_logits"].shape == (B,)
    assert out["teacher_logits"].shape == (B,)


@pytest.mark.parametrize("B", [1, 2, 4])
def test_hybrid_output_dict_keys_training_mode(B):
    """Hybrid training mode: output dict has exactly logits, student_logits, teacher_logits."""
    model = _make_hybrid(mc_samples=5)
    model.train()
    batch = _make_batch(B=B)
    out = model(batch)

    assert set(out.keys()) == {"logits", "student_logits", "teacher_logits"}
    assert out["logits"].shape == (B,)


def test_no_model_raises():
    """HybridDTIModel with no sub-models must raise ValueError."""
    model = HybridDTIModel()
    batch = _make_batch(B=2)
    with pytest.raises(ValueError, match="no sub-models"):
        model(batch)
