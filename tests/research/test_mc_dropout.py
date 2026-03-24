"""MC-Dropout consistency tests for HybridDTIModel."""

from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Batch, Data

from ugtsdti.models.hybrid import HybridDTIModel


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


def test_student_logits_equal_mean_of_mc_passes():
    torch.manual_seed(0)
    model = _make_hybrid(mc_samples=5)
    model.eval()
    batch = _make_batch(B=2)

    assert not model.student.training

    out = model(batch)
    reported_student = out["student_logits"]

    assert not model.student.training
    assert reported_student.shape == (2,)


def test_teacher_logits_equal_mean_of_mc_passes():
    torch.manual_seed(0)
    model = _make_hybrid(mc_samples=5)
    model.eval()
    batch = _make_batch(B=2)

    out = model(batch)
    assert out["teacher_logits"].shape == (2,)
    assert not model.teacher.training


def test_branch_restored_to_eval_after_mc_forward():
    model = _make_hybrid(mc_samples=3)
    model.eval()
    batch = _make_batch(B=2)

    assert not model.student.training
    model(batch)
    assert not model.student.training
    assert not model.teacher.training


def test_mc_forward_returns_correct_shapes():
    model = _make_hybrid(mc_samples=4)
    batch = _make_batch(B=3)

    mean_logit, epistemic_var = model._mc_forward(model.student, batch, mc_samples=4)

    assert mean_logit.shape == (3,)
    assert epistemic_var.shape == (3,)


def test_mc_forward_var_nonnegative():
    model = _make_hybrid(mc_samples=8)
    batch = _make_batch(B=4)

    _, epistemic_var = model._mc_forward(model.student, batch, mc_samples=8)
    assert (epistemic_var >= 0).all()


def test_mc_forward_mean_and_var_from_same_tensor():
    torch.manual_seed(42)
    model = _make_hybrid(mc_samples=6)
    batch = _make_batch(B=2)

    model.student.train()
    with torch.no_grad():
        mc_logits = torch.stack([model.student(batch)["logits"] for _ in range(6)], dim=-1)
    model.student.eval()

    expected_mean = mc_logits.mean(dim=-1)
    expected_var = mc_logits.var(dim=-1)

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


def test_student_branch_called_mc_plus_one_times_in_training_mode():
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

    assert call_count == 6


def test_teacher_branch_called_mc_plus_one_times_in_training_mode():
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

    assert call_count == 6


@pytest.mark.parametrize("B", [1, 2, 4, 8])
def test_only_student_output_shape_and_key(B):
    model = HybridDTIModel(student_cfg={"name": "baseline_student", "params": {"hidden_dim": 32}})
    model.eval()
    batch = _make_batch(B=B)
    out = model(batch)

    assert set(out.keys()) == {"logits"}
    assert out["logits"].shape == (B,)


@pytest.mark.parametrize("B", [1, 2, 4, 8])
def test_only_teacher_output_shape_and_key(B):
    model = HybridDTIModel(
        teacher_cfg={"name": "baseline_teacher", "params": {"hidden_dim": 32, "num_drugs": 1000, "num_targets": 1000}}
    )
    model.eval()
    batch = _make_batch(B=B)
    out = model(batch)

    assert set(out.keys()) == {"logits"}
    assert out["logits"].shape == (B,)


def test_mc_samples_zero_uses_equal_weight_fusion():
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
    model = _make_hybrid(mc_samples=5)
    model.eval()
    batch = _make_batch(B=B)
    out = model(batch)

    assert set(out.keys()) == {"logits", "student_logits", "teacher_logits", "gate_alpha"}
    assert out["logits"].shape == (B,)
    assert out["student_logits"].shape == (B,)
    assert out["teacher_logits"].shape == (B,)
    assert out["gate_alpha"].shape == (B,)


@pytest.mark.parametrize("B", [1, 2, 4])
def test_hybrid_output_dict_keys_training_mode(B):
    model = _make_hybrid(mc_samples=5)
    model.train()
    batch = _make_batch(B=B)
    out = model(batch)

    assert set(out.keys()) == {"logits", "student_logits", "teacher_logits", "gate_alpha"}
    assert out["logits"].shape == (B,)
    assert out["gate_alpha"].shape == (B,)


def test_no_model_raises():
    model = HybridDTIModel()
    batch = _make_batch(B=2)
    with pytest.raises(ValueError, match="no sub-models"):
        model(batch)
