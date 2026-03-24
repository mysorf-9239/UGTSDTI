"""Extended model tests for correctness and numerical stability."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from ugtsdti.models.hybrid import HybridDTIModel
from ugtsdti.models.student.baseline import BaselineStudent
from ugtsdti.models.teacher.baseline import BaselineTeacher


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


@pytest.mark.parametrize("B", [1, 2, 4])
def test_baseline_teacher_forward_shape(B):
    model = BaselineTeacher(hidden_dim=32, num_drugs=1000, num_targets=1000)
    batch = _make_batch(B=B)
    out = model(batch)
    assert "logits" in out
    assert out["logits"].shape == (B,)


def test_baseline_teacher_output_finite():
    model = BaselineTeacher(hidden_dim=32, num_drugs=1000, num_targets=1000)
    batch = _make_batch(B=4)
    out = model(batch)
    assert torch.isfinite(out["logits"]).all()


def test_baseline_teacher_gradient_flows():
    model = BaselineTeacher(hidden_dim=32, num_drugs=1000, num_targets=1000)
    batch = _make_batch(B=2)
    out = model(batch)
    out["logits"].sum().backward()
    assert model.drug_emb.weight.grad is not None
    assert model.target_emb.weight.grad is not None


def test_baseline_student_single_sample():
    model = BaselineStudent(hidden_dim=32)
    batch = _make_batch(B=1)
    out = model(batch)
    assert out["logits"].shape == (1,)


def test_baseline_student_output_finite():
    model = BaselineStudent(hidden_dim=32)
    batch = _make_batch(B=4)
    out = model(batch)
    assert torch.isfinite(out["logits"]).all()


def test_baseline_student_gradient_flows():
    model = BaselineStudent(hidden_dim=32)
    batch = _make_batch(B=2)
    out = model(batch)
    out["logits"].sum().backward()
    assert model.drug_proj.weight.grad is not None
    assert model.protein_proj.weight.grad is not None


def test_baseline_student_all_padding_mask():
    model = BaselineStudent(hidden_dim=32)
    B = 2
    graphs = [Data(x=torch.randn(3, 7), edge_index=torch.tensor([[0, 1], [1, 0]])) for _ in range(B)]
    batch = {
        "drug": Batch.from_data_list(graphs),
        "target_ids": torch.zeros(B, 32, dtype=torch.long),
        "target_mask": torch.zeros(B, 32, dtype=torch.long),
    }
    out = model(batch)
    assert out["logits"].shape == (B,)
    assert torch.isfinite(out["logits"]).all()


@pytest.mark.parametrize("B", [1, 2, 4])
def test_hybrid_forward_training_mode(B):
    model = _make_hybrid(mc_samples=5)
    model.train()
    batch = _make_batch(B=B)
    out = model(batch)

    assert set(out.keys()) == {"logits", "student_logits", "teacher_logits", "gate_alpha"}
    assert out["logits"].shape == (B,)
    assert out["student_logits"].shape == (B,)
    assert out["teacher_logits"].shape == (B,)
    assert out["gate_alpha"].shape == (B,)


@pytest.mark.parametrize("B", [1, 2, 4])
def test_hybrid_forward_eval_mode_mc(B):
    model = _make_hybrid(mc_samples=5)
    model.eval()
    batch = _make_batch(B=B)
    out = model(batch)

    assert set(out.keys()) == {"logits", "student_logits", "teacher_logits", "gate_alpha"}
    assert out["logits"].shape == (B,)
    assert out["gate_alpha"].shape == (B,)


def test_hybrid_forward_output_finite():
    model = _make_hybrid(mc_samples=5)
    model.eval()
    batch = _make_batch(B=4)
    out = model(batch)
    for key in ("logits", "student_logits", "teacher_logits", "gate_alpha"):
        assert torch.isfinite(out[key]).all()


def test_hybrid_gradient_flows_eval_mc():
    model = _make_hybrid(mc_samples=3)
    model.eval()
    batch = _make_batch(B=2)

    with torch.enable_grad():
        out = model(batch)
        loss = out["logits"].sum()
        loss.backward()

    assert model.fusion.gate_mlp[0].weight.grad is not None


def test_hybrid_only_student_forward():
    model = HybridDTIModel(student_cfg={"name": "baseline_student", "params": {"hidden_dim": 32}})
    batch = _make_batch(B=2)
    out = model(batch)
    assert "logits" in out
    assert out["logits"].shape == (2,)


def test_hybrid_only_teacher_forward():
    model = HybridDTIModel(
        teacher_cfg={"name": "baseline_teacher", "params": {"hidden_dim": 32, "num_drugs": 1000, "num_targets": 1000}}
    )
    batch = _make_batch(B=2)
    out = model(batch)
    assert "logits" in out
    assert out["logits"].shape == (2,)


def test_hybrid_is_hybrid_flag():
    full = _make_hybrid()
    assert full.is_hybrid is True

    student_only = HybridDTIModel(student_cfg={"name": "baseline_student", "params": {"hidden_dim": 32}})
    assert student_only.is_hybrid is False

    teacher_only = HybridDTIModel(
        teacher_cfg={"name": "baseline_teacher", "params": {"hidden_dim": 32, "num_drugs": 1000, "num_targets": 1000}}
    )
    assert teacher_only.is_hybrid is False
