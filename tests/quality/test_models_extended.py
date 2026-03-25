"""Extended model tests: output contract, numerical stability, gradient flow.

Plugin output contract enforced here:
  - Teacher/Student forward → {"logits": FloatTensor (B,)}  raw logits, NOT sigmoid, NOT (B,1)
  - HybridDTIModel forward → 6-key dict, all shapes (B,) or None
"""

import pytest
import torch
from torch_geometric.data import Batch, Data

from ugtsdti.models.hybrid import HybridDTIModel
from ugtsdti.models.student.baseline import BaselineStudent
from ugtsdti.models.teacher.baseline import BaselineTeacher

EXPECTED_OUTPUT_KEYS = {"logits", "student_logits", "teacher_logits", "gate_alpha", "student_var", "teacher_var"}


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
# Plugin output contract: Teacher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B", [1, 2, 4])
def test_baseline_teacher_output_contract(B):
    """Teacher contract: {"logits": FloatTensor (B,)} — raw logits, not sigmoid."""
    model = BaselineTeacher(hidden_dim=32, num_drugs=1000, num_targets=1000)
    out = model(_make_batch(B=B))
    assert set(out.keys()) == {"logits"}
    assert out["logits"].shape == (B,)
    assert out["logits"].dtype == torch.float32


def test_baseline_teacher_output_finite():
    model = BaselineTeacher(hidden_dim=32, num_drugs=1000, num_targets=1000)
    assert torch.isfinite(model(_make_batch(B=4))["logits"]).all()


def test_baseline_teacher_gradient_flows():
    model = BaselineTeacher(hidden_dim=32, num_drugs=1000, num_targets=1000)
    model(_make_batch(B=2))["logits"].sum().backward()
    assert model.drug_emb.weight.grad is not None
    assert model.target_emb.weight.grad is not None


def test_baseline_teacher_reads_only_index_keys():
    """Teacher must work with a batch that has ONLY drug_index and target_index."""
    model = BaselineTeacher(hidden_dim=32, num_drugs=1000, num_targets=1000)
    minimal_batch = {
        "drug_index": torch.randint(0, 1000, (2, 1)),
        "target_index": torch.randint(0, 1000, (2, 1)),
    }
    out = model(minimal_batch)
    assert out["logits"].shape == (2,)


# ---------------------------------------------------------------------------
# Plugin output contract: Student
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B", [1, 2, 4])
def test_baseline_student_output_contract(B):
    """Student contract: {"logits": FloatTensor (B,)} — raw logits, not sigmoid."""
    model = BaselineStudent(hidden_dim=32)
    out = model(_make_batch(B=B))
    assert set(out.keys()) == {"logits"}
    assert out["logits"].shape == (B,)
    assert out["logits"].dtype == torch.float32


def test_baseline_student_output_finite():
    model = BaselineStudent(hidden_dim=32)
    assert torch.isfinite(model(_make_batch(B=4))["logits"]).all()


def test_baseline_student_gradient_flows():
    model = BaselineStudent(hidden_dim=32)
    model(_make_batch(B=2))["logits"].sum().backward()
    assert model.drug_proj.weight.grad is not None
    assert model.protein_proj.weight.grad is not None


def test_baseline_student_all_padding_mask():
    """All-zero mask must not crash and must produce finite output."""
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


def test_baseline_student_reads_only_drug_sequence_keys():
    """Student must work without drug_index/target_index in batch."""
    model = BaselineStudent(hidden_dim=32)
    B = 2
    graphs = [Data(x=torch.randn(3, 7), edge_index=torch.tensor([[0, 1], [1, 0]])) for _ in range(B)]
    minimal_batch = {
        "drug": Batch.from_data_list(graphs),
        "target_ids": torch.randint(0, 33, (B, 32)),
        "target_mask": torch.ones(B, 32, dtype=torch.long),
    }
    out = model(minimal_batch)
    assert out["logits"].shape == (B,)


# ---------------------------------------------------------------------------
# HybridDTIModel output contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B", [1, 2, 4])
def test_hybrid_output_contract_eval_mode(B):
    """Hybrid eval: all 6 keys present, all tensors shape (B,)."""
    model = _make_hybrid(mc_samples=5)
    model.eval()
    out = model(_make_batch(B=B))
    assert set(out.keys()) == EXPECTED_OUTPUT_KEYS
    for key in ("logits", "student_logits", "teacher_logits", "gate_alpha", "student_var", "teacher_var"):
        assert out[key] is not None
        assert out[key].shape == (B,)
        assert out[key].dtype == torch.float32


@pytest.mark.parametrize("B", [1, 2, 4])
def test_hybrid_output_contract_training_mode_with_uncertainty(B):
    """Hybrid train with mc_samples>0 and use_uncertainty_in_train=True: all 6 keys present."""
    model = _make_hybrid(mc_samples=5)
    model.train()
    out = model(_make_batch(B=B))
    assert set(out.keys()) == EXPECTED_OUTPUT_KEYS
    assert out["logits"].shape == (B,)
    assert out["gate_alpha"].shape == (B,)
    assert out["student_var"].shape == (B,)
    assert out["teacher_var"].shape == (B,)


@pytest.mark.parametrize("B", [1, 2, 4])
def test_hybrid_output_contract_training_mode_no_uncertainty(B):
    """Hybrid train with mc_samples=0: student_var and teacher_var are None."""
    model = _make_hybrid(mc_samples=0)
    model.train()
    out = model(_make_batch(B=B))
    assert set(out.keys()) == EXPECTED_OUTPUT_KEYS
    assert out["logits"].shape == (B,)
    assert out["gate_alpha"].shape == (B,)
    assert out["student_var"] is None
    assert out["teacher_var"] is None


def test_hybrid_output_finite():
    model = _make_hybrid(mc_samples=5)
    model.eval()
    out = model(_make_batch(B=4))
    for key in ("logits", "student_logits", "teacher_logits", "gate_alpha", "student_var", "teacher_var"):
        assert torch.isfinite(out[key]).all()


def test_hybrid_gradient_flows_through_gate():
    model = _make_hybrid(mc_samples=3)
    model.eval()
    with torch.enable_grad():
        out = model(_make_batch(B=2))
        out["logits"].sum().backward()
    assert model.fusion.gate_mlp[0].weight.grad is not None


# ---------------------------------------------------------------------------
# HybridDTIModel — single-branch modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B", [1, 2, 4])
def test_hybrid_only_student_output_contract(B):
    """Student-only: logits and student_logits present; teacher fields are None."""
    model = HybridDTIModel(student_cfg={"name": "baseline_student", "params": {"hidden_dim": 32}})
    out = model(_make_batch(B=B))
    assert set(out.keys()) == EXPECTED_OUTPUT_KEYS
    assert out["logits"].shape == (B,)
    assert out["student_logits"].shape == (B,)
    assert out["teacher_logits"] is None
    assert out["gate_alpha"] is None
    assert out["student_var"] is None
    assert out["teacher_var"] is None


@pytest.mark.parametrize("B", [1, 2, 4])
def test_hybrid_only_teacher_output_contract(B):
    """Teacher-only: logits and teacher_logits present; student fields are None."""
    model = HybridDTIModel(
        teacher_cfg={"name": "baseline_teacher", "params": {"hidden_dim": 32, "num_drugs": 1000, "num_targets": 1000}}
    )
    out = model(_make_batch(B=B))
    assert set(out.keys()) == EXPECTED_OUTPUT_KEYS
    assert out["logits"].shape == (B,)
    assert out["teacher_logits"].shape == (B,)
    assert out["student_logits"] is None
    assert out["gate_alpha"] is None
    assert out["student_var"] is None
    assert out["teacher_var"] is None


def test_hybrid_no_submodels_raises():
    model = HybridDTIModel()
    with pytest.raises(ValueError, match="no sub-models"):
        model(_make_batch(B=2))


def test_hybrid_is_hybrid_flag():
    assert _make_hybrid().is_hybrid is True
    assert HybridDTIModel(student_cfg={"name": "baseline_student", "params": {"hidden_dim": 32}}).is_hybrid is False
    assert (
        HybridDTIModel(
            teacher_cfg={
                "name": "baseline_teacher",
                "params": {"hidden_dim": 32, "num_drugs": 1000, "num_targets": 1000},
            }
        ).is_hybrid
        is False
    )
