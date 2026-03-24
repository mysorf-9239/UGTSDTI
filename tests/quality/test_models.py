import torch
from torch_geometric.data import Batch, Data

from ugtsdti.models.hybrid import HybridDTIModel
from ugtsdti.models.student.baseline import BaselineStudent


def test_baseline_student_forward():
    model = BaselineStudent(hidden_dim=64)
    graphs = [
        Data(x=torch.randn(3, 7), edge_index=torch.tensor([[0, 1], [1, 0]])),
        Data(x=torch.randn(3, 7), edge_index=torch.tensor([[0, 1], [1, 0]])),
    ]
    batch_drug = Batch.from_data_list(graphs)
    target_ids = torch.randint(0, 33, (2, 128))
    target_mask = torch.ones((2, 128))

    batch = {"drug": batch_drug, "target_ids": target_ids, "target_mask": target_mask}
    out = model(batch)

    assert "logits" in out
    assert out["logits"].shape == (2,)


def test_hybrid_ablation_loading():
    # Only Student
    hybrid = HybridDTIModel(student_cfg={"name": "baseline_student", "params": {"hidden_dim": 64}})
    assert hybrid.student is not None
    assert hybrid.teacher is None

    # Only Teacher
    hybrid_teacher = HybridDTIModel(
        teacher_cfg={"name": "baseline_teacher", "params": {"hidden_dim": 64, "num_drugs": 1000, "num_targets": 1000}}
    )
    assert hybrid_teacher.teacher is not None
    assert hybrid_teacher.student is None

    # Both
    hybrid_full = HybridDTIModel(
        student_cfg={"name": "baseline_student", "params": {"hidden_dim": 64}},
        teacher_cfg={"name": "baseline_teacher", "params": {"hidden_dim": 64, "num_drugs": 1000, "num_targets": 1000}},
        fusion_cfg={"name": "ug_fusion", "params": {"gate_hidden": 16, "mc_samples": 5}},
    )
    assert hybrid_full.fusion is not None
