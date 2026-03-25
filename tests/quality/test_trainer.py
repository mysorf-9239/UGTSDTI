import os

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from ugtsdti.core.trainer import Trainer


class _TinyDataset(Dataset):
    unk_drug_index = 999
    unk_target_index = 888

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {
            "feature": torch.tensor([float(idx)]),
            "drug_index": torch.tensor([999 if idx == 3 else idx]),
            "target_index": torch.tensor([888 if idx == 2 else idx]),
            "label": torch.tensor([1.0 if idx % 2 == 0 else 0.0]),
        }


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, batch):
        logits = self.linear(batch["feature"].float()).view(-1)
        return {"logits": logits}


def test_load_best_checkpoint_restores_saved_weights(tmp_path):
    model = _TinyModel()
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        loss_fn=torch.nn.BCEWithLogitsLoss(),
        device=torch.device("cpu"),
        cfg_trainer={"epochs": 1, "output_dir": str(tmp_path)},
    )

    original_weight = model.linear.weight.detach().clone()
    trainer._save_checkpoint()

    with torch.no_grad():
        model.linear.weight.add_(10.0)

    loaded = trainer.load_best_checkpoint()

    assert loaded is True
    assert torch.allclose(model.linear.weight, original_weight)


def test_fit_without_val_still_saves_checkpoint(tmp_path):
    model = _TinyModel()
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        loss_fn=lambda outputs, y_true: torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["logits"], y_true.view(-1)
        ),
        device=torch.device("cpu"),
        cfg_trainer={"epochs": 1, "output_dir": str(tmp_path)},
    )
    loader = DataLoader(_TinyDataset(), batch_size=2, shuffle=False)

    trainer.fit(loader, val_loader=None)

    assert trainer.best_model_path
    assert os.path.exists(trainer.best_model_path)


class _TinyResearchModel(torch.nn.Module):
    def forward(self, batch):
        base = batch["feature"].float().view(-1)
        return {
            "logits": base,
            "student_logits": base + 0.1,
            "teacher_logits": base - 0.1,
            "gate_alpha": torch.full_like(base, 0.25),
            "student_var": torch.full_like(base, 0.1),
            "teacher_var": torch.full_like(base, 0.2),
        }


def test_evaluate_reports_research_metrics(tmp_path):
    trainer = Trainer(
        model=_TinyResearchModel(),
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1, requires_grad=True))], lr=0.1),
        loss_fn=lambda outputs, y_true: torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["logits"], y_true.view(-1)
        ),
        device=torch.device("cpu"),
        cfg_trainer={"epochs": 1, "output_dir": str(tmp_path)},
    )
    loader = DataLoader(_TinyDataset(), batch_size=2, shuffle=False)

    metrics = trainer.evaluate(loader, prefix="val")

    assert "val/gate_alpha_mean" in metrics
    assert metrics["val/gate_alpha_mean"] == pytest.approx(0.25)
    assert "val/student_var_mean" in metrics
    assert metrics["val/student_var_mean"] == pytest.approx(0.1)
    assert "val/teacher_var_mean" in metrics
    assert metrics["val/teacher_var_mean"] == pytest.approx(0.2)
    assert "val/unk_drug_rate" in metrics
    assert "val/unk_target_rate" in metrics
    assert "val/student_auroc" in metrics
    assert "val/teacher_auroc" in metrics
    assert "val/student_f1" in metrics
    assert "val/teacher_f1" in metrics
