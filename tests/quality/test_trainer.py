import os

import torch
from torch.utils.data import DataLoader, Dataset

from ugtsdti.core.trainer import Trainer


class _TinyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {
            "feature": torch.tensor([float(idx)]),
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
