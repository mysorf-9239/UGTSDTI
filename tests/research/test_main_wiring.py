from importlib import import_module

import torch
from omegaconf import OmegaConf

from ugtsdti.core.registry import LOSSES
from ugtsdti.main import (
    _inject_dataset_aware_model_params,
    build_experiment_components,
    infer_model_label,
    resolve_loss_cfg,
    resolve_model_cfg,
)


class _DummyTrainDataset:
    num_unique_drugs = 11
    num_unique_targets = 17


def test_inject_dataset_aware_model_params_updates_baseline_teacher_sizes():
    model_cfg = OmegaConf.create(
        {
            "name": "hybrid_dti",
            "params": {
                "student_cfg": {"name": "baseline_student", "params": {"hidden_dim": 32}},
                "teacher_cfg": {
                    "name": "baseline_teacher",
                    "params": {"hidden_dim": 32, "num_drugs": 1000, "num_targets": 1000},
                },
                "fusion_cfg": {"name": "ug_fusion", "params": {"gate_hidden": 8, "mc_samples": 3}},
            },
        }
    )

    model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
    _inject_dataset_aware_model_params(model_cfg, _DummyTrainDataset())

    teacher_params = model_cfg["params"]["teacher_cfg"]["params"]
    assert teacher_params["num_drugs"] == 11
    assert teacher_params["num_targets"] == 17


def test_loss_plugin_is_selected_from_cfg_trainer_loss():
    import_module("ugtsdti.losses")

    cfg = OmegaConf.create(
        {
            "trainer": {
                "params": {"epochs": 1},
                "loss": {"name": "kd", "params": {"alpha": 0.3}},
            }
        }
    )

    trainer_cfg = cfg.trainer.params if "params" in cfg.trainer else cfg.trainer
    loss_cfg = cfg.trainer.get("loss", trainer_cfg.get("loss", {"name": "bce"}))
    loss = LOSSES.build(loss_cfg)

    assert loss.__class__.__name__ == "KDLoss"
    assert loss.alpha == 0.3


def test_resolve_model_cfg_builds_slot_based_hybrid_config():
    cfg = OmegaConf.create(
        {
            "model": {"name": "hybrid_dti", "alias": "hybrid", "params": {}},
            "teacher": {"alias": "gcn", "name": "gcn_teacher", "params": {"hidden_dim": 64}},
            "student": {"alias": "baseline", "name": "baseline_student", "params": {"hidden_dim": 32}},
            "fusion": {"alias": "ug", "name": "ug_fusion", "params": {"gate_hidden": 8}},
        }
    )

    model_cfg = resolve_model_cfg(cfg)

    assert model_cfg["name"] == "hybrid_dti"
    assert model_cfg["params"]["teacher_cfg"]["name"] == "gcn_teacher"
    assert model_cfg["params"]["student_cfg"]["name"] == "baseline_student"
    assert model_cfg["params"]["fusion_cfg"]["name"] == "ug_fusion"


def test_resolve_loss_cfg_prefers_top_level_loss_group():
    cfg = OmegaConf.create(
        {
            "loss": {"name": "kd", "params": {"alpha": 0.7}},
            "trainer": {"params": {"epochs": 1}},
        }
    )

    loss_cfg = resolve_loss_cfg(cfg, cfg.trainer.params)

    assert loss_cfg["name"] == "kd"
    assert loss_cfg["params"]["alpha"] == 0.7


def test_resolve_loss_cfg_preserves_legacy_trainer_override_when_it_differs():
    cfg = OmegaConf.create(
        {
            "loss": {"name": "bce"},
            "trainer": {"params": {"epochs": 1}, "loss": {"name": "kd", "params": {"alpha": 0.4}}},
        }
    )

    loss_cfg = resolve_loss_cfg(cfg, cfg.trainer.params)

    assert loss_cfg["name"] == "kd"
    assert loss_cfg["params"]["alpha"] == 0.4


def test_infer_model_label_uses_slot_aliases_for_new_config_style():
    cfg = OmegaConf.create(
        {
            "model": {"name": "hybrid_dti", "alias": "hybrid", "params": {}},
            "teacher": {"alias": "gcn", "name": "gcn_teacher", "params": {}},
            "student": {"alias": "baseline", "name": "baseline_student", "params": {}},
            "fusion": {"alias": "ug", "name": "ug_fusion", "params": {}},
        }
    )

    assert infer_model_label(cfg) == "hybrid.gcn.baseline.ug"


def test_build_experiment_components_returns_optional_test_loader(monkeypatch):
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.num_unique_drugs = 5
            self.num_unique_targets = 7

        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return {
                "drug": torch.tensor([idx]),
                "target_ids": torch.tensor([idx]),
                "target_mask": torch.tensor([1]),
                "drug_index": torch.tensor([idx]),
                "target_index": torch.tensor([idx]),
                "label": torch.tensor([1.0]),
            }

    cfg = OmegaConf.create(
        {
            "data": {
                "train": {"name": "dummy_train"},
                "val": {"name": "dummy_val"},
                "test": {"name": "dummy_test"},
            },
            "trainer": {"params": {"batch_size": 2, "num_workers": 0}},
            "model": {
                "name": "hybrid_dti",
                "params": {
                    "teacher_cfg": {
                        "name": "baseline_teacher",
                        "params": {"hidden_dim": 8, "num_drugs": 1, "num_targets": 1},
                    }
                },
            },
        }
    )

    monkeypatch.setattr("ugtsdti.experiment.runtime.bootstrap_registries", lambda: None)
    monkeypatch.setattr("ugtsdti.experiment.runtime.DATASETS.build", lambda _: DummyDataset())
    monkeypatch.setattr("ugtsdti.experiment.runtime.MODELS.build", lambda cfg: {"built_from": cfg})

    components = build_experiment_components(cfg)
    train_loader = components.train_loader
    val_loader = components.val_loader
    test_loader = components.test_loader
    model = components.model

    assert len(train_loader.dataset) == 4
    assert len(val_loader.dataset) == 4
    assert len(test_loader.dataset) == 4
    teacher_params = model["built_from"]["params"]["teacher_cfg"]["params"]
    assert teacher_params["num_drugs"] == 5
    assert teacher_params["num_targets"] == 7
