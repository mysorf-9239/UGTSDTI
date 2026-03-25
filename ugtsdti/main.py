"""Thin training CLI for slot-based UGTSDTI experiments.

Runtime orchestration lives in :mod:`ugtsdti.experiment`. This module remains a
small Hydra entry point plus a compatibility export surface.
"""

import hydra
from omegaconf import DictConfig

from ugtsdti.experiment import bootstrap_registries, build_experiment_components, infer_model_label
from ugtsdti.experiment import inject_dataset_aware_model_params as _inject_dataset_aware_model_params
from ugtsdti.experiment import resolve_loss_cfg, resolve_model_cfg, run_experiment

__all__ = [
    "bootstrap_registries",
    "build_experiment_components",
    "infer_model_label",
    "resolve_loss_cfg",
    "resolve_model_cfg",
    "run_experiment",
    "_inject_dataset_aware_model_params",
]


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Compose Hydra config groups and execute a single experiment."""
    run_experiment(cfg)


if __name__ == "__main__":
    main()
