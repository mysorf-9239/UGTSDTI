from importlib import import_module

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader

from ugtsdti.core.registry import DATASETS, LOSSES, MODELS
from ugtsdti.core.trainer import Trainer
from ugtsdti.utils.logger import setup_logger, setup_wandb
from ugtsdti.utils.seed import set_random_seed


def bootstrap_registries() -> None:
    """Import plugin packages so registry decorators execute before build()."""
    import_module("ugtsdti.models")
    import_module("ugtsdti.data")
    import_module("ugtsdti.losses")


def build_experiment_components(cfg: DictConfig):
    """Instantiate datasets, dataloaders, and model from registry using Hydra config."""
    bootstrap_registries()

    logger.info("Building Datasets from config...")
    if not hasattr(cfg.data, "train") or not hasattr(cfg.data, "val"):
        raise ValueError("Config Data must include `train` and `val` keys!")

    train_dataset = DATASETS.build(cfg.data.train)
    val_dataset = DATASETS.build(cfg.data.val)

    batch_size = cfg.trainer.get("batch_size", 32)
    num_workers = cfg.trainer.get("num_workers", 0)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    logger.info("Building Model from config...")
    model = MODELS.build(cfg.model)
    logger.info(f"Model instantiated:\n{model.__class__.__name__}")

    return train_loader, val_loader, model


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    # 1. Setup global logger (Loguru) and WandB
    setup_logger(log_level=cfg.get("log_level", "INFO"))
    run = setup_wandb(cfg)

    # 2. Setup reproducibility
    if cfg.get("seed") is not None:
        set_random_seed(cfg.seed, strict_cudnn=cfg.get("strict_cudnn", False))

    logger.info(f"Starting experiment: {cfg.get('run_name', 'default_run')}")
    logger.debug(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # 3. Build components from standard registries
    train_loader, val_loader, model = build_experiment_components(cfg)

    # 4. Initialize Trainer and run
    trainer_cfg = cfg.trainer.params if "params" in cfg.trainer else cfg.trainer
    device = torch.device(trainer_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    loss_cfg = trainer_cfg.get("loss", {"name": "bce_with_logits"})
    loss_fn = LOSSES.build(loss_cfg)

    optimizer = torch.optim.Adam(model.parameters(), lr=trainer_cfg.get("lr", 1e-3))

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        cfg_trainer=trainer_cfg,
        run=run,
    )

    logger.info("Handing off to Trainer...")
    trainer.fit(train_loader, val_loader)

    logger.info("System bootstrap completed successfully.")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
