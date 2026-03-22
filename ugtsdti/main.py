from importlib import import_module

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from ugtsdti.core.registry import MODELS
from ugtsdti.utils.logger import setup_logger, setup_wandb
from ugtsdti.utils.seed import make_reproducible


def bootstrap_registries() -> None:
    """Import plugin packages so registry decorators execute before build()."""
    import_module("ugtsdti.models")
    import_module("ugtsdti.data")


def build_system(cfg: DictConfig):
    """Instantiate datasets, dataloaders, and models using the Registry."""
    bootstrap_registries()

    logger.info("Building Datasets from config...")
    # train_dataset = DATASETS.build(cfg.data.train)
    # val_dataset = DATASETS.build(cfg.data.val)
    if cfg.get("data") is not None:
        logger.debug(f"Dataset config selected: {cfg.data.get('name', 'unknown')}")

    logger.info("Building Model from config...")
    # Because of modularity, the top-level model handles routing
    model = MODELS.build(cfg.model)
    logger.info(f"Model instantiated:\n{model}")

    return model


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    # 1. Setup global logger (Loguru) and WandB
    setup_logger(log_level=cfg.get("log_level", "INFO"))
    run = setup_wandb(cfg)

    # 2. Setup reproducibility
    if cfg.get("seed") is not None:
        make_reproducible(cfg.seed, strict_cudnn=cfg.get("strict_cudnn", False))

    logger.info(f"Starting experiment: {cfg.get('run_name', 'default_run')}")
    # Print a beautiful representation of the config
    logger.debug(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # 3. Build components from standard registries
    model = build_system(cfg)

    # 4. Initialize Trainer and run
    # trainer = Trainer(model, cfg.trainer)
    # trainer.fit(...)

    logger.info(f"System bootstrap completed with model: {model.__class__.__name__}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
