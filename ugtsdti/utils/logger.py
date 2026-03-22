import sys
from typing import Any, cast

from loguru import logger
from omegaconf import DictConfig, OmegaConf


def setup_logger(log_level: str = "INFO", log_file: str = None):
    """
    Configure standard Loguru logger behavior globally.
    """
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
    )
    if log_file:
        logger.add(log_file, rotation="10 MB", level=log_level)

    return logger


def setup_wandb(cfg: DictConfig, project_name: str = "UGTSDTI"):
    """
    Initializes Weights & Biases for experiment tracking.
    Safely converts Hydra DictConfig to native Python dict for WandB storage.
    """
    try:
        import wandb
    except ImportError:
        logger.warning("WandB is not installed! Skipping remote logging. Run `pip install wandb`.")
        return None

    if not cfg.get("logging") or not cfg.logging.get("wandb_enabled", False):
        logger.info("WandB logging disabled via config.")
        return None

    run_name = cfg.get("run_name", "experiment")
    # Convert OmegaConf to primitive dict so wandb can flatten and display it
    config_dict = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))

    run = wandb.init(
        project=project_name, name=run_name, config=config_dict, dir=cfg.get("output_dir", "./output"), reinit=True
    )
    logger.info(f"WandB initialized successfully: Run [{run.name}]")
    return run
