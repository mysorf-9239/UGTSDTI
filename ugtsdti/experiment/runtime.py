"""Runtime orchestration for single UGTSDTI experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader

from ugtsdti.core.registry import DATASETS, LOSSES, MODELS
from ugtsdti.core.trainer import Trainer
from ugtsdti.data.protocols.audit import summarize_split_overlap
from ugtsdti.experiment.config import (
    bootstrap_registries,
    infer_model_label,
    inject_dataset_aware_model_params,
    resolve_loss_cfg,
    resolve_model_cfg,
)
from ugtsdti.utils.logger import setup_logger, setup_wandb
from ugtsdti.utils.seed import set_random_seed


@dataclass
class ExperimentComponents:
    """Concrete components needed to train and evaluate one experiment."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader | None
    model: torch.nn.Module


def build_experiment_components(cfg: DictConfig) -> ExperimentComponents:
    """Instantiate datasets, dataloaders, and model from registry using Hydra config."""
    bootstrap_registries()

    logger.info("Building Datasets from config...")
    if not hasattr(cfg.data, "train") or not hasattr(cfg.data, "val"):
        raise ValueError("Config Data must include `train` and `val` keys!")

    train_dataset = DATASETS.build(cfg.data.train)
    val_dataset = DATASETS.build(cfg.data.val)
    test_dataset = DATASETS.build(cfg.data.test) if hasattr(cfg.data, "test") and cfg.data.test is not None else None

    batch_size = cfg.trainer.params.get("batch_size", 32)
    num_workers = cfg.trainer.params.get("num_workers", 0)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = (
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if test_dataset is not None
        else None
    )

    logger.info("Building Model from config...")
    model_cfg = resolve_model_cfg(cfg)
    inject_dataset_aware_model_params(model_cfg, train_dataset)
    model = MODELS.build(model_cfg)
    logger.info(f"Model instantiated:\n{model.__class__.__name__}")

    return ExperimentComponents(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
    )


def wire_teacher_graphs(model, train_dataset, cfg: DictConfig) -> None:
    """Build and attach teacher graphs when the selected teacher is graph-based."""
    from ugtsdti.models.teacher.gcn_teacher import GCNTeacher

    teacher = getattr(model, "teacher", None) or (model if isinstance(model, GCNTeacher) else None)
    if teacher is None or not isinstance(teacher, GCNTeacher):
        return

    from ugtsdti.data.graph_builder import build_and_cache_graphs

    unique_smiles = getattr(train_dataset, "unique_smiles", None)
    unique_fasta = getattr(train_dataset, "unique_fasta", None)

    if unique_smiles is None or unique_fasta is None:
        logger.warning("train_dataset does not expose unique_smiles/unique_fasta. Cannot build GCNTeacher graphs.")
        return

    graph_cfg = cfg.get("trainer", {}).get("graph", {})
    cache_path = graph_cfg.get(
        "cache_path", getattr(train_dataset, "graph_cache_path", "data/cache/similarity_graphs.pt")
    )

    logger.info(f"Building DD/PP graphs for GCNTeacher ({len(unique_smiles)} drugs, {len(unique_fasta)} proteins)...")
    graphs = build_and_cache_graphs(unique_smiles, unique_fasta, cache_path=cache_path)
    teacher.set_graphs(graphs["dd"], graphs["pp"])
    logger.info("GCNTeacher graphs set successfully.")


def log_split_audit(train_dataset, eval_dataset, split_name: str) -> dict[str, Any]:
    """Compute and log overlap diagnostics between train and eval splits."""
    audit = summarize_split_overlap(train_dataset, eval_dataset, split_name=split_name)
    logger.info(
        f"Split audit [{split_name}] scenario={audit['scenario']} "
        f"drug_overlap={audit['drug_overlap_count']}/{audit['eval_drugs']} "
        f"({audit['drug_overlap_ratio']:.3f}) | "
        f"target_overlap={audit['target_overlap_count']}/{audit['eval_targets']} "
        f"({audit['target_overlap_ratio']:.3f})"
    )
    return audit


def numeric_prefixed(prefix: str, payload: dict[str, Any]) -> dict[str, float]:
    """Filter numeric values and prefix keys for logging/report output."""
    return {f"{prefix}/{k}": float(v) for k, v in payload.items() if isinstance(v, (int, float))}


def run_experiment(cfg: DictConfig) -> dict[str, Any]:
    """Run a single experiment and return a benchmark-friendly result dict."""
    setup_logger(log_level=cfg.get("log_level", "INFO"))
    run = setup_wandb(cfg)

    if cfg.get("seed") is not None:
        set_random_seed(cfg.seed, strict_cudnn=cfg.get("strict_cudnn", False))

    logger.info(f"Starting experiment: {cfg.get('run_name', 'default_run')}")
    logger.debug(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    components = build_experiment_components(cfg)
    train_loader = components.train_loader
    val_loader = components.val_loader
    test_loader = components.test_loader
    model = components.model

    train_dataset = train_loader.dataset
    wire_teacher_graphs(model, train_dataset, cfg)
    val_audit = log_split_audit(train_dataset, val_loader.dataset, split_name="val")
    test_audit = (
        log_split_audit(train_dataset, test_loader.dataset, split_name="test") if test_loader is not None else None
    )

    trainer_cfg = cfg.trainer.params if "params" in cfg.trainer else cfg.trainer
    device = torch.device(trainer_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    loss_cfg = resolve_loss_cfg(cfg, trainer_cfg)
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

    audit_val_metrics = numeric_prefixed("audit/val", val_audit)
    trainer._log_metrics({}, audit_val_metrics)
    audit_test_metrics = numeric_prefixed("audit/test", test_audit or {})
    if audit_test_metrics:
        trainer._log_metrics({}, audit_test_metrics)

    final_metrics: dict[str, Any] = {
        "run_name": cfg.get("run_name", "default_run"),
        "seed": int(cfg.get("seed", -1)),
        "data_config": getattr(cfg, "data_name", "unknown"),
        "model_config": infer_model_label(cfg),
        "loss_name": loss_cfg.get("name", "unknown"),
        **audit_val_metrics,
        **audit_test_metrics,
    }

    if val_loader is not None:
        trainer.load_best_checkpoint()
        logger.info("Running final evaluation on validation split with best checkpoint...")
        val_metrics = trainer.evaluate(val_loader, prefix="val_best")
        trainer._log_metrics({}, val_metrics)
        final_metrics.update(val_metrics)

    if test_loader is not None:
        if val_loader is None:
            trainer.load_best_checkpoint()
        logger.info("Running final evaluation on test split...")
        test_metrics = trainer.evaluate(test_loader, prefix="test")
        trainer._log_metrics({}, test_metrics)
        final_metrics.update(test_metrics)

    logger.info("System bootstrap completed successfully.")
    if run is not None:
        run.finish()

    return final_metrics
