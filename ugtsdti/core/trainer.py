import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import wandb
from loguru import logger
from tqdm import tqdm

from ugtsdti.core.metrics import compute_dti_metrics


def batch_to_device(batch, device):
    """Recursively move tensors and dictionaries to the specified device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    elif hasattr(batch, "to"):
        # For PyG Data/Batch objects
        return batch.to(device)
    elif isinstance(batch, list):
        return [batch_to_device(v, device) for v in batch]
    return batch


class Trainer:
    """
    A professional, boilerplate-free training loop for UGTSDTI.
    Handles epochs, early stopping, checkpoint saving, and metric logging.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        cfg_trainer: Dict[str, Any],
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        run=None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device
        self.run = run

        # Hyperparameters
        self.epochs = cfg_trainer.get("epochs", 100)
        self.patience = cfg_trainer.get("patience", 10)
        self.val_check_interval = cfg_trainer.get("val_check_interval", 1)
        self.grad_clip = cfg_trainer.get("grad_clip", 1.0)

        # Paths
        self.output_dir = cfg_trainer.get("output_dir", "./outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.output_dir, "best_model.pt")

        # State
        self.best_metric = -float("inf")
        self.no_improve_epochs = 0
        self.current_epoch = 0

    def fit(self, train_loader, val_loader=None):
        logger.info(f"Starting Training for {self.epochs} epochs.")

        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch

            # 1. Train Pulse
            train_metrics = self._train_epoch(train_loader)

            # 2. Validation Pulse
            val_metrics = {}
            if val_loader is not None and epoch % self.val_check_interval == 0:
                val_metrics = self.evaluate(val_loader, prefix="val")
                self._check_early_stopping(val_metrics)

            # 3. Log to WandB
            self._log_metrics(train_metrics, val_metrics)

            if self.no_improve_epochs >= self.patience:
                logger.info(f"Early Stopping triggered at epoch {epoch}!")
                break

        logger.info(f"Training Complete. Best Validation Metric: {self.best_metric:.4f}")
        return self.best_metric

    def _train_epoch(self, loader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {self.current_epoch}/{self.epochs} [Train]")
        for batch in pbar:
            batch = batch_to_device(batch, self.device)
            y_true = batch.pop("label").float()
            inputs = batch

            self.optimizer.zero_grad()
            y_prob_dict = self.model(inputs)

            loss = self.loss_fn(y_prob_dict, y_true)
            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            total_loss += loss.item() * y_true.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if self.scheduler is not None:
            self.scheduler.step()

        return {"train/loss": total_loss / len(loader.dataset)}

    @torch.no_grad()
    def evaluate(self, loader, prefix="val") -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_trues = []
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {self.current_epoch} [{prefix.capitalize()}]")
        for batch in pbar:
            batch = batch_to_device(batch, self.device)
            y_true = batch.pop("label").float()
            inputs = batch

            y_prob_dict = self.model(inputs)
            loss = self.loss_fn(y_prob_dict, y_true)

            total_loss += loss.item() * y_true.size(0)

            # For AUROC/metrics, we specifically extract the main predictions
            all_preds.append(y_prob_dict["logits"].detach().cpu().numpy())
            all_trues.append(y_true.cpu().numpy())

        y_prob_full = np.concatenate(all_preds)
        y_true_full = np.concatenate(all_trues)

        # Calculate DTI metrics
        metrics = compute_dti_metrics(y_true_full, y_prob_full)
        metrics["loss"] = total_loss / len(loader.dataset)

        # Format keys for logging
        result = {f"{prefix}/{k}": v for k, v in metrics.items()}

        log_str = " | ".join([f"{k}: {v:.4f}" for k, v in result.items()])
        logger.info(f"Epoch {self.current_epoch} Metrics: {log_str}")
        return result

    def _check_early_stopping(self, val_metrics: Dict[str, float]):
        # Target metric is usually Validation MSE (lower is better) or AUROC (higher is better).
        # We assume AUROC for now.
        target_metric = val_metrics.get("val/auroc", 0.0)

        if target_metric > self.best_metric:
            self.best_metric = target_metric
            self.no_improve_epochs = 0
            self._save_checkpoint()
            logger.info(f"New best model saved! (AUROC: {self.best_metric:.4f})")
        else:
            self.no_improve_epochs += 1

    def _save_checkpoint(self):
        torch.save(self.model.state_dict(), self.best_model_path)

    def _log_metrics(self, train_metrics, val_metrics):
        if self.run is not None:
            log_dict = {**train_metrics, **val_metrics, "epoch": self.current_epoch}
            if self.scheduler is not None:
                log_dict["lr"] = self.scheduler.get_last_lr()[0]
            wandb.log(log_dict, step=self.current_epoch)
