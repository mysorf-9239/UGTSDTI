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
    """Recursively move a batch payload to the target device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    elif hasattr(batch, "to"):
        # Covers PyG Data and Batch objects.
        return batch.to(device)
    elif isinstance(batch, list):
        return [batch_to_device(v, device) for v in batch]
    return batch


class Trainer:
    """Training and evaluation loop for UGTSDTI experiments.

    Beyond standard fit/evaluate behavior, the trainer records branch-wise
    metrics and uncertainty diagnostics so hybrid runs remain analyzable across
    cold-start scenarios.
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

    @staticmethod
    def _summarize_array(values: list[np.ndarray], prefix: str) -> dict[str, float]:
        """Return summary statistics for a list of per-batch arrays."""
        if not values:
            return {}
        merged = np.concatenate(values).astype(np.float64)
        return {
            f"{prefix}_mean": float(merged.mean()),
            f"{prefix}_std": float(merged.std()),
            f"{prefix}_p50": float(np.percentile(merged, 50)),
            f"{prefix}_p90": float(np.percentile(merged, 90)),
        }

    @staticmethod
    def _compute_branch_metrics(
        y_true: np.ndarray,
        logits_list: list[np.ndarray],
        branch_name: str,
    ) -> dict[str, float]:
        """Compute metrics for an auxiliary branch when logits are available."""
        if not logits_list:
            return {}
        branch_logits = np.concatenate(logits_list)
        branch_metrics = compute_dti_metrics(y_true, y_score=branch_logits)
        return {f"{branch_name}_{key}": value for key, value in branch_metrics.items()}

    def fit(self, train_loader, val_loader=None):
        """Train for at most ``epochs`` and return the best validation AUROC."""
        logger.info(f"Starting training for {self.epochs} epochs.")

        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch

            train_metrics = self._train_epoch(train_loader)

            val_metrics = {}
            if val_loader is not None and epoch % self.val_check_interval == 0:
                val_metrics = self.evaluate(val_loader, prefix="val")
                self._check_early_stopping(val_metrics)

            self._log_metrics(train_metrics, val_metrics)

            if self.no_improve_epochs >= self.patience:
                logger.info(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs).")
                break

        if val_loader is None and not os.path.exists(self.best_model_path):
            self._save_checkpoint()

        logger.info(f"Training complete. Best val/auroc: {self.best_metric:.4f}")
        return self.best_metric

    def _train_epoch(self, loader) -> Dict[str, float]:
        """Run one optimization epoch and return aggregate training metrics."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {self.current_epoch}/{self.epochs} [Train]")
        for batch in pbar:
            batch = batch_to_device(batch, self.device)
            y_true = batch.pop("label").float()

            self.optimizer.zero_grad()
            model_output = self.model(batch)

            loss = self.loss_fn(model_output, y_true)
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
        """Evaluate one split and return prefixed scalar metrics.

        Besides fused metrics, the method emits branch-wise scores, uncertainty
        summaries, and UNK coverage rates when the model/dataset expose them.
        """
        self.model.eval()
        all_preds = []
        all_trues = []
        total_loss = 0.0
        gate_alphas = []
        student_vars = []
        teacher_vars = []
        student_logits = []
        teacher_logits = []
        drug_indices = []
        target_indices = []

        pbar = tqdm(loader, desc=f"Epoch {self.current_epoch} [{prefix.capitalize()}]")
        for batch in pbar:
            batch = batch_to_device(batch, self.device)
            y_true = batch.pop("label").float()

            model_output = self.model(batch)
            loss = self.loss_fn(model_output, y_true)

            total_loss += loss.item() * y_true.size(0)
            all_preds.append(model_output["logits"].detach().cpu().numpy())
            all_trues.append(y_true.cpu().numpy())
            if model_output.get("gate_alpha") is not None:
                gate_alphas.append(model_output["gate_alpha"].detach().cpu().numpy())
            if model_output.get("student_var") is not None:
                student_vars.append(model_output["student_var"].detach().cpu().numpy())
            if model_output.get("teacher_var") is not None:
                teacher_vars.append(model_output["teacher_var"].detach().cpu().numpy())
            if model_output.get("student_logits") is not None:
                student_logits.append(model_output["student_logits"].detach().cpu().numpy())
            if model_output.get("teacher_logits") is not None:
                teacher_logits.append(model_output["teacher_logits"].detach().cpu().numpy())
            if "drug_index" in batch:
                drug_indices.append(batch["drug_index"].detach().cpu().numpy().reshape(-1))
            if "target_index" in batch:
                target_indices.append(batch["target_index"].detach().cpu().numpy().reshape(-1))

        y_score_full = np.concatenate(all_preds)
        y_true_full = np.concatenate(all_trues)

        # Compute fused and auxiliary research metrics.
        metrics = compute_dti_metrics(y_true_full, y_score=y_score_full)
        metrics["loss"] = total_loss / len(loader.dataset)
        metrics.update(self._summarize_array(gate_alphas, "gate_alpha"))
        metrics.update(self._summarize_array(student_vars, "student_var"))
        metrics.update(self._summarize_array(teacher_vars, "teacher_var"))
        metrics.update(self._compute_branch_metrics(y_true_full, student_logits, "student"))
        metrics.update(self._compute_branch_metrics(y_true_full, teacher_logits, "teacher"))

        unk_drug_index = getattr(loader.dataset, "unk_drug_index", None)
        unk_target_index = getattr(loader.dataset, "unk_target_index", None)
        if drug_indices and unk_drug_index is not None:
            merged_drug_indices = np.concatenate(drug_indices)
            metrics["unk_drug_rate"] = float((merged_drug_indices == unk_drug_index).mean())
        if target_indices and unk_target_index is not None:
            merged_target_indices = np.concatenate(target_indices)
            metrics["unk_target_rate"] = float((merged_target_indices == unk_target_index).mean())

        # Prefix keys so callers can safely merge multiple split payloads.
        result = {f"{prefix}/{k}": v for k, v in metrics.items()}

        log_str = " | ".join([f"{k}: {v:.4f}" for k, v in result.items()])
        logger.info(f"Epoch {self.current_epoch} Metrics: {log_str}")
        return result

    def _check_early_stopping(self, val_metrics: Dict[str, float]):
        """Track the best validation AUROC and update early-stopping state."""
        target_metric = val_metrics.get("val/auroc", 0.0)

        if target_metric > self.best_metric:
            self.best_metric = target_metric
            self.no_improve_epochs = 0
            self._save_checkpoint()
            logger.info(f"New best model saved! (AUROC: {self.best_metric:.4f})")
        else:
            self.no_improve_epochs += 1

    def _save_checkpoint(self):
        """Persist the current model weights as the best-known checkpoint."""
        torch.save(self.model.state_dict(), self.best_model_path)

    def load_best_checkpoint(self) -> bool:
        """Load the best checkpoint if it exists."""
        if not os.path.exists(self.best_model_path):
            logger.warning(f"Best checkpoint not found at {self.best_model_path}")
            return False

        state_dict = torch.load(self.best_model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict)
        logger.info(f"Loaded best checkpoint from {self.best_model_path}")
        return True

    def _log_metrics(self, train_metrics, val_metrics):
        """Forward metrics to WandB when a run is active."""
        if self.run is not None:
            log_dict = {**train_metrics, **val_metrics, "epoch": self.current_epoch}
            if self.scheduler is not None:
                log_dict["lr"] = self.scheduler.get_last_lr()[0]
            wandb.log(log_dict, step=self.current_epoch)
