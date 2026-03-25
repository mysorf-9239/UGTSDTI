import torch
import torch.nn as nn

from ugtsdti.core.registry import LOSSES


@LOSSES.register("kd")
class KDLoss(nn.Module):
    """Knowledge Distillation loss.

    Convex combination of:
    - Task loss: BCE between fused logits and true labels
    - Distillation loss: MSE between student and teacher logits

    When ``student_logits`` / ``teacher_logits`` are absent (e.g. ablation
    without a teacher), falls back to task loss only.

    Args:
        alpha: Weight of the distillation term. ``0.0`` = pure task loss,
            ``1.0`` = pure distillation loss.
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.task_loss = nn.BCEWithLogitsLoss()
        self.distill_loss = nn.MSELoss()

    def forward(self, model_outputs: dict, y_true: torch.Tensor) -> torch.Tensor:
        logits = model_outputs["logits"].view(-1)
        task = self.task_loss(logits, y_true.view(-1))

        s_logits = model_outputs.get("student_logits")
        t_logits = model_outputs.get("teacher_logits")
        if s_logits is not None and t_logits is not None:
            distill = self.distill_loss(s_logits.view(-1), t_logits.view(-1))
            return (1.0 - self.alpha) * task + self.alpha * distill

        return task
