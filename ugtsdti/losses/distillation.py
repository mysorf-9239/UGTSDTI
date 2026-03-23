import torch
import torch.nn as nn

from ugtsdti.core.registry import LOSSES


@LOSSES.register("kd_dual_loss")
class KDDualLoss(nn.Module):
    """
    Knowledge Distillation Loss that computes a convex combination of:
    1. Task Loss (BCE) between Fusion Logits and True Labels
    2. Distillation Loss (MSE) between Student Logits and Teacher Logits
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.task_loss = nn.BCEWithLogitsLoss()
        self.kd_loss = nn.MSELoss()

    def forward(self, model_outputs: dict, y_true: torch.Tensor) -> torch.Tensor:
        task_loss = self.task_loss(model_outputs["logits"], y_true)

        if "student_logits" in model_outputs and "teacher_logits" in model_outputs:
            kd_loss = self.kd_loss(model_outputs["student_logits"], model_outputs["teacher_logits"])
            return (1.0 - self.alpha) * task_loss + self.alpha * kd_loss

        return task_loss
