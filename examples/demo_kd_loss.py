"""
Demo: KDDualLoss — task loss + knowledge distillation loss.

Usage:
    conda run -n ugtsdti python examples/demo_kd_loss.py
"""
import torch
from ugtsdti.losses.distillation import KDDualLoss

if __name__ == "__main__":
    B = 8
    y_true = torch.randint(0, 2, (B, 1)).float()

    loss_fn = KDDualLoss(alpha=0.5)

    # Hybrid output (student + teacher logits available → KD active)
    hybrid_out = {
        "logits": torch.randn(B, 1),
        "student_logits": torch.randn(B, 1),
        "teacher_logits": torch.randn(B, 1),
    }
    loss_hybrid = loss_fn(hybrid_out, y_true)
    print(f"Hybrid (KD active) loss: {loss_hybrid.item():.4f}")

    # Student-only output (no KD term → falls back to pure BCE)
    student_out = {"logits": torch.randn(B, 1)}
    loss_student = loss_fn(student_out, y_true)
    print(f"Student-only (BCE only) loss: {loss_student.item():.4f}")
