"""Uncertainty-Gated (UG) fusion module for UGTS-DTI.

This module implements the core fusion strategy of the UGTS-DTI architecture:
blending Teacher and Student predictions using MC-Dropout epistemic uncertainty
as a soft gate signal.

Architecture:
    α = σ(MLP([var_student, var_teacher]))   # gate weight ∈ (0, 1)
    ŷ = α · logit_teacher + (1 − α) · logit_student

When Teacher is confident (low var_teacher) and Student is uncertain
(high var_student), α → 1 → Teacher dominates (warm-start / S1 split).
When Teacher is uncertain (cold-start / S4 split), α → 0 → Student dominates.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ugtsdti.core.registry import MODELS


@MODELS.register("ug_fusion")
class UncertaintyGatedFusion(nn.Module):
    """Uncertainty-Gated fusion module (UG).

    Computes a soft gate weight ``α`` from the MC-Dropout epistemic uncertainty
    pair ``(var_student, var_teacher)`` via a small MLP, then blends Student
    and Teacher logits adaptively per sample:

        α = σ(MLP([var_s, var_t]))
        ŷ = α · logit_teacher + (1 − α) · logit_student

    When ``student_var`` / ``teacher_var`` are not provided (e.g., ablation
    without MC-Dropout), falls back to a simple equal-weight average.

    Args:
        gate_hidden: Hidden dimension of the gate MLP.
        mc_samples: Backward-compatible default MC sample count used for both
            train and eval when split-specific values are not provided.
        train_mc_samples: Optional MC sample count for train-time uncertainty.
        eval_mc_samples: Optional MC sample count for eval-time uncertainty.
        input_dim: Unused; kept for config backward-compatibility.
    """

    def __init__(
        self,
        gate_hidden: int,
        mc_samples: int = 5,
        train_mc_samples: int | None = None,
        eval_mc_samples: int | None = None,
        input_dim: int = 1,
    ):
        super().__init__()
        self.mc_samples = mc_samples
        self.train_mc_samples = mc_samples if train_mc_samples is None else train_mc_samples
        self.eval_mc_samples = mc_samples if eval_mc_samples is None else eval_mc_samples
        self.use_uncertainty_in_train = self.train_mc_samples > 0

        # Gate MLP: (var_s, var_t) → α ∈ (0, 1)
        self.gate_mlp = nn.Sequential(
            nn.Linear(2, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_var: torch.Tensor | None = None,
        teacher_var: torch.Tensor | None = None,
        return_alpha: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Fuse Student and Teacher predictions via uncertainty-gated blending.

        Args:
            student_logits: Shape ``(B,)``.
            teacher_logits: Shape ``(B,)``.
            student_var: MC-Dropout epistemic variance of Student. Shape ``(B,)``.
            teacher_var: MC-Dropout epistemic variance of Teacher. Shape ``(B,)``.

        Returns:
            Fused logits, same shape as inputs.
        """
        if student_var is None or teacher_var is None:
            # Fallback: equal-weight average (no uncertainty information)
            alpha = torch.full_like(student_logits.view(-1), 0.5)
            fused = (0.5 * student_logits + 0.5 * teacher_logits).view(-1)
            return (fused, alpha) if return_alpha else fused

        # Stack uncertainty pair → gate MLP → α
        uncertainty_pair = torch.stack([student_var, teacher_var], dim=-1)  # (B, 2)
        alpha = self.gate_mlp(uncertainty_pair).view(-1)  # (B,), α ∈ (0, 1)

        fused = alpha * teacher_logits + (1.0 - alpha) * student_logits
        fused = fused.view(-1)  # ensure (B,) even when B=1
        return (fused, alpha) if return_alpha else fused
