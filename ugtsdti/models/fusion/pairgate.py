import torch
import torch.nn as nn

from ugtsdti.core.registry import MODELS


@MODELS.register("pairgate_fusion")
class PairGateFusion(nn.Module):
    """Uncertainty-gated fusion module (PairGate).

    Computes a soft gate weight ``α`` from the epistemic uncertainty pair
    ``(var_s, var_t)`` via a small MLP, then blends student and teacher logits:

        α = σ(MLP([var_s, var_t]))
        ŷ = α · logit_t + (1 − α) · logit_s

    When ``student_var`` / ``teacher_var`` are not provided (e.g., during
    ablation without MC-Dropout), falls back to a simple average.
    """

    def __init__(self, gate_hidden: int, mc_samples: int = 5, input_dim: int = 1):
        """
        Args:
            gate_hidden: Hidden dimension of the gate MLP.
            mc_samples: Number of MC-Dropout passes used upstream to estimate
                uncertainty. Stored here so ``HybridDTIModel`` can detect it
                via duck-typing.
            input_dim: Unused; kept for config backward-compatibility.
        """
        super().__init__()
        self.mc_samples = mc_samples

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
    ) -> torch.Tensor:
        """Fuse student and teacher predictions.

        Args:
            student_logits: Shape ``(B,)`` or ``(B, 1)``.
            teacher_logits: Shape ``(B,)`` or ``(B, 1)``.
            student_var: Epistemic variance of student. Shape ``(B,)``.
            teacher_var: Epistemic variance of teacher. Shape ``(B,)``.

        Returns:
            Fused logits of shape ``(B,)`` or ``(B, 1)``.
        """
        if student_var is None or teacher_var is None:
            # Fallback: equal-weight average (no uncertainty information)
            return 0.5 * student_logits + 0.5 * teacher_logits

        # uncertainty_pair: [B, 2]
        uncertainty_pair = torch.stack([student_var, teacher_var], dim=-1)
        gate_weight = self.gate_mlp(uncertainty_pair).squeeze(-1)  # α ∈ (0, 1)

        return gate_weight * teacher_logits + (1.0 - gate_weight) * student_logits
