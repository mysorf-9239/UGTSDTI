import torch
import torch.nn as nn

from ugtsdti.core.registry import MODELS


@MODELS.register("pairgate_fusion")
class PairGateFusion(nn.Module):
    """
    Uncertainty-Gated Fusion mechanism utilizing PairGate concepts.
    """

    def __init__(self, input_dim: int, gate_hidden: int, mc_samples: int = 5):
        super().__init__()
        self.mc_samples = mc_samples

        # Very simple MLP to compute gating weight from uncertainties
        self.gate_mlp = nn.Sequential(nn.Linear(2, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 1), nn.Sigmoid())

    def forward(self, student_logits, teacher_logits, student_var=None, teacher_var=None):
        """
        Fuses predictions using uncertainties.
        Mock implementation for the architecture demo.
        """
        # If no variance mapping provided, just average
        if student_var is None or teacher_var is None:
            return 0.5 * student_logits + 0.5 * teacher_logits

        # [Batch, 2] -> Pair of variances
        var_pair = torch.stack([student_var, teacher_var], dim=-1)

        # [Batch, 1]
        gate_weight = self.gate_mlp(var_pair).squeeze(-1)

        # Adaptive combination
        fused = gate_weight * teacher_logits + (1 - gate_weight) * student_logits
        return fused
