from typing import Any

import torch
import torch.nn as nn

from ugtsdti.core.registry import MODELS


@MODELS.register("baseline_teacher")
class BaselineTeacher(nn.Module):
    """
    Dummy/Baseline Teacher Model for Pipeline Validation.
    The Teacher branch canonically uses global Graph embeddings (transductive lookup).
    We simulate this by using nn.Embedding over the globally consistent hash IDs.
    """

    def __init__(
        self,
        num_drugs: int = 100003,  # Large prime used in modulo hashing
        num_targets: int = 100003,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Transductive Node Embeddings
        self.drug_emb = nn.Embedding(num_drugs, hidden_dim)
        self.target_emb = nn.Embedding(num_targets, hidden_dim)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        Forward pass expecting drug_index and target_index.
        """
        d_idx = batch["drug_index"].squeeze(-1)
        t_idx = batch["target_index"].squeeze(-1)

        d_features = self.drug_emb(d_idx)
        t_features = self.target_emb(t_idx)

        fused_features = torch.cat([d_features, t_features], dim=1)
        logits = self.fusion(fused_features)

        return {"logits": logits}
