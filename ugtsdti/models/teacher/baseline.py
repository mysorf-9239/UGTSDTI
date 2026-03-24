from __future__ import annotations

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
        """Forward pass using transductive node indices.

        Args:
            batch: Must contain ``drug_index`` and ``target_index`` (LongTensor [B]).
        """
        drug_idx = batch["drug_index"].squeeze(-1)
        target_idx = batch["target_index"].squeeze(-1)

        drug_emb = self.drug_emb(drug_idx)  # [B, hidden_dim]
        target_emb = self.target_emb(target_idx)  # [B, hidden_dim]

        pair_emb = torch.cat([drug_emb, target_emb], dim=1)  # [B, hidden_dim * 2]
        logits = self.fusion(pair_emb)

        return {"logits": logits}
