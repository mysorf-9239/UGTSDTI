"""
GCN-based Teacher Model for UGTSDTI (Phase 11).

Transductive encoder operating on global Drug-Drug and Protein-Protein
similarity graphs. Node embeddings are looked up by sequential index.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from ugtsdti.core.registry import MODELS


@MODELS.register("gcn_teacher")
class GCNTeacher(nn.Module):
    """GCN Teacher model for Drug-Target Interaction prediction.

    Uses two GCN encoders (one for drugs, one for proteins) operating on
    global similarity graphs. Node embeddings are looked up by index.

    Args:
        drug_feat_dim: Input feature dimension for drug nodes (default 2048).
        protein_feat_dim: Input feature dimension for protein nodes (default 8000).
        hidden_dim: Hidden/output embedding dimension (default 128).
        num_layers: Number of GCNConv layers (default 2).
        dropout: Dropout probability, clamped to >= 0.2 (default 0.3).
    """

    def __init__(
        self,
        drug_feat_dim: int = 2048,
        protein_feat_dim: int = 8000,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Clamp dropout to ensure MC-Dropout uncertainty is meaningful
        dropout = max(dropout, 0.2)

        self.drug_feat_dim = drug_feat_dim
        self.protein_feat_dim = protein_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Drug GCN encoder
        self.drug_convs = nn.ModuleList()
        in_dim = drug_feat_dim
        for _ in range(num_layers):
            self.drug_convs.append(GCNConv(in_dim, hidden_dim))
            in_dim = hidden_dim

        # Protein GCN encoder
        self.protein_convs = nn.ModuleList()
        in_dim = protein_feat_dim
        for _ in range(num_layers):
            self.protein_convs.append(GCNConv(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)

        # Predictor MLP
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Graphs are set via set_graphs()
        self.dd_graph: Data | None = None
        self.pp_graph: Data | None = None

    def set_graphs(self, dd_graph: Data, pp_graph: Data) -> None:
        """Store global DD and PP graphs for use in forward().

        Args:
            dd_graph: Drug-Drug similarity graph (PyG Data).
            pp_graph: Protein-Protein similarity graph (PyG Data).
        """
        self.dd_graph = dd_graph
        self.pp_graph = pp_graph

    def _encode_drug(self) -> torch.Tensor:
        """Run GCN over the full DD graph and return node embeddings."""
        x = self.dd_graph.x
        edge_index = self.dd_graph.edge_index
        for conv in self.drug_convs:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = self.drop(x)
        return x  # (n_drugs, hidden_dim)

    def _encode_protein(self) -> torch.Tensor:
        """Run GCN over the full PP graph and return node embeddings."""
        x = self.pp_graph.x
        edge_index = self.pp_graph.edge_index
        for conv in self.protein_convs:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = self.drop(x)
        return x  # (n_proteins, hidden_dim)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Forward pass: lookup embeddings by index, predict DTI.

        Args:
            batch: Dict with keys:
                - "drug_index": LongTensor (B, 1) or (B,)
                - "target_index": LongTensor (B, 1) or (B,)

        Returns:
            {"logits": FloatTensor (B,)}

        Raises:
            RuntimeError: If set_graphs() has not been called.
        """
        if self.dd_graph is None or self.pp_graph is None:
            raise RuntimeError(
                "Graphs not set. Call set_graphs(dd_graph, pp_graph) before forward."
            )

        drug_idx = batch["drug_index"].squeeze(-1)    # (B,)
        target_idx = batch["target_index"].squeeze(-1)  # (B,)

        # Encode full graphs
        drug_embs = self._encode_drug()      # (n_drugs, H)
        protein_embs = self._encode_protein()  # (n_proteins, H)

        # Lookup by index
        drug_emb = drug_embs[drug_idx]      # (B, H)
        prot_emb = protein_embs[target_idx]  # (B, H)

        # Concatenate and predict
        pair_emb = torch.cat([drug_emb, prot_emb], dim=1)  # (B, 2H)
        logits = self.predictor(pair_emb).squeeze(-1)        # (B,)

        return {"logits": logits}
