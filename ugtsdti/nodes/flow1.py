"""Flow 1 graph node runtimes: SeqBiLSTMEncoderRuntime, GNNDrugEncoderRuntime, DenseHeadRuntime.

REQ-FLOW1-001, REQ-FLOW1-002, REQ-FLOW1-003
"""

from __future__ import annotations

from typing import Any

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.errors import InvalidConfigError, MissingDependencyError
from ugtsdti.nodes.baseline import (
    _require_torch,
    _to_token_ids,
    _TorchRuntime,
)


class SeqBiLSTMEncoderRuntime(_TorchRuntime):
    """Sequence encoder: Embedding → Conv1d×2 → BiLSTM → mean pool → projection.

    Handles both drug sequences (vocab_size=64) and protein sequences (vocab_size=32)
    depending on the params supplied at construction time.

    REQ-FLOW1-001
    """

    def __init__(
        self,
        *,
        vocab_size: int = 64,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        kernel_size_1: int = 3,
        kernel_size_2: int = 5,
        lstm_dim: int = 64,
        proj_dim: int = 64,
    ) -> None:
        torch, nn = _require_torch()
        self._vocab_size = int(vocab_size)
        self._proj_dim = int(proj_dim)

        self.embedding = nn.Embedding(self._vocab_size, int(embedding_dim))

        pad1 = max(0, int(kernel_size_1) // 2)
        pad2 = max(0, int(kernel_size_2) // 2)
        self.conv1 = nn.Conv1d(int(embedding_dim), int(hidden_dim), kernel_size=int(kernel_size_1), padding=pad1)
        self.conv2 = nn.Conv1d(int(hidden_dim), int(hidden_dim), kernel_size=int(kernel_size_2), padding=pad2)
        self.relu = nn.ReLU()

        # bidirectional=True → output dim = 2 * lstm_dim
        self.bilstm = nn.LSTM(
            int(hidden_dim),
            int(lstm_dim),
            bidirectional=True,
            batch_first=True,
        )
        self.projection = nn.Linear(2 * int(lstm_dim), int(proj_dim))

    def forward(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        del context
        torch, _ = _require_torch()

        # Single declared input — assert exactly one key is present
        assert len(inputs) == 1, f"SeqBiLSTMEncoderRuntime expects exactly 1 declared input, got {list(inputs.keys())}"
        x = next(iter(inputs.values()))

        # Validate rank
        raw = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        if raw.ndim != 2:
            raise InvalidConfigError(
                f"SeqBiLSTMEncoderRuntime expects rank-2 input (B, L), got shape {tuple(raw.shape)}.",
                stage="graph",
                component="SeqBiLSTMEncoderRuntime",
            )

        tokens = _to_token_ids(x, vocab_size=self._vocab_size)  # (B, L) long, clamped

        emb = self.embedding(tokens)  # (B, L, E)
        emb = emb.transpose(1, 2)  # (B, E, L)
        emb = self.relu(self.conv1(emb))  # (B, H, L)
        emb = self.relu(self.conv2(emb))  # (B, H, L)
        emb = emb.transpose(1, 2)  # (B, L, H)

        lstm_out, _ = self.bilstm(emb)  # (B, L, 2*lstm_dim)
        pooled = lstm_out.mean(dim=1)  # (B, 2*lstm_dim)
        out = self.projection(pooled)  # (B, proj_dim)

        return {"embedding": out}


class GNNDrugEncoderRuntime(_TorchRuntime):
    """GCN-style drug graph encoder: 2 GCN blocks → mean pool → projection.

    No torch_geometric dependency — uses only PyTorch core bmm.

    REQ-FLOW1-002
    """

    def __init__(
        self,
        *,
        node_feat_dim: int = 9,
        hidden_dim: int = 64,
        proj_dim: int = 64,
        normalize_adj: bool = True,
        allow_missing_input: bool = False,
    ) -> None:
        _, nn = _require_torch()
        self._proj_dim = int(proj_dim)
        self._normalize_adj = bool(normalize_adj)
        self._allow_missing_input = bool(allow_missing_input)

        self.linear1 = nn.Linear(int(node_feat_dim), int(hidden_dim))
        self.linear2 = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.relu = nn.ReLU()
        self.projection = nn.Linear(int(hidden_dim), int(proj_dim))

    def _infer_batch_size(self, inputs: dict[str, Any], context: ExecutionContext) -> int:
        """Infer batch size from context or any tensor present in inputs."""
        if context.batch_size > 1:
            return int(context.batch_size)
        torch, _ = _require_torch()
        for v in inputs.values():
            if isinstance(v, torch.Tensor):
                return int(v.shape[0])
        return int(context.batch_size)  # fallback to context value (default 1)

    def forward(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        torch, _ = _require_torch()

        drug_graph = inputs.get("drug_graph")

        if drug_graph is None:
            if self._allow_missing_input:
                B = self._infer_batch_size(inputs, context)
                return {"embedding": torch.zeros(B, self._proj_dim)}
            raise MissingDependencyError(
                "drug_graph is missing from inputs and allow_missing_input=False.",
                stage="graph",
                component="GNNDrugEncoderRuntime",
                key="drug_graph",
            )

        adj = drug_graph["adj"]  # (B, N, N)
        node_feat = drug_graph["node_feat"]  # (B, N, F)

        if not isinstance(adj, torch.Tensor):
            adj = torch.as_tensor(adj, dtype=torch.float32)
        if not isinstance(node_feat, torch.Tensor):
            node_feat = torch.as_tensor(node_feat, dtype=torch.float32)

        if adj.ndim != 3:
            raise InvalidConfigError(
                f"GNNDrugEncoderRuntime expects adj rank 3 (B,N,N), got shape {tuple(adj.shape)}.",
                stage="graph",
                component="GNNDrugEncoderRuntime",
                key="adj",
            )
        if node_feat.ndim != 3:
            raise InvalidConfigError(
                f"GNNDrugEncoderRuntime expects node_feat rank 3 (B,N,F), got shape {tuple(node_feat.shape)}.",
                stage="graph",
                component="GNNDrugEncoderRuntime",
                key="node_feat",
            )

        adj = adj.to(dtype=torch.float32)
        node_feat = node_feat.to(dtype=torch.float32)

        if self._normalize_adj:
            deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)  # (B, N, 1)
            adj_norm = adj / deg
        else:
            adj_norm = adj

        # GCN block 1
        h = self.linear1(node_feat)  # (B, N, H)
        h = torch.bmm(adj_norm, h)  # (B, N, H)
        h = self.relu(h)

        # GCN block 2
        h = self.linear2(h)  # (B, N, H)
        h = torch.bmm(adj_norm, h)  # (B, N, H)
        h = self.relu(h)

        pooled = h.mean(dim=1)  # (B, H)
        out = self.projection(pooled)  # (B, proj_dim)

        return {"embedding": out}


class DenseHeadRuntime(_TorchRuntime):
    """Two-layer dense MLP head producing raw logits (no sigmoid).

    REQ-FLOW1-003
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        output_dim: int = 1,
        input_dim: int | None = None,
    ) -> None:
        _, nn = _require_torch()
        if input_dim is None:
            self.hidden = nn.LazyLinear(int(hidden_dim))
        else:
            self.hidden = nn.Linear(int(input_dim), int(hidden_dim))
        self.relu = nn.ReLU()
        self.output = nn.Linear(int(hidden_dim), int(output_dim))

    def forward(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        del context
        torch, _ = _require_torch()

        # Single declared input
        assert len(inputs) == 1, f"DenseHeadRuntime expects exactly 1 declared input, got {list(inputs.keys())}"
        x = next(iter(inputs.values()))

        tensor = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
        if tensor.ndim != 2:
            raise InvalidConfigError(
                f"DenseHeadRuntime expects rank-2 input (B, D), got shape {tuple(tensor.shape)}.",
                stage="graph",
                component="DenseHeadRuntime",
            )

        tensor = tensor.to(dtype=torch.float32)
        hidden = self.relu(self.hidden(tensor))  # (B, hidden_dim)
        logits = self.output(hidden)  # (B, output_dim)

        return {"logits": logits}
