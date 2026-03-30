"""Trainable baseline graph node runtimes."""

from __future__ import annotations

from typing import Any

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.errors import InvalidConfigError
from ugtsdti.nodes.base import NodeRuntime


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - torch-backed runtimes require torch
        raise RuntimeError("Baseline node runtimes require torch to be installed.") from exc
    return torch, nn


def _to_token_ids(value: Any, *, vocab_size: int) -> Any:
    torch, _ = _require_torch()

    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    else:
        tensor = torch.as_tensor(value)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise InvalidConfigError(
            f"Expected token input with rank 2 (B, L), got shape {tuple(tensor.shape)}.",
            stage="graph",
            component="baseline_nodes",
        )
    tensor = tensor.to(dtype=torch.long)
    return torch.remainder(tensor, max(1, int(vocab_size)))


class _TorchRuntime(NodeRuntime):
    """Common helpers for torch-backed node runtimes."""

    def parameters(self) -> list[Any]:
        torch, nn = _require_torch()
        parameters: list[Any] = []
        for value in self.__dict__.values():
            if isinstance(value, nn.Module):
                parameters.extend(list(value.parameters()))
            elif isinstance(value, torch.nn.Parameter):
                parameters.append(value)
        seen: set[int] = set()
        unique: list[Any] = []
        for parameter in parameters:
            marker = id(parameter)
            if marker in seen:
                continue
            seen.add(marker)
            unique.append(parameter)
        return unique

    def state_dict(self) -> dict[str, Any]:
        _, nn = _require_torch()
        state: dict[str, Any] = {}
        for name, value in self.__dict__.items():
            if isinstance(value, nn.Module):
                state[name] = value.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        _, nn = _require_torch()
        for name, payload in state.items():
            module = getattr(self, name, None)
            if isinstance(module, nn.Module):
                module.load_state_dict(payload)


class SimpleDrugEncoderRuntime(_TorchRuntime):
    """Embedding + mean pooling + projection for drug sequences."""

    def __init__(self, *, vocab_size: int = 64, embedding_dim: int = 32, hidden_dim: int = 64) -> None:
        _, nn = _require_torch()
        self._vocab_size = int(vocab_size)
        self.embedding = nn.Embedding(self._vocab_size, int(embedding_dim))
        self.projection = nn.Linear(int(embedding_dim), int(hidden_dim))

    def forward(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        del context
        tokens = _to_token_ids(inputs["drug_seq"], vocab_size=self._vocab_size)
        embedded = self.embedding(tokens)
        pooled = embedded.mean(dim=1)
        return {"embedding": self.projection(pooled)}


class CNNProteinEncoderRuntime(_TorchRuntime):
    """Embedding + Conv1d + pooling encoder for protein sequences."""

    def __init__(
        self,
        *,
        vocab_size: int = 32,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        kernel_size: int = 5,
        pooling: str = "mean",
    ) -> None:
        _, nn = _require_torch()
        self._vocab_size = int(vocab_size)
        self._pooling = str(pooling).lower()
        if self._pooling not in {"mean", "max"}:
            raise InvalidConfigError(
                f"Unsupported pooling mode {pooling!r}. Expected 'mean' or 'max'.",
                stage="graph",
                component="CNNProteinEncoderRuntime",
                key="pooling",
            )
        padding = max(0, int(kernel_size) // 2)
        self.embedding = nn.Embedding(self._vocab_size, int(embedding_dim))
        self.conv = nn.Conv1d(int(embedding_dim), int(hidden_dim), kernel_size=int(kernel_size), padding=padding)
        self.activation = nn.ReLU()

    def forward(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        del context
        tokens = _to_token_ids(inputs["protein_seq"], vocab_size=self._vocab_size)
        embedded = self.embedding(tokens).transpose(1, 2)
        convolved = self.activation(self.conv(embedded))
        if self._pooling == "max":
            pooled = convolved.max(dim=2).values
        else:
            pooled = convolved.mean(dim=2)
        return {"embedding": pooled}


class ConcatFusionRuntime(_TorchRuntime):
    """Concatenate graph embeddings and optionally project them."""

    def __init__(self, *, project_dim: int | None = None, input_dim: int | None = None) -> None:
        _, nn = _require_torch()
        self._project_dim = None if project_dim is None else int(project_dim)
        self._input_dim = None if input_dim is None else int(input_dim)
        if self._project_dim is None:
            self.projection = nn.Identity()
        elif self._input_dim is None:
            self.projection = nn.LazyLinear(self._project_dim)
        else:
            self.projection = nn.Linear(self._input_dim, self._project_dim)

    def forward(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        del context
        torch, _ = _require_torch()
        tensors = [
            value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
            for value in inputs.values()
        ]
        if len(tensors) != 2:
            raise InvalidConfigError(
                f"fusion.concat expects exactly 2 inputs, got {len(tensors)}.",
                stage="graph",
                component="ConcatFusionRuntime",
            )
        left, right = tensors
        if left.ndim != 2 or right.ndim != 2:
            raise InvalidConfigError(
                "fusion.concat expects rank-2 embeddings shaped (B, D).",
                stage="graph",
                component="ConcatFusionRuntime",
            )
        if left.shape[0] != right.shape[0]:
            raise InvalidConfigError(
                f"fusion.concat batch mismatch: {tuple(left.shape)} vs {tuple(right.shape)}.",
                stage="graph",
                component="ConcatFusionRuntime",
            )
        fused = torch.cat([left.to(dtype=torch.float32), right.to(dtype=torch.float32)], dim=1)
        return {"embedding": self.projection(fused)}


class MLPHeadRuntime(_TorchRuntime):
    """Two-layer MLP head producing raw logits."""

    def __init__(self, *, hidden_dim: int = 64, output_dim: int = 1, input_dim: int | None = None) -> None:
        _, nn = _require_torch()
        if input_dim is None:
            self.hidden = nn.LazyLinear(int(hidden_dim))
        else:
            self.hidden = nn.Linear(int(input_dim), int(hidden_dim))
        self.activation = nn.ReLU()
        self.output = nn.Linear(int(hidden_dim), int(output_dim))

    def forward(self, inputs: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        del context
        torch, _ = _require_torch()
        embedding = next(iter(inputs.values()))
        tensor = embedding if isinstance(embedding, torch.Tensor) else torch.as_tensor(embedding, dtype=torch.float32)
        if tensor.ndim != 2:
            raise InvalidConfigError(
                f"head.mlp expects rank-2 embedding input, got shape {tuple(tensor.shape)}.",
                stage="graph",
                component="MLPHeadRuntime",
            )
        hidden = self.activation(self.hidden(tensor.to(dtype=torch.float32)))
        return {"logits": self.output(hidden)}
