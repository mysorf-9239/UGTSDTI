"""Default runtime wiring for shipped config type keys."""
from __future__ import annotations

from typing import Any

from ugtsdti.graph.registry import NodeRegistry
from ugtsdti.graph.specs import NodePluginSpec
from ugtsdti.interaction.base import InteractionPluginSpec
from ugtsdti.interaction.diagnostics import DiagnosticsInteraction, diagnostics_output_keys
from ugtsdti.interaction.kd import KDInteraction, kd_output_keys
from ugtsdti.interaction.noop import NoOpInteraction
from ugtsdti.interaction.registry import InteractionRegistry
from ugtsdti.interaction.uncertainty import UncertaintyInteraction, uncertainty_output_keys
from ugtsdti.nodes.base import NodeRuntime


class SimpleConcatEncoderRuntime(NodeRuntime):
    """Convert batch inputs into numeric features and concatenate them."""

    def forward(self, inputs: dict[str, Any], context: Any) -> dict[str, Any]:
        del context
        import torch

        features = [_to_feature_tensor(value) for value in inputs.values()]
        return {"embedding": torch.cat(features, dim=1)}


class SimpleLinearHeadRuntime(NodeRuntime):
    """Produce scalar logits from flattened embeddings."""

    def __init__(self, *, scale: float = 0.1, bias: float = 0.0) -> None:
        self._scale = float(scale)
        self._bias = float(bias)

    def forward(self, inputs: dict[str, Any], context: Any) -> dict[str, Any]:
        del context
        import torch

        embedding = next(iter(inputs.values()))
        tensor = embedding if isinstance(embedding, torch.Tensor) else _to_feature_tensor(embedding)
        flattened = tensor.reshape(tensor.shape[0], -1)
        logits = flattened.mean(dim=1, keepdim=True) * self._scale + self._bias
        return {"logits": logits}


def build_default_graph_registry() -> NodeRegistry:
    registry = NodeRegistry()
    registry.register(
        NodePluginSpec(type_key="encoder.student", output_attrs=["embedding"], input_kinds=["drug_seq", "protein_seq"]),
        SimpleConcatEncoderRuntime,
    )
    registry.register(
        NodePluginSpec(
            type_key="encoder.teacher", output_attrs=["embedding"], input_kinds=["drug_graph", "protein_seq"]
        ),
        SimpleConcatEncoderRuntime,
    )
    registry.register(
        NodePluginSpec(
            type_key="encoder.baseline", output_attrs=["embedding"], input_kinds=["drug_seq", "protein_seq"]
        ),
        SimpleConcatEncoderRuntime,
    )
    registry.register(
        NodePluginSpec(type_key="head.student", output_attrs=["logits"], input_kinds=["embedding"]),
        SimpleLinearHeadRuntime,
    )
    registry.register(
        NodePluginSpec(type_key="head.teacher", output_attrs=["logits"], input_kinds=["embedding"]),
        SimpleLinearHeadRuntime,
    )
    registry.register(
        NodePluginSpec(type_key="head.linear", output_attrs=["logits"], input_kinds=["embedding"]),
        SimpleLinearHeadRuntime,
    )
    return registry


def build_default_interaction_registry() -> InteractionRegistry:
    registry = InteractionRegistry()
    registry.register(
        InteractionPluginSpec(type_key="noop", output_keys_fn=lambda params: []),
        NoOpInteraction,
    )
    registry.register(
        InteractionPluginSpec(type_key="kd.standard", output_keys_fn=kd_output_keys),
        KDInteraction,
    )
    registry.register(
        InteractionPluginSpec(type_key="uncertainty.mc_dropout", output_keys_fn=uncertainty_output_keys),
        UncertaintyInteraction,
    )
    registry.register(
        InteractionPluginSpec(type_key="diagnostics.basic", output_keys_fn=diagnostics_output_keys),
        DiagnosticsInteraction,
    )
    return registry


def _to_feature_tensor(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(dtype=torch.float32)
        if tensor.ndim == 0:
            return tensor.reshape(1, 1)
        if tensor.ndim == 1:
            return tensor.unsqueeze(-1)
        return tensor.reshape(tensor.shape[0], -1)

    if isinstance(value, list):
        if not value:
            return torch.zeros((0, 1), dtype=torch.float32)
        first = value[0]
        if isinstance(first, str):
            return _string_batch_features(value)
        try:
            tensor = torch.as_tensor(value, dtype=torch.float32)
            if tensor.ndim == 0:
                return tensor.reshape(1, 1)
            if tensor.ndim == 1:
                return tensor.unsqueeze(-1)
            return tensor.reshape(tensor.shape[0], -1)
        except Exception:
            return _string_batch_features([str(item) for item in value])

    if isinstance(value, str):
        return _string_batch_features([value])

    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim == 0:
            return tensor.reshape(1, 1)
        if tensor.ndim == 1:
            return tensor.unsqueeze(-1)
        return tensor.reshape(tensor.shape[0], -1)
    except Exception:
        return _string_batch_features([str(value)])


def _string_batch_features(values: list[str]) -> Any:
    import torch

    features: list[list[float]] = []
    for item in values:
        text = str(item)
        length = float(len(text))
        score = float(sum(ord(char) for char in text)) / max(length, 1.0)
        features.append([length, score / 127.0])
    return torch.tensor(features, dtype=torch.float32)
