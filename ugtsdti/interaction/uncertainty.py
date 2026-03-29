"""Uncertainty estimation interaction modules."""
from __future__ import annotations

from typing import Any

from ugtsdti.core.context import ExecutionContext
from ugtsdti.interaction.base import InteractionRuntime


def uncertainty_output_keys(params: dict[str, Any]) -> list[str]:
    """Return uncertainty keys for the enabled targets."""
    if params.get("enabled", True) is False:
        return []
    targets = params.get("targets", {})
    if not isinstance(targets, dict):
        targets = {}
    emit_teacher = targets.get("teacher", True)
    emit_student = targets.get("student", True)

    keys: list[str] = []
    if emit_teacher:
        keys.append("teacher.var")
    if emit_student:
        keys.append("student.var")
    return keys


class UncertaintyInteraction(InteractionRuntime):
    """Estimate per-branch uncertainty from logits or sampled predictions."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        targets: dict[str, bool] | None = None,
        sample_dim: int = 0,
    ) -> None:
        self._enabled = enabled
        self._targets = dict(targets or {})
        self._sample_dim = sample_dim

    def forward(
        self,
        inputs: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        del context
        if not self._enabled:
            return {}

        outputs: dict[str, Any] = {}
        emit_teacher = self._targets.get("teacher", True)
        emit_student = self._targets.get("student", True)

        if emit_teacher and "teacher.logits" in inputs:
            outputs["teacher.var"] = _estimate_variance(inputs["teacher.logits"], sample_dim=self._sample_dim)
        if emit_student and "student.logits" in inputs:
            outputs["student.var"] = _estimate_variance(inputs["student.logits"], sample_dim=self._sample_dim)
        return outputs


def _estimate_variance(value: Any, *, sample_dim: int) -> Any:
    try:
        import torch

        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
        tensor = tensor.to(dtype=torch.float32)
        if tensor.ndim >= 3:
            variance = tensor.var(dim=sample_dim, unbiased=False)
        else:
            probs = torch.sigmoid(tensor)
            variance = probs * (1.0 - probs)
        return variance.clamp_min(0.0)
    except ImportError as exc:
        raise RuntimeError("UncertaintyInteraction requires torch.") from exc
