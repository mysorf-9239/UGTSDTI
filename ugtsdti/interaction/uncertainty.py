"""Uncertainty estimation interaction modules."""

from __future__ import annotations

from typing import Any

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.errors import InvalidInteractionGraphError
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


class _BaseUncertaintyInteraction(InteractionRuntime):
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
            outputs["teacher.var"] = self._estimate(inputs["teacher.logits"])
        if emit_student and "student.logits" in inputs:
            outputs["student.var"] = self._estimate(inputs["student.logits"])
        return outputs

    def _estimate(self, value: Any) -> Any:
        raise NotImplementedError


class SampleVarianceUncertaintyInteraction(_BaseUncertaintyInteraction):
    """Estimate uncertainty from repeated sampled forward passes."""

    def _estimate(self, value: Any) -> Any:
        try:
            import torch

            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
            tensor = tensor.to(dtype=torch.float32)
            if tensor.ndim < 3:
                raise InvalidInteractionGraphError(
                    "uncertainty.sample_variance requires sampled logits with an explicit sample dimension.",
                    stage="interaction",
                    component="SampleVarianceUncertaintyInteraction",
                    key="logits",
                )
            variance = tensor.var(dim=self._sample_dim, unbiased=False)
            return variance.clamp_min(0.0)
        except ImportError as exc:
            raise RuntimeError("SampleVarianceUncertaintyInteraction requires torch.") from exc


class ConfidenceProxyUncertaintyInteraction(_BaseUncertaintyInteraction):
    """Estimate uncertainty as a confidence-derived proxy from logits."""

    def _estimate(self, value: Any) -> Any:
        try:
            import torch

            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
            tensor = tensor.to(dtype=torch.float32)
            probs = torch.sigmoid(tensor)
            variance = probs * (1.0 - probs)
            return variance.clamp_min(0.0)
        except ImportError as exc:
            raise RuntimeError("ConfidenceProxyUncertaintyInteraction requires torch.") from exc


class UncertaintyInteraction(ConfidenceProxyUncertaintyInteraction):
    """Backward-compatible alias for the confidence-proxy uncertainty path."""
