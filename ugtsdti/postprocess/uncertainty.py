"""Uncertainty estimation modules for post-process pipeline."""

from __future__ import annotations

from typing import Any

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.errors import InvalidInteractionGraphError
from ugtsdti.core.state import State, StateWriter
from ugtsdti.interaction.base import InteractionRuntime


def _validate_input_logits(logits: Any, *, key: str, component: str) -> None:
    """Validate input logits are finite and non-empty before computation."""
    try:
        import torch

        tensor = logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits, dtype=torch.float32)

        # Check for empty tensor
        if tensor.numel() == 0:
            raise InvalidInteractionGraphError(
                "Empty tensor in uncertainty estimation.",
                stage="postprocess",
                component=component,
                key=key,
            )

        # Check for NaN/Inf values
        if not torch.all(torch.isfinite(tensor)):
            raise InvalidInteractionGraphError(
                "Uncertainty received non-finite logits.",
                stage="postprocess",
                component=component,
                key=key,
            )

    except ImportError as exc:
        raise RuntimeError("Uncertainty estimation requires torch.") from exc


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
            # Validate input logits before computation
            _validate_input_logits(inputs["teacher.logits"], key="teacher.logits", component=self.__class__.__name__)
            outputs["teacher.var"] = self._estimate(inputs["teacher.logits"])
        if emit_student and "student.logits" in inputs:
            # Validate input logits before computation
            _validate_input_logits(inputs["student.logits"], key="student.logits", component=self.__class__.__name__)
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
                    stage="postprocess",
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


def run_uncertainty(state: State, cfg: dict[str, Any]) -> State:
    """Run uncertainty estimation on state."""
    context = ExecutionContext(mode="eval", seed=42, device="cpu", deterministic=False)

    # Create uncertainty interaction
    uncertainty = UncertaintyInteraction(
        enabled=cfg.get("enabled", True),
        targets=cfg.get("targets", {"teacher": True, "student": True}),
        sample_dim=cfg.get("sample_dim", 0),
    )

    # Prepare inputs from state
    inputs = {}
    if state.has("teacher.logits"):
        inputs["teacher.logits"] = state.get("teacher.logits")
    if state.has("student.logits"):
        inputs["student.logits"] = state.get("student.logits")

    # Run uncertainty estimation
    outputs = uncertainty.forward(inputs, context)

    # Write outputs to state
    writer = StateWriter(state)
    writer.commit("uncertainty", outputs)

    return state
