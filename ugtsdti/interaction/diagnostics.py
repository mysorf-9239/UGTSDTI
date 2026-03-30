"""Diagnostics interaction modules."""

from __future__ import annotations

from typing import Any

from ugtsdti.core.context import ExecutionContext
from ugtsdti.interaction.base import InteractionRuntime


def diagnostics_output_keys(params: dict[str, Any]) -> list[str]:
    """Return diagnostics keys emitted by the module."""
    if params.get("enabled", True) is False:
        return []
    keys = ["interaction.disagreement"]
    if params.get("emit_calibration", True):
        keys.extend(
            [
                "diagnostics.teacher_confidence",
                "diagnostics.student_confidence",
            ]
        )
    return keys


class DiagnosticsInteraction(InteractionRuntime):
    """Emit disagreement and simple calibration helper statistics."""

    def __init__(self, *, emit_calibration: bool = True, enabled: bool = True) -> None:
        self._emit_calibration = emit_calibration
        self._enabled = enabled

    def forward(
        self,
        inputs: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        del context
        if not self._enabled:
            return {}
        teacher_logits = inputs["teacher.logits"]
        student_logits = inputs["student.logits"]

        teacher_prob = _sigmoid(teacher_logits)
        student_prob = _sigmoid(student_logits, device=teacher_prob.device)
        disagreement = (teacher_prob - student_prob).abs().mean()

        outputs: dict[str, Any] = {
            "interaction.disagreement": disagreement,
        }
        if self._emit_calibration:
            outputs["diagnostics.teacher_confidence"] = teacher_prob.mean()
            outputs["diagnostics.student_confidence"] = student_prob.mean()
        return outputs


def _sigmoid(value: Any, *, device: Any | None = None) -> Any:
    try:
        import torch

        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
        tensor = tensor.to(dtype=torch.float32)
        if device is not None:
            tensor = tensor.to(device=device)
        return torch.sigmoid(tensor)
    except ImportError as exc:
        raise RuntimeError("DiagnosticsInteraction requires torch.") from exc
