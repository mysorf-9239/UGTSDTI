"""Trust estimation primitives for advanced decision modules."""

from __future__ import annotations

from typing import Any

from ugtsdti.core.errors import InvalidDecisionOutputError
from ugtsdti.core.state import State


class TrustEstimator:
    """Estimate trust signals from branch logits and optional uncertainty."""

    def __init__(
        self,
        *,
        use_uncertainty: bool = False,
        uncertainty_source: str = "none",
        epsilon: float = 1e-6,
    ) -> None:
        self._use_uncertainty = use_uncertainty
        self._uncertainty_source = uncertainty_source
        self._epsilon = epsilon

    def estimate(self, state: State) -> dict[str, Any]:
        if not state.has("teacher.logits") or not state.has("student.logits"):
            raise InvalidDecisionOutputError(
                "TrustEstimator requires both 'teacher.logits' and 'student.logits'.",
                stage="decision",
                component="TrustEstimator",
                key="teacher.logits",
            )

        teacher_logits = _as_tensor(state.get("teacher.logits"))
        student_logits = _as_tensor(state.get("student.logits"), device=teacher_logits.device)

        teacher_score = _confidence_score(teacher_logits)
        student_score = _confidence_score(student_logits)

        if self._use_uncertainty:
            if not state.has("teacher.var") or not state.has("student.var"):
                raise InvalidDecisionOutputError(
                    "TrustEstimator requires 'teacher.var' and 'student.var' when use_uncertainty=True.",
                    stage="decision",
                    component="TrustEstimator",
                    key="teacher.var",
                )
            teacher_var = _reduce_like_logits(state.get("teacher.var"), teacher_score.device)
            student_var = _reduce_like_logits(state.get("student.var"), student_score.device)
            teacher_score = teacher_score / (1.0 + teacher_var)
            student_score = student_score / (1.0 + student_var)

        alpha = teacher_score / (teacher_score + student_score + self._epsilon)
        outputs = {
            "gate.alpha": alpha.clamp(0.0, 1.0),
            "gate.teacher_score": teacher_score,
            "gate.student_score": student_score,
        }
        if self._use_uncertainty:
            outputs["gate.uncertainty_source"] = self._uncertainty_source
        return outputs


def _as_tensor(value: Any, *, device: Any | None = None) -> Any:
    try:
        import torch

        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
        tensor = tensor.to(dtype=torch.float32)
        if device is not None:
            tensor = tensor.to(device=device)
        return tensor
    except ImportError as exc:
        raise RuntimeError("Advanced decision modules require torch.") from exc


def _confidence_score(logits: Any) -> Any:
    try:
        import torch

        probs = torch.sigmoid(logits)
        confidence = (probs - 0.5).abs() * 2.0
        if confidence.ndim == 1:
            confidence = confidence.unsqueeze(-1)
        return confidence
    except ImportError as exc:
        raise RuntimeError("Advanced decision modules require torch.") from exc


def _reduce_like_logits(value: Any, device: Any) -> Any:
    try:
        tensor = _as_tensor(value, device=device)
        if tensor.ndim == 0:
            return tensor.reshape(1, 1)
        if tensor.ndim == 1:
            return tensor.unsqueeze(-1)
        if tensor.ndim > 2:
            dims = tuple(range(1, tensor.ndim))
            return tensor.mean(dim=dims, keepdim=True)
        return tensor.mean(dim=-1, keepdim=True) if tensor.ndim == 2 and tensor.shape[-1] != 1 else tensor
    except ImportError as exc:
        raise RuntimeError("Advanced decision modules require torch.") from exc
