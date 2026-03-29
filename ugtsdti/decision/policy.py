"""Decision policies that turn trust signals into final logits."""
from __future__ import annotations

from typing import Any

from ugtsdti.core.errors import InvalidAlphaRangeError


class DecisionPolicy:
    """Apply soft or hard teacher/student selection policies."""

    def soft_blend(self, teacher_logits: Any, student_logits: Any, alpha: Any) -> Any:
        alpha_tensor = self._validate_alpha(alpha)
        teacher_tensor = _as_tensor(teacher_logits, device=alpha_tensor.device)
        student_tensor = _as_tensor(student_logits, device=alpha_tensor.device)
        return alpha_tensor * teacher_tensor + (1.0 - alpha_tensor) * student_tensor

    def hard_select(
        self,
        teacher_logits: Any,
        student_logits: Any,
        alpha: Any,
        *,
        threshold: float = 0.5,
    ) -> Any:
        alpha_tensor = self._validate_alpha(alpha)
        teacher_tensor = _as_tensor(teacher_logits, device=alpha_tensor.device)
        student_tensor = _as_tensor(student_logits, device=alpha_tensor.device)
        try:
            import torch

            choose_teacher = alpha_tensor >= threshold
            return torch.where(choose_teacher, teacher_tensor, student_tensor)
        except ImportError as exc:
            raise RuntimeError("Advanced decision modules require torch.") from exc

    def _validate_alpha(self, alpha: Any) -> Any:
        try:
            import torch

            alpha_tensor = _as_tensor(alpha)
            if not torch.all(torch.isfinite(alpha_tensor)):
                raise InvalidAlphaRangeError(
                    "gate.alpha must be finite.",
                    stage="decision",
                    component="DecisionPolicy",
                    key="gate.alpha",
                )
            if not torch.all((alpha_tensor >= 0.0) & (alpha_tensor <= 1.0)):
                raise InvalidAlphaRangeError(
                    "gate.alpha must lie in [0, 1].",
                    stage="decision",
                    component="DecisionPolicy",
                    key="gate.alpha",
                )
            return alpha_tensor
        except ImportError as exc:
            raise RuntimeError("Advanced decision modules require torch.") from exc


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
