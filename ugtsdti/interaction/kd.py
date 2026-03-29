"""Knowledge distillation interaction modules."""
from __future__ import annotations

from typing import Any

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.errors import InvalidConfigError, MissingDependencyError
from ugtsdti.interaction.base import InteractionRuntime


def kd_output_keys(params: dict[str, Any]) -> list[str]:
    """Return stable output keys for the configured KD module."""
    if params.get("enabled", True) is False:
        return []
    return [
        "interaction.kd.loss_component",
        "kd.teacher_target",
        "kd.student_target",
    ]


def binary_logits_to_dist(logits: Any, temperature: float) -> Any:
    """Convert binary logits to a two-class probability distribution."""
    if temperature <= 0:
        raise InvalidConfigError(
            "KD temperature must be > 0.",
            stage="interaction",
            component="KDInteraction",
            key="interaction.kd.temperature",
        )

    try:
        import torch

        if not isinstance(logits, torch.Tensor):
            logits = torch.as_tensor(logits, dtype=torch.float32)
        logits = logits.to(dtype=torch.float32)
        if logits.ndim == 0:
            logits = logits.reshape(1)
        if logits.ndim > 1 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        baseline = torch.zeros_like(logits)
        stacked = torch.stack([baseline, logits], dim=-1) / float(temperature)
        return torch.softmax(stacked, dim=-1)
    except ImportError as exc:
        raise RuntimeError("KDInteraction requires torch.") from exc


class KDInteraction(InteractionRuntime):
    """Compute explicit KD outputs for logits, feature, or relation distillation."""

    def __init__(
        self,
        *,
        mode: str = "logits",
        temperature: float = 1.0,
        enabled: bool = True,
        teacher_key: str | None = None,
        student_key: str | None = None,
    ) -> None:
        self._mode = mode
        self._temperature = float(temperature)
        self._enabled = enabled
        self._teacher_key = teacher_key
        self._student_key = student_key

    def forward(
        self,
        inputs: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        del context
        if not self._enabled:
            return {}

        teacher_key, student_key = self._resolve_keys(inputs)
        teacher_value = inputs[teacher_key]
        student_value = inputs[student_key]

        if self._mode == "logits":
            return self._forward_logits(teacher_value, student_value)
        if self._mode == "feature":
            return self._forward_feature(teacher_value, student_value)
        if self._mode == "relation":
            return self._forward_relation(teacher_value, student_value)
        raise InvalidConfigError(
            f"Unsupported KD mode {self._mode!r}.",
            stage="interaction",
            component="KDInteraction",
            key="interaction.kd.mode",
        )

    def _resolve_keys(self, inputs: dict[str, Any]) -> tuple[str, str]:
        teacher_key = self._teacher_key or self._find_key(inputs, preferred_prefix="teacher.")
        student_key = self._student_key or self._find_key(inputs, preferred_prefix="student.")

        if teacher_key not in inputs:
            raise MissingDependencyError(
                f"KDInteraction requires teacher input {teacher_key!r}.",
                stage="interaction",
                component="KDInteraction",
                key=teacher_key,
            )
        if student_key not in inputs:
            raise MissingDependencyError(
                f"KDInteraction requires student input {student_key!r}.",
                stage="interaction",
                component="KDInteraction",
                key=student_key,
            )
        return teacher_key, student_key

    def _find_key(self, inputs: dict[str, Any], *, preferred_prefix: str) -> str:
        for key in inputs:
            if key.startswith(preferred_prefix):
                return key
        if len(inputs) >= 2:
            return list(inputs)[0 if preferred_prefix == "teacher." else 1]
        raise MissingDependencyError(
            "KDInteraction requires teacher and student inputs.",
            stage="interaction",
            component="KDInteraction",
            key=preferred_prefix.rstrip("."),
        )

    def _forward_logits(self, teacher_logits: Any, student_logits: Any) -> dict[str, Any]:
        try:
            import torch.nn.functional as F
        except ImportError as exc:
            raise RuntimeError("KDInteraction requires torch.") from exc

        teacher_target = binary_logits_to_dist(teacher_logits, self._temperature)
        student_target = binary_logits_to_dist(student_logits, self._temperature)
        loss = F.kl_div(
            student_target.clamp_min(1e-12).log(),
            teacher_target,
            reduction="batchmean",
        ) * (self._temperature**2)
        return {
            "interaction.kd.loss_component": loss,
            "kd.teacher_target": teacher_target,
            "kd.student_target": student_target,
        }

    def _forward_feature(self, teacher_features: Any, student_features: Any) -> dict[str, Any]:
        try:
            import torch.nn.functional as F
        except ImportError as exc:
            raise RuntimeError("KDInteraction requires torch.") from exc

        teacher_tensor = _as_tensor(teacher_features)
        student_tensor = _as_tensor(student_features, device=teacher_tensor.device)
        loss = F.mse_loss(student_tensor, teacher_tensor)
        return {
            "interaction.kd.loss_component": loss,
            "kd.teacher_target": teacher_tensor,
            "kd.student_target": student_tensor,
        }

    def _forward_relation(self, teacher_features: Any, student_features: Any) -> dict[str, Any]:
        teacher_relation = _relation_matrix(teacher_features)
        student_relation = _relation_matrix(student_features, device=teacher_relation.device)
        try:
            import torch.nn.functional as F
        except ImportError as exc:
            raise RuntimeError("KDInteraction requires torch.") from exc

        loss = F.mse_loss(student_relation, teacher_relation)
        return {
            "interaction.kd.loss_component": loss,
            "kd.teacher_target": teacher_relation,
            "kd.student_target": student_relation,
        }


def _as_tensor(value: Any, *, device: Any | None = None) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            tensor = value.to(dtype=torch.float32)
        else:
            tensor = torch.as_tensor(value, dtype=torch.float32)
        if device is not None:
            tensor = tensor.to(device=device)
        return tensor
    except ImportError as exc:
        raise RuntimeError("KDInteraction requires torch.") from exc


def _relation_matrix(value: Any, *, device: Any | None = None) -> Any:
    tensor = _as_tensor(value, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    flattened = tensor.reshape(tensor.shape[0], -1)
    denominator = flattened.norm(dim=1, keepdim=True).clamp_min(1e-12)
    normalized = flattened / denominator
    return normalized @ normalized.transpose(0, 1)
