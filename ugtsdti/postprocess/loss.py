"""Minimal loss composition for the baseline pipeline.

REQ-POST-001, REQ-ARCH-003
"""
from __future__ import annotations

from typing import Any

from ugtsdti.core.errors import InvalidLossMappingError
from ugtsdti.core.state import State


class LossMapValidator:
    """Validate that configured loss mappings reference available state keys."""

    def validate(self, loss_cfg: dict[str, Any], state: State) -> None:
        for name, mapping in loss_cfg.get("map", {}).items():
            source_key = mapping.get("from", "") if isinstance(mapping, dict) else str(mapping)
            if source_key and not state.has(source_key):
                raise InvalidLossMappingError(
                    f"Loss map entry {name!r} references missing key {source_key!r}.",
                    stage="postprocess",
                    component="LossMapValidator",
                    key=source_key,
                )


class LossComposer:
    """Compose explicit hard and auxiliary losses without hidden behavior."""

    def __init__(self) -> None:
        self._validator = LossMapValidator()

    def compose(
        self,
        loss_cfg: dict[str, Any],
        state: State,
        labels: Any,
    ) -> dict[str, Any]:
        self._validator.validate(loss_cfg, state)
        hard_loss = _hard_loss(state.get("logits"), labels)
        hard_weight = float(loss_cfg.get("hard_weight", 1.0))
        outputs: dict[str, Any] = {
            "loss.hard": hard_loss,
            "loss.total": hard_weight * hard_loss,
        }

        for name, mapping in loss_cfg.get("map", {}).items():
            source_key = mapping.get("from", "") if isinstance(mapping, dict) else str(mapping)
            weight = mapping.get("weight", 1.0) if isinstance(mapping, dict) else 1.0
            if source_key:
                component = state.get(source_key)
                outputs[f"loss.{name}"] = component
                outputs["loss.total"] = outputs["loss.total"] + float(weight) * component

        return outputs


def _hard_loss(logits: Any, labels: Any) -> Any:
    try:
        import torch
        import torch.nn.functional as F

        if isinstance(logits, torch.Tensor):
            target = labels if isinstance(labels, torch.Tensor) else torch.as_tensor(labels, dtype=logits.dtype)
            target = target.to(dtype=logits.dtype, device=logits.device)
            target = target.reshape_as(logits)
            return F.binary_cross_entropy_with_logits(logits, target)
    except ImportError:
        pass

    if isinstance(logits, (int, float)) and isinstance(labels, (int, float)):
        diff = float(logits) - float(labels)
        return diff * diff

    raise TypeError("Unsupported logits/labels types for minimal hard loss.")
