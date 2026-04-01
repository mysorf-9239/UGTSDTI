"""Minimal loss composition for the baseline pipeline.

REQ-POST-001, REQ-ARCH-003
"""

from __future__ import annotations

from typing import Any

from ugtsdti.core.errors import InvalidLossMappingError
from ugtsdti.core.state import State, StateWriter


def _validate_logits(logits: Any) -> None:
    """Validate logits are finite and non-empty."""
    try:
        import torch

        if isinstance(logits, torch.Tensor):
            # Check for empty tensor
            if logits.numel() == 0:
                raise InvalidLossMappingError(
                    "Empty batch in loss computation.",
                    stage="postprocess",
                    component="LossComposer",
                    key="logits",
                )

            # Check for NaN/Inf values
            if not torch.all(torch.isfinite(logits)):
                raise InvalidLossMappingError(
                    "Loss received non-finite logits.",
                    stage="postprocess",
                    component="LossComposer",
                    key="logits",
                )
    except ImportError as exc:
        raise RuntimeError("Loss computation requires torch.") from exc


def _validate_labels(labels: Any) -> Any:
    """Validate and convert labels to tensor if needed."""
    try:
        import torch

        if isinstance(labels, torch.Tensor):
            # Check for NaN/Inf in labels
            if not torch.all(torch.isfinite(labels)):
                raise InvalidLossMappingError(
                    "Loss received non-finite labels.",
                    stage="postprocess",
                    component="LossComposer",
                    key="labels",
                )
            return labels
        else:
            # Convert to tensor
            return torch.as_tensor(labels, dtype=torch.float32)
    except ImportError as exc:
        raise RuntimeError("Loss computation requires torch.") from exc


def _validate_auxiliary_component(component: Any, name: str) -> None:
    """Validate auxiliary loss component is finite."""
    try:
        import torch

        if isinstance(component, torch.Tensor):
            if not torch.all(torch.isfinite(component)):
                raise InvalidLossMappingError(
                    f"Loss component '{name}' contains NaN/Inf.",
                    stage="postprocess",
                    component="LossComposer",
                    key=f"loss.{name}",
                )
    except ImportError as exc:
        raise RuntimeError("Loss computation requires torch.") from exc


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
        # Validate loss mapping configuration
        self._validator.validate(loss_cfg, state)

        # Check for required logits
        if not state.has("logits"):
            raise InvalidLossMappingError(
                "Loss requires 'logits' in state",
                stage="postprocess",
                component="LossComposer",
                key="logits",
            )

        # Get and validate logits
        logits = state.get("logits")
        _validate_logits(logits)

        # Validate and convert labels
        validated_labels = _validate_labels(labels)

        # Compute hard loss
        hard_loss = _hard_loss(logits, validated_labels)
        hard_weight = float(loss_cfg.get("hard_weight", 1.0))
        outputs: dict[str, Any] = {
            "loss.hard": hard_loss,
            "loss.total": hard_weight * hard_loss,
        }

        # Process auxiliary loss components
        for name, mapping in loss_cfg.get("map", {}).items():
            source_key = mapping.get("from", "") if isinstance(mapping, dict) else str(mapping)
            weight = mapping.get("weight", 1.0) if isinstance(mapping, dict) else 1.0
            if source_key:
                component = state.get(source_key)

                # Validate auxiliary component
                _validate_auxiliary_component(component, name)

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


def run_loss(state: State, cfg: dict[str, Any], labels: Any) -> State:
    """Run loss composition on state."""
    composer = LossComposer()
    outputs = composer.compose(cfg, state, labels)

    # Write outputs to state under loss namespace
    writer = StateWriter(state)
    writer.commit("loss", outputs)

    return state
