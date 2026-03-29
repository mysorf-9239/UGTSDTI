"""Checkpoint bundle contracts and I/O."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ugtsdti.core.errors import CheckpointCorruptedError
from ugtsdti.runtime.artifacts import _serialize_payload


@dataclass
class CheckpointBundle:
    """Serializable checkpoint bundle."""

    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any]
    rng_state: dict[str, Any]
    epoch: int
    step: int
    identity: dict[str, Any]
    config: dict[str, Any]
    dataset_metadata: dict[str, Any]
    split_metadata: dict[str, Any]
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CheckpointBundle":
        required = {
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "rng_state",
            "epoch",
            "step",
            "identity",
            "config",
            "dataset_metadata",
            "split_metadata",
        }
        missing = required - set(payload)
        if missing:
            raise CheckpointCorruptedError(
                f"Checkpoint bundle is missing required fields: {sorted(missing)}",
                stage="runtime",
                component="CheckpointBundle",
                key="checkpoint",
            )
        return cls(
            model_state=dict(payload["model_state"]),
            optimizer_state=dict(payload["optimizer_state"]),
            scheduler_state=dict(payload["scheduler_state"]),
            rng_state=dict(payload["rng_state"]),
            epoch=int(payload["epoch"]),
            step=int(payload["step"]),
            identity=dict(payload["identity"]),
            config=dict(payload["config"]),
            dataset_metadata=dict(payload["dataset_metadata"]),
            split_metadata=dict(payload["split_metadata"]),
            extras=dict(payload.get("extras", {})),
        )


class CheckpointIO:
    """Atomic checkpoint save/load with basic compatibility validation."""

    def save(self, bundle: CheckpointBundle, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.with_suffix(destination.suffix + ".tmp")
        payload = bundle.to_dict()
        try:
            import torch

            torch.save(payload, temp_path)
        except ImportError:
            temp_path.write_text(json.dumps(_serialize_payload(payload), sort_keys=True, indent=2), encoding="utf-8")
        os.replace(temp_path, destination)
        return destination

    def load(
        self,
        path: str | Path,
        *,
        expected_config_hash: str | None = None,
        expected_dataset: str | None = None,
    ) -> CheckpointBundle:
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise CheckpointCorruptedError(
                f"Checkpoint file does not exist: {checkpoint_path}",
                stage="runtime",
                component="CheckpointIO",
                key=str(checkpoint_path),
            )

        payload = self._load_payload(checkpoint_path)

        bundle = CheckpointBundle.from_dict(payload)
        if expected_config_hash is not None and bundle.identity.get("config_hash") != expected_config_hash:
            raise CheckpointCorruptedError(
                "Checkpoint config hash does not match requested run.",
                stage="runtime",
                component="CheckpointIO",
                key="identity.config_hash",
            )
        if expected_dataset is not None and bundle.dataset_metadata.get("dataset") != expected_dataset:
            raise CheckpointCorruptedError(
                "Checkpoint dataset metadata does not match requested dataset.",
                stage="runtime",
                component="CheckpointIO",
                key="dataset_metadata.dataset",
            )
        return bundle

    def _load_payload(self, checkpoint_path: Path) -> dict[str, Any]:
        try:
            import torch

            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass

        try:
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CheckpointCorruptedError(
                "Checkpoint file is not a valid checkpoint payload.",
                stage="runtime",
                component="CheckpointIO",
                key=str(checkpoint_path),
            ) from exc
        if not isinstance(payload, dict):
            raise CheckpointCorruptedError(
                "Checkpoint payload must deserialize to a mapping.",
                stage="runtime",
                component="CheckpointIO",
                key=str(checkpoint_path),
            )
        return payload
