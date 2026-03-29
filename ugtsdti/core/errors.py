"""Error taxonomy for UGTSDTI framework.

All errors carry stage, component, key, message, and optional debug_payload
so that validation failures can be traced to the exact boundary.

REQ-QUAL-001
"""
from __future__ import annotations

from typing import Any


class UGTSDTIError(Exception):
    """Base error for all UGTSDTI framework errors."""

    def __init__(
        self,
        message: str,
        *,
        stage: str = "",
        component: str = "",
        key: str = "",
        debug_payload: Any = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.component = component
        self.key = key
        self.message = message
        self.debug_payload = debug_payload

    def __str__(self) -> str:
        parts = [self.message]
        if self.stage:
            parts.append(f"stage={self.stage!r}")
        if self.component:
            parts.append(f"component={self.component!r}")
        if self.key:
            parts.append(f"key={self.key!r}")
        return " | ".join(parts)


class MissingDependencyError(UGTSDTIError):
    """A required dependency (key, module, or artifact) is missing."""


class KeyCollisionError(UGTSDTIError):
    """Attempt to write a key that already exists in the current forward pass."""


class InvalidConfigError(UGTSDTIError):
    """Config is malformed, missing required sections, or fails cross-section validation."""


class InvalidRoleBindingError(UGTSDTIError):
    """Role binding references non-existent or shape-incompatible graph outputs."""


class InvalidInteractionGraphError(UGTSDTIError):
    """Interaction graph has cycles, unresolved dependencies, or ordering violations."""


class InvalidDecisionOutputError(UGTSDTIError):
    """Decision module produced invalid or missing outputs (e.g. non-finite logits)."""


class InvalidAlphaRangeError(UGTSDTIError):
    """Gate alpha is outside [0, 1] or is non-finite."""


class InvalidLossMappingError(UGTSDTIError):
    """Loss config references keys that are not produced by any stage."""


class BatchSchemaError(UGTSDTIError):
    """Batch is missing required keys, contains NaN/Inf, or violates the batch contract."""


class InvalidStateSchemaError(UGTSDTIError):
    """State does not satisfy the declared schema at a stage boundary."""


class NumericalInstabilityError(UGTSDTIError):
    """NaN or Inf detected in a tensor that must be finite."""


class DeviceInconsistencyError(UGTSDTIError):
    """Tensors from different devices are used together without explicit transfer."""


class CheckpointCorruptedError(UGTSDTIError):
    """Checkpoint bundle is incomplete, corrupted, or config-incompatible."""


class MissingRawSnapshotError(UGTSDTIError):
    """Raw dataset snapshot is absent when preprocessing or splitting is attempted."""


class ProcessedSplitMismatchError(UGTSDTIError):
    """Processed data version does not match the split manifest."""


class InconsistentSplitError(UGTSDTIError):
    """Split artifacts are internally inconsistent (e.g. overlapping train/test sets)."""
