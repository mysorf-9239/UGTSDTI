"""Validation for materialized data artifacts and batches."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ugtsdti.core.errors import BatchSchemaError, MissingRawSnapshotError, ProcessedSplitMismatchError
from ugtsdti.core.schema import BatchSpec
from ugtsdti.data.contracts import DatasetVersion, SplitManifest


class DataValidator:
    """Validate artifact presence, split metadata, and batch contents."""

    def validate_artifacts(
        self,
        *,
        dataset_version_path: str | Path,
        split_manifest_path: str | Path,
    ) -> tuple[DatasetVersion, SplitManifest]:
        dataset_path = Path(dataset_version_path)
        manifest_path = Path(split_manifest_path)
        if not dataset_path.exists():
            raise MissingRawSnapshotError(
                f"Dataset version artifact is missing: {dataset_path}",
                stage="data",
                component="DataValidator",
                key=str(dataset_path),
            )
        if not manifest_path.exists():
            raise ProcessedSplitMismatchError(
                f"Split manifest is missing: {manifest_path}",
                stage="data",
                component="DataValidator",
                key=str(manifest_path),
            )

        dataset_version = DatasetVersion.from_dict(json.loads(dataset_path.read_text(encoding="utf-8")))
        manifest = SplitManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
        if dataset_version.dataset != manifest.dataset:
            raise ProcessedSplitMismatchError(
                "Dataset version metadata does not match split manifest dataset.",
                stage="data",
                component="DataValidator",
                key="dataset",
            )
        if dataset_version.preprocessing_version != manifest.preprocessing_version:
            raise ProcessedSplitMismatchError(
                "Dataset preprocessing_version does not match split manifest.",
                stage="data",
                component="DataValidator",
                key="preprocessing_version",
            )
        for scenario, path in manifest.split_paths.items():
            if not Path(path).exists():
                raise ProcessedSplitMismatchError(
                    f"Split artifact for scenario {scenario!r} is missing: {path}",
                    stage="data",
                    component="DataValidator",
                    key=scenario,
                )
        return dataset_version, manifest

    def validate_batch(self, batch: dict[str, Any], batch_spec: BatchSpec) -> None:
        for key in batch_spec.required_common:
            if key not in batch:
                raise BatchSchemaError(
                    f"Batch is missing required key {key!r}.",
                    stage="data",
                    component="DataValidator",
                    key=key,
                )
        for key in batch_spec.conditional_keys:
            if key not in batch:
                raise BatchSchemaError(
                    f"Batch is missing conditional key {key!r}.",
                    stage="data",
                    component="DataValidator",
                    key=key,
                )
        for key, value in batch.items():
            _ensure_finite(key, value)


def _ensure_finite(key: str, value: Any) -> None:
    try:
        import math

        import torch

        if isinstance(value, torch.Tensor):
            if not torch.all(torch.isfinite(value)):
                raise BatchSchemaError(
                    f"Batch key {key!r} contains NaN/Inf values.",
                    stage="data",
                    component="DataValidator",
                    key=key,
                )
            return
    except ImportError:
        pass

    if isinstance(value, list):
        for item in value:
            _ensure_finite(key, item)
        return
    if isinstance(value, (float, int)) and not math.isfinite(float(value)):
        raise BatchSchemaError(
            f"Batch key {key!r} contains NaN/Inf values.",
            stage="data",
            component="DataValidator",
            key=key,
        )
