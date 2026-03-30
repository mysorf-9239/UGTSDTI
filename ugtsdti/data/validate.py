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
        for scenario, partitions in manifest.scenario_partitions.items():
            for partition, path in partitions.items():
                if not Path(path).exists():
                    raise ProcessedSplitMismatchError(
                        f"Split artifact for scenario {scenario!r} partition {partition!r} is missing: {path}",
                        stage="data",
                        component="DataValidator",
                        key=f"{scenario}.{partition}",
                    )
        self._validate_leakage(manifest)
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

    def _validate_leakage(self, manifest: SplitManifest) -> None:
        for scenario in manifest.scenarios:
            partitions = manifest.scenario_partitions.get(scenario, {})
            train_rows = _read_rows(partitions.get("train"))
            val_rows = _read_rows(partitions.get("val"))
            test_rows = _read_rows(partitions.get("test"))
            eval_rows = val_rows + test_rows

            _ensure_no_overlap(
                scenario=scenario,
                invariant="pair",
                left_partition="train",
                right_partition="eval",
                left_rows=train_rows,
                right_rows=eval_rows,
                identity_fn=_pair_identity,
            )
            if scenario == "s2":
                _ensure_no_overlap(
                    scenario=scenario,
                    invariant="drug",
                    left_partition="train",
                    right_partition="eval",
                    left_rows=train_rows,
                    right_rows=eval_rows,
                    identity_fn=_drug_identity,
                )
            if scenario == "s3":
                _ensure_no_overlap(
                    scenario=scenario,
                    invariant="target",
                    left_partition="train",
                    right_partition="eval",
                    left_rows=train_rows,
                    right_rows=eval_rows,
                    identity_fn=_target_identity,
                )
            if scenario == "s4":
                _ensure_no_overlap(
                    scenario=scenario,
                    invariant="drug",
                    left_partition="train",
                    right_partition="eval",
                    left_rows=train_rows,
                    right_rows=eval_rows,
                    identity_fn=_drug_identity,
                )
                _ensure_no_overlap(
                    scenario=scenario,
                    invariant="target",
                    left_partition="train",
                    right_partition="eval",
                    left_rows=train_rows,
                    right_rows=eval_rows,
                    identity_fn=_target_identity,
                )
            _ensure_no_overlap(
                scenario=scenario,
                invariant="pair",
                left_partition="val",
                right_partition="test",
                left_rows=val_rows,
                right_rows=test_rows,
                identity_fn=_pair_identity,
            )


def _read_rows(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _ensure_no_overlap(
    *,
    scenario: str,
    invariant: str,
    left_partition: str,
    right_partition: str,
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    identity_fn: Any,
) -> None:
    left_ids = {str(identity_fn(row)) for row in left_rows}
    right_ids = {str(identity_fn(row)) for row in right_rows}
    overlap = sorted(left_ids & right_ids)
    if overlap:
        sample = ", ".join(overlap[:5])
        raise ProcessedSplitMismatchError(
            f"Leakage detected for {scenario} on {invariant} between {left_partition} and {right_partition}: {sample}",
            stage="data",
            component="DataValidator",
            key=f"{scenario}.{invariant}",
            debug_payload={
                "scenario": scenario,
                "invariant": invariant,
                "left_partition": left_partition,
                "right_partition": right_partition,
                "offending_ids": overlap,
            },
        )


def _drug_identity(row: dict[str, Any]) -> str:
    return str(row.get("drug_id", row.get("drug", row.get("drug_seq", ""))))


def _target_identity(row: dict[str, Any]) -> str:
    return str(row.get("protein_id", row.get("target", row.get("protein_seq", ""))))


def _pair_identity(row: dict[str, Any]) -> str:
    return f"{_drug_identity(row)}::{_target_identity(row)}"


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
