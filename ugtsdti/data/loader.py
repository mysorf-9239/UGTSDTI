"""Artifact-backed data loading without raw-data side effects."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ugtsdti.core.errors import ProcessedSplitMismatchError
from ugtsdti.core.schema import BatchSpec, StateSchemaBuilder
from ugtsdti.data.contracts import DatasetVersion, SplitManifest
from ugtsdti.data.validate import DataValidator


class DataLoaderFactory:
    """Build simple batch iterables from materialized artifacts only."""

    def build(
        self,
        *,
        cfg: dict[str, Any],
        dataset_version_path: str | Path,
        split_manifest_path: str | Path,
        batch_size: int = 32,
        scenarios: list[str] | None = None,
        partition: str = "test",
    ) -> tuple[list[dict[str, Any]], BatchSpec, DatasetVersion, SplitManifest]:
        dataset_version, manifest = DataValidator().validate_artifacts(
            dataset_version_path=dataset_version_path,
            split_manifest_path=split_manifest_path,
        )
        selected_scenarios = list(scenarios or manifest.scenarios)
        records: list[dict[str, Any]] = []
        for scenario in selected_scenarios:
            partitions = manifest.scenario_partitions.get(scenario, {})
            split_path = partitions.get(partition)
            if split_path is None:
                raise ProcessedSplitMismatchError(
                    f"Requested scenario {scenario!r} partition {partition!r} is not present in the split manifest.",
                    stage="data",
                    component="DataLoaderFactory",
                    key=f"{scenario}.{partition}",
                )
            records.extend(_read_jsonl(Path(split_path)))
        batches = [_collate(records[index : index + batch_size]) for index in range(0, len(records), batch_size)]
        batch_spec = StateSchemaBuilder().build_batch_spec(cfg)
        return batches, batch_spec, dataset_version, manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    batch: dict[str, Any] = {}
    for key in rows[0]:
        batch[key] = [row.get(key) for row in rows]
    return batch
