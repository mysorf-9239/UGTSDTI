"""Artifact-backed data loading without raw-data side effects."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ugtsdti.core.schema import BatchSpec, StateSchemaBuilder
from ugtsdti.data.contracts import DatasetVersion, SplitManifest


class DataLoaderFactory:
    """Build simple batch iterables from materialized artifacts only."""

    def build(
        self,
        *,
        cfg: dict[str, Any],
        dataset_version_path: str | Path,
        split_manifest_path: str | Path,
        batch_size: int = 32,
    ) -> tuple[list[dict[str, Any]], BatchSpec, DatasetVersion, SplitManifest]:
        dataset_version = DatasetVersion.from_dict(json.loads(Path(dataset_version_path).read_text(encoding="utf-8")))
        manifest = SplitManifest.from_dict(json.loads(Path(split_manifest_path).read_text(encoding="utf-8")))
        records = _read_jsonl(Path(manifest.split_paths[manifest.scenarios[0]]))
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
