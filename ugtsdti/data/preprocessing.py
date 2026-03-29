"""Raw-to-processed dataset materialization."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

from ugtsdti.data.contracts import DatasetVersion


class DataPreprocessor:
    """Transform raw CSV snapshots into materialized JSONL feature artifacts."""

    def __init__(self, processed_root: str | Path = "data/processed") -> None:
        self._processed_root = Path(processed_root)

    def materialize(
        self,
        dataset: str,
        *,
        raw_snapshot: str | Path,
        preprocessing_version: str,
        transform_fn: Callable[[dict[str, str]], dict[str, Any]] | None = None,
    ) -> tuple[Path, DatasetVersion]:
        raw_path = Path(raw_snapshot)
        destination_dir = self._processed_root / dataset / preprocessing_version
        destination_dir.mkdir(parents=True, exist_ok=True)
        records_path = destination_dir / "records.jsonl"

        materialized: list[dict[str, Any]] = []
        with raw_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                materialized.append(transform_fn(row) if transform_fn is not None else dict(row))

        with records_path.open("w", encoding="utf-8") as handle:
            for row in materialized:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

        feature_keys = sorted({key for row in materialized for key in row.keys()})
        version = DatasetVersion(
            dataset=dataset,
            raw_version=raw_path.stem,
            preprocessing_version=preprocessing_version,
            record_count=len(materialized),
            feature_keys=feature_keys,
        )
        metadata_path = destination_dir / "dataset_version.json"
        metadata_path.write_text(json.dumps(version.to_dict(), sort_keys=True, indent=2), encoding="utf-8")
        return records_path, version
