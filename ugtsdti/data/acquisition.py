"""Offline-first raw data acquisition helpers."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable

from ugtsdti.core.errors import MissingRawSnapshotError


class DataAcquisition:
    """Materialize raw dataset snapshots outside the train/eval runtime path."""

    def __init__(self, raw_root: str | Path = "data/raw") -> None:
        self._raw_root = Path(raw_root)

    def export_raw_snapshot(
        self,
        dataset: str,
        *,
        records: list[dict[str, Any]] | None = None,
        fetch_fn: Callable[[], list[dict[str, Any]]] | None = None,
    ) -> Path:
        rows = records if records is not None else fetch_fn() if fetch_fn is not None else None
        if not rows:
            raise MissingRawSnapshotError(
                f"No raw records available for dataset {dataset!r}.",
                stage="data",
                component="DataAcquisition",
                key=dataset,
            )

        destination = self._raw_root / f"{dataset}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row.keys()})

        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        return destination
