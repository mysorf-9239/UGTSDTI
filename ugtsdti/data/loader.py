"""Artifact-backed data loading without raw-data side effects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ugtsdti.core.errors import ProcessedSplitMismatchError
from ugtsdti.core.schema import BatchSpec, StateSchemaBuilder
from ugtsdti.data.contracts import DatasetVersion, SplitManifest
from ugtsdti.data.validate import DataValidator

# Import deterministic loading utilities
try:
    import ugtsdti.data.deterministic_loader  # noqa: F401

    HAS_DETERMINISTIC_LOADER = True
except ImportError:
    HAS_DETERMINISTIC_LOADER = False


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
        allow_missing_scenarios: bool = False,
        deterministic: bool = False,
        seed: int | None = None,
    ) -> tuple[list[dict[str, Any]], BatchSpec, DatasetVersion, SplitManifest]:
        dataset_version, manifest = DataValidator().validate_artifacts(
            dataset_version_path=dataset_version_path,
            split_manifest_path=split_manifest_path,
        )
        selected_scenarios = list(scenarios or manifest.scenarios)
        if not selected_scenarios:
            batch_spec = StateSchemaBuilder().build_batch_spec(cfg)
            return [], batch_spec, dataset_version, manifest
        records: list[dict[str, Any]] = []
        scenario_counts: dict[str, int] = {}
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
            rows = _read_jsonl(Path(split_path))
            scenario_counts[scenario] = len(rows)
            records.extend(rows)
        missing_scenarios = sorted(scenario for scenario, count in scenario_counts.items() if count == 0)
        if missing_scenarios and not allow_missing_scenarios:
            raise ProcessedSplitMismatchError(
                f"Requested scenarios {missing_scenarios} produced zero rows for partition {partition!r}.",
                stage="data",
                component="DataLoaderFactory",
                key=f"partition.{partition}",
                debug_payload={
                    "requested_scenarios": selected_scenarios,
                    "missing_scenarios": missing_scenarios,
                    "partition": partition,
                },
            )

        # Apply deterministic ordering if requested
        if deterministic and seed is not None and HAS_DETERMINISTIC_LOADER:
            # Set global seed for deterministic behavior
            import random

            import numpy as np

            try:
                import torch

                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True)
            except ImportError:
                # Fallback for systems without torch
                random.seed(seed)
                np.random.seed(seed)

            # ⚠️  REMOVED: Hash-based sorting (data transformation)
            # Determinism should be handled by DataLoader, not record sorting
        batch_spec = StateSchemaBuilder().build_batch_spec(cfg)
        return records, batch_spec, dataset_version, manifest


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
