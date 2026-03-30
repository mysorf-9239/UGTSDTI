"""Deterministic partition-aware split artifact generation for S1-S4 protocols."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ugtsdti.core.errors import InconsistentSplitError
from ugtsdti.data.contracts import SplitManifest

_PARTITIONS = ("train", "val", "test")
_SCENARIOS = ("s1", "s2", "s3", "s4")


class DataSplitter:
    """Create deterministic scenario-partition artifacts from materialized records."""

    def __init__(self, splits_root: str | Path = "data/splits") -> None:
        self._splits_root = Path(splits_root)

    def create_splits(
        self,
        dataset: str,
        *,
        records: list[dict[str, Any]],
        preprocessing_version: str,
        split_version: str,
        seed: int,
        scenarios: list[str] | None = None,
    ) -> SplitManifest:
        scenario_list = list(scenarios or list(_SCENARIOS))
        _validate_requested_scenarios(scenario_list)

        destination_dir = self._splits_root / dataset / preprocessing_version / split_version
        destination_dir.mkdir(parents=True, exist_ok=True)

        buckets = _empty_bucket_map(scenario_list)
        dropped_records = 0
        for row in records:
            placed = _assign_protocol_row(row, seed, buckets)
            if not placed:
                dropped_records += 1

        _materialize_cold_start_train_rows(buckets, scenario_list)

        scenario_partitions: dict[str, dict[str, str]] = {}
        counts: dict[str, dict[str, int]] = {}
        scenario_reports: dict[str, dict[str, Any]] = {}
        protocol_report: dict[str, Any] = {
            "protocol_version": "cold-start.v2",
            "dropped_record_count": dropped_records,
            "scenario_reports": scenario_reports,
        }

        for scenario in scenario_list:
            scenario_dir = destination_dir / scenario
            scenario_dir.mkdir(parents=True, exist_ok=True)
            scenario_partitions[scenario] = {}
            counts[scenario] = {}
            scenario_rows = buckets[scenario]
            for partition in _PARTITIONS:
                rows = scenario_rows[partition]
                path = scenario_dir / f"{partition}.jsonl"
                with path.open("w", encoding="utf-8") as handle:
                    for record in rows:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                scenario_partitions[scenario][partition] = str(path)
                counts[scenario][partition] = len(rows)
            scenario_reports[scenario] = _build_scenario_report(scenario_rows)

        for scenario in scenario_list:
            if scenario in {"s2", "s3", "s4"} and counts[scenario]["train"] == 0:
                raise InconsistentSplitError(
                    f"Requested split produced zero train rows for cold-start scenario {scenario!r}. "
                    "Generate a warm S1 split alongside cold scenarios or include 's1' in the requested scenarios.",
                    stage="data",
                    component="DataSplitter",
                    key=f"{scenario}.train",
                )
        if all(counts[scenario]["test"] == 0 for scenario in scenario_list):
            raise InconsistentSplitError(
                "Requested split produced zero evaluation records across all scenarios.",
                stage="data",
                component="DataSplitter",
                key="test",
            )

        manifest = SplitManifest(
            dataset=dataset,
            preprocessing_version=preprocessing_version,
            split_version=split_version,
            seed=seed,
            scenarios=scenario_list,
            partitions=list(_PARTITIONS),
            scenario_partitions=scenario_partitions,
            counts=counts,
            protocol_report=protocol_report,
        )
        manifest_path = destination_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_dict(), sort_keys=True, indent=2), encoding="utf-8")
        return manifest


def _validate_requested_scenarios(scenarios: list[str]) -> None:
    unsupported = sorted(set(scenarios) - set(_SCENARIOS))
    if unsupported:
        raise InconsistentSplitError(
            f"Unsupported scenarios requested: {', '.join(unsupported)}.",
            stage="data",
            component="DataSplitter",
            key="scenario",
        )


def _empty_bucket_map(scenarios: list[str]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    return {scenario: {partition: [] for partition in _PARTITIONS} for scenario in scenarios}


def _materialize_cold_start_train_rows(
    buckets: dict[str, dict[str, list[dict[str, Any]]]],
    scenarios: list[str],
) -> None:
    cold_scenarios = [scenario for scenario in ("s2", "s3", "s4") if scenario in buckets]
    if not cold_scenarios:
        return
    if "s1" not in buckets:
        return
    warm_train_rows = list(buckets["s1"]["train"])
    for scenario in cold_scenarios:
        buckets[scenario]["train"] = list(warm_train_rows)


def _assign_protocol_row(
    row: dict[str, Any],
    seed: int,
    buckets: dict[str, dict[str, list[dict[str, Any]]]],
) -> bool:
    drug_id = _drug_identity(row)
    target_id = _target_identity(row)
    drug_partition = _entity_partition(drug_id, seed)
    target_partition = _entity_partition(target_id, seed + 1)
    pair_partition = _entity_partition(_pair_identity(row), seed + 2)

    if drug_partition == "train" and target_partition == "train":
        scenario = "s1"
        partition = pair_partition
    elif drug_partition != "train" and target_partition == "train":
        scenario = "s2"
        partition = drug_partition
    elif drug_partition == "train" and target_partition != "train":
        scenario = "s3"
        partition = target_partition
    else:
        scenario = "s4"
        partition = _eval_partition(_pair_identity(row), seed + 3)

    if scenario not in buckets:
        return False

    serialized = dict(row)
    serialized["scenario"] = scenario
    serialized["partition"] = partition
    buckets[scenario][partition].append(serialized)
    return True


def _build_scenario_report(scenario_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    report: dict[str, Any] = {"partitions": {}}
    for partition, rows in scenario_rows.items():
        report["partitions"][partition] = {
            "count": len(rows),
            "unique_pairs": len({_pair_identity(row) for row in rows}),
            "unique_drugs": len({_drug_identity(row) for row in rows}),
            "unique_targets": len({_target_identity(row) for row in rows}),
        }
    return report


def _entity_partition(key: str, seed: int) -> str:
    bucket = _bucket(key, seed)
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def _eval_partition(key: str, seed: int) -> str:
    return "val" if _bucket(key, seed) < 50 else "test"


def _drug_identity(row: dict[str, Any]) -> str:
    return str(row.get("drug_id", row.get("drug", row.get("drug_seq", ""))))


def _target_identity(row: dict[str, Any]) -> str:
    return str(row.get("protein_id", row.get("target", row.get("protein_seq", ""))))


def _pair_identity(row: dict[str, Any]) -> str:
    return f"{_drug_identity(row)}::{_target_identity(row)}"


def _bucket(key: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100
