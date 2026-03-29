"""Deterministic split artifact generation for S1-S4 protocols."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ugtsdti.core.errors import InconsistentSplitError
from ugtsdti.data.contracts import SplitManifest


class DataSplitter:
    """Create deterministic scenario splits from materialized records."""

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
        scenario_list = list(scenarios or ["s1", "s2", "s3", "s4"])
        destination_dir = self._splits_root / dataset / preprocessing_version / split_version
        destination_dir.mkdir(parents=True, exist_ok=True)

        split_paths: dict[str, str] = {}
        counts: dict[str, int] = {}
        seen_keys: dict[str, set[str]] = {}

        for scenario in scenario_list:
            selected = [row for row in records if _belongs_to_scenario(row, scenario, seed)]
            path = destination_dir / f"{scenario}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in selected:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            split_paths[scenario] = str(path)
            counts[scenario] = len(selected)
            seen_keys[scenario] = {_record_key(row) for row in selected}

        if len(scenario_list) == 1 and counts[scenario_list[0]] == 0:
            raise InconsistentSplitError(
                "Requested split produced zero records.",
                stage="data",
                component="DataSplitter",
                key=scenario_list[0],
            )

        manifest = SplitManifest(
            dataset=dataset,
            preprocessing_version=preprocessing_version,
            split_version=split_version,
            seed=seed,
            scenarios=scenario_list,
            split_paths=split_paths,
            counts=counts,
        )
        manifest_path = destination_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_dict(), sort_keys=True, indent=2), encoding="utf-8")
        return manifest


def _belongs_to_scenario(row: dict[str, Any], scenario: str, seed: int) -> bool:
    bucket = _bucket(_group_key(row, scenario), seed)
    if scenario == "s1":
        return bucket < 80
    if scenario == "s2":
        return bucket < 75
    if scenario == "s3":
        return bucket < 70
    if scenario == "s4":
        return bucket < 65
    raise InconsistentSplitError(
        f"Unsupported scenario {scenario!r}.",
        stage="data",
        component="DataSplitter",
        key=scenario,
    )


def _group_key(row: dict[str, Any], scenario: str) -> str:
    drug = str(row.get("drug_id", row.get("drug", row.get("drug_seq", ""))))
    protein = str(row.get("protein_id", row.get("target", row.get("protein_seq", ""))))
    pair = _record_key(row)
    if scenario == "s1":
        return pair
    if scenario == "s2":
        return drug
    if scenario == "s3":
        return protein
    if scenario == "s4":
        return f"{drug}::{protein}"
    return pair


def _record_key(row: dict[str, Any]) -> str:
    return "::".join(f"{key}={row[key]}" for key in sorted(row))


def _bucket(key: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100
