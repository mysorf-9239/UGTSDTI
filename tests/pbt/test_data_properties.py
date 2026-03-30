"""Property-based tests for scenario split leakage invariants."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from ugtsdti.data import DataSplitter, SplitManifest


def _grid_records(drugs: int, targets: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for drug_index in range(drugs):
        for target_index in range(targets):
            records.append(
                {
                    "drug_id": f"d{drug_index}",
                    "protein_id": f"p{target_index}",
                    "drug_seq": f"DRUG{drug_index}",
                    "protein_seq": f"PROT{target_index}",
                    "labels": float((drug_index + target_index) % 2),
                }
            )
    return records


def _read_rows(path: str) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@given(seed=st.integers(min_value=0, max_value=5000))
def test_s2_train_and_eval_drugs_remain_disjoint(seed):
    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest = _build_manifest(Path(tmp_dir), seed)
        train_rows = _read_rows(manifest.scenario_partitions["s2"]["train"])
        eval_rows = _read_rows(manifest.scenario_partitions["s2"]["val"]) + _read_rows(
            manifest.scenario_partitions["s2"]["test"]
        )

        assert {row["drug_id"] for row in train_rows}.isdisjoint({row["drug_id"] for row in eval_rows})


@given(seed=st.integers(min_value=0, max_value=5000))
def test_s3_train_and_eval_targets_remain_disjoint(seed):
    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest = _build_manifest(Path(tmp_dir), seed)
        train_rows = _read_rows(manifest.scenario_partitions["s3"]["train"])
        eval_rows = _read_rows(manifest.scenario_partitions["s3"]["val"]) + _read_rows(
            manifest.scenario_partitions["s3"]["test"]
        )

        assert {row["protein_id"] for row in train_rows}.isdisjoint({row["protein_id"] for row in eval_rows})


@given(seed=st.integers(min_value=0, max_value=5000))
def test_s4_train_and_eval_entities_remain_fully_disjoint(seed):
    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest = _build_manifest(Path(tmp_dir), seed)
        train_rows = _read_rows(manifest.scenario_partitions["s4"]["train"])
        eval_rows = _read_rows(manifest.scenario_partitions["s4"]["val"]) + _read_rows(
            manifest.scenario_partitions["s4"]["test"]
        )

        assert {row["drug_id"] for row in train_rows}.isdisjoint({row["drug_id"] for row in eval_rows})
        assert {row["protein_id"] for row in train_rows}.isdisjoint({row["protein_id"] for row in eval_rows})


def _build_manifest(tmp_path: Path, seed: int) -> SplitManifest:
    splitter = DataSplitter(tmp_path / "splits")
    return splitter.create_splits(
        "davis",
        records=_grid_records(drugs=8, targets=8),
        preprocessing_version="prep-v1",
        split_version=f"split-{seed}",
        seed=seed,
    )
