"""Unit tests for data contracts, splitting, and validation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ugtsdti.core.errors import BatchSchemaError, MissingRawSnapshotError, ProcessedSplitMismatchError
from ugtsdti.core.schema import BatchSpec
from ugtsdti.data import DataLoaderFactory, DatasetVersion, DataSplitter, DataValidator, SplitManifest


def _grid_records(drugs: int = 10, targets: int = 10) -> list[dict[str, object]]:
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
                    "scenario": "raw",
                }
            )
    return records


def _cfg() -> dict:
    return {
        "graph": {
            "nodes": [
                {"name": "student_encoder", "type_key": "encoder.baseline", "inputs": ["drug_seq", "protein_seq"]},
            ]
        }
    }


def _write_dataset_version(tmp_path: Path, record_count: int) -> Path:
    dataset_version = DatasetVersion(
        dataset="davis",
        dataset_version="raw-v1",
        preprocessing_version="prep-v1",
        record_count=record_count,
        feature_keys=["drug_seq", "protein_seq", "labels", "scenario", "partition"],
    )
    dataset_path = tmp_path / "dataset_version.json"
    dataset_path.write_text(json.dumps(dataset_version.to_dict(), sort_keys=True), encoding="utf-8")
    return dataset_path


def _read_rows(path: str) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _build_partition_manifest(tmp_path: Path, *, rows: dict[tuple[str, str], list[dict[str, object]]]) -> SplitManifest:
    scenario_partitions: dict[str, dict[str, str]] = {}
    counts: dict[str, dict[str, int]] = {}
    for (scenario, partition), partition_rows in rows.items():
        scenario_dir = tmp_path / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        path = scenario_dir / f"{partition}.jsonl"
        payload = "\n".join(json.dumps(row, sort_keys=True) for row in partition_rows)
        path.write_text((payload + "\n") if payload else "", encoding="utf-8")
        scenario_partitions.setdefault(scenario, {})[partition] = str(path)
        counts.setdefault(scenario, {})[partition] = len(partition_rows)
    return SplitManifest(
        dataset="davis",
        preprocessing_version="prep-v1",
        split_version="split-v1",
        seed=7,
        scenarios=sorted(scenario_partitions),
        partitions=["train", "val", "test"],
        scenario_partitions=scenario_partitions,
        counts=counts,
        protocol_report={"protocol_version": "cold-start.v2"},
    )


def test_splitter_is_deterministic_for_same_seed(tmp_path):
    splitter = DataSplitter(tmp_path / "splits")
    records = _grid_records()
    manifest1 = splitter.create_splits(
        "davis",
        records=records,
        preprocessing_version="prep-v1",
        split_version="split-v1",
        seed=7,
    )
    manifest2 = splitter.create_splits(
        "davis",
        records=records,
        preprocessing_version="prep-v1",
        split_version="split-v1",
        seed=7,
    )

    assert manifest1.to_dict() == manifest2.to_dict()


def test_split_manifest_contains_required_protocol_metadata(tmp_path):
    splitter = DataSplitter(tmp_path / "splits")
    manifest = splitter.create_splits(
        "davis",
        records=_grid_records(),
        preprocessing_version="prep-v1",
        split_version="split-v2",
        seed=13,
        scenarios=["s1", "s2"],
    )

    assert manifest.dataset == "davis"
    assert manifest.preprocessing_version == "prep-v1"
    assert manifest.split_version == "split-v2"
    assert manifest.seed == 13
    assert manifest.scenarios == ["s1", "s2"]
    assert manifest.partitions == ["train", "val", "test"]
    assert set(manifest.scenario_partitions["s1"]) == {"train", "val", "test"}
    assert manifest.protocol_report["protocol_version"] == "cold-start.v2"
    assert "scenario_reports" in manifest.protocol_report


def test_splitter_s2_enforces_cold_drug_separation(tmp_path):
    splitter = DataSplitter(tmp_path / "splits")
    manifest = splitter.create_splits(
        "davis",
        records=_grid_records(),
        preprocessing_version="prep-v1",
        split_version="split-v1",
        seed=7,
    )

    train_rows = _read_rows(manifest.scenario_partitions["s2"]["train"])
    eval_rows = _read_rows(manifest.scenario_partitions["s2"]["val"]) + _read_rows(
        manifest.scenario_partitions["s2"]["test"]
    )

    train_drugs = {row["drug_id"] for row in train_rows}
    eval_drugs = {row["drug_id"] for row in eval_rows}

    assert eval_rows
    assert train_drugs.isdisjoint(eval_drugs)


def test_splitter_s3_enforces_cold_target_separation(tmp_path):
    splitter = DataSplitter(tmp_path / "splits")
    manifest = splitter.create_splits(
        "davis",
        records=_grid_records(),
        preprocessing_version="prep-v1",
        split_version="split-v1",
        seed=7,
    )

    train_rows = _read_rows(manifest.scenario_partitions["s3"]["train"])
    eval_rows = _read_rows(manifest.scenario_partitions["s3"]["val"]) + _read_rows(
        manifest.scenario_partitions["s3"]["test"]
    )

    train_targets = {row["protein_id"] for row in train_rows}
    eval_targets = {row["protein_id"] for row in eval_rows}

    assert eval_rows
    assert train_targets.isdisjoint(eval_targets)


def test_splitter_s4_enforces_fully_cold_separation(tmp_path):
    splitter = DataSplitter(tmp_path / "splits")
    manifest = splitter.create_splits(
        "davis",
        records=_grid_records(),
        preprocessing_version="prep-v1",
        split_version="split-v1",
        seed=7,
    )

    train_rows = _read_rows(manifest.scenario_partitions["s4"]["train"])
    eval_rows = _read_rows(manifest.scenario_partitions["s4"]["val"]) + _read_rows(
        manifest.scenario_partitions["s4"]["test"]
    )

    train_drugs = {row["drug_id"] for row in train_rows}
    eval_drugs = {row["drug_id"] for row in eval_rows}
    train_targets = {row["protein_id"] for row in train_rows}
    eval_targets = {row["protein_id"] for row in eval_rows}

    assert eval_rows
    assert train_drugs.isdisjoint(eval_drugs)
    assert train_targets.isdisjoint(eval_targets)


def test_data_validator_detects_s2_drug_leakage(tmp_path):
    manifest = _build_partition_manifest(
        tmp_path,
        rows={
            ("s2", "train"): [{"drug_id": "d1", "protein_id": "p1", "labels": 1.0, "scenario": "s2"}],
            ("s2", "val"): [{"drug_id": "d1", "protein_id": "p2", "labels": 0.0, "scenario": "s2"}],
            ("s2", "test"): [],
        },
    )
    dataset_path = _write_dataset_version(tmp_path, record_count=2)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), sort_keys=True), encoding="utf-8")

    with pytest.raises(ProcessedSplitMismatchError, match="Leakage detected for s2 on drug"):
        DataValidator().validate_artifacts(
            dataset_version_path=dataset_path,
            split_manifest_path=manifest_path,
        )


def test_data_validator_missing_labels_raises():
    validator = DataValidator()
    with pytest.raises(BatchSchemaError, match="labels"):
        validator.validate_batch(
            {"scenario": ["s1"], "drug_seq": ["AA"], "protein_seq": ["MK"]},
            BatchSpec(required_common=["labels", "scenario"], conditional_keys=["drug_seq", "protein_seq"]),
        )


def test_data_validator_version_mismatch_raises(tmp_path):
    dataset_version = DatasetVersion(
        dataset="davis",
        dataset_version="raw-v1",
        preprocessing_version="prep-v1",
        record_count=4,
        feature_keys=["labels"],
    )
    manifest = _build_partition_manifest(
        tmp_path,
        rows={
            ("s1", "train"): [{"drug_id": "d1", "protein_id": "p1", "labels": 1.0, "scenario": "s1"}],
            ("s1", "val"): [],
            ("s1", "test"): [],
        },
    )
    manifest = SplitManifest(
        dataset="kiba",
        preprocessing_version=manifest.preprocessing_version,
        split_version=manifest.split_version,
        seed=manifest.seed,
        scenarios=manifest.scenarios,
        partitions=manifest.partitions,
        scenario_partitions=manifest.scenario_partitions,
        counts=manifest.counts,
        protocol_report=manifest.protocol_report,
    )
    dataset_path = tmp_path / "dataset_version.json"
    manifest_path = tmp_path / "manifest.json"
    dataset_path.write_text(json.dumps(dataset_version.to_dict()), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

    with pytest.raises(ProcessedSplitMismatchError):
        DataValidator().validate_artifacts(
            dataset_version_path=dataset_path,
            split_manifest_path=manifest_path,
        )


def test_data_validator_nan_in_batch_raises():
    validator = DataValidator()
    with pytest.raises(BatchSchemaError, match="NaN/Inf"):
        validator.validate_batch(
            {"labels": [float("nan")], "scenario": ["s1"], "drug_seq": ["AA"], "protein_seq": ["MK"]},
            BatchSpec(required_common=["labels", "scenario"], conditional_keys=["drug_seq", "protein_seq"]),
        )


def test_loader_factory_reads_materialized_partition(tmp_path):
    train_rows = [
        {
            "drug_id": "d1",
            "protein_id": "p1",
            "drug_seq": "AA",
            "protein_seq": "MK",
            "labels": 1.0,
            "scenario": "s1",
            "partition": "train",
        },
        {
            "drug_id": "d2",
            "protein_id": "p2",
            "drug_seq": "BB",
            "protein_seq": "ML",
            "labels": 0.0,
            "scenario": "s1",
            "partition": "train",
        },
    ]
    manifest = _build_partition_manifest(
        tmp_path,
        rows={
            ("s1", "train"): train_rows,
            ("s1", "val"): [],
            ("s1", "test"): [],
        },
    )
    dataset_path = _write_dataset_version(tmp_path, record_count=2)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

    batches, batch_spec, loaded_version, loaded_manifest = DataLoaderFactory().build(
        cfg=_cfg(),
        dataset_version_path=dataset_path,
        split_manifest_path=manifest_path,
        batch_size=1,
        scenarios=["s1"],
        partition="train",
    )

    assert len(batches) == 2
    assert batch_spec.required_common == ["labels", "scenario"]
    assert loaded_version.dataset == "davis"
    assert loaded_manifest.split_version == "split-v1"


def test_loader_factory_can_select_multiple_scenarios(tmp_path):
    manifest = _build_partition_manifest(
        tmp_path,
        rows={
            ("s1", "train"): [],
            ("s1", "val"): [],
            ("s1", "test"): [
                {
                    "drug_id": "d1",
                    "protein_id": "p1",
                    "drug_seq": "AA",
                    "protein_seq": "MK",
                    "labels": 1.0,
                    "scenario": "s1",
                    "partition": "test",
                }
            ],
            ("s2", "train"): [],
            ("s2", "val"): [],
            ("s2", "test"): [
                {
                    "drug_id": "d2",
                    "protein_id": "p2",
                    "drug_seq": "BB",
                    "protein_seq": "ML",
                    "labels": 0.0,
                    "scenario": "s2",
                    "partition": "test",
                }
            ],
        },
    )
    dataset_path = _write_dataset_version(tmp_path, record_count=2)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

    batches, _, _, _ = DataLoaderFactory().build(
        cfg=_cfg(),
        dataset_version_path=dataset_path,
        split_manifest_path=manifest_path,
        batch_size=2,
        scenarios=["s1", "s2"],
        partition="test",
    )

    assert len(batches) == 1
    assert batches[0]["scenario"] == ["s1", "s2"]


def test_data_validator_missing_artifact_raises(tmp_path):
    with pytest.raises(MissingRawSnapshotError):
        DataValidator().validate_artifacts(
            dataset_version_path=tmp_path / "missing_dataset_version.json",
            split_manifest_path=tmp_path / "missing_manifest.json",
        )


def test_dataset_version_supports_backward_compatible_raw_version_alias():
    version = DatasetVersion.from_dict(
        {
            "dataset": "davis",
            "raw_version": "raw-v1",
            "preprocessing_version": "prep-v1",
            "record_count": 2,
            "feature_keys": ["labels"],
        }
    )

    assert version.dataset_version == "raw-v1"
    assert version.raw_version == "raw-v1"
    assert version.to_dict()["dataset_version"] == "raw-v1"
    assert version.to_dict()["raw_version"] == "raw-v1"
