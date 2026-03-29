"""Unit tests for data contracts, splitting, and validation."""
from __future__ import annotations

import json

import pytest

from ugtsdti.core.errors import BatchSchemaError, MissingRawSnapshotError, ProcessedSplitMismatchError
from ugtsdti.core.schema import BatchSpec
from ugtsdti.data import DataLoaderFactory, DatasetVersion, DataSplitter, DataValidator, SplitManifest


def _records() -> list[dict[str, object]]:
    return [
        {"drug_id": "d1", "protein_id": "p1", "drug_seq": "AA", "protein_seq": "MK", "labels": 1.0, "scenario": "s1"},
        {"drug_id": "d1", "protein_id": "p2", "drug_seq": "AA", "protein_seq": "ML", "labels": 0.0, "scenario": "s1"},
        {"drug_id": "d2", "protein_id": "p1", "drug_seq": "BB", "protein_seq": "MK", "labels": 1.0, "scenario": "s1"},
        {"drug_id": "d2", "protein_id": "p2", "drug_seq": "BB", "protein_seq": "ML", "labels": 0.0, "scenario": "s1"},
    ]


def _cfg() -> dict:
    return {
        "graph": {
            "nodes": [
                {"name": "student_encoder", "type_key": "encoder.baseline", "inputs": ["drug_seq", "protein_seq"]},
            ]
        }
    }


def test_splitter_is_deterministic_for_same_seed(tmp_path):
    splitter = DataSplitter(tmp_path / "splits")
    manifest1 = splitter.create_splits(
        "davis",
        records=_records(),
        preprocessing_version="prep-v1",
        split_version="split-v1",
        seed=7,
    )
    manifest2 = splitter.create_splits(
        "davis",
        records=_records(),
        preprocessing_version="prep-v1",
        split_version="split-v1",
        seed=7,
    )

    assert manifest1.to_dict() == manifest2.to_dict()


def test_split_manifest_contains_required_metadata(tmp_path):
    splitter = DataSplitter(tmp_path / "splits")
    manifest = splitter.create_splits(
        "davis",
        records=_records(),
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
    assert set(manifest.split_paths) == {"s1", "s2"}


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
        raw_version="raw-v1",
        preprocessing_version="prep-v1",
        record_count=4,
        feature_keys=["labels"],
    )
    manifest = SplitManifest(
        dataset="kiba",
        preprocessing_version="prep-v1",
        split_version="split-v1",
        seed=1,
        scenarios=["s1"],
        split_paths={"s1": str(tmp_path / "s1.jsonl")},
        counts={"s1": 1},
    )
    (tmp_path / "s1.jsonl").write_text('{"labels": 1}\n', encoding="utf-8")
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


def test_loader_factory_reads_materialized_split(tmp_path):
    records_path = tmp_path / "s1.jsonl"
    records_path.write_text(
        "\n".join(
            [
                json.dumps(_records()[0], sort_keys=True),
                json.dumps(_records()[1], sort_keys=True),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_version = DatasetVersion(
        dataset="davis",
        raw_version="davis",
        preprocessing_version="prep-v1",
        record_count=2,
        feature_keys=["drug_seq", "protein_seq", "labels", "scenario"],
    )
    manifest = SplitManifest(
        dataset="davis",
        preprocessing_version="prep-v1",
        split_version="split-v1",
        seed=1,
        scenarios=["s1"],
        split_paths={"s1": str(records_path)},
        counts={"s1": 2},
    )
    dataset_path = tmp_path / "dataset_version.json"
    manifest_path = tmp_path / "manifest.json"
    dataset_path.write_text(json.dumps(dataset_version.to_dict()), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

    batches, batch_spec, loaded_version, loaded_manifest = DataLoaderFactory().build(
        cfg=_cfg(),
        dataset_version_path=dataset_path,
        split_manifest_path=manifest_path,
        batch_size=1,
    )

    assert len(batches) == 2
    assert batch_spec.required_common == ["labels", "scenario"]
    assert loaded_version.dataset == "davis"
    assert loaded_manifest.split_version == "split-v1"


def test_data_validator_missing_artifact_raises(tmp_path):
    with pytest.raises(MissingRawSnapshotError):
        DataValidator().validate_artifacts(
            dataset_version_path=tmp_path / "missing_dataset_version.json",
            split_manifest_path=tmp_path / "missing_manifest.json",
        )
