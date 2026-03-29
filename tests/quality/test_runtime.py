"""Unit tests for runtime identity, checkpointing, and logging."""
from __future__ import annotations

import json

import pytest

from ugtsdti.core.errors import CheckpointCorruptedError
from ugtsdti.logging import CompositeLogger, FileLogger, WandbLogger
from ugtsdti.runtime import (
    CheckpointBundle,
    CheckpointIO,
    RuntimeAdapter,
    build_experiment_identity,
    build_reproducibility_key,
    seed_worker,
)


def test_reproducibility_key_is_stable():
    key1 = build_reproducibility_key(
        config_hash="abc",
        dataset_version="dataset-v1",
        preprocessing_version="prep-v1",
        split_version="split-v1",
        seed=7,
    )
    key2 = build_reproducibility_key(
        config_hash="abc",
        dataset_version="dataset-v1",
        preprocessing_version="prep-v1",
        split_version="split-v1",
        seed=7,
    )
    assert key1 == key2


def test_experiment_identity_separates_run_id_from_config_hash():
    identity1 = build_experiment_identity({"model": "a"})
    identity2 = build_experiment_identity({"model": "a"})
    assert identity1.config_hash == identity2.config_hash
    assert identity1.run_id != identity2.run_id


def test_runtime_adapter_resolves_operational_knobs():
    adapted = RuntimeAdapter().adapt(
        {
            "runtime": {
                "device": "cpu",
                "seed": 3,
                "artifacts_dir": "artifacts/test",
                "pin_memory": True,
                "data_dir": "data/input",
            }
        }
    )
    assert adapted["device"] == "cpu"
    assert adapted["seed"] == 3
    assert adapted["artifacts_dir"].endswith("artifacts/test")
    assert adapted["pin_memory"] is True
    assert adapted["data_dir"].endswith("data/input")


def test_checkpoint_corrupt_bundle_raises(tmp_path):
    path = tmp_path / "broken.ckpt.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CheckpointCorruptedError):
        CheckpointIO().load(path)


def test_checkpoint_config_or_dataset_mismatch_raises(tmp_path):
    bundle = CheckpointBundle(
        model_state={"w": 1},
        optimizer_state={},
        scheduler_state={},
        rng_state={},
        epoch=1,
        step=2,
        identity={"config_hash": "abc"},
        config={"model": "baseline"},
        dataset_metadata={"dataset": "davis"},
        split_metadata={"split_version": "v1"},
    )
    path = CheckpointIO().save(bundle, tmp_path / "checkpoint.json")

    with pytest.raises(CheckpointCorruptedError):
        CheckpointIO().load(path, expected_config_hash="other")

    with pytest.raises(CheckpointCorruptedError):
        CheckpointIO().load(path, expected_dataset="kiba")


def test_file_logger_and_composite_logger_write_outputs(tmp_path):
    file_logger = FileLogger(tmp_path / "logs")
    composite = CompositeLogger([file_logger, WandbLogger(project="test", enabled=False)])
    composite.log_metrics({"loss.total": 0.5}, step=3)
    composite.log_text("summary", "hello")
    composite.close()

    metrics_lines = (tmp_path / "logs" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(metrics_lines[0])
    assert payload["step"] == 3
    assert payload["metrics"]["loss.total"] == 0.5
    assert (tmp_path / "logs" / "summary.txt").read_text(encoding="utf-8") == "hello"


def test_seed_worker_returns_unique_derived_seed():
    assert seed_worker(1, 10) == 11
