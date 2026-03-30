"""Tests for CLI validation and sweep handling."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import yaml

from ugtsdti.cli import resolve_sweep_targets, run_cli
from ugtsdti.config.loader import ConfigLoader
from ugtsdti.config.normalize import ConfigNormalizer
from ugtsdti.config.validate import ConfigValidator
from ugtsdti.runtime import CheckpointIO


def _valid_cfg() -> dict:
    return {
        "version": "1.0",
        "data": {"dataset": "davis"},
        "scenario": {"train": "s1", "eval": ["s1"]},
        "modalities": {"available": ["sequence"], "student": {"uses": ["sequence"]}},
        "graph": {
            "nodes": {
                "student_encoder": {
                    "type_key": "encoder.student",
                    "inputs": ["drug_seq", "protein_seq"],
                    "output_attrs": ["embedding"],
                },
                "student_head": {
                    "type_key": "head.student",
                    "inputs": ["student_encoder.embedding"],
                    "output_attrs": ["logits"],
                },
            }
        },
        "roles": {"student": {"outputs": ["student_head.logits"]}},
        "interaction": {
            "order": ["noop"],
            "dependencies": {},
            "noop": {"type": "noop", "inputs": ["student.logits"], "params": {"enabled": True}},
        },
        "decision": {"type": "identity", "strategy": "identity", "source_key": "student.logits"},
        "training": {
            "teacher": {"freeze": True},
            "student": {"freeze": False},
            "kd": {"schedule": "constant"},
            "gate": {"trainable": False},
        },
        "loss": {"type": "hard", "hard_weight": 1.0, "map": {}},
        "sweep": {"parameters": {"decision.strategy": ["identity", "soft"]}},
    }


def _write_partition_manifest(
    base_dir: Path,
    *,
    train_rows: list[dict[str, object]],
    test_rows: list[dict[str, object]] | None = None,
) -> None:
    scenario_dir = base_dir / "s1"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    val_rows: list[dict[str, object]] = []
    test_payload = list(test_rows or [])
    for partition, rows in (("train", train_rows), ("val", val_rows), ("test", test_payload)):
        path = scenario_dir / f"{partition}.jsonl"
        payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
        path.write_text((payload + "\n") if payload else "", encoding="utf-8")
    (base_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "davis",
                "preprocessing_version": "v1",
                "split_version": "s1_v1",
                "seed": 7,
                "scenarios": ["s1"],
                "partitions": ["train", "val", "test"],
                "scenario_partitions": {
                    "s1": {
                        "train": str(scenario_dir / "train.jsonl"),
                        "val": str(scenario_dir / "val.jsonl"),
                        "test": str(scenario_dir / "test.jsonl"),
                    }
                },
                "counts": {"s1": {"train": len(train_rows), "val": 0, "test": len(test_payload)}},
                "protocol_report": {"protocol_version": "cold-start.v2"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_cli_validate_invalid_config_fails_fast(tmp_path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump({"version": "1.0"}), encoding="utf-8")
    buffer = io.StringIO()

    exit_code = run_cli(["validate", str(config_path)], stdout=buffer)

    assert exit_code == 1
    assert "ERROR:" in buffer.getvalue()


def test_cli_validate_does_not_dispatch_runtime_handlers(tmp_path):
    config_path = tmp_path / "valid.yaml"
    config_path.write_text(yaml.safe_dump(_valid_cfg()), encoding="utf-8")
    buffer = io.StringIO()
    called: list[str] = []

    exit_code = run_cli(
        ["validate", str(config_path)],
        stdout=buffer,
        handlers={"validate": lambda cfg, args: called.append("validate")},
    )

    assert exit_code == 0
    assert called == []
    assert buffer.getvalue().strip() == "VALID"


def test_cli_validate_custom_plugin_without_registrar_fails_closed(tmp_path):
    cfg = _valid_cfg()
    cfg["graph"]["nodes"] = [
        {
            "name": "student_encoder",
            "type_key": "encoder.echo",
            "inputs": ["drug_seq"],
            "output_attrs": ["embedding"],
        },
        {
            "name": "student_head",
            "type_key": "head.linear",
            "inputs": ["student_encoder.embedding"],
            "output_attrs": ["logits"],
        },
    ]
    config_path = tmp_path / "missing_registrar.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    buffer = io.StringIO()

    exit_code = run_cli(["validate", str(config_path)], stdout=buffer)

    assert exit_code == 1
    assert "unregistered plugin type 'encoder.echo'" in buffer.getvalue()


def test_cli_validate_invalid_registrar_path_fails_closed(tmp_path):
    cfg = _valid_cfg()
    cfg["graph"]["nodes"] = [
        {
            "name": "student_encoder",
            "type_key": "encoder.echo",
            "inputs": ["drug_seq"],
            "output_attrs": ["embedding"],
        },
        {
            "name": "student_head",
            "type_key": "head.linear",
            "inputs": ["student_encoder.embedding"],
            "output_attrs": ["logits"],
        },
    ]
    cfg["runtime"] = {"plugin_registrars": ["tests.support_plugins.missing_registrar"]}
    config_path = tmp_path / "invalid_registrar.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    buffer = io.StringIO()

    exit_code = run_cli(["validate", str(config_path)], stdout=buffer)

    assert exit_code == 1
    assert "not callable" in buffer.getvalue() or "Could not import runtime registrar module" in buffer.getvalue()


def test_cli_validate_uses_injected_validator_with_runtime_registries(tmp_path):
    class TrackingValidator(ConfigValidator):
        def __init__(self) -> None:
            super().__init__()
            self.called = False

        def validate(self, cfg: dict[str, object]) -> None:
            self.called = True
            assert self._graph_registry is not None
            assert self._interaction_registry is not None
            self._graph_registry.get_spec("encoder.echo")
            self._interaction_registry.get_spec("noop.echo")
            super().validate(cfg)

    cfg = _valid_cfg()
    cfg["graph"]["nodes"] = [
        {
            "name": "student_encoder",
            "type_key": "encoder.echo",
            "inputs": ["drug_seq"],
            "output_attrs": ["embedding"],
        },
        {
            "name": "student_head",
            "type_key": "head.linear",
            "inputs": ["student_encoder.embedding"],
            "output_attrs": ["logits"],
        },
    ]
    cfg["interaction"]["noop"]["type"] = "noop.echo"
    cfg["runtime"] = {"plugin_registrars": ["tests.support_plugins.register_test_runtime_plugins"]}
    config_path = tmp_path / "custom_validator.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    buffer = io.StringIO()
    validator = TrackingValidator()

    exit_code = run_cli(["validate", str(config_path)], stdout=buffer, validator=validator)

    assert exit_code == 0
    assert validator.called is True
    assert buffer.getvalue().strip() == "VALID"


def test_cli_sweep_resolves_normalized_parameter_targets(tmp_path):
    cfg = _valid_cfg()
    cfg["sweep"] = {
        "parameters": [
            {"target": "loss.map.kd.weight", "values": [0.1, 0.3]},
            {"target": "decision.strategy", "values": ["identity", "soft"]},
        ]
    }
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    buffer = io.StringIO()
    called: list[str] = []

    exit_code = run_cli(
        ["sweep", str(config_path)],
        stdout=buffer,
        handlers={"sweep": lambda cfg, args: called.append(args.command)},
    )

    assert exit_code == 0
    assert resolve_sweep_targets(ConfigNormalizer().normalize(ConfigLoader().load(config_path)).to_dict()) == [
        "decision.strategy",
        "loss.map.kd.weight",
    ]
    assert called == ["sweep"]
    assert "run_id=" in buffer.getvalue()


def test_cli_train_without_handler_fails_closed(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(_valid_cfg()), encoding="utf-8")
    buffer = io.StringIO()

    exit_code = run_cli(["train", str(config_path)], stdout=buffer, handlers={})

    assert exit_code == 1
    assert "No execution handler is configured" in buffer.getvalue()


def test_cli_train_runs_with_default_handlers_when_artifacts_exist(tmp_path):
    records_dir = tmp_path / "data" / "splits" / "davis" / "v1" / "s1_v1"
    processed_dir = tmp_path / "data" / "processed" / "davis" / "v1"
    records_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    (processed_dir / "dataset_version.json").write_text(
        json.dumps(
            {
                "dataset": "davis",
                "dataset_version": "raw-v1",
                "preprocessing_version": "v1",
                "record_count": 2,
                "feature_keys": ["drug_seq", "protein_seq", "labels", "scenario"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_partition_manifest(
        records_dir,
        train_rows=[
            {"drug_seq": "AA", "protein_seq": "MK", "labels": 1.0, "scenario": "s1", "partition": "train"},
            {"drug_seq": "BB", "protein_seq": "ML", "labels": 0.0, "scenario": "s1", "partition": "train"},
        ],
    )

    cfg = _valid_cfg()
    cfg["data"]["preprocessing_version"] = "v1"
    cfg["data"]["split_version"] = "s1_v1"
    cfg["graph"]["nodes"] = [
        {
            "name": "student_encoder",
            "type_key": "encoder.baseline",
            "inputs": ["drug_seq", "protein_seq"],
            "output_attrs": ["embedding"],
        },
        {
            "name": "student_head",
            "type_key": "head.linear",
            "inputs": ["student_encoder.embedding"],
            "output_attrs": ["logits"],
        },
    ]
    cfg["runtime"] = {
        "data_dir": str(tmp_path / "data"),
        "artifacts_dir": str(tmp_path / "artifacts"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "seed": 7,
    }
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    buffer = io.StringIO()

    exit_code = run_cli(["train", str(config_path)], stdout=buffer)

    assert exit_code == 0
    lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
    assert lines[0].startswith("run_id=")
    summary = json.loads(lines[-1])
    run_id = lines[0].split()[0].split("=", 1)[1]
    assert summary["command"] == "train"
    assert run_id in summary["checkpoint"]
    assert run_id in summary["logs_dir"]
    assert summary["artifact_bundle"].endswith(run_id)
    artifacts_dir = tmp_path / "artifacts"
    assert any(path.is_dir() for path in artifacts_dir.iterdir())
    checkpoint_dir = tmp_path / "checkpoints"
    assert checkpoint_dir.exists()
    artifact_identity = json.loads((artifacts_dir / run_id / "identity.json").read_text(encoding="utf-8"))
    log_identity = json.loads((Path(summary["logs_dir"]) / "identity.json").read_text(encoding="utf-8"))
    checkpoint_bundle = CheckpointIO().load(summary["checkpoint"])

    assert artifact_identity["run_id"] == run_id
    assert log_identity["run_id"] == run_id
    assert checkpoint_bundle.identity["run_id"] == run_id
    assert artifact_identity["config_hash"] == checkpoint_bundle.identity["config_hash"]
    assert artifact_identity["reproducibility_key"] == checkpoint_bundle.identity["reproducibility_key"]


def test_cli_train_final_bundle_prefers_eval_metrics_when_eval_cadence_is_enabled(tmp_path):
    records_dir = tmp_path / "data" / "splits" / "davis" / "v1" / "s1_v1"
    processed_dir = tmp_path / "data" / "processed" / "davis" / "v1"
    records_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    (processed_dir / "dataset_version.json").write_text(
        json.dumps(
            {
                "dataset": "davis",
                "dataset_version": "raw-v1",
                "preprocessing_version": "v1",
                "record_count": 3,
                "feature_keys": ["drug_seq", "protein_seq", "labels", "scenario"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_partition_manifest(
        records_dir,
        train_rows=[
            {"drug_seq": "AA", "protein_seq": "MK", "labels": 1.0, "scenario": "s1", "partition": "train"},
            {"drug_seq": "BB", "protein_seq": "ML", "labels": 0.0, "scenario": "s1", "partition": "train"},
        ],
        test_rows=[
            {"drug_seq": "CC", "protein_seq": "MM", "labels": 1.0, "scenario": "s1", "partition": "test"},
        ],
    )

    cfg = _valid_cfg()
    cfg["data"]["preprocessing_version"] = "v1"
    cfg["data"]["split_version"] = "s1_v1"
    cfg["graph"]["nodes"] = [
        {
            "name": "student_encoder",
            "type_key": "encoder.baseline",
            "inputs": ["drug_seq", "protein_seq"],
            "output_attrs": ["embedding"],
        },
        {
            "name": "student_head",
            "type_key": "head.linear",
            "inputs": ["student_encoder.embedding"],
            "output_attrs": ["logits"],
        },
    ]
    cfg["training"]["loop"] = {
        "epochs": 1,
        "checkpoint_every_epochs": 1,
        "summary_every_steps": 1,
        "eval_every_epochs": 1,
        "eval_partition": "test",
    }
    cfg["runtime"] = {
        "data_dir": str(tmp_path / "data"),
        "artifacts_dir": str(tmp_path / "artifacts"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "batch_size": 1,
        "seed": 7,
    }
    config_path = tmp_path / "train_eval_bundle.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    buffer = io.StringIO()

    assert run_cli(["train", str(config_path)], stdout=buffer) == 0
    lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
    run_id = lines[0].split()[0].split("=", 1)[1]
    summary = json.loads(lines[-1])
    metrics_payload = json.loads((tmp_path / "artifacts" / run_id / "metrics.json").read_text(encoding="utf-8"))

    assert summary["final_eval_metrics"]
    for key, value in summary["final_eval_metrics"].items():
        assert metrics_payload[key] == pytest.approx(value)


def test_cli_train_runs_epoch_loop_with_summary_and_eval_cadence(tmp_path):
    records_dir = tmp_path / "data" / "splits" / "davis" / "v1" / "s1_v1"
    processed_dir = tmp_path / "data" / "processed" / "davis" / "v1"
    records_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    (processed_dir / "dataset_version.json").write_text(
        json.dumps(
            {
                "dataset": "davis",
                "dataset_version": "raw-v1",
                "preprocessing_version": "v1",
                "record_count": 4,
                "feature_keys": ["drug_seq", "protein_seq", "labels", "scenario"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_partition_manifest(
        records_dir,
        train_rows=[
            {"drug_seq": "AA", "protein_seq": "MK", "labels": 1.0, "scenario": "s1", "partition": "train"},
            {"drug_seq": "BB", "protein_seq": "ML", "labels": 0.0, "scenario": "s1", "partition": "train"},
        ],
        test_rows=[
            {"drug_seq": "CC", "protein_seq": "MM", "labels": 1.0, "scenario": "s1", "partition": "test"},
        ],
    )

    cfg = _valid_cfg()
    cfg["data"]["preprocessing_version"] = "v1"
    cfg["data"]["split_version"] = "s1_v1"
    cfg["graph"]["nodes"] = [
        {
            "name": "student_encoder",
            "type_key": "encoder.baseline",
            "inputs": ["drug_seq", "protein_seq"],
            "output_attrs": ["embedding"],
        },
        {
            "name": "student_head",
            "type_key": "head.linear",
            "inputs": ["student_encoder.embedding"],
            "output_attrs": ["logits"],
        },
    ]
    cfg["training"]["loop"] = {
        "epochs": 2,
        "checkpoint_every_epochs": 1,
        "summary_every_steps": 1,
        "eval_every_epochs": 1,
        "eval_partition": "test",
    }
    cfg["runtime"] = {
        "data_dir": str(tmp_path / "data"),
        "artifacts_dir": str(tmp_path / "artifacts"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "batch_size": 1,
        "seed": 7,
    }
    config_path = tmp_path / "train_loop.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    buffer = io.StringIO()

    exit_code = run_cli(["train", str(config_path)], stdout=buffer)

    assert exit_code == 0
    lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
    assert any('"event": "train_step"' in line for line in lines[1:])
    assert any('"event": "eval_epoch"' in line for line in lines[1:])
    summary = json.loads(lines[-1])
    assert summary["epochs"] == 2
    assert summary["steps"] == 4
    assert summary["checkpoint"].endswith(".pt")
    assert summary["eval_partition"] == "test"
    metrics_lines = (Path(summary["logs_dir"]) / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(metrics_lines) >= 4


def test_cli_train_checkpoint_can_be_reused_by_eval(tmp_path):
    records_dir = tmp_path / "data" / "splits" / "davis" / "v1" / "s1_v1"
    processed_dir = tmp_path / "data" / "processed" / "davis" / "v1"
    records_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    (processed_dir / "dataset_version.json").write_text(
        json.dumps(
            {
                "dataset": "davis",
                "dataset_version": "raw-v1",
                "preprocessing_version": "v1",
                "record_count": 4,
                "feature_keys": ["drug_seq", "protein_seq", "labels", "scenario"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_partition_manifest(
        records_dir,
        train_rows=[
            {"drug_seq": "AA", "protein_seq": "MK", "labels": 1.0, "scenario": "s1", "partition": "train"},
            {"drug_seq": "BB", "protein_seq": "ML", "labels": 0.0, "scenario": "s1", "partition": "train"},
        ],
        test_rows=[
            {"drug_seq": "CC", "protein_seq": "MM", "labels": 1.0, "scenario": "s1", "partition": "test"},
        ],
    )

    cfg = _valid_cfg()
    cfg["data"]["preprocessing_version"] = "v1"
    cfg["data"]["split_version"] = "s1_v1"
    cfg["graph"]["nodes"] = [
        {
            "name": "student_encoder",
            "type_key": "encoder.baseline",
            "inputs": ["drug_seq", "protein_seq"],
            "output_attrs": ["embedding"],
        },
        {
            "name": "student_head",
            "type_key": "head.linear",
            "inputs": ["student_encoder.embedding"],
            "output_attrs": ["logits"],
        },
    ]
    cfg["training"]["loop"] = {"epochs": 1, "checkpoint_every_epochs": 1, "summary_every_steps": 1}
    cfg["runtime"] = {
        "data_dir": str(tmp_path / "data"),
        "artifacts_dir": str(tmp_path / "artifacts"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "batch_size": 1,
        "seed": 7,
    }
    train_path = tmp_path / "train.yaml"
    train_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    train_buffer = io.StringIO()

    assert run_cli(["train", str(train_path)], stdout=train_buffer) == 0
    train_summary = json.loads([line for line in train_buffer.getvalue().splitlines() if line.strip()][-1])

    eval_cfg = json.loads(json.dumps(cfg))
    eval_cfg["runtime"]["checkpoint_path"] = train_summary["checkpoint"]
    eval_path = tmp_path / "eval.yaml"
    eval_path.write_text(yaml.safe_dump(eval_cfg), encoding="utf-8")
    eval_buffer = io.StringIO()

    assert run_cli(["eval", str(eval_path)], stdout=eval_buffer) == 0
    eval_summary = json.loads([line for line in eval_buffer.getvalue().splitlines() if line.strip()][-1])
    assert eval_summary["checkpoint"] == train_summary["checkpoint"]
    artifact_run_dirs = [
        path for path in (tmp_path / "artifacts").iterdir() if path.is_dir() and not path.name.startswith("_")
    ]
    assert artifact_run_dirs
    latest_bundle = max(artifact_run_dirs, key=lambda path: path.stat().st_mtime)
    model_path = latest_bundle / "model.pt"
    assert model_path.exists()
    checkpoint_bundle = CheckpointIO().load(train_summary["checkpoint"])
    artifact_model = CheckpointIO()._load_payload(model_path)
    assert artifact_model["student_head"]["scale"].shape == checkpoint_bundle.model_state["student_head"]["scale"].shape


def test_cli_train_can_register_runtime_plugins_from_config(tmp_path):
    records_dir = tmp_path / "data" / "splits" / "davis" / "v1" / "s1_v1"
    processed_dir = tmp_path / "data" / "processed" / "davis" / "v1"
    records_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    (processed_dir / "dataset_version.json").write_text(
        json.dumps(
            {
                "dataset": "davis",
                "dataset_version": "raw-v1",
                "preprocessing_version": "v1",
                "record_count": 1,
                "feature_keys": ["drug_seq", "labels", "scenario"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_partition_manifest(
        records_dir,
        train_rows=[{"drug_seq": "AA", "labels": 1.0, "scenario": "s1", "partition": "train"}],
    )
    cfg = _valid_cfg()
    cfg["data"]["preprocessing_version"] = "v1"
    cfg["data"]["split_version"] = "s1_v1"
    cfg["graph"]["nodes"] = [
        {
            "name": "student_encoder",
            "type_key": "encoder.echo",
            "inputs": ["drug_seq"],
            "output_attrs": ["embedding"],
        },
        {
            "name": "student_head",
            "type_key": "head.linear",
            "inputs": ["student_encoder.embedding"],
            "output_attrs": ["logits"],
        },
    ]
    cfg["interaction"] = {
        "order": ["noop"],
        "dependencies": {},
        "noop": {"type": "noop.echo", "inputs": ["student.logits"], "params": {"enabled": True}},
    }
    cfg["runtime"] = {
        "data_dir": str(tmp_path / "data"),
        "artifacts_dir": str(tmp_path / "artifacts"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "seed": 7,
        "plugin_registrars": ["tests.support_plugins.register_test_runtime_plugins"],
    }
    config_path = tmp_path / "plugin_train.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    buffer = io.StringIO()

    exit_code = run_cli(["train", str(config_path)], stdout=buffer)

    assert exit_code == 0
    assert '"command": "train"' in buffer.getvalue()


def test_cli_eval_fails_when_requested_scenario_has_no_materialized_rows(tmp_path):
    records_dir = tmp_path / "data" / "splits" / "davis" / "v1" / "s1_v1"
    processed_dir = tmp_path / "data" / "processed" / "davis" / "v1"
    records_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    (processed_dir / "dataset_version.json").write_text(
        json.dumps(
            {
                "dataset": "davis",
                "dataset_version": "raw-v1",
                "preprocessing_version": "v1",
                "record_count": 1,
                "feature_keys": ["drug_seq", "protein_seq", "labels", "scenario"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_partition_manifest(
        records_dir,
        train_rows=[{"drug_seq": "AA", "protein_seq": "MK", "labels": 1.0, "scenario": "s1", "partition": "train"}],
        test_rows=[{"drug_seq": "BB", "protein_seq": "ML", "labels": 0.0, "scenario": "s1", "partition": "test"}],
    )

    cfg = _valid_cfg()
    cfg["data"]["preprocessing_version"] = "v1"
    cfg["data"]["split_version"] = "s1_v1"
    cfg["scenario"]["eval"] = ["s1", "s4"]
    cfg["graph"]["nodes"] = [
        {
            "name": "student_encoder",
            "type_key": "encoder.baseline",
            "inputs": ["drug_seq", "protein_seq"],
            "output_attrs": ["embedding"],
        },
        {
            "name": "student_head",
            "type_key": "head.linear",
            "inputs": ["student_encoder.embedding"],
            "output_attrs": ["logits"],
        },
    ]
    cfg["runtime"] = {
        "data_dir": str(tmp_path / "data"),
        "artifacts_dir": str(tmp_path / "artifacts"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "seed": 7,
    }
    config_path = tmp_path / "eval_missing_scenario.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    buffer = io.StringIO()

    exit_code = run_cli(["eval", str(config_path)], stdout=buffer)

    assert exit_code == 1
    assert "s4" in buffer.getvalue()
