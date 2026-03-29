"""Tests for CLI validation and sweep handling."""
from __future__ import annotations

import io
import json
from pathlib import Path

import yaml

from ugtsdti.cli import resolve_sweep_targets, run_cli
from ugtsdti.config.loader import ConfigLoader
from ugtsdti.config.normalize import ConfigNormalizer


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
    assert '"command": "train"' in buffer.getvalue()
    assert '"checkpoint":' in buffer.getvalue()
    assert '"logs_dir":' in buffer.getvalue()
    artifacts_dir = tmp_path / "artifacts"
    assert any(path.is_dir() for path in artifacts_dir.iterdir())
    checkpoint_dir = tmp_path / "checkpoints"
    assert checkpoint_dir.exists()


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
