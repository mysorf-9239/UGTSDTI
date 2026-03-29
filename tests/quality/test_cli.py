"""Tests for CLI validation and sweep handling."""
from __future__ import annotations

import io

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

    exit_code = run_cli(["sweep", str(config_path)], stdout=buffer)

    assert exit_code == 0
    assert resolve_sweep_targets(ConfigNormalizer().normalize(ConfigLoader().load(config_path)).to_dict()) == [
        "decision.strategy",
        "loss.map.kd.weight",
    ]
    assert "run_id=" in buffer.getvalue()
