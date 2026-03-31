"""Integration coverage for baseline portability configs and workflows."""

from __future__ import annotations

import builtins
import io
import json
import re
from pathlib import Path

import yaml

from ugtsdti.cli import run_cli
from ugtsdti.config.loader import ConfigLoader
from ugtsdti.config.normalize import ConfigNormalizer
from ugtsdti.config.validate import ConfigValidator


def _profile_root() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "profiles"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _baseline_profile(relative_path: str) -> dict:
    return ConfigLoader().load(_profile_root() / relative_path)


def _write_artifact_fixture(data_dir: Path) -> None:
    processed_dir = data_dir / "processed" / "davis" / "v1"
    split_dir = data_dir / "splits" / "davis" / "v1" / "v1"
    processed_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    rows = _scenario_rows()
    counts: dict[str, dict[str, int]] = {}
    scenario_partitions: dict[str, dict[str, str]] = {}
    for scenario, partitions in rows.items():
        scenario_dir = split_dir / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        counts[scenario] = {}
        scenario_partitions[scenario] = {}
        for partition, records in partitions.items():
            path = scenario_dir / f"{partition}.jsonl"
            payload = "\n".join(json.dumps(row, sort_keys=True) for row in records)
            path.write_text((payload + "\n") if payload else "", encoding="utf-8")
            counts[scenario][partition] = len(records)
            scenario_partitions[scenario][partition] = str(path)

    dataset_version = {
        "dataset": "davis",
        "dataset_version": "baseline-portability-fixture",
        "preprocessing_version": "v1",
        "record_count": sum(sum(len(records) for records in partitions.values()) for partitions in rows.values()),
        "feature_keys": ["drug_id", "protein_id", "drug_seq", "protein_seq", "labels", "scenario"],
    }
    (processed_dir / "dataset_version.json").write_text(
        json.dumps(dataset_version, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    (processed_dir / "records.jsonl").write_text("", encoding="utf-8")

    manifest = {
        "dataset": "davis",
        "preprocessing_version": "v1",
        "split_version": "v1",
        "seed": 7,
        "scenarios": ["s1", "s2", "s3", "s4"],
        "partitions": ["train", "val", "test"],
        "scenario_partitions": scenario_partitions,
        "counts": counts,
        "protocol_report": {
            "protocol_version": "baseline-portability-fixture.v1",
            "note": "Artifact-backed fixture for baseline portability tests.",
        },
    }
    (split_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")


def _scenario_rows() -> dict[str, dict[str, list[dict[str, object]]]]:
    base = {
        "s1": 1,
        "s2": 5,
        "s3": 9,
        "s4": 13,
    }
    scenario_rows: dict[str, dict[str, list[dict[str, object]]]] = {}
    for scenario, token in base.items():
        train_rows = [
            _row(f"{scenario}-drug-a", f"{scenario}-protein-a", token, 1.0, scenario),
            _row(f"{scenario}-drug-b", f"{scenario}-protein-b", token + 8, 0.0, scenario),
        ]
        val_rows = [
            _row(f"{scenario}-drug-c", f"{scenario}-protein-c", token + 1, 1.0, scenario),
            _row(f"{scenario}-drug-d", f"{scenario}-protein-d", token + 9, 0.0, scenario),
        ]
        test_rows = [
            _row(f"{scenario}-drug-e", f"{scenario}-protein-e", token + 2, 1.0, scenario),
            _row(f"{scenario}-drug-f", f"{scenario}-protein-f", token + 10, 0.0, scenario),
        ]
        scenario_rows[scenario] = {"train": train_rows, "val": val_rows, "test": test_rows}
    return scenario_rows


def _row(drug_id: str, protein_id: str, token: int, label: float, scenario: str) -> dict[str, object]:
    return {
        "drug_id": drug_id,
        "protein_id": protein_id,
        "drug_seq": [token, token, token, token],
        "protein_seq": [token, token, token, token, token],
        "labels": label,
        "scenario": scenario,
    }


def _write_temp_config(tmp_path: Path, cfg: dict, name: str) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _summary_lines(payload: str) -> list[str]:
    return [line for line in payload.splitlines() if line.strip()]


def _last_json_line(payload: str) -> dict:
    for line in reversed(_summary_lines(payload)):
        if not line.startswith("{"):
            continue
        decoded = json.loads(line)
        if isinstance(decoded, dict):
            return decoded
    return {}


def test_baseline_profiles_smoke_validate_and_preserve_baseline_surface():
    loader = ConfigLoader()
    validator = ConfigValidator()
    normalizer = ConfigNormalizer()
    expected_type_keys = ["encoder.simple_drug", "encoder.cnn_protein", "fusion.concat", "head.mlp"]

    for relative_path in (
        "baseline_local.yaml",
        "baseline_kaggle.yaml",
    ):
        raw = loader.load(_profile_root() / relative_path)
        validator.validate(raw)
        normalized = normalizer.normalize(raw).to_dict()
        node_types = [node["type_key"] for node in normalized["graph"]["nodes"]]

        assert node_types == expected_type_keys
        assert normalized["interaction"]["order"] == ["noop"]
        assert normalized["decision"]["type"] == "identity"
        assert normalized["roles"] == {"student": {"outputs": ["head.logits"], "aggregation": "first"}}
        assert normalized["data"]["dataset"] == "davis"
        assert normalized["scenario"]["eval"] == ["s1", "s2", "s3", "s4"]


def test_baseline_local_profile_runs_train_then_eval_on_artifact_fixture(tmp_path):
    _write_artifact_fixture(tmp_path / "data")
    cfg = _baseline_profile("baseline_local.yaml")
    cfg["runtime"]["data_dir"] = str(tmp_path / "data")
    cfg["runtime"]["artifacts_dir"] = str(tmp_path / "artifacts")
    cfg["runtime"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
    cfg["training"]["loop"]["epochs"] = 2
    cfg["training"]["loop"]["summary_every_steps"] = 1
    config_path = _write_temp_config(tmp_path, cfg, "baseline_local.yaml")

    train_buffer = io.StringIO()
    assert run_cli(["train", str(config_path)], stdout=train_buffer) == 0
    train_summary = _last_json_line(train_buffer.getvalue())
    assert train_summary["command"] == "train"
    assert train_summary["best_checkpoint"].endswith(".best.pt")
    assert train_summary["final_eval_metrics"]["metrics.auroc"] >= 0.0
    assert Path(train_summary["artifact_bundle"]).exists()

    cfg["runtime"]["checkpoint_path"] = train_summary["best_checkpoint"]
    eval_config_path = _write_temp_config(tmp_path, cfg, "baseline_local_eval.yaml")
    eval_buffer = io.StringIO()
    assert run_cli(["eval", str(eval_config_path)], stdout=eval_buffer) == 0
    eval_summary = _last_json_line(eval_buffer.getvalue())

    assert eval_summary["command"] == "eval"
    assert eval_summary["checkpoint"] == train_summary["best_checkpoint"]
    assert "metrics.auroc" in eval_summary["metrics"]
    assert "metrics.s4.auprc" in eval_summary["metrics"]


def test_baseline_local_profile_supports_wandb_override_when_wandb_is_unavailable(tmp_path, monkeypatch):
    _write_artifact_fixture(tmp_path / "data")
    cfg = _baseline_profile("baseline_local.yaml")
    cfg["runtime"]["data_dir"] = str(tmp_path / "data")
    cfg["runtime"]["artifacts_dir"] = str(tmp_path / "artifacts")
    cfg["runtime"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
    cfg["training"]["loop"]["epochs"] = 1
    cfg["logging"]["backend"] = "wandb"
    config_path = _write_temp_config(tmp_path, cfg, "baseline_local_wandb.yaml")

    original_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "wandb":
            raise ImportError("wandb unavailable in test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    buffer = io.StringIO()

    assert run_cli(["train", str(config_path)], stdout=buffer) == 0
    summary = _last_json_line(buffer.getvalue())
    assert summary["command"] == "train"
    assert Path(summary["artifact_bundle"]).exists()


def test_baseline_real_profiles_fail_closed_when_required_artifacts_are_missing(tmp_path):
    cfg = _baseline_profile("baseline_local.yaml")
    cfg["runtime"]["data_dir"] = str(tmp_path / "missing-data")
    cfg["runtime"]["artifacts_dir"] = str(tmp_path / "artifacts")
    cfg["runtime"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
    config_path = _write_temp_config(tmp_path, cfg, "baseline_missing.yaml")
    buffer = io.StringIO()

    assert run_cli(["train", str(config_path)], stdout=buffer) == 1
    assert "auto_prepare is disabled" in buffer.getvalue()


def test_readme_references_existing_baseline_files():
    repo_root = _repo_root()
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    referenced = set(
        re.findall(r"(configs/[A-Za-z0-9_./-]+\.ya?ml|scripts/[A-Za-z0-9_./-]+|examples/[A-Za-z0-9_./-]+)", readme)
    )

    expected = {
        "configs/baseline_reference.yaml",
        "configs/profiles/baseline_local.yaml",
        "configs/profiles/baseline_kaggle.yaml",
        "scripts/baseline.sh",
        "scripts/prepare_baseline_artifacts.py",
        "examples/baseline.py",
    }

    assert expected.issubset(referenced)
    for relative_path in expected:
        assert (repo_root / relative_path).exists(), relative_path
