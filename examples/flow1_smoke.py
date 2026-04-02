"""Flow 1 Teacher-Student smoke run: synthetic fixture → validate → train → eval."""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ugtsdti.cli.main import run_cli  # noqa: E402


def main() -> int:
    root = ROOT_DIR
    data_dir = root / "data"
    artifacts_dir = root / "artifacts" / "flow1_smoke"
    checkpoint_dir = root / "checkpoints" / "flow1_smoke"
    config_path = root / "configs" / "flow1_teacher_student.yaml"

    if not config_path.exists():
        print(f"[flow1_smoke] ERROR: config not found at {config_path}")
        return 1

    print("[flow1_smoke] Preparing synthetic fixture artifacts...")
    _prepare_fixture_artifacts(data_dir)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["runtime"] = {
        "data_dir": str(data_dir),
        "artifacts_dir": str(artifacts_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "batch_size": 2,
        "seed": 42,
        "device": "cpu",
        "deterministic": True,
    }
    loop_cfg = dict(config.get("training", {}).get("loop", {}))
    loop_cfg.update(
        {
            "epochs": 3,
            "checkpoint_every_epochs": 1,
            "summary_every_steps": 1,
            "eval_every_epochs": 1,
            "eval_partition": "test",
            "early_stopping": {"enabled": False},
        }
    )
    config["training"]["loop"] = loop_cfg

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        temp_config = Path(handle.name)
        yaml.safe_dump(config, handle, sort_keys=False)

    try:
        print(f"[flow1_smoke] validate -> {temp_config}")
        if run_cli(["validate", str(temp_config)]) != 0:
            print("[flow1_smoke] FAILED: validation")
            return 1

        print("[flow1_smoke] train (3 epochs, synthetic data)...")
        train_buffer = io.StringIO()
        if run_cli(["train", str(temp_config)], stdout=train_buffer) != 0:
            print(train_buffer.getvalue(), end="")
            print("[flow1_smoke] FAILED: training")
            return 1
        print(train_buffer.getvalue(), end="")

        train_summary = _last_json_line(train_buffer.getvalue())
        checkpoint_path = train_summary.get("best_checkpoint") or train_summary.get("checkpoint")
        if checkpoint_path:
            config.setdefault("runtime", {})["checkpoint_path"] = str(checkpoint_path)
            temp_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        print("[flow1_smoke] eval...")
        eval_buffer = io.StringIO()
        result = run_cli(["eval", str(temp_config)], stdout=eval_buffer)
        print(eval_buffer.getvalue(), end="")

        if result == 0:
            print("\n[flow1_smoke] ✓ Flow 1 smoke run PASSED")
        else:
            print("\n[flow1_smoke] ✗ Flow 1 smoke run FAILED at eval")
        return result

    finally:
        temp_config.unlink(missing_ok=True)


def _prepare_fixture_artifacts(data_dir: Path) -> None:
    """Create synthetic fixture with drug_seq, protein_seq, drug_graph, labels."""
    processed_dir = data_dir / "processed" / "davis" / "v1"
    split_dir = data_dir / "splits" / "davis" / "v1" / "v1"
    processed_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    rows = _build_rows()
    counts: dict[str, dict[str, int]] = {}
    scenario_partitions: dict[str, dict[str, str]] = {}
    for scenario, partitions in rows.items():
        scenario_dir = split_dir / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        counts[scenario] = {}
        scenario_partitions[scenario] = {}
        for partition, records in partitions.items():
            path = scenario_dir / f"{partition}.jsonl"
            _write_jsonl(path, records)
            counts[scenario][partition] = len(records)
            scenario_partitions[scenario][partition] = str(path)

    dataset_version = {
        "dataset": "davis",
        "dataset_version": "synthetic-flow1-v1",
        "preprocessing_version": "v1",
        "record_count": sum(sum(len(records) for records in partitions.values()) for partitions in rows.values()),
        "feature_keys": ["drug_seq", "protein_seq", "drug_graph", "labels", "scenario"],
    }
    (processed_dir / "dataset_version.json").write_text(
        json.dumps(dataset_version, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "dataset": "davis",
        "preprocessing_version": "v1",
        "split_version": "v1",
        "seed": 42,
        "scenarios": ["s1", "s2", "s3", "s4"],
        "partitions": ["train", "val", "test"],
        "scenario_partitions": scenario_partitions,
        "counts": counts,
        "protocol_report": {
            "protocol_version": "flow1-fixture.v1",
            "note": "Synthetic fixture for Flow 1 smoke runs only.",
        },
    }
    (split_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    path.write_text((payload + "\n") if payload else "", encoding="utf-8")


def _last_json_line(payload: str) -> dict[str, Any]:
    for line in reversed(payload.splitlines()):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            return decoded
    return {}


def _drug_graph(n_atoms: int = 4, n_feat: int = 9) -> dict[str, Any]:
    """Synthetic drug graph: adjacency matrix + node features."""
    adj = [[float(i != j) for j in range(n_atoms)] for i in range(n_atoms)]
    node_feat = [[float((i + j) % 3) / 2.0 for j in range(n_feat)] for i in range(n_atoms)]
    return {"adj": adj, "node_feat": node_feat}


def _record(
    drug_tokens: list[int],
    protein_tokens: list[int],
    label: float,
    scenario: str,
) -> dict[str, Any]:
    return {
        "drug_seq": drug_tokens,
        "protein_seq": protein_tokens,
        "drug_graph": _drug_graph(),
        "labels": label,
        "scenario": scenario,
    }


def _build_rows() -> dict[str, dict[str, list[dict[str, Any]]]]:
    positives = [
        _record([1, 1, 1, 1], [1, 1, 1, 1, 1], 1.0, "s1"),
        _record([2, 2, 2, 2], [2, 2, 2, 2, 2], 1.0, "s1"),
        _record([3, 3, 3, 3], [3, 3, 3, 3, 3], 1.0, "s1"),
        _record([4, 4, 4, 4], [4, 4, 4, 4, 4], 1.0, "s1"),
    ]
    negatives = [
        _record([8, 8, 8, 8], [8, 8, 8, 8, 8], 0.0, "s1"),
        _record([9, 9, 9, 9], [9, 9, 9, 9, 9], 0.0, "s1"),
        _record([10, 10, 10, 10], [10, 10, 10, 10, 10], 0.0, "s1"),
        _record([11, 11, 11, 11], [11, 11, 11, 11, 11], 0.0, "s1"),
    ]
    scenario_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for index, scenario in enumerate(("s1", "s2", "s3", "s4"), start=0):
        # S2/S4: drug_graph=None to test missing modality handling
        use_graph = scenario in ("s1", "s3")
        train_rows = [
            {
                **positives[index % len(positives)],
                "scenario": scenario,
                "drug_graph": _drug_graph() if use_graph else None,
            },
            {
                **negatives[index % len(negatives)],
                "scenario": scenario,
                "drug_graph": _drug_graph() if use_graph else None,
            },
            {
                **positives[(index + 1) % len(positives)],
                "scenario": scenario,
                "drug_graph": _drug_graph() if use_graph else None,
            },
            {
                **negatives[(index + 1) % len(negatives)],
                "scenario": scenario,
                "drug_graph": _drug_graph() if use_graph else None,
            },
        ]
        val_rows = [
            {
                **positives[(index + 2) % len(positives)],
                "scenario": scenario,
                "drug_graph": _drug_graph() if use_graph else None,
            },
            {
                **negatives[(index + 2) % len(negatives)],
                "scenario": scenario,
                "drug_graph": _drug_graph() if use_graph else None,
            },
        ]
        test_rows = [
            {
                **positives[(index + 3) % len(positives)],
                "scenario": scenario,
                "drug_graph": _drug_graph() if use_graph else None,
            },
            {
                **negatives[(index + 3) % len(negatives)],
                "scenario": scenario,
                "drug_graph": _drug_graph() if use_graph else None,
            },
        ]
        scenario_rows[scenario] = {"train": train_rows, "val": val_rows, "test": test_rows}
    return scenario_rows


if __name__ == "__main__":
    raise SystemExit(main())
