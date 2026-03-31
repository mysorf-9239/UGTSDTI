"""One-file baseline workflow: prepare fixture artifacts and run validate/train/eval."""

from __future__ import annotations

import argparse
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the baseline reference end-to-end on a synthetic fixture.")
    parser.add_argument("--root", default=None, help="Repo root. Defaults to the parent of examples/.")
    parser.add_argument("--data-dir", default=None, help="Artifact data dir. Defaults to <root>/data.")
    parser.add_argument("--artifacts-dir", default=None, help="Artifacts output dir. Defaults to <root>/artifacts.")
    parser.add_argument("--checkpoint-dir", default=None, help="Checkpoint output dir. Defaults to <root>/checkpoints.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs for the smoke run.")
    parser.add_argument("--batch-size", type=int, default=2, help="Runtime batch size.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir).resolve() if args.data_dir else root / "data"
    artifacts_dir = Path(args.artifacts_dir).resolve() if args.artifacts_dir else root / "artifacts"
    checkpoint_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else root / "checkpoints"
    config_path = root / "configs" / "baseline_reference.yaml"

    _prepare_fixture_artifacts(data_dir)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["runtime"] = {
        "data_dir": str(data_dir),
        "artifacts_dir": str(artifacts_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "batch_size": int(args.batch_size),
        "seed": 7,
    }
    loop_cfg = dict(config["training"].get("loop", {}))
    loop_cfg.update(
        {
            "epochs": max(1, int(args.epochs)),
            "checkpoint_every_epochs": 1,
            "summary_every_steps": 1,
            "eval_every_epochs": 1,
            "eval_partition": "test",
        }
    )
    config["training"]["loop"] = loop_cfg

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        temp_config = Path(handle.name)
        yaml.safe_dump(config, handle, sort_keys=False)

    try:
        print(f"[baseline] validate -> {temp_config}")
        if run_cli(["validate", str(temp_config)]) != 0:
            return 1

        print("[baseline] train")
        train_buffer = io.StringIO()
        if run_cli(["train", str(temp_config)], stdout=train_buffer) != 0:
            print(train_buffer.getvalue(), end="")
            return 1
        print(train_buffer.getvalue(), end="")
        train_summary = _last_json_line(train_buffer.getvalue())
        checkpoint_policy = str(config["training"]["loop"].get("select_checkpoint", "last")).lower()
        checkpoint_path = (
            train_summary.get("best_checkpoint") if checkpoint_policy == "best" else train_summary.get("checkpoint")
        ) or train_summary.get("checkpoint")
        if checkpoint_path:
            config.setdefault("runtime", {})["checkpoint_path"] = str(checkpoint_path)
            temp_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        print("[baseline] eval")
        return run_cli(["eval", str(temp_config)])
    finally:
        temp_config.unlink(missing_ok=True)


def _prepare_fixture_artifacts(data_dir: Path) -> None:
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
        "dataset_version": "synthetic-baseline-v1",
        "preprocessing_version": "v1",
        "record_count": sum(sum(len(records) for records in partitions.values()) for partitions in rows.values()),
        "feature_keys": ["drug_seq", "protein_seq", "labels", "scenario"],
    }
    (processed_dir / "dataset_version.json").write_text(
        json.dumps(dataset_version, sort_keys=True, indent=2),
        encoding="utf-8",
    )

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
            "protocol_version": "baseline-fixture.v1",
            "note": "Synthetic fixture for smoke runs only; not for research reporting.",
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


def _record(drug_tokens: list[int], protein_tokens: list[int], label: float, scenario: str) -> dict[str, Any]:
    return {
        "drug_seq": drug_tokens,
        "protein_seq": protein_tokens,
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
        train_rows = [
            {**positives[index % len(positives)], "scenario": scenario},
            {**negatives[index % len(negatives)], "scenario": scenario},
            {**positives[(index + 1) % len(positives)], "scenario": scenario},
            {**negatives[(index + 1) % len(negatives)], "scenario": scenario},
        ]
        val_rows = [
            {**positives[(index + 2) % len(positives)], "scenario": scenario},
            {**negatives[(index + 2) % len(negatives)], "scenario": scenario},
        ]
        test_rows = [
            {**positives[(index + 3) % len(positives)], "scenario": scenario},
            {**negatives[(index + 3) % len(negatives)], "scenario": scenario},
        ]
        scenario_rows[scenario] = {"train": train_rows, "val": val_rows, "test": test_rows}
    return scenario_rows


if __name__ == "__main__":
    raise SystemExit(main())
