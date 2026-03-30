#!/usr/bin/env python3
"""Materialize a tiny synthetic artifact set for the baseline reference config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create synthetic processed/split artifacts for baseline smoke runs.")
    parser.add_argument("--data-dir", default="data", help="Root artifact directory to populate.")
    parser.add_argument("--dataset", default="davis", help="Dataset name used by the baseline config.")
    parser.add_argument("--preprocessing-version", default="v1", help="Preprocessing version folder.")
    parser.add_argument("--split-version", default="v1", help="Split version folder.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir).resolve()
    processed_dir = data_dir / "processed" / args.dataset / args.preprocessing_version
    split_dir = data_dir / "splits" / args.dataset / args.preprocessing_version / args.split_version

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
        "dataset": args.dataset,
        "dataset_version": "synthetic-baseline-v1",
        "preprocessing_version": args.preprocessing_version,
        "record_count": sum(sum(len(records) for records in partitions.values()) for partitions in rows.values()),
        "feature_keys": ["drug_seq", "protein_seq", "labels", "scenario"],
    }
    (processed_dir / "dataset_version.json").write_text(
        json.dumps(dataset_version, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "dataset": args.dataset,
        "preprocessing_version": args.preprocessing_version,
        "split_version": args.split_version,
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

    print(
        json.dumps(
            {
                "data_dir": str(data_dir),
                "dataset_version": str(processed_dir / "dataset_version.json"),
                "manifest": str(split_dir / "manifest.json"),
            },
            sort_keys=True,
        )
    )
    return 0


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    path.write_text((payload + "\n") if payload else "", encoding="utf-8")


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
