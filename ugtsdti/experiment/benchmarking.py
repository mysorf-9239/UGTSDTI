"""Benchmark matrix execution, aggregation, and artifact writing."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from ugtsdti.data.protocols.probe import probe_tdc_split
from ugtsdti.experiment.runtime import run_experiment


def repo_root() -> Path:
    """Return the repository root used to load configs and write artifacts."""
    return Path(__file__).resolve().parents[2]


def load_named_cfg(group: str, name: str) -> DictConfig:
    """Load a named config file from one Hydra config group."""
    return OmegaConf.load(repo_root() / "configs" / group / f"{name}.yaml")


def normalize_experiment_spec(model_name: str | dict[str, Any]) -> dict[str, Any]:
    """Normalize benchmark matrix entries into a dict-based experiment spec."""
    if isinstance(model_name, str):
        return {"name": model_name, "model": model_name}
    return dict(model_name)


def load_experiment_cfg(
    model_name: str | dict[str, Any],
    data_name: str,
    trainer_name: str,
    loss_name: str,
    seed: int,
    output_dir: Path,
    wandb_enabled: bool,
) -> DictConfig:
    """Compose one fully resolved experiment config for benchmark execution."""
    experiment_spec = normalize_experiment_spec(model_name)
    experiment_name = str(experiment_spec.get("name") or experiment_spec.get("model") or "experiment")

    cfg = OmegaConf.load(repo_root() / "configs" / "default.yaml")
    cfg.data = load_named_cfg("data", data_name)
    cfg.trainer = load_named_cfg("trainer", trainer_name)
    cfg.loss = load_named_cfg("loss", loss_name)

    model_ref = str(experiment_spec.get("model", "hybrid"))
    cfg.model = load_named_cfg("model", model_ref)
    cfg.teacher = load_named_cfg("teacher", str(experiment_spec.get("teacher", "none")))
    cfg.student = load_named_cfg("student", str(experiment_spec.get("student", "none")))
    cfg.fusion = load_named_cfg("fusion", str(experiment_spec.get("fusion", "none")))

    cfg.seed = seed
    cfg.model_name = experiment_name
    cfg.data_name = data_name
    cfg.run_name = f"{experiment_name}.{loss_name}.{data_name}.seed{seed}"
    cfg.logging.wandb_enabled = wandb_enabled
    cfg.trainer.loss = cfg.loss
    cfg.trainer.loss.name = loss_name
    cfg.trainer.params.output_dir = str(output_dir)
    cfg.output_dir = str(output_dir)
    return cfg


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Write benchmark rows to both JSON and CSV summary files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate repeated runs over seeds into mean/std benchmark rows."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_key = (
            str(row.get("model_config", "unknown")),
            str(row.get("data_config", "unknown")),
            str(row.get("loss_name", "unknown")),
        )
        grouped[group_key].append(row)

    aggregate_rows_out: list[dict[str, Any]] = []
    for (model_name, data_name, loss_name), group_rows in grouped.items():
        numeric_keys = sorted(
            {
                metric_key
                for row in group_rows
                for metric_key, value in row.items()
                if isinstance(value, (int, float)) and metric_key not in {"seed"}
            }
        )
        aggregate: dict[str, Any] = {
            "model_config": model_name,
            "data_config": data_name,
            "loss_name": loss_name,
            "num_seeds": len(group_rows),
            "seeds": [int(row["seed"]) for row in group_rows if "seed" in row],
        }
        for metric_key in numeric_keys:
            values = [float(row[metric_key]) for row in group_rows if metric_key in row]
            if not values:
                continue
            aggregate[f"{metric_key}__mean"] = sum(values) / len(values)
            if len(values) > 1:
                mean = aggregate[f"{metric_key}__mean"]
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                aggregate[f"{metric_key}__std"] = variance**0.5
            else:
                aggregate[f"{metric_key}__std"] = 0.0
        aggregate_rows_out.append(aggregate)

    return aggregate_rows_out


def infer_scenario_name(data_config: str) -> str:
    """Extract a short scenario tag from a data config name."""
    parts = str(data_config).split("_")
    if parts and parts[-1].startswith("s"):
        return parts[-1]
    return str(data_config)


def build_report_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert wide benchmark rows into a tidy research-report table."""
    report_rows: list[dict[str, Any]] = []
    branch_prefixes = {
        "fused": "",
        "student": "student_",
        "teacher": "teacher_",
    }
    metrics = ("auroc", "auprc", "f1", "mse", "ci", "loss")
    splits = ("val_best", "test")

    for row in rows:
        scenario_name = infer_scenario_name(str(row.get("data_config", "unknown")))
        base = {
            "model_config": row.get("model_config", "unknown"),
            "data_config": row.get("data_config", "unknown"),
            "scenario_name": scenario_name,
            "loss_name": row.get("loss_name", "unknown"),
        }
        if "seed" in row:
            base["seed"] = row["seed"]
        if "num_seeds" in row:
            base["num_seeds"] = row["num_seeds"]

        for split in splits:
            for branch_name, metric_prefix in branch_prefixes.items():
                present = False
                entry = {
                    **base,
                    "split": split,
                    "branch": branch_name,
                }
                for metric_name in metrics:
                    raw_key = f"{split}/{metric_prefix}{metric_name}"
                    aggregate_mean_key = f"{raw_key}__mean"
                    std_key = f"{raw_key}__std"
                    if raw_key in row:
                        entry[metric_name] = row[raw_key]
                        present = True
                    elif aggregate_mean_key in row:
                        entry[metric_name] = row[aggregate_mean_key]
                        present = True
                    if std_key in row:
                        entry[f"{metric_name}__std"] = row[std_key]
                if present:
                    report_rows.append(entry)

    return report_rows


def write_manifest(cfg: DictConfig, output_dir: Path, protocol_summary: list[dict[str, Any]]) -> None:
    """Persist benchmark config provenance and protocol probe output."""
    manifest = {
        "benchmark_config": OmegaConf.to_container(cfg, resolve=True),
        "protocol_summary": protocol_summary,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def probe_protocols(cfg: DictConfig) -> list[dict[str, Any]]:
    """Probe configured scenarios directly from PyTDC before running models."""
    protocol_rows: list[dict[str, Any]] = []
    if not cfg.get("probe_protocols", True):
        return protocol_rows

    root = repo_root()
    for data_name in cfg.data_configs:
        data_cfg = OmegaConf.load(root / "configs" / "data" / f"{data_name}.yaml")
        train_params = OmegaConf.to_container(data_cfg.train.params, resolve=True)
        protocol_rows.append(
            probe_tdc_split(
                name=str(train_params["name"]),
                split_type=str(train_params["split_type"]),
                column_name=train_params.get("column_name"),
                frac=list(train_params.get("frac", [0.8, 0.1, 0.1])),
                seed=int(train_params.get("seed", 42)),
                scenario_name=str(train_params.get("scenario_name", data_name)),
            )
        )
    return protocol_rows


def run_benchmark_matrix(cfg: DictConfig) -> list[dict[str, Any]]:
    """Execute the configured benchmark matrix and materialize all artifacts."""
    output_root = Path(cfg.output_root)
    results: list[dict[str, Any]] = []
    protocol_rows = probe_protocols(cfg)
    experiments = list(cfg.get("experiments", []))
    if not experiments:
        raise ValueError("Benchmark config must define a non-empty `experiments` list.")

    for experiment_spec, data_name, loss_name, seed in product(experiments, cfg.data_configs, cfg.losses, cfg.seeds):
        experiment_name = str(experiment_spec.get("name") or experiment_spec.get("model"))
        run_dir = output_root / data_name / experiment_name / loss_name / f"seed_{seed}"
        logger.info(
            f"Benchmark run: model={experiment_name} data={data_name} loss={loss_name} seed={seed} output={run_dir}"
        )
        experiment_cfg = load_experiment_cfg(
            model_name=experiment_spec,
            data_name=data_name,
            trainer_name=cfg.trainer_config,
            loss_name=loss_name,
            seed=int(seed),
            output_dir=run_dir,
            wandb_enabled=bool(cfg.logging.wandb_enabled),
        )
        result = run_experiment(experiment_cfg)
        results.append(result)

    aggregate_rows_out = aggregate_rows(results)
    write_summary(results, output_root)
    write_summary(aggregate_rows_out, output_root / "aggregate")
    write_summary(build_report_rows(results), output_root / "report")
    write_summary(build_report_rows(aggregate_rows_out), output_root / "report_aggregate")
    write_summary(protocol_rows, output_root / "protocol")
    write_manifest(cfg, output_root, protocol_rows)
    return results
