"""Command-line entrypoints for train/eval/validate/sweep."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Literal, cast

from ugtsdti.config.loader import ConfigLoader
from ugtsdti.config.normalize import ConfigNormalizer
from ugtsdti.config.validate import ConfigValidator
from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.errors import UGTSDTIError
from ugtsdti.data import DataLoaderFactory
from ugtsdti.graph.builder import GraphBuilder
from ugtsdti.graph.planner import GraphPlanner
from ugtsdti.interaction.registry import InteractionPlanner
from ugtsdti.logging import CompositeLogger, FileLogger, Logger, WandbLogger
from ugtsdti.runtime import (
    ArtifactWriter,
    CheckpointIO,
    RuntimeAdapter,
    apply_runtime_registrars,
    build_default_graph_registry,
    build_default_interaction_registry,
    build_experiment_identity,
    seed_everything,
)
from ugtsdti.trainer import Evaluator, PipelineExecutor, Trainer

CommandHandler = Callable[[dict[str, Any], argparse.Namespace], Any]


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(prog="ugtsdti")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command_name in ("train", "eval", "sweep"):
        command = subparsers.add_parser(command_name)
        command.add_argument("config")
        command.add_argument("--config", dest="config_flag")

    validate = subparsers.add_parser("validate")
    validate.add_argument("config", nargs="*")
    validate.add_argument("--config", dest="config_flag", nargs="+")

    return parser


def run_cli(
    argv: list[str] | None = None,
    *,
    stdout: Any = None,
    config_loader: ConfigLoader | None = None,
    validator: ConfigValidator | None = None,
    normalizer: ConfigNormalizer | None = None,
    handlers: dict[str, CommandHandler] | None = None,
) -> int:
    """Run the CLI command and return a process-like exit code."""
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    loader = config_loader or ConfigLoader()
    validator = validator or ConfigValidator()
    normalizer = normalizer or ConfigNormalizer()
    command_handlers = _default_handlers(stream) if handlers is None else handlers

    try:
        config_paths = _resolve_config_paths(args)
        if not config_paths:
            raise UGTSDTIError("Missing config path.", stage="cli", component="main", key="config")
        if args.command == "validate":
            for config_path in config_paths:
                raw_cfg = loader.load(Path(config_path))
                validator.validate(raw_cfg)
            stream.write("VALID\n")
            return 0

        raw_cfg = loader.load(Path(config_paths[0]))
        validator.validate(raw_cfg)
        normalized_cfg = normalizer.normalize(raw_cfg).to_dict()
        identity = build_experiment_identity(normalized_cfg)
        stream.write(
            f"run_id={identity.run_id} config_hash={identity.config_hash} "
            f"git_commit={identity.git_commit or 'unknown'}\n"
        )

        if args.command == "sweep":
            targets = resolve_sweep_targets(normalized_cfg)
            stream.write(json.dumps({"targets": targets}, sort_keys=True) + "\n")

        handler = command_handlers.get(args.command)
        if handler is None:
            raise UGTSDTIError(
                f"No execution handler is configured for command {args.command!r}.",
                stage="cli",
                component="main",
                key=args.command,
            )
        handler(normalized_cfg, args)
        return 0
    except UGTSDTIError as exc:
        stream.write(f"ERROR: {exc}\n")
        return 1


def resolve_sweep_targets(cfg: dict[str, Any]) -> list[str]:
    """Resolve normalized sweep parameter targets."""
    sweep_cfg = cfg.get("sweep", {})
    parameters = sweep_cfg.get("parameters", {})

    if isinstance(parameters, dict):
        return sorted(str(target) for target in parameters.keys())
    if isinstance(parameters, list):
        targets = []
        for entry in parameters:
            if isinstance(entry, dict) and "target" in entry:
                targets.append(str(entry["target"]))
        return sorted(targets)
    return []


def _resolve_config_paths(args: argparse.Namespace) -> list[str]:
    explicit = getattr(args, "config_flag", None)
    if explicit:
        if isinstance(explicit, list):
            return [str(path) for path in explicit]
        return [str(explicit)]
    positional = getattr(args, "config", None)
    if positional is None:
        return []
    if isinstance(positional, list):
        return [str(path) for path in positional]
    return [str(positional)]


def main(argv: list[str] | None = None) -> int:
    """Console-script entrypoint."""
    return run_cli(argv)


def _default_handlers(stream: Any) -> dict[str, CommandHandler]:
    return {
        "train": lambda cfg, args: _run_train(cfg, args, stream),
        "eval": lambda cfg, args: _run_eval(cfg, args, stream),
        "sweep": lambda cfg, args: _run_sweep(cfg, args, stream),
    }


def _run_train(cfg: dict[str, Any], args: argparse.Namespace, stream: Any) -> None:
    runtime_cfg, executor, runtime_state = _prepare_runtime(cfg)
    context = _build_context(runtime_state, mode="train")
    train_scenario = str(runtime_cfg.get("scenario", {}).get("train", "s1"))
    batches, split_manifest = _load_batches(runtime_cfg, runtime_state, scenarios=[train_scenario], partition="train")
    identity = build_experiment_identity(cfg)
    logs_dir = _logs_dir(runtime_state, identity.run_id)
    logger = _build_logger(cfg, logs_dir)
    checkpoint_io = CheckpointIO()
    artifact_writer = ArtifactWriter(runtime_state["artifacts_dir"])
    trainer = Trainer(
        executor,
        logger=logger,
        checkpoint_io=checkpoint_io,
        artifact_writer=artifact_writer,
        parameter_groups=executor.parameter_groups(runtime_cfg),
    )
    optimizer = _build_optimizer(cfg, executor, runtime_cfg)
    checkpoint_path = Path(runtime_state["checkpoint_dir"]) / f"{identity.run_id}.pt"

    try:
        for step_idx, batch in enumerate(batches):
            trainer.step(
                batch,
                runtime_cfg,
                context,
                optimizer=optimizer,
                step_idx=step_idx,
                epoch=0,
                checkpoint_path=str(checkpoint_path),
                identity=identity,
                normalized_config=runtime_cfg,
                split_manifest=split_manifest,
                model_state=lambda: executor.model_state(runtime_cfg),
                logs_dir=str(logs_dir),
            )
        stream.write(
            json.dumps(
                {
                    "batches": len(batches),
                    "checkpoint": str(checkpoint_path),
                    "command": args.command,
                    "logs_dir": str(logs_dir),
                },
                sort_keys=True,
            )
            + "\n"
        )
    finally:
        logger.close()


def _run_eval(cfg: dict[str, Any], args: argparse.Namespace, stream: Any) -> None:
    runtime_cfg, executor, runtime_state = _prepare_runtime(cfg)
    context = _build_context(runtime_state, mode="eval")
    eval_scenarios = list(runtime_cfg.get("scenario", {}).get("eval", []))
    batches, split_manifest = _load_batches(runtime_cfg, runtime_state, scenarios=eval_scenarios, partition="test")
    identity = build_experiment_identity(cfg)
    logs_dir = _logs_dir(runtime_state, identity.run_id)
    logger = _build_logger(cfg, logs_dir)
    artifact_writer = ArtifactWriter(runtime_state["artifacts_dir"])
    evaluator = Evaluator(executor, logger=logger, artifact_writer=artifact_writer)
    try:
        result = evaluator.evaluate(
            batches,
            runtime_cfg,
            context,
            identity=identity,
            normalized_config=runtime_cfg,
            split_manifest=split_manifest,
            model_state=executor.model_state(runtime_cfg),
            logs_dir=str(logs_dir),
        )
        summary = {key: value for key, value in result.metrics.items() if key.startswith("metrics.")}
        stream.write(
            json.dumps({"command": args.command, "logs_dir": str(logs_dir), "metrics": summary}, sort_keys=True) + "\n"
        )
    finally:
        logger.close()


def _run_sweep(cfg: dict[str, Any], args: argparse.Namespace, stream: Any) -> None:
    del cfg
    stream.write(json.dumps({"command": args.command, "status": "ready"}, sort_keys=True) + "\n")


def _prepare_runtime(cfg: dict[str, Any]) -> tuple[dict[str, Any], PipelineExecutor, dict[str, Any]]:
    runtime_state = RuntimeAdapter().adapt(cfg)
    seed_everything(runtime_state["seed"], deterministic=runtime_state["deterministic"])

    graph_registry = build_default_graph_registry()
    interaction_registry = build_default_interaction_registry()
    registrars = list(cfg.get("runtime", {}).get("plugin_registrars", []))
    if registrars:
        apply_runtime_registrars(
            registrars,
            graph_registry=graph_registry,
            interaction_registry=interaction_registry,
        )

    runtime_cfg = dict(cfg)
    runtime_cfg["graph"] = _graph_nodes_as_list(dict(cfg.get("graph", {})))
    runtime_cfg["graph_plan"] = GraphPlanner(graph_registry).plan(
        GraphBuilder(graph_registry).build(runtime_cfg["graph"])
    )
    role_outputs = {f"{role}.logits" for role in runtime_cfg.get("roles", {})}
    runtime_cfg["interaction_plan"] = InteractionPlanner(interaction_registry).plan(
        runtime_cfg.get("interaction", {}),
        available_inputs=role_outputs,
    )
    executor = PipelineExecutor(graph_registry=graph_registry, interaction_registry=interaction_registry)
    return runtime_cfg, executor, runtime_state


def _logs_dir(runtime_state: dict[str, Any], run_id: str) -> Path:
    return Path(runtime_state["artifacts_dir"]) / "_logs" / run_id


def _build_logger(cfg: dict[str, Any], logs_dir: Path) -> CompositeLogger:
    logging_cfg = dict(cfg.get("logging", {}))
    backend = str(logging_cfg.get("backend", "file")).lower()
    loggers: list[Logger] = [FileLogger(logs_dir)]
    if backend == "wandb":
        project = str(cfg.get("experiment", {}).get("name", "ugtsdti"))
        loggers.append(WandbLogger(project=project, enabled=True))
    return CompositeLogger(loggers)


def _build_optimizer(
    cfg: dict[str, Any],
    executor: PipelineExecutor,
    runtime_cfg: dict[str, Any],
) -> Any | None:
    try:
        importlib.import_module("torch")
        from torch.optim import SGD as SgdOptimizer  # type: ignore[attr-defined]
    except ImportError:
        return None

    parameter_groups = executor.parameter_groups(runtime_cfg)
    trainable = [parameter for group in parameter_groups.values() for parameter in group if _is_trainable(parameter)]
    if not trainable:
        return None
    optimizer_cfg = cfg.get("training", {}).get("optimizer", {})
    lr = float(optimizer_cfg.get("lr", 0.01))
    return SgdOptimizer(trainable, lr=lr)


def _is_trainable(parameter: Any) -> bool:
    return bool(getattr(parameter, "requires_grad", False))


def _build_context(runtime_state: dict[str, Any], *, mode: str) -> ExecutionContext:
    precision: Literal["fp32", "mixed"] = "mixed" if runtime_state.get("precision") == "mixed" else "fp32"
    execution_mode = cast(Literal["train", "eval", "infer"], mode)
    return ExecutionContext(
        mode=execution_mode,
        seed=int(runtime_state["seed"]),
        device=str(runtime_state["device"]),
        deterministic=bool(runtime_state["deterministic"]),
        precision=precision,
    )


def _load_batches(
    cfg: dict[str, Any],
    runtime_state: dict[str, Any],
    *,
    scenarios: list[str],
    partition: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_cfg = dict(cfg.get("data", {}))
    dataset = str(data_cfg.get("dataset", "dataset"))
    preprocessing_version = str(data_cfg.get("preprocessing_version", "v1"))
    split_version = str(data_cfg.get("split_version", "v1"))
    data_root = Path(runtime_state["data_dir"])
    dataset_version_path = data_root / "processed" / dataset / preprocessing_version / "dataset_version.json"
    split_manifest_path = data_root / "splits" / dataset / preprocessing_version / split_version / "manifest.json"
    batch_size = int(runtime_state.get("batch_size", 32))
    raw_batches, _, dataset_version, manifest = DataLoaderFactory().build(
        cfg=cfg,
        dataset_version_path=dataset_version_path,
        split_manifest_path=split_manifest_path,
        batch_size=batch_size,
        scenarios=scenarios,
        partition=partition,
    )
    batches = [_tensorize_batch(batch) for batch in raw_batches]
    if not batches:
        raise UGTSDTIError(
            "No materialized rows were loaded for the requested scenarios.",
            stage="cli",
            component="main",
            key=f"data.{partition}",
        )
    split_manifest = manifest.to_dict()
    split_manifest["dataset_version"] = dataset_version.to_dict()
    split_manifest["selected_scenarios"] = list(scenarios)
    split_manifest["selected_partition"] = partition
    return batches, split_manifest


def _tensorize_batch(batch: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return batch

    converted: dict[str, Any] = {}
    for key, value in batch.items():
        if key == "scenario":
            converted[key] = list(value)
            continue
        if key == "labels":
            tensor = torch.as_tensor(value, dtype=torch.float32)
            converted[key] = tensor.reshape(-1, 1)
            continue
        converted[key] = _maybe_tensorize_value(value, torch)
    return converted


def _maybe_tensorize_value(value: Any, torch_module: Any) -> Any:
    if isinstance(value, list) and value and isinstance(value[0], str):
        return value
    try:
        tensor = torch_module.as_tensor(value, dtype=torch_module.float32)
        if tensor.ndim == 0:
            return tensor.reshape(1, 1)
        return tensor
    except Exception:
        return value


def _graph_nodes_as_list(graph_cfg: dict[str, Any]) -> dict[str, Any]:
    nodes = graph_cfg.get("nodes", {})
    if not isinstance(nodes, dict):
        return graph_cfg
    graph = dict(graph_cfg)
    graph["nodes"] = [{**dict(node_cfg), "name": name} for name, node_cfg in nodes.items()]
    return graph
