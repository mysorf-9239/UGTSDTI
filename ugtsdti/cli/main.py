"""Command-line entrypoints for train/eval/validate/sweep."""

from __future__ import annotations

import argparse
import importlib
import inspect
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
    build_reproducibility_key,
    seed_everything,
)
from ugtsdti.trainer import Evaluator, PipelineExecutor, Trainer

CommandHandler = Callable[..., Any]


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
                _validate_raw_config(raw_cfg, validator)
            stream.write("VALID\n")
            return 0

        raw_cfg = loader.load(Path(config_paths[0]))
        _validate_raw_config(raw_cfg, validator)
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
        _call_handler(handler, normalized_cfg, args, identity)
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
        "train": lambda cfg, args, identity: _run_train(cfg, args, stream, identity),
        "eval": lambda cfg, args, identity: _run_eval(cfg, args, stream, identity),
        "sweep": lambda cfg, args, identity: _run_sweep(cfg, args, stream, identity),
    }


def _call_handler(
    handler: CommandHandler,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    identity: Any,
) -> Any:
    try:
        positional = [
            parameter
            for parameter in inspect.signature(handler).parameters.values()
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        positional = []

    if len(positional) >= 3:
        return handler(cfg, args, identity)
    return handler(cfg, args)


def _run_train(cfg: dict[str, Any], args: argparse.Namespace, stream: Any, identity: Any) -> None:
    runtime_cfg, executor, runtime_state = _prepare_runtime(cfg)
    context = _build_context(runtime_state, mode="train")
    train_scenario = str(runtime_cfg.get("scenario", {}).get("train", "s1"))
    batches, split_manifest = _load_batches(runtime_cfg, runtime_state, scenarios=[train_scenario], partition="train")
    loop_cfg = _resolve_loop_cfg(runtime_cfg)
    eval_batches: list[dict[str, Any]] = []
    eval_partition = str(loop_cfg["eval_partition"])
    if int(loop_cfg["eval_every_epochs"]) > 0:
        eval_batches, _ = _load_batches(
            runtime_cfg,
            runtime_state,
            scenarios=list(runtime_cfg.get("scenario", {}).get("eval", [])),
            partition=eval_partition,
            allow_empty=True,
        )
    runtime_identity = _materialize_runtime_identity(identity, runtime_cfg, runtime_state, split_manifest)
    logs_dir = _logs_dir(runtime_state, runtime_identity["run_id"])
    _write_identity_log(logs_dir, runtime_identity)
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
    evaluator = Evaluator(executor, logger=logger) if eval_batches else None
    checkpoint_path = Path(runtime_state["checkpoint_dir"]) / f"{runtime_identity['run_id']}.pt"
    global_step = 0
    last_checkpoint = None
    last_eval_metrics: dict[str, Any] = {}

    try:
        for epoch_idx in range(int(loop_cfg["epochs"])):
            epoch_number = epoch_idx + 1
            last_train_result = None
            for batch_idx, batch in enumerate(batches):
                should_checkpoint = (
                    batch_idx == len(batches) - 1 and epoch_number % int(loop_cfg["checkpoint_every_epochs"]) == 0
                )
                last_train_result = trainer.step(
                    batch,
                    runtime_cfg,
                    context,
                    optimizer=optimizer,
                    step_idx=global_step,
                    epoch=epoch_number,
                    checkpoint_path=str(checkpoint_path) if should_checkpoint else None,
                    identity=runtime_identity,
                    normalized_config=runtime_cfg,
                    split_manifest=split_manifest,
                    model_state=lambda: executor.model_state(runtime_cfg),
                    logs_dir=str(logs_dir),
                )
                global_step += 1
                if global_step % int(loop_cfg["summary_every_steps"]) == 0:
                    stream.write(
                        json.dumps(
                            {
                                "command": args.command,
                                "epoch": epoch_number,
                                "event": "train_step",
                                "step": global_step,
                                "metrics": last_train_result.logged_metrics,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                if should_checkpoint:
                    last_checkpoint = str(checkpoint_path)

            if evaluator is not None and epoch_number % int(loop_cfg["eval_every_epochs"]) == 0:
                eval_result = evaluator.evaluate(
                    eval_batches,
                    runtime_cfg,
                    _build_context(runtime_state, mode="eval"),
                )
                last_eval_metrics = {
                    key: value for key, value in eval_result.metrics.items() if key.startswith("metrics.")
                }
                logger.log_metrics(
                    _scalarize_metrics(last_eval_metrics),
                    step=global_step,
                )
                stream.write(
                    json.dumps(
                        {
                            "command": args.command,
                            "epoch": epoch_number,
                            "event": "eval_epoch",
                            "metrics": _scalarize_metrics(last_eval_metrics),
                            "partition": eval_partition,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        stream.write(
            json.dumps(
                {
                    "batches_per_epoch": len(batches),
                    "checkpoint": last_checkpoint,
                    "command": args.command,
                    "epochs": int(loop_cfg["epochs"]),
                    "eval_partition": eval_partition if eval_batches else None,
                    "final_eval_metrics": _scalarize_metrics(last_eval_metrics),
                    "logs_dir": str(logs_dir),
                    "steps": global_step,
                },
                sort_keys=True,
            )
            + "\n"
        )
    finally:
        logger.close()


def _run_eval(cfg: dict[str, Any], args: argparse.Namespace, stream: Any, identity: Any) -> None:
    runtime_cfg, executor, runtime_state = _prepare_runtime(cfg)
    eval_scenarios = list(runtime_cfg.get("scenario", {}).get("eval", []))
    batches, split_manifest = _load_batches(runtime_cfg, runtime_state, scenarios=eval_scenarios, partition="test")
    runtime_identity = _materialize_runtime_identity(identity, runtime_cfg, runtime_state, split_manifest)
    checkpoint_bundle = _maybe_load_checkpoint(runtime_cfg, runtime_state, executor, runtime_identity)
    context = _build_context(runtime_state, mode="eval")
    logs_dir = _logs_dir(runtime_state, runtime_identity["run_id"])
    _write_identity_log(logs_dir, runtime_identity)
    logger = _build_logger(cfg, logs_dir)
    artifact_writer = ArtifactWriter(runtime_state["artifacts_dir"])
    evaluator = Evaluator(executor, logger=logger, artifact_writer=artifact_writer)
    try:
        result = evaluator.evaluate(
            batches,
            runtime_cfg,
            context,
            identity=runtime_identity,
            normalized_config=runtime_cfg,
            split_manifest=split_manifest,
            model_state=checkpoint_bundle.model_state
            if checkpoint_bundle is not None
            else executor.model_state(runtime_cfg),
            logs_dir=str(logs_dir),
        )
        summary = {key: value for key, value in result.metrics.items() if key.startswith("metrics.")}
        stream.write(
            json.dumps(
                {
                    "command": args.command,
                    "checkpoint": runtime_state.get("checkpoint_path"),
                    "logs_dir": str(logs_dir),
                    "metrics": _scalarize_metrics(summary),
                },
                sort_keys=True,
            )
            + "\n"
        )
    finally:
        logger.close()


def _run_sweep(cfg: dict[str, Any], args: argparse.Namespace, stream: Any, identity: Any) -> None:
    del cfg, identity
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


def _validate_raw_config(raw_cfg: dict[str, Any], validator: ConfigValidator) -> None:
    graph_registry = build_default_graph_registry()
    interaction_registry = build_default_interaction_registry()
    registrars = list(raw_cfg.get("runtime", {}).get("plugin_registrars", []))
    if registrars:
        apply_runtime_registrars(
            registrars,
            graph_registry=graph_registry,
            interaction_registry=interaction_registry,
        )
    ConfigValidator(
        graph_registry=graph_registry,
        interaction_registry=interaction_registry,
    ).validate(raw_cfg)


def _logs_dir(runtime_state: dict[str, Any], run_id: str) -> Path:
    return Path(runtime_state["artifacts_dir"]) / "_logs" / run_id


def _materialize_runtime_identity(
    identity: Any,
    cfg: dict[str, Any],
    runtime_state: dict[str, Any],
    split_manifest: dict[str, Any],
) -> dict[str, Any]:
    identity_payload = identity.to_dict() if hasattr(identity, "to_dict") else dict(identity)
    dataset_metadata = dict(split_manifest.get("dataset_version", {}))
    identity_payload["reproducibility_key"] = build_reproducibility_key(
        config_hash=str(identity_payload["config_hash"]),
        dataset_version=str(dataset_metadata.get("dataset_version", "unknown")),
        preprocessing_version=str(
            dataset_metadata.get("preprocessing_version", cfg.get("data", {}).get("preprocessing_version", "unknown"))
        ),
        split_version=str(split_manifest.get("split_version", cfg.get("data", {}).get("split_version", "unknown"))),
        seed=int(runtime_state["seed"]),
    )
    return identity_payload


def _write_identity_log(logs_dir: Path, identity: dict[str, Any]) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "identity.json").write_text(json.dumps(identity, sort_keys=True, indent=2), encoding="utf-8")


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
    allow_empty: bool = False,
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
    if not batches and not allow_empty:
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


def _resolve_loop_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    training_cfg = dict(cfg.get("training", {}))
    loop_cfg = dict(training_cfg.get("loop", {}))
    epochs = max(1, int(loop_cfg.get("epochs", 1)))
    checkpoint_every_epochs = max(1, int(loop_cfg.get("checkpoint_every_epochs", 1)))
    summary_every_steps = max(1, int(loop_cfg.get("summary_every_steps", 1)))
    eval_every_epochs = max(0, int(loop_cfg.get("eval_every_epochs", 0)))
    eval_partition = str(loop_cfg.get("eval_partition", "val"))
    return {
        "epochs": epochs,
        "checkpoint_every_epochs": checkpoint_every_epochs,
        "summary_every_steps": summary_every_steps,
        "eval_every_epochs": eval_every_epochs,
        "eval_partition": eval_partition,
    }


def _maybe_load_checkpoint(
    cfg: dict[str, Any],
    runtime_state: dict[str, Any],
    executor: PipelineExecutor,
    runtime_identity: dict[str, Any],
) -> Any | None:
    checkpoint_path = runtime_state.get("checkpoint_path")
    if not checkpoint_path:
        return None
    checkpoint = CheckpointIO().load(
        checkpoint_path,
        expected_config_hash=str(runtime_identity["config_hash"]),
        expected_dataset=str(cfg.get("data", {}).get("dataset", "")),
        expected_reproducibility_key=str(runtime_identity["reproducibility_key"]),
    )
    executor.load_model_state(cfg, checkpoint.model_state)
    return checkpoint


def _scalarize_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    scalarized: dict[str, float] = {}
    for key, value in metrics.items():
        try:
            scalarized[key] = float(value)
        except (TypeError, ValueError):
            try:
                import torch

                if isinstance(value, torch.Tensor):
                    scalarized[key] = float(value.detach().to(dtype=torch.float32).mean().cpu().item())
                    continue
            except ImportError:
                pass
    return scalarized


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
