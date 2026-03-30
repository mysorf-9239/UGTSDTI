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
    CheckpointBundle,
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
    scheduler = _build_scheduler(cfg, optimizer)
    evaluator = Evaluator(executor, logger=logger) if eval_batches else None
    checkpoint_paths = _checkpoint_paths(runtime_state, runtime_identity["run_id"])
    resume_bundle = _maybe_load_checkpoint(
        runtime_cfg,
        runtime_state,
        executor,
        runtime_identity,
        strict_identity=False,
    )
    if resume_bundle is not None:
        _restore_component_state(optimizer, resume_bundle.optimizer_state)
        _restore_component_state(scheduler, resume_bundle.scheduler_state)
    global_step = int(resume_bundle.step) + 1 if resume_bundle is not None else 0
    start_epoch = int(resume_bundle.epoch) + 1 if resume_bundle is not None else 1
    last_checkpoint = None
    last_eval_metrics: dict[str, Any] = {}
    last_train_result = None
    final_bundle_dir: str | None = None
    loop_cfg = _resolve_loop_cfg(runtime_cfg)
    best_checkpoint: str | None = None
    best_metric_name = str(loop_cfg["best_metric"])
    best_metric_value: float | None = None
    stopped_early = False
    early_wait = 0
    completed_epochs = 0

    try:
        for epoch_number in range(start_epoch, int(loop_cfg["epochs"]) + 1):
            completed_epochs = epoch_number
            for batch_idx, batch in enumerate(batches):
                should_checkpoint = (
                    batch_idx == len(batches) - 1 and epoch_number % int(loop_cfg["checkpoint_every_epochs"]) == 0
                )
                last_train_result = trainer.step(
                    batch,
                    runtime_cfg,
                    context,
                    optimizer=optimizer,
                    scheduler=None,
                    step_idx=global_step,
                    epoch=epoch_number,
                    checkpoint_path=str(checkpoint_paths["last"]) if should_checkpoint else None,
                    checkpoint_bundle=_build_checkpoint_bundle(
                        identity=runtime_identity,
                        normalized_config=runtime_cfg,
                        split_manifest=split_manifest,
                        model_state=executor.model_state(runtime_cfg),
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch_number,
                        step=global_step,
                        extras={
                            "best_metric_name": best_metric_name,
                            "best_metric_value": best_metric_value,
                        },
                    )
                    if should_checkpoint
                    else None,
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
                    last_checkpoint = str(checkpoint_paths["last"])

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
                current_metric = _metric_value(last_eval_metrics, best_metric_name)
                if _is_better_metric(
                    current_metric,
                    best_metric_value,
                    mode=str(loop_cfg["best_mode"]),
                    min_delta=float(loop_cfg["early_stopping"]["min_delta"]),
                ):
                    best_metric_value = current_metric
                    best_checkpoint = str(checkpoint_paths["best"])
                    early_wait = 0
                    checkpoint_io.save(
                        _build_checkpoint_bundle(
                            identity=runtime_identity,
                            normalized_config=runtime_cfg,
                            split_manifest=split_manifest,
                            model_state=executor.model_state(runtime_cfg),
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch_number,
                            step=max(0, global_step - 1),
                            extras={
                                "best_metric_name": best_metric_name,
                                "best_metric_value": best_metric_value,
                            },
                        ),
                        checkpoint_paths["best"],
                    )
                elif bool(loop_cfg["early_stopping"]["enabled"]):
                    early_wait += 1

                if _is_plateau_scheduler(scheduler):
                    _step_plateau_scheduler(scheduler, current_metric)
                elif scheduler is not None:
                    scheduler.step()

                if bool(loop_cfg["early_stopping"]["enabled"]) and early_wait > int(
                    loop_cfg["early_stopping"]["patience"]
                ):
                    stopped_early = True
                    break
            elif scheduler is not None and not _is_plateau_scheduler(scheduler):
                scheduler.step()
        selected_checkpoint = _select_final_checkpoint(
            loop_cfg=loop_cfg,
            last_checkpoint=last_checkpoint,
            best_checkpoint=best_checkpoint,
        )
        final_eval_metrics = dict(last_eval_metrics)
        if eval_batches and selected_checkpoint is not None:
            selected_bundle = checkpoint_io.load(
                selected_checkpoint,
                expected_config_hash=str(runtime_identity["config_hash"]),
                expected_dataset=str(runtime_cfg.get("data", {}).get("dataset", "")),
                expected_reproducibility_key=str(runtime_identity["reproducibility_key"]),
            )
            executor.load_model_state(runtime_cfg, selected_bundle.model_state)
            final_result = Evaluator(executor, logger=logger).evaluate(
                eval_batches,
                runtime_cfg,
                _build_context(runtime_state, mode="eval"),
            )
            final_eval_metrics = {
                key: value for key, value in final_result.metrics.items() if key.startswith("metrics.")
            }
        if last_train_result is not None:
            safe_snapshot = last_train_result.state.snapshot_isolated()
            final_metrics = (
                final_eval_metrics
                if final_eval_metrics
                else {key: value for key, value in safe_snapshot.items() if key.startswith("metrics.")}
            )
            final_bundle_dir = str(
                artifact_writer.write_bundle(
                    identity=runtime_identity,
                    config=runtime_cfg,
                    metrics=final_metrics,
                    diagnostics={key: value for key, value in safe_snapshot.items() if key.startswith("diagnostics.")},
                    split_manifest=split_manifest,
                    model_state=executor.model_state(runtime_cfg),
                    execution_trace=last_train_result.trace.__dict__,
                    state_boundary_summaries=last_train_result.trace.state_boundary_summaries,
                    logs_dir=str(logs_dir),
                    bundle_kind="final",
                )
            )
        stream.write(
            json.dumps(
                {
                    "artifact_bundle": final_bundle_dir,
                    "batches_per_epoch": len(batches),
                    "best_checkpoint": best_checkpoint,
                    "best_metric_name": best_metric_name if best_metric_value is not None else None,
                    "best_metric_value": best_metric_value,
                    "checkpoint": last_checkpoint,
                    "command": args.command,
                    "epochs": int(loop_cfg["epochs"]),
                    "epochs_ran": completed_epochs,
                    "epochs_requested": int(loop_cfg["epochs"]),
                    "eval_partition": eval_partition if eval_batches else None,
                    "final_eval_metrics": _scalarize_metrics(final_eval_metrics),
                    "logs_dir": str(logs_dir),
                    "steps": global_step,
                    "stopped_early": stopped_early,
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

    graph_registry, interaction_registry = _build_runtime_registries(cfg)

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
    graph_registry, interaction_registry = _build_runtime_registries(raw_cfg)
    validator.bind_registries(
        graph_registry=graph_registry,
        interaction_registry=interaction_registry,
    ).validate(raw_cfg)


def _build_runtime_registries(cfg: dict[str, Any]) -> tuple[Any, Any]:
    graph_registry = build_default_graph_registry()
    interaction_registry = build_default_interaction_registry()
    registrars = list(cfg.get("runtime", {}).get("plugin_registrars", []))
    if registrars:
        apply_runtime_registrars(
            registrars,
            graph_registry=graph_registry,
            interaction_registry=interaction_registry,
        )
    return graph_registry, interaction_registry


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
        optim_module = importlib.import_module("torch.optim")
    except ImportError:
        return None

    parameter_groups = executor.parameter_groups(runtime_cfg)
    trainable = [parameter for group in parameter_groups.values() for parameter in group if _is_trainable(parameter)]
    if not trainable:
        return None
    optimizer_cfg = cfg.get("training", {}).get("optimizer", {})
    optimizer_type = str(optimizer_cfg.get("type", "adam")).lower()
    lr = float(optimizer_cfg.get("lr", 0.01))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))
    if optimizer_type == "sgd":
        return optim_module.SGD(trainable, lr=lr, weight_decay=weight_decay)
    if optimizer_type == "adam":
        return optim_module.Adam(trainable, lr=lr, weight_decay=weight_decay)
    if optimizer_type == "adamw":
        return optim_module.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    raise UGTSDTIError(
        f"Unsupported optimizer type {optimizer_type!r}.",
        stage="cli",
        component="main",
        key="training.optimizer.type",
    )


def _build_scheduler(cfg: dict[str, Any], optimizer: Any | None) -> Any | None:
    if optimizer is None:
        return None
    scheduler_cfg = cfg.get("training", {}).get("scheduler", {})
    scheduler_type = str(scheduler_cfg.get("type", "none")).lower()
    if scheduler_type == "none":
        return None
    try:
        lr_scheduler_module = importlib.import_module("torch.optim.lr_scheduler")
    except ImportError:
        return None
    if scheduler_type == "step":
        return lr_scheduler_module.StepLR(
            optimizer,
            step_size=int(scheduler_cfg.get("step_size", 10)),
            gamma=float(scheduler_cfg.get("gamma", 0.5)),
        )
    if scheduler_type == "plateau":
        return lr_scheduler_module.ReduceLROnPlateau(
            optimizer,
            mode=cast(Literal["min", "max"], str(scheduler_cfg.get("mode", "max")).lower()),
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 1)),
        )
    raise UGTSDTIError(
        f"Unsupported scheduler type {scheduler_type!r}.",
        stage="cli",
        component="main",
        key="training.scheduler.type",
    )


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
    allow_missing_scenarios: bool = False,
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
        allow_missing_scenarios=allow_missing_scenarios,
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
    early_cfg = dict(loop_cfg.get("early_stopping", {}))
    return {
        "epochs": epochs,
        "checkpoint_every_epochs": checkpoint_every_epochs,
        "summary_every_steps": summary_every_steps,
        "eval_every_epochs": eval_every_epochs,
        "eval_partition": eval_partition,
        "select_checkpoint": str(loop_cfg.get("select_checkpoint", "last")).lower(),
        "best_metric": str(loop_cfg.get("best_metric", "metrics.auroc")),
        "best_mode": str(loop_cfg.get("best_mode", "max")).lower(),
        "early_stopping": {
            "enabled": bool(early_cfg.get("enabled", False)),
            "patience": max(0, int(early_cfg.get("patience", 0))),
            "min_delta": max(0.0, float(early_cfg.get("min_delta", 0.0))),
        },
    }


def _maybe_load_checkpoint(
    cfg: dict[str, Any],
    runtime_state: dict[str, Any],
    executor: PipelineExecutor,
    runtime_identity: dict[str, Any],
    *,
    strict_identity: bool = True,
) -> Any | None:
    checkpoint_path = runtime_state.get("checkpoint_path")
    if not checkpoint_path:
        return None
    checkpoint = CheckpointIO().load(
        checkpoint_path,
        expected_config_hash=str(runtime_identity["config_hash"]) if strict_identity else None,
        expected_dataset=str(cfg.get("data", {}).get("dataset", "")),
        expected_reproducibility_key=str(runtime_identity["reproducibility_key"]) if strict_identity else None,
    )
    executor.load_model_state(cfg, checkpoint.model_state)
    return checkpoint


def _restore_component_state(component: Any | None, state: dict[str, Any]) -> None:
    load_state_dict = getattr(component, "load_state_dict", None)
    if callable(load_state_dict) and state:
        load_state_dict(state)


def _checkpoint_paths(runtime_state: dict[str, Any], run_id: str) -> dict[str, Path]:
    checkpoint_dir = Path(runtime_state["checkpoint_dir"])
    return {
        "last": checkpoint_dir / f"{run_id}.pt",
        "best": checkpoint_dir / f"{run_id}.best.pt",
    }


def _build_checkpoint_bundle(
    *,
    identity: dict[str, Any],
    normalized_config: dict[str, Any],
    split_manifest: dict[str, Any],
    model_state: dict[str, Any],
    optimizer: Any | None,
    scheduler: Any | None,
    epoch: int,
    step: int,
    extras: dict[str, Any] | None = None,
) -> Any:
    return CheckpointBundle(
        model_state=model_state,
        optimizer_state=_component_state_dict(optimizer),
        scheduler_state=_component_state_dict(scheduler),
        rng_state={},
        epoch=epoch,
        step=step,
        identity=dict(identity),
        config=normalized_config,
        dataset_metadata=(split_manifest or {}).get("dataset_version", {}),
        split_metadata=split_manifest or {},
        extras=dict(extras or {}),
    )


def _component_state_dict(component: Any | None) -> dict[str, Any]:
    state_dict = getattr(component, "state_dict", None)
    if callable(state_dict):
        return dict(state_dict())
    return {}


def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
    scalarized = _scalarize_metrics(metrics)
    return scalarized.get(key)


def _is_better_metric(current: float | None, best: float | None, *, mode: str, min_delta: float) -> bool:
    if current is None:
        return False
    if best is None:
        return True
    if mode == "max":
        return current > best + min_delta
    return current < best - min_delta


def _is_plateau_scheduler(scheduler: Any | None) -> bool:
    return scheduler is not None and scheduler.__class__.__name__.lower() == "reducelronplateau"


def _step_plateau_scheduler(scheduler: Any | None, metric: float | None) -> None:
    if scheduler is None or metric is None:
        return
    scheduler.step(metric)


def _select_final_checkpoint(
    *,
    loop_cfg: dict[str, Any],
    last_checkpoint: str | None,
    best_checkpoint: str | None,
) -> str | None:
    if loop_cfg.get("select_checkpoint") == "best":
        return best_checkpoint or last_checkpoint
    return last_checkpoint or best_checkpoint


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
