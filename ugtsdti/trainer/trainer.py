"""Pipeline execution and training orchestration."""

from __future__ import annotations

import copy
import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.errors import (
    BatchSchemaError,
    CheckpointCorruptedError,
    InvalidConfigError,
    NumericalInstabilityError,
)
from ugtsdti.core.schema import StateSchemaBuilder
from ugtsdti.core.state import State, StateWriter
from ugtsdti.decision.base import DecisionModule
from ugtsdti.decision.module import (
    HardSelectionDecisionModule,
    IdentityDecisionModule,
    SoftBlendingDecisionModule,
)
from ugtsdti.decision.policy import DecisionPolicy
from ugtsdti.decision.trust import TrustEstimator
from ugtsdti.graph.engine import GraphEngine, GraphTrace
from ugtsdti.interaction.engine import InteractionEngine
from ugtsdti.logging.base import Logger
from ugtsdti.postprocess.loss import LossComposer
from ugtsdti.postprocess.metrics import MetricsReporter
from ugtsdti.roles.binder import RoleBinder, RoleBinding
from ugtsdti.runtime.artifacts import ArtifactWriter
from ugtsdti.runtime.checkpoint import CheckpointBundle, CheckpointIO
from ugtsdti.runtime.identity import ExperimentIdentity


@dataclass
class PipelineTrace:
    """Human-readable trace across stage boundaries."""

    stage_order: list[str] = field(default_factory=list)
    graph_trace: GraphTrace | None = None
    state_boundary_summaries: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class TrainStepResult:
    """Structured result of a single training step."""

    state: State
    trace: PipelineTrace
    loss: Any
    logged_metrics: dict[str, float]
    freeze_policy: dict[str, bool]
    kd_weight: float | None = None


@dataclass
class ModelRestoreReport:
    """Summary of graph runtime state restored from a checkpoint payload."""

    restored_nodes: list[str] = field(default_factory=list)


class PipelineExecutor:
    """Run the minimal fixed pipeline through the decision stage."""

    def __init__(
        self,
        *,
        graph_registry: Any,
        interaction_registry: Any,
        debug: bool = False,
        strict_mode: bool = False,
    ) -> None:
        self._graph_registry = graph_registry
        self._interaction_registry = interaction_registry
        self._debug = debug
        self._strict_mode = strict_mode
        self._schema_builder = StateSchemaBuilder()
        self._graph_engine = GraphEngine(self._graph_registry, debug=self._debug, strict_mode=self._strict_mode)
        self._interaction_engine = InteractionEngine(self._interaction_registry)

    def run_until_decision(
        self,
        batch: dict[str, Any],
        cfg: dict[str, Any],
        context: ExecutionContext,
    ) -> tuple[State, PipelineTrace]:
        state, _, trace = self._run_core_pipeline(batch, cfg, context)
        return state, trace

    def run_batch(
        self,
        batch: dict[str, Any],
        cfg: dict[str, Any],
        context: ExecutionContext,
    ) -> tuple[State, PipelineTrace]:
        state, writer, trace = self._run_core_pipeline(batch, cfg, context)
        outputs: dict[str, Any] = {}
        outputs.update(LossComposer().compose(cfg.get("loss", {}), state, batch["labels"]))
        metrics_cfg = cfg.get("metrics", {})
        if metrics_cfg:
            outputs.update(MetricsReporter().report(metrics_cfg, state, batch["labels"]))
        postprocess_outputs = {
            key: value for key, value in outputs.items() if not (key.startswith("diagnostics.") and state.has(key))
        }
        writer.commit("postprocess", postprocess_outputs)
        trace.stage_order.append("postprocess")
        trace.state_boundary_summaries["postprocess"] = state.keys()
        return state, trace

    def _run_core_pipeline(
        self,
        batch: dict[str, Any],
        cfg: dict[str, Any],
        context: ExecutionContext,
    ) -> tuple[State, StateWriter, PipelineTrace]:
        state = State()
        writer = StateWriter(state)
        trace = PipelineTrace(
            stage_order=["batch", "graph", "role_binding", "interaction", "decision"],
        )

        self._validate_batch(batch, cfg)
        writer.commit("batch", dict(batch))
        trace.state_boundary_summaries["batch"] = state.keys()

        graph_plan = cfg["graph_plan"]
        trace.graph_trace = self._graph_engine.run(graph_plan, state, writer, context)
        trace.state_boundary_summaries["graph"] = state.keys()

        bindings = self._build_role_bindings(cfg)
        RoleBinder(bindings).bind(state, writer)
        trace.state_boundary_summaries["role_binding"] = state.keys()

        self._run_interactions(state, writer, cfg, context)
        trace.state_boundary_summaries["interaction"] = state.keys()

        decision = self._build_decision_module(cfg)
        writer.commit("decision", decision.forward(state, context))
        trace.state_boundary_summaries["decision"] = state.keys()

        return state, writer, trace

    def _validate_batch(self, batch: dict[str, Any], cfg: dict[str, Any]) -> None:
        batch_spec = self._schema_builder.build_batch_spec(cfg)
        for key in batch_spec.required_common:
            if key not in batch:
                raise BatchSchemaError(
                    f"Batch is missing required key {key!r}.",
                    stage="batch",
                    component="PipelineExecutor",
                    key=key,
                )
        for key in batch_spec.conditional_keys:
            if key not in batch:
                raise BatchSchemaError(
                    f"Batch is missing conditional key {key!r} required by the selected graph.",
                    stage="batch",
                    component="PipelineExecutor",
                    key=key,
                )

    def _build_role_bindings(self, cfg: dict[str, Any]) -> list[RoleBinding]:
        bindings: list[RoleBinding] = []
        for role_name, role_cfg in cfg.get("roles", {}).items():
            outputs = role_cfg.get("outputs")
            if outputs is None and "source" in role_cfg:
                outputs = [role_cfg["source"]]
            bindings.append(
                RoleBinding(
                    role=role_name,
                    outputs=list(outputs or []),
                    aggregation=role_cfg.get("aggregation", "first"),
                    allow_multi_output_first=bool(role_cfg.get("allow_multi_output_first", False)),
                )
            )
        return bindings

    def _run_interactions(
        self,
        state: State,
        writer: StateWriter,
        cfg: dict[str, Any],
        context: ExecutionContext,
    ) -> None:
        self._interaction_engine.run(
            cfg["interaction_plan"],
            state=state,
            writer=writer,
            context=context,
        )

    def _build_decision_module(self, cfg: dict[str, Any]) -> DecisionModule:
        decision_cfg = cfg.get("decision", {})
        decision_type = decision_cfg.get("type", "identity")
        strategy = decision_cfg.get("strategy", "identity")
        fallback = dict(decision_cfg.get("fallback", {}))

        if decision_type == "identity" or strategy == "identity":
            source_key = decision_cfg.get("source_key", "student.logits")
            return IdentityDecisionModule(source_key=source_key)

        trust_estimator = TrustEstimator(
            use_uncertainty=bool(decision_cfg.get("use_uncertainty", False)),
            uncertainty_source=_resolve_uncertainty_source(cfg),
        )
        policy = DecisionPolicy()

        if strategy == "soft":
            return SoftBlendingDecisionModule(
                trust_estimator=trust_estimator,
                policy=policy,
                fallback=fallback,
            )
        if strategy == "hard":
            return HardSelectionDecisionModule(
                threshold=float(decision_cfg.get("threshold", 0.5)),
                trust_estimator=trust_estimator,
                policy=policy,
                fallback=fallback,
            )

        raise InvalidConfigError(
            f"Unsupported decision strategy {strategy!r} for decision type {decision_type!r}.",
            stage="decision",
            component="PipelineExecutor",
            key="decision.strategy",
        )

    def parameter_groups(self, cfg: dict[str, Any]) -> dict[str, list[Any]]:
        """Collect graph runtime parameters grouped by role."""
        plan = cfg["graph_plan"]
        definitions = plan.definition_map()
        producers = plan.producers
        dependencies = plan.edges
        groups: dict[str, list[Any]] = {"teacher": [], "student": [], "gate": []}
        for role_name in ("teacher", "student"):
            visited_nodes: set[str] = set()
            role_cfg = cfg.get("roles", {}).get(role_name, {})
            for output_key in role_cfg.get("outputs", []):
                node_name = producers.get(output_key)
                if node_name is None:
                    continue
                for ancestor in _collect_upstream_nodes(node_name, dependencies):
                    if ancestor in visited_nodes:
                        continue
                    visited_nodes.add(ancestor)
                    runtime = self._graph_engine.ensure_runtime(definitions[ancestor])
                    parameters = getattr(runtime, "parameters", None)
                    if callable(parameters):
                        groups[role_name].extend(list(parameters()))
        return groups

    def model_state(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """Collect serializable runtime state for graph nodes."""
        plan = cfg["graph_plan"]
        definitions = plan.definition_map()
        state: dict[str, Any] = {}
        for node_name in plan.order:
            runtime = self._graph_engine.ensure_runtime(definitions[node_name])
            state_dict = getattr(runtime, "state_dict", None)
            if callable(state_dict):
                state[node_name] = state_dict()
        return state

    def load_model_state(self, cfg: dict[str, Any], model_state: dict[str, Any]) -> ModelRestoreReport:
        """Load runtime state into graph nodes with fail-closed restore semantics."""
        if not isinstance(model_state, dict):
            raise InvalidConfigError(
                "Model state must be a mapping of node_name -> state_dict.",
                stage="runtime",
                component="PipelineExecutor",
                key="model_state",
            )
        plan = cfg["graph_plan"]
        definitions = plan.definition_map()
        report = ModelRestoreReport()
        for node_name, node_state in model_state.items():
            if node_name not in definitions:
                raise CheckpointCorruptedError(
                    f"Checkpoint model state references unknown graph node {node_name!r}.",
                    stage="runtime",
                    component="PipelineExecutor",
                    key=f"model_state.{node_name}",
                )
            runtime = self._graph_engine.ensure_runtime(definitions[node_name])
            load_state_dict = getattr(runtime, "load_state_dict", None)
            if not callable(load_state_dict):
                raise CheckpointCorruptedError(
                    f"Checkpoint contains state for graph node {node_name!r}, "
                    "but its runtime does not support load_state_dict().",
                    stage="runtime",
                    component="PipelineExecutor",
                    key=f"model_state.{node_name}",
                )
            load_state_dict(node_state)
            report.restored_nodes.append(node_name)
        return report


class Trainer:
    """Training orchestration on top of the shared pipeline executor."""

    def __init__(
        self,
        pipeline_executor: PipelineExecutor,
        *,
        logger: Logger | None = None,
        checkpoint_io: CheckpointIO | None = None,
        artifact_writer: ArtifactWriter | None = None,
        parameter_groups: dict[str, list[Any]] | None = None,
    ) -> None:
        self._pipeline_executor = pipeline_executor
        self._logger = logger
        self._checkpoint_io = checkpoint_io
        self._artifact_writer = artifact_writer
        self._parameter_groups = dict(parameter_groups or {})

    def step(
        self,
        batch: dict[str, Any],
        cfg: dict[str, Any],
        context: ExecutionContext,
        *,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        step_idx: int = 0,
        epoch: int = 0,
        checkpoint_path: str | None = None,
        checkpoint_bundle: CheckpointBundle | None = None,
        identity: ExperimentIdentity | dict[str, Any] | None = None,
        normalized_config: dict[str, Any] | None = None,
        split_manifest: dict[str, Any] | None = None,
        model_state: Any | None = None,
        logs_dir: str | None = None,
    ) -> TrainStepResult:
        scheduled_cfg = self._apply_kd_schedule(cfg, step_idx)
        freeze_policy = _resolve_freeze_policy(scheduled_cfg)
        _apply_freeze_policy(self._parameter_groups, freeze_policy)

        if optimizer is not None:
            _zero_grad(optimizer)

        with _autocast_context(context):
            state, trace = self._pipeline_executor.run_batch(batch, scheduled_cfg, context)
            loss = state.get("loss.total")

        if _strict_training_flag(scheduled_cfg, "fail_on_nonfinite_loss") and not _is_finite_value(loss):
            raise NumericalInstabilityError(
                "Training loss is non-finite.",
                stage="train",
                component="Trainer",
                key="loss.total",
            )

        grad_norm: float | None = None
        if optimizer is not None and _requires_grad(loss):
            loss.backward()
            grad_norm = _grad_norm(self._parameter_groups)
            max_grad_norm = _max_grad_norm(scheduled_cfg)
            if max_grad_norm is not None:
                grad_norm = _clip_grad_norm(self._parameter_groups, max_grad_norm)
            if _strict_training_flag(scheduled_cfg, "fail_on_nonfinite_grad") and not _is_finite_number(grad_norm):
                raise NumericalInstabilityError(
                    "Training gradients are non-finite.",
                    stage="train",
                    component="Trainer",
                    key="grad_norm",
                )
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        resolved_model_state = model_state() if callable(model_state) else model_state

        logged_metrics = _collect_scalar_metrics(state)
        logged_metrics["trainer.epoch"] = float(epoch)
        logged_metrics["trainer.step"] = float(step_idx)
        logged_metrics["trainer.teacher_frozen"] = 1.0 if freeze_policy["teacher"] else 0.0
        logged_metrics["trainer.student_frozen"] = 1.0 if freeze_policy["student"] else 0.0
        logged_metrics["trainer.gate_trainable"] = 1.0 if freeze_policy["gate"] else 0.0
        if grad_norm is not None and _is_finite_number(grad_norm):
            logged_metrics["trainer.grad_norm"] = float(grad_norm)
        if self._logger is not None:
            self._logger.log_metrics(logged_metrics, step=step_idx)

        if self._checkpoint_io is not None and checkpoint_path:
            bundle = checkpoint_bundle
            if bundle is None and identity is not None and normalized_config is not None:
                bundle = CheckpointBundle(
                    model_state=resolved_model_state or {},
                    optimizer_state=_component_state_dict(optimizer),
                    scheduler_state=_component_state_dict(scheduler),
                    rng_state={},
                    epoch=epoch,
                    step=step_idx,
                    identity=_identity_dict(identity),
                    config=normalized_config,
                    dataset_metadata=(split_manifest or {}).get("dataset_version", {}),
                    split_metadata=split_manifest or {},
                )
            elif bundle is not None and resolved_model_state is not None:
                bundle = replace(
                    bundle,
                    model_state=resolved_model_state,
                    optimizer_state=_component_state_dict(optimizer),
                    scheduler_state=_component_state_dict(scheduler),
                    epoch=epoch,
                    step=step_idx,
                )
            if bundle is not None:
                self._checkpoint_io.save(bundle, checkpoint_path)
        if self._artifact_writer is not None and identity is not None and normalized_config is not None:
            safe_snapshot = state.snapshot_isolated()
            self._artifact_writer.write_bundle(
                identity=_identity_dict(identity),
                config=normalized_config,
                metrics={key: value for key, value in safe_snapshot.items() if key.startswith("metrics.")},
                diagnostics={key: value for key, value in safe_snapshot.items() if key.startswith("diagnostics.")},
                split_manifest=split_manifest or {},
                model_state=resolved_model_state
                if resolved_model_state is not None
                else checkpoint_bundle.model_state
                if checkpoint_bundle
                else {},
                execution_trace=asdict(trace),
                state_boundary_summaries=trace.state_boundary_summaries,
                logs_dir=logs_dir,
                bundle_kind="snapshot",
                snapshot_label=f"epoch-{epoch:04d}-step-{step_idx:08d}",
            )

        return TrainStepResult(
            state=state,
            trace=trace,
            loss=loss,
            logged_metrics=logged_metrics,
            freeze_policy=freeze_policy,
            kd_weight=_read_kd_weight(scheduled_cfg),
        )

    def _apply_kd_schedule(self, cfg: dict[str, Any], step_idx: int) -> dict[str, Any]:
        scheduled_cfg = copy.deepcopy(cfg)
        weight = _scheduled_kd_weight(scheduled_cfg, step_idx)
        if weight is None:
            return scheduled_cfg
        scheduled_cfg.setdefault("loss", {}).setdefault("map", {}).setdefault("kd", {})["weight"] = weight
        return scheduled_cfg


def _resolve_freeze_policy(cfg: dict[str, Any]) -> dict[str, bool]:
    training_cfg = cfg.get("training", {})
    teacher_cfg = training_cfg.get("teacher", {})
    student_cfg = training_cfg.get("student", {})
    gate_cfg = training_cfg.get("gate", {})
    return {
        "teacher": bool(teacher_cfg.get("freeze", True)),
        "student": bool(student_cfg.get("freeze", False)),
        "gate": bool(gate_cfg.get("trainable", False)),
    }


def _resolve_uncertainty_source(cfg: dict[str, Any]) -> str:
    interaction_cfg = cfg.get("interaction", {})
    for module_name in interaction_cfg.get("order", []):
        module_cfg = interaction_cfg.get(module_name, {})
        if not isinstance(module_cfg, dict):
            continue
        type_key = str(module_cfg.get("type", module_cfg.get("type_key", "")))
        if type_key == "uncertainty.sample_variance":
            return "sample_variance"
        if type_key == "uncertainty.confidence_proxy":
            return "confidence_proxy"
    return "none"


def _scheduled_kd_weight(cfg: dict[str, Any], step_idx: int) -> float | None:
    kd_cfg = cfg.get("training", {}).get("kd", {})
    schedule = str(kd_cfg.get("schedule", "constant")).lower()
    mapping = cfg.get("loss", {}).get("map", {}).get("kd")
    if not isinstance(mapping, dict):
        return None
    target_weight = float(mapping.get("weight", 1.0))
    if schedule == "constant":
        return target_weight
    if schedule == "warmup":
        warmup_steps = max(1, int(kd_cfg.get("warmup_steps", 1)))
        progress = min(1.0, float(step_idx + 1) / float(warmup_steps))
        return target_weight * progress
    raise InvalidConfigError(
        f"Unsupported KD schedule {schedule!r}.",
        stage="train",
        component="Trainer",
        key="training.kd.schedule",
    )


def _read_kd_weight(cfg: dict[str, Any]) -> float | None:
    mapping = cfg.get("loss", {}).get("map", {}).get("kd")
    if not isinstance(mapping, dict):
        return None
    return float(mapping.get("weight", 1.0))


def _apply_freeze_policy(parameter_groups: dict[str, list[Any]], freeze_policy: dict[str, bool]) -> None:
    for parameter in parameter_groups.get("teacher", []):
        if hasattr(parameter, "requires_grad"):
            parameter.requires_grad = not freeze_policy["teacher"]
    for parameter in parameter_groups.get("student", []):
        if hasattr(parameter, "requires_grad"):
            parameter.requires_grad = not freeze_policy["student"]
    for parameter in parameter_groups.get("gate", []):
        if hasattr(parameter, "requires_grad"):
            parameter.requires_grad = freeze_policy["gate"]


def _zero_grad(optimizer: Any) -> None:
    zero_grad = getattr(optimizer, "zero_grad", None)
    if callable(zero_grad):
        zero_grad(set_to_none=True)


def _requires_grad(value: Any) -> bool:
    return bool(getattr(value, "requires_grad", False))


def _autocast_context(context: ExecutionContext) -> Any:
    if context.precision != "mixed":
        return nullcontext()
    try:
        import torch
    except ImportError:
        return nullcontext()
    device_type = context.device.split(":", 1)[0]
    if device_type not in {"cpu", "cuda"}:
        return nullcontext()
    return torch.autocast(device_type=device_type, enabled=True)


def _collect_scalar_metrics(state: State) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in state.snapshot_isolated().items():
        if not (key.startswith("loss.") or key.startswith("metrics.") or key.startswith("diagnostics.")):
            continue
        scalar = _to_scalar(value)
        if scalar is not None:
            metrics[key] = scalar
    return metrics


def _to_scalar(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return float(value.detach().to(dtype=torch.float32).mean().cpu().item())
    except ImportError:
        return None
    return None


def _identity_dict(identity: ExperimentIdentity | dict[str, Any]) -> dict[str, Any]:
    if isinstance(identity, ExperimentIdentity):
        return identity.to_dict()
    return dict(identity)


def _collect_upstream_nodes(node_name: str, edges: dict[str, list[str]]) -> list[str]:
    ordered: list[str] = []
    visited: set[str] = set()

    def visit(current: str) -> None:
        if current in visited:
            return
        visited.add(current)
        for dependency in edges.get(current, []):
            visit(dependency)
        ordered.append(current)

    visit(node_name)
    return ordered


def _component_state_dict(component: Any | None) -> dict[str, Any]:
    state_dict = getattr(component, "state_dict", None)
    if callable(state_dict):
        return dict(state_dict())
    return {}


def _strict_training_flag(cfg: dict[str, Any], key: str) -> bool:
    return bool(cfg.get("training", {}).get("loop", {}).get(key, False))


def _max_grad_norm(cfg: dict[str, Any]) -> float | None:
    value = cfg.get("training", {}).get("loop", {}).get("max_grad_norm", None)
    if value is None:
        return None
    return float(value)


def _all_parameters(parameter_groups: dict[str, list[Any]]) -> list[Any]:
    parameters: list[Any] = []
    seen: set[int] = set()
    for group in parameter_groups.values():
        for parameter in group:
            marker = id(parameter)
            if marker in seen:
                continue
            seen.add(marker)
            parameters.append(parameter)
    return parameters


def _grad_norm(parameter_groups: dict[str, list[Any]]) -> float | None:
    try:
        import torch
    except ImportError:
        return None
    values: list[float] = []
    for parameter in _all_parameters(parameter_groups):
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        values.append(float(torch.linalg.vector_norm(grad.detach()).cpu().item()))
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values))


def _clip_grad_norm(parameter_groups: dict[str, list[Any]], max_grad_norm: float) -> float | None:
    try:
        import torch
    except ImportError:
        return None
    parameters = [
        parameter for parameter in _all_parameters(parameter_groups) if getattr(parameter, "grad", None) is not None
    ]
    if not parameters:
        return None
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_grad_norm)
    return float(norm.detach().cpu().item()) if hasattr(norm, "detach") else float(norm)


def _is_finite_value(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return value == value and value not in {float("inf"), float("-inf")}
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return bool(torch.all(torch.isfinite(value)).item())
    except ImportError:
        return False
    return True


def _is_finite_number(value: float | None) -> bool:
    if value is None:
        return True
    return value == value and value not in {float("inf"), float("-inf")}
