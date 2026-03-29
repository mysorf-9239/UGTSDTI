"""Shared pipeline executor for train/eval flows."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.errors import BatchSchemaError, InvalidConfigError
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
from ugtsdti.postprocess.loss import LossComposer
from ugtsdti.postprocess.metrics import MetricsReporter
from ugtsdti.roles.binder import RoleBinder, RoleBinding


@dataclass
class PipelineTrace:
    """Human-readable trace across stage boundaries."""

    stage_order: list[str] = field(default_factory=list)
    graph_trace: GraphTrace | None = None
    state_boundary_summaries: dict[str, list[str]] = field(default_factory=dict)


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
        writer.commit("postprocess", outputs)
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
        graph_engine = GraphEngine(self._graph_registry, debug=self._debug, strict_mode=self._strict_mode)
        trace.graph_trace = graph_engine.run(graph_plan, state, writer, context)
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
        interaction_plan = cfg["interaction_plan"]
        definition_map = interaction_plan.definition_map()
        for name in interaction_plan.order:
            definition = definition_map[name]
            runtime = self._interaction_registry.build_runtime(definition)
            inputs = {key: state.get(key) for key in definition.inputs if state.has(key)}
            outputs = runtime.forward(inputs, context)
            if outputs:
                writer.commit(f"interaction.{name}", outputs)

    def _build_decision_module(self, cfg: dict[str, Any]) -> DecisionModule:
        decision_cfg = cfg.get("decision", {})
        decision_type = decision_cfg.get("type", "identity")
        strategy = decision_cfg.get("strategy", "identity")
        fallback = dict(decision_cfg.get("fallback", {}))

        if decision_type == "identity" or strategy == "identity":
            source_key = decision_cfg.get("source_key", "student.logits")
            return IdentityDecisionModule(source_key=source_key)

        trust_estimator = TrustEstimator(use_uncertainty=bool(decision_cfg.get("use_uncertainty", False)))
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
