"""Evaluation orchestration on top of the shared pipeline executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.state import State, StateWriter
from ugtsdti.logging.base import Logger
from ugtsdti.postprocess.metrics import MetricsReporter
from ugtsdti.runtime.artifacts import ArtifactWriter
from ugtsdti.runtime.identity import ExperimentIdentity
from ugtsdti.trainer.trainer import PipelineExecutor, PipelineTrace, _collect_scalar_metrics


@dataclass
class EvaluationResult:
    """Aggregated evaluator output."""

    state: State
    metrics: dict[str, Any]
    traces: list[PipelineTrace] = field(default_factory=list)
    highlighted_scenarios: list[str] = field(default_factory=list)
    aggregate_trace: dict[str, Any] = field(default_factory=dict)


class Evaluator:
    """Run scenario-aware evaluation with the shared pipeline executor."""

    def __init__(
        self,
        pipeline_executor: PipelineExecutor,
        *,
        logger: Logger | None = None,
        artifact_writer: ArtifactWriter | None = None,
    ) -> None:
        self._pipeline_executor = pipeline_executor
        self._logger = logger
        self._artifact_writer = artifact_writer
        self._reporter = MetricsReporter()

    def evaluate(
        self,
        batches: list[dict[str, Any]],
        cfg: dict[str, Any],
        context: ExecutionContext,
        *,
        step: int | None = None,
        identity: ExperimentIdentity | dict[str, Any] | None = None,
        normalized_config: dict[str, Any] | None = None,
        split_manifest: dict[str, Any] | None = None,
        model_state: dict[str, Any] | None = None,
        logs_dir: str | None = None,
    ) -> EvaluationResult:
        traces: list[PipelineTrace] = []
        collected: dict[str, list[Any]] = {
            "logits": [],
            "scenario": [],
        }
        labels: list[Any] = []

        for batch in batches:
            state, trace = self._pipeline_executor.run_until_decision(batch, cfg, context)
            traces.append(trace)
            labels.append(batch["labels"])
            collected["logits"].append(state.get("logits"))
            collected["scenario"].extend(list(batch.get("scenario", [])))
            for optional_key in (
                "gate.alpha",
                "diagnostics.disagreement",
                "teacher.var",
                "student.var",
            ):
                if state.has(optional_key):
                    collected.setdefault(optional_key, []).append(state.get(optional_key))

        aggregate_state = _build_aggregate_state(collected)
        metrics = self._reporter.report(cfg.get("metrics", {}), aggregate_state, _concat_values(labels))
        aggregate_state_boundaries: dict[str, Any] = {
            f"batch_{index:04d}": trace.state_boundary_summaries for index, trace in enumerate(traces)
        }
        aggregate_trace: dict[str, Any] = {
            "num_batches": len(traces),
            "num_samples": len(collected.get("scenario", [])),
            "scenario_counts": {
                str(label): list(collected.get("scenario", [])).count(label)
                for label in sorted(set(collected.get("scenario", [])))
            },
            "num_traces_kept": len(traces),
            "stage_orders": [list(trace.stage_order) for trace in traces],
            "state_boundary_summaries": aggregate_state_boundaries,
        }
        if self._logger is not None:
            self._logger.log_metrics(_collect_scalar_metrics_from_dict(metrics), step=step)
        if self._artifact_writer is not None and identity is not None and normalized_config is not None and traces:
            self._artifact_writer.write_bundle(
                identity=_identity_dict(identity),
                config=normalized_config,
                metrics={key: value for key, value in metrics.items() if key.startswith("metrics.")},
                diagnostics={key: value for key, value in metrics.items() if key.startswith("diagnostics.")},
                split_manifest=split_manifest or {},
                model_state=model_state or {},
                execution_trace=aggregate_trace,
                state_boundary_summaries=aggregate_state_boundaries,
                logs_dir=logs_dir,
                bundle_kind="final",
            )

        highlighted = ["s4"] if any(key.startswith("metrics.s4.") for key in metrics) else []
        return EvaluationResult(
            state=aggregate_state,
            metrics=metrics,
            traces=traces,
            highlighted_scenarios=highlighted,
            aggregate_trace=aggregate_trace,
        )


def _build_aggregate_state(collected: dict[str, list[Any]]) -> State:
    state = State()
    writer = StateWriter(state)
    outputs = {
        "logits": _concat_values(collected.get("logits", [])),
        "scenario": list(collected.get("scenario", [])),
    }
    for key in ("gate.alpha", "diagnostics.disagreement", "teacher.var", "student.var"):
        values = collected.get(key, [])
        if values:
            outputs[key] = _concat_values(values)
    writer.commit("evaluation.aggregate", outputs)
    return state


def _concat_values(values: list[Any]) -> Any:
    if not values:
        return []
    try:
        import torch

        if all(isinstance(value, torch.Tensor) for value in values):
            if all(value.ndim == 0 for value in values):
                return torch.stack(values, dim=0)
            return torch.cat(values, dim=0)
    except ImportError:
        pass

    merged: list[Any] = []
    for value in values:
        if isinstance(value, list):
            merged.extend(value)
        else:
            merged.append(value)
    return merged


def _collect_scalar_metrics_from_dict(metrics: dict[str, Any]) -> dict[str, float]:
    state = State()
    writer = StateWriter(state)
    writer.commit("evaluation.metrics", metrics)
    return _collect_scalar_metrics(state)


def _identity_dict(identity: ExperimentIdentity | dict[str, Any]) -> dict[str, Any]:
    if isinstance(identity, ExperimentIdentity):
        return identity.to_dict()
    return dict(identity)
