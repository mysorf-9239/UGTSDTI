"""Evaluation orchestration on top of the shared pipeline executor."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.state import State, StateWriter
from ugtsdti.logging.base import Logger
from ugtsdti.postprocess.metrics import MetricsReporter
from ugtsdti.trainer.trainer import PipelineExecutor, PipelineTrace, _collect_scalar_metrics


@dataclass
class EvaluationResult:
    """Aggregated evaluator output."""

    state: State
    metrics: dict[str, Any]
    traces: list[PipelineTrace] = field(default_factory=list)
    highlighted_scenarios: list[str] = field(default_factory=list)


class Evaluator:
    """Run scenario-aware evaluation with the shared pipeline executor."""

    def __init__(self, pipeline_executor: PipelineExecutor, *, logger: Logger | None = None) -> None:
        self._pipeline_executor = pipeline_executor
        self._logger = logger
        self._reporter = MetricsReporter()

    def evaluate(
        self,
        batches: list[dict[str, Any]],
        cfg: dict[str, Any],
        context: ExecutionContext,
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
                "interaction.disagreement",
                "teacher.var",
                "student.var",
            ):
                if state.has(optional_key):
                    collected.setdefault(optional_key, []).append(state.get(optional_key))

        aggregate_state = _build_aggregate_state(collected)
        metrics = self._reporter.report(cfg.get("metrics", {}), aggregate_state, _concat_values(labels))
        if self._logger is not None:
            self._logger.log_metrics(_collect_scalar_metrics_from_dict(metrics), step=0)

        highlighted = ["s4"] if any(key.startswith("metrics.s4.") for key in metrics) else []
        return EvaluationResult(
            state=aggregate_state,
            metrics=metrics,
            traces=traces,
            highlighted_scenarios=highlighted,
        )


def _build_aggregate_state(collected: dict[str, list[Any]]) -> State:
    state = State()
    writer = StateWriter(state)
    outputs = {
        "logits": _concat_values(collected.get("logits", [])),
        "scenario": list(collected.get("scenario", [])),
    }
    for key in ("gate.alpha", "interaction.disagreement", "teacher.var", "student.var"):
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
