"""Metrics and diagnostics reporting for postprocess stage."""
from __future__ import annotations

from typing import Any

from ugtsdti.core.state import State


class MetricsReporter:
    """Compute split-level and optional per-scenario metrics from state."""

    def report(
        self,
        metrics_cfg: dict[str, Any],
        state: State,
        labels: Any,
    ) -> dict[str, Any]:
        enabled = list(metrics_cfg.get("enabled", ["auroc", "auprc", "f1"]))
        by_scenario = bool(metrics_cfg.get("by_scenario", False))

        logits = _as_tensor(state.get("logits"))
        targets = _prepare_targets(labels, device=logits.device)
        probabilities = _sigmoid(logits)

        outputs: dict[str, Any] = {}
        for metric_name in enabled:
            outputs[f"metrics.{metric_name}"] = _compute_metric(metric_name, probabilities, targets)

        if by_scenario and state.has("scenario"):
            scenario_labels = state.get("scenario")
            outputs.update(_scenario_metrics(enabled, probabilities, targets, scenario_labels))

        outputs.update(_diagnostics_from_state(state))
        return outputs


def _scenario_metrics(
    enabled: list[str],
    probabilities: Any,
    targets: Any,
    scenario_labels: Any,
) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    labels_list = list(scenario_labels)
    for scenario in sorted(set(labels_list)):
        indices = [index for index, label in enumerate(labels_list) if label == scenario]
        scenario_probs = probabilities[indices]
        scenario_targets = targets[indices]
        for metric_name in enabled:
            outputs[f"metrics.{scenario}.{metric_name}"] = _compute_metric(
                metric_name, scenario_probs, scenario_targets
            )
    return outputs


def _diagnostics_from_state(state: State) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    if state.has("interaction.disagreement"):
        outputs["diagnostics.disagreement"] = state.get("interaction.disagreement")
    if state.has("gate.alpha"):
        outputs["diagnostics.gate_alpha"] = _as_tensor(state.get("gate.alpha")).mean()
    if state.has("teacher.var") and state.has("student.var"):
        teacher_var = _reduce_scalar(state.get("teacher.var"))
        student_var = _reduce_scalar(state.get("student.var"), device=teacher_var.device)
        outputs["diagnostics.uncertainty_error"] = (teacher_var - student_var).abs()
    return outputs


def _compute_metric(metric_name: str, probabilities: Any, targets: Any) -> Any:
    if metric_name == "f1":
        return _binary_f1(probabilities, targets)
    if metric_name == "auroc":
        return _binary_auroc(probabilities, targets)
    if metric_name == "auprc":
        return _binary_auprc(probabilities, targets)
    raise ValueError(f"Unsupported metric {metric_name!r}.")


def _binary_f1(probabilities: Any, targets: Any, threshold: float = 0.5) -> Any:
    try:
        import torch

        preds = (probabilities >= threshold).to(dtype=torch.float32)
        target = targets.to(dtype=torch.float32)
        tp = (preds * target).sum()
        fp = (preds * (1.0 - target)).sum()
        fn = ((1.0 - preds) * target).sum()
        denom = 2.0 * tp + fp + fn
        if denom.item() == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=probabilities.device)
        return 2.0 * tp / denom
    except ImportError as exc:
        raise RuntimeError("MetricsReporter requires torch.") from exc


def _binary_auroc(probabilities: Any, targets: Any) -> Any:
    try:
        import torch

        probs = probabilities.reshape(-1)
        target = targets.reshape(-1)
        positives = probs[target >= 0.5]
        negatives = probs[target < 0.5]
        if positives.numel() == 0 or negatives.numel() == 0:
            return torch.tensor(0.5, dtype=torch.float32, device=probs.device)
        wins = (positives[:, None] > negatives[None, :]).to(dtype=torch.float32).mean()
        ties = (positives[:, None] == negatives[None, :]).to(dtype=torch.float32).mean()
        return wins + 0.5 * ties
    except ImportError as exc:
        raise RuntimeError("MetricsReporter requires torch.") from exc


def _binary_auprc(probabilities: Any, targets: Any) -> Any:
    try:
        import torch

        probs = probabilities.reshape(-1)
        target = targets.reshape(-1)
        order = torch.argsort(probs, descending=True)
        sorted_target = target[order]
        positives = sorted_target.sum()
        if positives.item() == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=probs.device)
        cum_tp = torch.cumsum(sorted_target, dim=0)
        precision = cum_tp / torch.arange(1, sorted_target.numel() + 1, device=probs.device, dtype=torch.float32)
        return precision[sorted_target >= 0.5].mean()
    except ImportError as exc:
        raise RuntimeError("MetricsReporter requires torch.") from exc


def _prepare_targets(labels: Any, *, device: Any) -> Any:
    try:
        import torch

        target = labels if isinstance(labels, torch.Tensor) else torch.as_tensor(labels, dtype=torch.float32)
        target = target.to(dtype=torch.float32, device=device)
        if target.ndim == 0:
            target = target.reshape(1)
        if target.ndim > 1 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        return target
    except ImportError as exc:
        raise RuntimeError("MetricsReporter requires torch.") from exc


def _sigmoid(logits: Any) -> Any:
    try:
        import torch

        tensor = logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits, dtype=torch.float32)
        tensor = tensor.to(dtype=torch.float32)
        if tensor.ndim > 1 and tensor.shape[-1] == 1:
            tensor = tensor.squeeze(-1)
        return torch.sigmoid(tensor)
    except ImportError as exc:
        raise RuntimeError("MetricsReporter requires torch.") from exc


def _as_tensor(value: Any) -> Any:
    try:
        import torch

        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
        return tensor.to(dtype=torch.float32)
    except ImportError as exc:
        raise RuntimeError("MetricsReporter requires torch.") from exc


def _reduce_scalar(value: Any, *, device: Any | None = None) -> Any:
    tensor = _as_tensor(value)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor.mean()
