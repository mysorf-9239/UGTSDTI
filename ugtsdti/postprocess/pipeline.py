"""Post-process pipeline for deterministic ML framework."""

from __future__ import annotations

from typing import Any

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.state import State
from ugtsdti.postprocess.decision import run_decision
from ugtsdti.postprocess.loss import run_loss
from ugtsdti.postprocess.metrics import run_metrics
from ugtsdti.postprocess.uncertainty import run_uncertainty


def run_postprocess(state: State, config: Any, context: ExecutionContext, labels: Any = None) -> State:
    """
    Executes post-process in strict order:
    1) uncertainty
    2) decision
    3) loss (train only)
    4) metrics (eval only)

    Must be deterministic and side-effect-free except state writes.
    """
    # Handle both dict and object config
    uncertainty_cfg = getattr(config, "uncertainty", None) or config.get("uncertainty", {})
    decision_cfg = getattr(config, "decision", None) or config.get("decision", {})
    loss_cfg = getattr(config, "loss", None) or config.get("loss", {})
    metrics_cfg = getattr(config, "metrics", None) or config.get("metrics", {})

    # 1. Uncertainty
    if uncertainty_cfg and uncertainty_cfg.get("enabled", True):
        state = run_uncertainty(state, uncertainty_cfg)

    # 2. Decision (ALWAYS)
    state = run_decision(state, decision_cfg)

    # 3. Loss
    if context.mode == "train":
        state = run_loss(state, loss_cfg, labels)

    # 4. Metrics — run regardless of mode when enabled
    if metrics_cfg and metrics_cfg.get("enabled"):
        state = run_metrics(state, metrics_cfg, context, labels)

    return state
