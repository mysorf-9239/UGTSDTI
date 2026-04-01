"""Decision modules wrapper for post-process pipeline."""

from __future__ import annotations

from typing import Any

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.state import State, StateWriter
from ugtsdti.decision.module import SoftBlendingDecisionModule


def run_decision(state: State, cfg: dict[str, Any]) -> State:
    """Run decision module on state."""
    context = ExecutionContext(mode="eval", seed=42, device="cpu", deterministic=False)

    # Create decision module
    decision = SoftBlendingDecisionModule(fallback=cfg.get("fallback", {"no_uncertainty": "student"}))

    # Run decision
    outputs = decision.forward(state, context)

    # Write outputs to state
    writer = StateWriter(state)
    writer.commit("decision", outputs)

    return state
