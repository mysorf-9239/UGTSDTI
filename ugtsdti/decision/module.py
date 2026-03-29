"""Concrete decision modules."""
from __future__ import annotations

from typing import Any

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.errors import InvalidDecisionOutputError
from ugtsdti.core.state import State
from ugtsdti.decision.base import DecisionModule
from ugtsdti.decision.policy import DecisionPolicy
from ugtsdti.decision.trust import TrustEstimator


class IdentityDecisionModule(DecisionModule):
    """Copy one canonical branch logits tensor to final `logits`."""

    def __init__(self, source_key: str = "student.logits") -> None:
        self._source_key = source_key

    def forward(
        self,
        state: State,
        context: ExecutionContext,
    ) -> dict[str, object]:
        del context
        if not state.has(self._source_key):
            raise InvalidDecisionOutputError(
                f"Identity decision requires source key {self._source_key!r}.",
                stage="decision",
                component="IdentityDecisionModule",
                key=self._source_key,
            )
        return {"logits": state.get(self._source_key)}


class SoftBlendingDecisionModule(DecisionModule):
    """Blend teacher/student logits using trust-derived alpha."""

    def __init__(
        self,
        *,
        trust_estimator: TrustEstimator | None = None,
        policy: DecisionPolicy | None = None,
        fallback: dict[str, str] | None = None,
    ) -> None:
        self._trust_estimator = trust_estimator or TrustEstimator()
        self._policy = policy or DecisionPolicy()
        self._fallback = dict(fallback or {})

    def forward(
        self,
        state: State,
        context: ExecutionContext,
    ) -> dict[str, Any]:
        del context
        fallback_outputs = _single_branch_fallback(state, self._fallback)
        if fallback_outputs is not None:
            return fallback_outputs

        try:
            trust_outputs = self._trust_estimator.estimate(state)
        except InvalidDecisionOutputError:
            fallback_key = self._fallback.get("no_uncertainty")
            if fallback_key:
                return _fallback_from_key(state, fallback_key, self._fallback)
            raise

        logits = self._policy.soft_blend(
            state.get("teacher.logits"),
            state.get("student.logits"),
            trust_outputs["gate.alpha"],
        )
        _validate_logits(logits, component="SoftBlendingDecisionModule")
        return {"logits": logits, **trust_outputs}


class HardSelectionDecisionModule(DecisionModule):
    """Select teacher or student logits with a hard decision rule."""

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        trust_estimator: TrustEstimator | None = None,
        policy: DecisionPolicy | None = None,
        fallback: dict[str, str] | None = None,
    ) -> None:
        self._threshold = threshold
        self._trust_estimator = trust_estimator or TrustEstimator()
        self._policy = policy or DecisionPolicy()
        self._fallback = dict(fallback or {})

    def forward(
        self,
        state: State,
        context: ExecutionContext,
    ) -> dict[str, Any]:
        del context
        fallback_outputs = _single_branch_fallback(state, self._fallback)
        if fallback_outputs is not None:
            return fallback_outputs

        try:
            trust_outputs = self._trust_estimator.estimate(state)
        except InvalidDecisionOutputError:
            fallback_key = self._fallback.get("no_uncertainty")
            if fallback_key:
                return _fallback_from_key(state, fallback_key, self._fallback)
            raise

        logits = self._policy.hard_select(
            state.get("teacher.logits"),
            state.get("student.logits"),
            trust_outputs["gate.alpha"],
            threshold=self._threshold,
        )
        _validate_logits(logits, component="HardSelectionDecisionModule")
        return {"logits": logits, **trust_outputs}


def _single_branch_fallback(state: State, fallback: dict[str, str]) -> dict[str, Any] | None:
    has_teacher = state.has("teacher.logits")
    has_student = state.has("student.logits")
    if has_teacher and has_student:
        return None
    if not has_teacher and has_student:
        return _fallback_from_key(state, fallback.get("no_teacher", "student"), fallback)
    if has_teacher and not has_student:
        return _fallback_from_key(state, fallback.get("no_student", "teacher"), fallback)
    raise InvalidDecisionOutputError(
        "Decision module requires at least one branch logits tensor.",
        stage="decision",
        component="DecisionModule",
        key="student.logits",
    )


def _fallback_from_key(state: State, fallback_key: str, fallback: dict[str, str]) -> dict[str, Any]:
    del fallback
    source_key = {
        "teacher": "teacher.logits",
        "student": "student.logits",
    }.get(fallback_key, fallback_key)
    if not state.has(source_key):
        raise InvalidDecisionOutputError(
            f"Configured fallback source {source_key!r} is unavailable.",
            stage="decision",
            component="DecisionModule",
            key=source_key,
        )
    logits = state.get(source_key)
    _validate_logits(logits, component="DecisionModule")
    return {"logits": logits}


def _validate_logits(logits: Any, *, component: str) -> None:
    try:
        import torch

        if isinstance(logits, torch.Tensor) and not torch.all(torch.isfinite(logits)):
            raise InvalidDecisionOutputError(
                "Decision module produced non-finite logits.",
                stage="decision",
                component=component,
                key="logits",
            )
    except ImportError as exc:
        raise RuntimeError("Advanced decision modules require torch.") from exc
