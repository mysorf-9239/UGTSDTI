"""Concrete decision modules."""
from __future__ import annotations

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.errors import InvalidDecisionOutputError
from ugtsdti.core.state import State
from ugtsdti.decision.base import DecisionModule


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
