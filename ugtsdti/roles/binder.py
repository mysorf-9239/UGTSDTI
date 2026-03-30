"""Role binding — map graph outputs into canonical role logits.

REQ-ROLE-001, REQ-STATE-004, REQ-ARCH-004
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Any, Callable, Iterable, Literal

from ugtsdti.core.errors import InvalidConfigError, InvalidRoleBindingError
from ugtsdti.core.schema import validate_graph_key, validate_role_key
from ugtsdti.core.state import State, StateWriter


@dataclass(frozen=True)
class RoleBinding:
    """Bind one semantic role to one or more graph outputs."""

    role: str
    outputs: list[str]
    aggregation: Literal["first", "mean"] = "first"
    allow_multi_output_first: bool = False


class RoleBinder:
    """Produce canonical `<role>.logits` keys from graph outputs."""

    def __init__(self, bindings: Iterable[RoleBinding]) -> None:
        self._bindings = list(bindings)

    def bind(self, state: State, writer: StateWriter) -> None:
        """Validate bindings and commit canonical role logits."""
        for binding in self._bindings:
            role_key = f"{binding.role}.logits"
            try:
                validate_role_key(role_key)
            except InvalidConfigError as exc:
                raise InvalidRoleBindingError(
                    f"Invalid role binding target {role_key!r}.",
                    stage="role_binding",
                    component=binding.role,
                    key=role_key,
                ) from exc

            if binding.aggregation not in {"first", "mean"}:
                raise InvalidRoleBindingError(
                    f"Unsupported aggregation {binding.aggregation!r} for role {binding.role!r}.",
                    stage="role_binding",
                    component=binding.role,
                    key=role_key,
                )

            if not binding.outputs:
                raise InvalidRoleBindingError(
                    f"Role {binding.role!r} must bind at least one graph output.",
                    stage="role_binding",
                    component=binding.role,
                    key=role_key,
                )
            if binding.aggregation == "first" and len(binding.outputs) > 1 and not binding.allow_multi_output_first:
                raise InvalidRoleBindingError(
                    f"Role {binding.role!r} config is ambiguous: aggregation='first' with multiple outputs "
                    "requires allow_multi_output_first=true.",
                    stage="role_binding",
                    component=binding.role,
                    key=role_key,
                )

            values: list[Any] = []
            for output_key in binding.outputs:
                try:
                    validate_graph_key(output_key)
                except InvalidConfigError as exc:
                    raise InvalidRoleBindingError(
                        f"Role {binding.role!r} references invalid graph key {output_key!r}.",
                        stage="role_binding",
                        component=binding.role,
                        key=output_key,
                    ) from exc

                if not state.has(output_key):
                    raise InvalidRoleBindingError(
                        f"Role {binding.role!r} references missing graph output {output_key!r}.",
                        stage="role_binding",
                        component=binding.role,
                        key=output_key,
                    )
                values.append(state.get(output_key))

            bound_value = self._aggregate(binding, values)
            writer.commit(f"role_binding.{binding.role}", {role_key: bound_value})

    def _aggregate(self, binding: RoleBinding, values: list[Any]) -> Any:
        if binding.aggregation == "first":
            return values[0]
        return _mean_values(binding.role, values)


def _mean_values(role: str, values: list[Any]) -> Any:
    first = values[0]

    try:
        import torch

        if isinstance(first, torch.Tensor):
            _ensure_same_shape(role, values, lambda v: tuple(v.shape))
            return torch.stack(values, dim=0).mean(dim=0)
    except ImportError:
        pass

    try:
        import numpy as np

        if isinstance(first, np.ndarray):
            _ensure_same_shape(role, values, lambda v: tuple(v.shape))
            return np.stack(values, axis=0).mean(axis=0)
    except ImportError:
        pass

    if isinstance(first, Number):
        if not all(isinstance(v, Number) for v in values):
            raise InvalidRoleBindingError(
                f"Role {role!r} cannot mean-aggregate mixed value types.",
                stage="role_binding",
                component=role,
                key=f"{role}.logits",
            )
        return sum(values) / len(values)

    raise InvalidRoleBindingError(
        f"Role {role!r} received unsupported value type {type(first).__name__!r} for mean aggregation.",
        stage="role_binding",
        component=role,
        key=f"{role}.logits",
    )


def _ensure_same_shape(
    role: str,
    values: list[Any],
    shape_fn: Callable[[Any], tuple[Any, ...]],
) -> None:
    expected = shape_fn(values[0])
    for value in values[1:]:
        if shape_fn(value) != expected:
            raise InvalidRoleBindingError(
                f"Role {role!r} requires shape-compatible outputs for mean aggregation.",
                stage="role_binding",
                component=role,
                key=f"{role}.logits",
            )
