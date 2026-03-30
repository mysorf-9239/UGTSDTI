"""Runtime execution for the interaction stage with contract enforcement."""

from __future__ import annotations

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.errors import InvalidInteractionGraphError, MissingDependencyError
from ugtsdti.core.state import State, StateWriter
from ugtsdti.interaction.base import InteractionPlan
from ugtsdti.interaction.registry import InteractionRegistry


class InteractionEngine:
    """Execute a planned interaction stage and enforce runtime output contracts."""

    def __init__(self, registry: InteractionRegistry) -> None:
        self._registry = registry

    def run(
        self,
        plan: InteractionPlan,
        *,
        state: State,
        writer: StateWriter,
        context: ExecutionContext,
    ) -> None:
        definition_map = plan.definition_map()
        for name in plan.order:
            definition = definition_map[name]
            runtime = self._registry.build_runtime(definition)
            inputs = self._materialize_inputs(name, definition.inputs, state)
            outputs = runtime.forward(inputs, context)
            if not isinstance(outputs, dict):
                raise InvalidInteractionGraphError(
                    f"Interaction runtime {name!r} returned a non-mapping output.",
                    stage="interaction",
                    component=name,
                    key="outputs",
                )

            expected_keys = set(plan.produced_keys.get(name, []))
            actual_keys = set(outputs)
            unexpected = sorted(actual_keys - expected_keys)
            missing = sorted(expected_keys - actual_keys)

            if unexpected:
                raise InvalidInteractionGraphError(
                    f"Interaction runtime {name!r} produced undeclared keys {unexpected}.",
                    stage="interaction",
                    component=name,
                    key=unexpected[0],
                )
            if missing:
                raise InvalidInteractionGraphError(
                    f"Interaction runtime {name!r} did not produce required keys {missing}.",
                    stage="interaction",
                    component=name,
                    key=missing[0],
                )
            if outputs:
                writer.commit(f"interaction.{name}", outputs)

    def _materialize_inputs(
        self,
        interaction_name: str,
        declared_inputs: list[str],
        state: State,
    ) -> dict[str, object]:
        inputs: dict[str, object] = {}
        for key in declared_inputs:
            if not state.has(key):
                raise MissingDependencyError(
                    f"Interaction runtime {interaction_name!r} requires input {key!r} which is not present in State.",
                    stage="interaction",
                    component=interaction_name,
                    key=key,
                )
            inputs[key] = state.get(key)
        return inputs
