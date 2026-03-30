"""Registry and planner for the interaction stage.

REQ-INT-001, REQ-ARCH-003
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

from ugtsdti.core.errors import InvalidInteractionGraphError, MissingDependencyError
from ugtsdti.interaction.base import (
    InteractionDefinition,
    InteractionPlan,
    InteractionPluginSpec,
)

if TYPE_CHECKING:
    from ugtsdti.interaction.base import InteractionRuntime


_INTERACTION_ALLOWED_KEYS = {"type", "type_key", "inputs", "params", "output_keys"}


class InteractionRegistry:
    """Registry for interaction plugin specs and runtime classes."""

    def __init__(self) -> None:
        self._specs: dict[str, InteractionPluginSpec] = {}
        self._runtime_classes: dict[str, type["InteractionRuntime"]] = {}

    def register(
        self,
        spec: InteractionPluginSpec,
        runtime_cls: type["InteractionRuntime"],
    ) -> None:
        self._specs[spec.type_key] = spec
        self._runtime_classes[spec.type_key] = runtime_cls

    def get_spec(self, type_key: str) -> InteractionPluginSpec:
        if type_key not in self._specs:
            raise MissingDependencyError(
                f"Interaction plugin type {type_key!r} is not registered.",
                stage="interaction",
                component="InteractionRegistry",
                key=type_key,
            )
        return self._specs[type_key]

    def build_runtime(self, definition: InteractionDefinition) -> "InteractionRuntime":
        if definition.type_key not in self._runtime_classes:
            raise MissingDependencyError(
                f"Interaction plugin type {definition.type_key!r} is not registered.",
                stage="interaction",
                component="InteractionRegistry",
                key=definition.type_key,
            )
        runtime_cls = self._runtime_classes[definition.type_key]
        return runtime_cls(**definition.params)


class InteractionPlanner:
    """Build and validate an interaction plan without instantiating runtimes."""

    def __init__(self, registry: InteractionRegistry) -> None:
        self._registry = registry

    def plan(
        self,
        interaction_cfg: dict[str, Any],
        *,
        available_inputs: set[str] | None = None,
    ) -> InteractionPlan:
        available = set(available_inputs or set())
        order = list(interaction_cfg.get("order", []))
        dependencies_cfg = interaction_cfg.get("dependencies", {})

        definitions: list[InteractionDefinition] = []
        seen_names: set[str] = set()
        for name in order:
            if name in seen_names:
                raise InvalidInteractionGraphError(
                    f"Duplicate interaction module name {name!r}.",
                    stage="interaction",
                    component="InteractionPlanner",
                    key=name,
                )
            seen_names.add(name)
            module_cfg = interaction_cfg.get(name, {})
            if not isinstance(module_cfg, dict):
                raise InvalidInteractionGraphError(
                    f"Interaction module {name!r} config must be a mapping.",
                    stage="interaction",
                    component="InteractionPlanner",
                    key=name,
                )
            extra_keys = set(module_cfg) - _INTERACTION_ALLOWED_KEYS
            if extra_keys:
                raise InvalidInteractionGraphError(
                    f"Interaction module {name!r} has unsupported top-level fields {sorted(extra_keys)}. "
                    "Runtime options must be nested under 'params'.",
                    stage="interaction",
                    component="InteractionPlanner",
                    key=name,
                )
            type_key = str(module_cfg.get("type", module_cfg.get("type_key", "")))
            if not type_key:
                raise InvalidInteractionGraphError(
                    f"Interaction module {name!r} is missing a type key.",
                    stage="interaction",
                    component="InteractionPlanner",
                    key=name,
                )
            deps = dependencies_cfg.get(name, [])
            if isinstance(deps, str):
                deps = [deps]
            params = _resolve_module_params(module_cfg)
            definitions.append(
                InteractionDefinition(
                    name=name,
                    type_key=type_key,
                    inputs=list(module_cfg.get("inputs", [])),
                    params=params,
                    dependencies=list(deps),
                )
            )

        definition_map = {definition.name: definition for definition in definitions}
        for definition in definitions:
            self._registry.get_spec(definition.type_key)
            for dep in definition.dependencies:
                if dep not in definition_map:
                    raise MissingDependencyError(
                        f"Interaction module {definition.name!r} depends on {dep!r} which is not configured.",
                        stage="interaction",
                        component=definition.name,
                        key=dep,
                    )

        topological_order = _topological_sort(definitions)

        producers: dict[str, str] = {}
        produced_keys: dict[str, list[str]] = {}
        resolved_keys = set(available)

        for name in topological_order:
            definition = definition_map[name]
            for input_key in definition.inputs:
                if input_key not in resolved_keys:
                    raise MissingDependencyError(
                        f"Interaction module {name!r} requires input {input_key!r} "
                        "which is not available from role outputs or prior interaction outputs.",
                        stage="interaction",
                        component=name,
                        key=input_key,
                    )

            spec = self._registry.get_spec(definition.type_key)
            output_keys = spec.output_keys_fn(definition.params)
            produced_keys[name] = output_keys
            for output_key in output_keys:
                if output_key in producers:
                    raise InvalidInteractionGraphError(
                        f"Interaction output key {output_key!r} is produced by both "
                        f"{producers[output_key]!r} and {name!r}.",
                        stage="interaction",
                        component=name,
                        key=output_key,
                    )
                producers[output_key] = name
                resolved_keys.add(output_key)

        ordered_definitions = [definition_map[name] for name in topological_order]
        return InteractionPlan(
            definitions=ordered_definitions,
            order=topological_order,
            producers=producers,
            produced_keys=produced_keys,
        )


def _topological_sort(definitions: list[InteractionDefinition]) -> list[str]:
    names = [definition.name for definition in definitions]
    in_degree: dict[str, int] = {name: 0 for name in names}
    successors: dict[str, list[str]] = {name: [] for name in names}

    for definition in definitions:
        for dep in definition.dependencies:
            in_degree[definition.name] += 1
            successors[dep].append(definition.name)

    queue: deque[str] = deque(name for name in names if in_degree[name] == 0)
    ordered: list[str] = []

    while queue:
        name = queue.popleft()
        ordered.append(name)
        for successor in successors[name]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)

    if len(ordered) != len(names):
        raise InvalidInteractionGraphError(
            "Interaction dependency graph contains a cycle.",
            stage="interaction",
            component="InteractionPlanner",
            key="interaction.dependencies",
        )

    return ordered


def _resolve_module_params(module_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return canonical nested runtime params for an interaction module."""
    params = module_cfg.get("params", {})
    return dict(params) if isinstance(params, dict) else {}
