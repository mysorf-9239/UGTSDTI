"""Unit tests for interaction planning and no-op interaction.

REQ-INT-001, REQ-ARCH-001
"""
from __future__ import annotations

import pytest

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.errors import InvalidInteractionGraphError, MissingDependencyError
from ugtsdti.interaction.base import InteractionPluginSpec, InteractionRuntime
from ugtsdti.interaction.noop import NoOpInteraction
from ugtsdti.interaction.registry import InteractionPlanner, InteractionRegistry


class _StubRuntime(InteractionRuntime):
    def forward(self, inputs, context):
        return {}


def _make_registry(*specs):
    registry = InteractionRegistry()
    for spec in specs:
        registry.register(spec, _StubRuntime)
    return registry


NOOP_SPEC = InteractionPluginSpec(type_key="noop", output_keys_fn=lambda params: [])
DIAG_SPEC = InteractionPluginSpec(type_key="diag", output_keys_fn=lambda params: ["interaction.disagreement"])
VAR_SPEC = InteractionPluginSpec(type_key="var", output_keys_fn=lambda params: ["teacher.var"])


class TestInteractionPlanner:
    def test_noop_plan_passes_with_empty_outputs(self):
        registry = _make_registry(NOOP_SPEC)
        planner = InteractionPlanner(registry)
        plan = planner.plan(
            {
                "order": ["noop"],
                "dependencies": {},
                "noop": {"type": "noop", "inputs": ["student.logits"]},
            },
            available_inputs={"student.logits"},
        )
        assert plan.order == ["noop"]
        assert plan.produced_keys["noop"] == []

    def test_dependency_cycle_raises(self):
        registry = _make_registry(NOOP_SPEC)
        planner = InteractionPlanner(registry)
        with pytest.raises(InvalidInteractionGraphError, match="cycle"):
            planner.plan(
                {
                    "order": ["a", "b"],
                    "dependencies": {"a": ["b"], "b": ["a"]},
                    "a": {"type": "noop"},
                    "b": {"type": "noop"},
                },
                available_inputs={"student.logits"},
            )

    def test_unresolved_input_raises(self):
        registry = _make_registry(DIAG_SPEC)
        planner = InteractionPlanner(registry)
        with pytest.raises(MissingDependencyError, match="teacher.logits"):
            planner.plan(
                {
                    "order": ["diag"],
                    "dependencies": {},
                    "diag": {"type": "diag", "inputs": ["teacher.logits"]},
                },
                available_inputs={"student.logits"},
            )

    def test_single_producer_per_output_key_enforced(self):
        registry = _make_registry(
            VAR_SPEC, InteractionPluginSpec(type_key="var2", output_keys_fn=lambda params: ["teacher.var"])
        )
        planner = InteractionPlanner(registry)
        with pytest.raises(InvalidInteractionGraphError, match="teacher.var"):
            planner.plan(
                {
                    "order": ["a", "b"],
                    "dependencies": {},
                    "a": {"type": "var", "inputs": ["teacher.logits"]},
                    "b": {"type": "var2", "inputs": ["teacher.logits"]},
                },
                available_inputs={"teacher.logits"},
            )

    def test_prior_interaction_outputs_resolve_downstream_inputs(self):
        registry = _make_registry(VAR_SPEC, DIAG_SPEC)
        planner = InteractionPlanner(registry)
        plan = planner.plan(
            {
                "order": ["var", "diag"],
                "dependencies": {"diag": ["var"]},
                "var": {"type": "var", "inputs": ["teacher.logits"]},
                "diag": {"type": "diag", "inputs": ["teacher.var"]},
            },
            available_inputs={"teacher.logits"},
        )
        assert plan.order == ["var", "diag"]
        assert plan.producers["teacher.var"] == "var"


class TestNoOpInteraction:
    def test_noop_forward_returns_no_outputs(self):
        runtime = NoOpInteraction()
        outputs = runtime.forward(
            {"student.logits": 1.0},
            ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False),
        )
        assert outputs == {}
