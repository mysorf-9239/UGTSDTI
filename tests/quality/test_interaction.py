"""Unit tests for interaction planning and no-op interaction.

REQ-INT-001, REQ-ARCH-001
"""
from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.errors import InvalidInteractionGraphError, MissingDependencyError
from ugtsdti.interaction.base import InteractionPluginSpec, InteractionRuntime
from ugtsdti.interaction.diagnostics import DiagnosticsInteraction, diagnostics_output_keys
from ugtsdti.interaction.kd import KDInteraction, binary_logits_to_dist, kd_output_keys
from ugtsdti.interaction.noop import NoOpInteraction
from ugtsdti.interaction.registry import InteractionPlanner, InteractionRegistry
from ugtsdti.interaction.uncertainty import UncertaintyInteraction, uncertainty_output_keys


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


class TestKDInteraction:
    def test_binary_logits_to_dist_returns_valid_distribution(self):
        torch = pytest.importorskip("torch")
        logits = torch.tensor([[0.0], [2.0], [-1.0]])
        dist = binary_logits_to_dist(logits, temperature=2.0)

        assert dist.shape == (3, 2)
        assert torch.all(dist >= 0)
        assert torch.allclose(dist.sum(dim=-1), torch.ones(3))

    def test_kd_logits_mode_emits_explicit_targets_and_loss(self):
        torch = pytest.importorskip("torch")
        runtime = KDInteraction(mode="logits", temperature=2.0)
        outputs = runtime.forward(
            {
                "teacher.logits": torch.tensor([[2.0], [0.0]]),
                "student.logits": torch.tensor([[0.5], [-0.5]]),
            },
            ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False),
        )

        assert set(outputs) == set(kd_output_keys({}))
        assert outputs["kd.teacher_target"].shape == (2, 2)
        assert outputs["kd.student_target"].shape == (2, 2)
        assert outputs["interaction.kd.loss_component"].shape == ()
        assert outputs["interaction.kd.loss_component"].item() >= 0.0

    def test_kd_feature_and_relation_modes_are_supported(self):
        torch = pytest.importorskip("torch")
        features = {
            "teacher.features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "student.features": torch.tensor([[0.9, 0.1], [0.2, 0.8]]),
        }

        feature_outputs = KDInteraction(
            mode="feature",
            teacher_key="teacher.features",
            student_key="student.features",
        ).forward(features, ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False))
        relation_outputs = KDInteraction(
            mode="relation",
            teacher_key="teacher.features",
            student_key="student.features",
        ).forward(features, ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False))

        assert feature_outputs["interaction.kd.loss_component"].item() >= 0.0
        assert relation_outputs["interaction.kd.loss_component"].item() >= 0.0

    def test_kd_default_direction_is_teacher_to_student(self):
        torch = pytest.importorskip("torch")
        runtime = KDInteraction(mode="logits", temperature=1.0)
        teacher_logits = torch.tensor([[4.0], [-2.0]])
        student_logits = torch.tensor([[0.0], [0.0]])
        outputs = runtime.forward(
            {"teacher.logits": teacher_logits, "student.logits": student_logits},
            ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False),
        )

        expected_teacher = binary_logits_to_dist(teacher_logits, temperature=1.0)
        assert torch.allclose(outputs["kd.teacher_target"], expected_teacher)


class TestUncertaintyInteraction:
    def test_uncertainty_outputs_are_finite_and_non_negative(self):
        torch = pytest.importorskip("torch")
        runtime = UncertaintyInteraction(targets={"teacher": True, "student": True})
        outputs = runtime.forward(
            {
                "teacher.logits": torch.tensor(
                    [
                        [[0.0], [1.0]],
                        [[0.2], [1.1]],
                        [[-0.1], [0.9]],
                    ]
                ),
                "student.logits": torch.tensor(
                    [
                        [[0.5], [0.4]],
                        [[0.7], [0.3]],
                        [[0.6], [0.2]],
                    ]
                ),
            },
            ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False),
        )

        assert set(outputs) == set(uncertainty_output_keys({"targets": {"teacher": True, "student": True}}))
        assert torch.all(torch.isfinite(outputs["teacher.var"]))
        assert torch.all(torch.isfinite(outputs["student.var"]))
        assert torch.all(outputs["teacher.var"] >= 0)
        assert torch.all(outputs["student.var"] >= 0)

    def test_uncertainty_for_plain_logits_is_not_forced_to_zero(self):
        torch = pytest.importorskip("torch")
        runtime = UncertaintyInteraction(targets={"teacher": True, "student": False})
        outputs = runtime.forward(
            {"teacher.logits": torch.tensor([[2.0], [0.0], [-2.0]])},
            ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False),
        )

        assert torch.any(outputs["teacher.var"] > 0)

    def test_uncertainty_supports_teacher_only_path(self):
        torch = pytest.importorskip("torch")
        runtime = UncertaintyInteraction(targets={"teacher": True, "student": False})
        outputs = runtime.forward(
            {"teacher.logits": torch.tensor([[[0.0]], [[0.1]], [[-0.1]]])},
            ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False),
        )

        assert set(outputs) == {"teacher.var"}


class TestDiagnosticsInteraction:
    def test_diagnostics_emit_disagreement_and_confidence_stats(self):
        torch = pytest.importorskip("torch")
        runtime = DiagnosticsInteraction()
        outputs = runtime.forward(
            {
                "teacher.logits": torch.tensor([[2.0], [1.0]]),
                "student.logits": torch.tensor([[1.0], [-1.0]]),
            },
            ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False),
        )

        assert set(outputs) == set(diagnostics_output_keys({"emit_calibration": True}))
        assert outputs["interaction.disagreement"].item() >= 0.0
        assert 0.0 <= outputs["diagnostics.teacher_confidence"].item() <= 1.0
        assert 0.0 <= outputs["diagnostics.student_confidence"].item() <= 1.0

    def test_diagnostics_supports_enabled_flag(self):
        runtime = DiagnosticsInteraction(enabled=False)
        outputs = runtime.forward(
            {"teacher.logits": 1.0, "student.logits": 0.0},
            ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False),
        )
        assert outputs == {}


@st.composite
def _interaction_dag_cfgs(draw):
    names = draw(
        st.lists(
            st.sampled_from(["mod_a", "mod_b", "mod_c", "mod_d", "mod_e"]),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    dependencies = {}
    cfg = {"order": names, "dependencies": dependencies}

    for index, name in enumerate(names):
        deps = [] if index == 0 else draw(st.lists(st.sampled_from(names[:index]), unique=True))
        dependencies[name] = deps
        inputs = ["student.logits"] if not deps else [f"interaction.{dep}.out" for dep in deps]
        cfg[name] = {
            "type": "prop",
            "inputs": inputs,
            "params": {"output_key": f"interaction.{name}.out"},
        }
    return cfg


@given(_interaction_dag_cfgs())
def test_interaction_planner_keeps_dependencies_before_generated_modules(cfg):
    registry = _make_registry(
        InteractionPluginSpec(type_key="prop", output_keys_fn=lambda params: [params["output_key"]])
    )
    planner = InteractionPlanner(registry)

    plan = planner.plan(cfg, available_inputs={"student.logits"})
    positions = {name: index for index, name in enumerate(plan.order)}

    for name, deps in cfg["dependencies"].items():
        for dep in deps:
            assert positions[dep] < positions[name]


@given(st.lists(st.sampled_from(["a", "b", "c", "d"]), min_size=2, max_size=4, unique=True))
def test_interaction_planner_rejects_cycles_for_generated_modules(names):
    registry = _make_registry(
        InteractionPluginSpec(type_key="prop", output_keys_fn=lambda params: [params["output_key"]])
    )
    planner = InteractionPlanner(registry)

    cfg = {
        "order": names,
        "dependencies": {},
    }
    for index, name in enumerate(names):
        deps = [names[index - 1]] if index > 0 else []
        cfg["dependencies"][name] = deps
        cfg[name] = {
            "type": "prop",
            "inputs": ["student.logits"] if not deps else [f"interaction.{deps[0]}.out"],
            "params": {"output_key": f"interaction.{name}.out"},
        }
    cfg["dependencies"][names[0]] = [names[-1]]
    cfg[names[0]]["inputs"] = [f"interaction.{names[-1]}.out"]

    with pytest.raises(InvalidInteractionGraphError):
        planner.plan(cfg, available_inputs={"student.logits"})
