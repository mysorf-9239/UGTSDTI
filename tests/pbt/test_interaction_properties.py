"""Property-based tests for interaction modules and planning."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.errors import InvalidInteractionGraphError
from ugtsdti.interaction.base import InteractionPluginSpec, InteractionRuntime
from ugtsdti.interaction.kd import binary_logits_to_dist
from ugtsdti.interaction.registry import InteractionPlanner, InteractionRegistry
from ugtsdti.interaction.uncertainty import UncertaintyInteraction


class _StubRuntime(InteractionRuntime):
    def forward(self, inputs, context):
        return {}


def _make_registry(*specs):
    registry = InteractionRegistry()
    for spec in specs:
        registry.register(spec, _StubRuntime)
    return registry


@given(
    logits=st.lists(
        st.floats(min_value=-20, max_value=20, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=8,
    ),
    temperature=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_binary_logits_distribution_property(logits, temperature):
    torch = pytest.importorskip("torch")
    tensor = torch.tensor([[value] for value in logits], dtype=torch.float32)

    dist = binary_logits_to_dist(tensor, temperature=temperature)

    assert torch.all(dist >= 0)
    assert torch.all(torch.isfinite(dist))
    assert torch.allclose(dist.sum(dim=-1), torch.ones(dist.shape[0]))


@given(
    teacher_logits=st.lists(
        st.floats(min_value=-20, max_value=20, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=6,
    ),
    student_logits=st.lists(
        st.floats(min_value=-20, max_value=20, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=6,
    ),
)
def test_uncertainty_outputs_property_for_valid_logits(teacher_logits, student_logits):
    torch = pytest.importorskip("torch")
    teacher = torch.tensor([[value] for value in teacher_logits], dtype=torch.float32)
    student = torch.tensor([[value] for value in student_logits], dtype=torch.float32)
    runtime = UncertaintyInteraction(targets={"teacher": True, "student": True})

    outputs = runtime.forward(
        {"teacher.logits": teacher, "student.logits": student},
        ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False),
    )

    for key in ("teacher.var", "student.var"):
        assert torch.all(torch.isfinite(outputs[key]))
        assert torch.all(outputs[key] >= 0)


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
            "inputs": ["student.logits"],
            "params": {"output_key": f"interaction.{name}.out"},
        }
    cfg["dependencies"][names[0]] = [names[-1]]

    with pytest.raises(InvalidInteractionGraphError):
        planner.plan(cfg, available_inputs={"student.logits"})
