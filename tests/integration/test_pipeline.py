"""Integration tests for the minimal baseline pipeline."""

from __future__ import annotations

import pytest

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.errors import CheckpointCorruptedError, InvalidInteractionGraphError
from ugtsdti.graph.builder import GraphBuilder
from ugtsdti.graph.planner import GraphPlanner
from ugtsdti.graph.registry import NodeRegistry
from ugtsdti.graph.specs import NodePluginSpec
from ugtsdti.interaction.base import InteractionPluginSpec
from ugtsdti.interaction.kd import KDInteraction, kd_output_keys
from ugtsdti.interaction.noop import NoOpInteraction
from ugtsdti.interaction.registry import InteractionPlanner, InteractionRegistry
from ugtsdti.interaction.uncertainty import UncertaintyInteraction, uncertainty_output_keys
from ugtsdti.nodes.base import NodeRuntime
from ugtsdti.postprocess.loss import LossComposer
from ugtsdti.runtime.defaults import build_default_graph_registry, build_default_interaction_registry
from ugtsdti.trainer.trainer import PipelineExecutor


class EncoderRuntime(NodeRuntime):
    def forward(self, inputs, context):
        del context
        drug = inputs["drug_seq"]
        protein = inputs["protein_seq"]
        import torch

        return {"embedding": torch.cat([drug, protein], dim=1)}


class TeacherEncoderRuntime(NodeRuntime):
    def forward(self, inputs, context):
        del context
        drug_graph = inputs["drug_graph"]
        protein = inputs["protein_seq"]
        import torch

        return {"embedding": torch.cat([drug_graph, protein], dim=1)}


class HeadRuntime(NodeRuntime):
    def forward(self, inputs, context):
        del context
        import torch

        embedding = next(iter(inputs.values()))
        return {"logits": embedding.sum(dim=1, keepdim=True) * 0.1 + torch.tensor([[0.2], [0.2]])}


class UnexpectedInteractionRuntime:
    def forward(self, inputs, context):
        del inputs, context
        return {"unexpected.key": 1.0}


def _make_registries():
    graph_registry = NodeRegistry()
    graph_registry.register(
        NodePluginSpec(type_key="encoder.baseline", output_attrs=["embedding"]),
        EncoderRuntime,
    )
    graph_registry.register(
        NodePluginSpec(type_key="encoder.teacher", output_attrs=["embedding"]),
        TeacherEncoderRuntime,
    )
    graph_registry.register(
        NodePluginSpec(type_key="head.linear", output_attrs=["logits"]),
        HeadRuntime,
    )

    interaction_registry = InteractionRegistry()
    interaction_registry.register(
        InteractionPluginSpec(type_key="noop", output_keys_fn=lambda params: []),
        NoOpInteraction,
    )
    interaction_registry.register(
        InteractionPluginSpec(type_key="kd.standard", output_keys_fn=kd_output_keys),
        KDInteraction,
    )
    interaction_registry.register(
        InteractionPluginSpec(type_key="uncertainty.confidence_proxy", output_keys_fn=uncertainty_output_keys),
        UncertaintyInteraction,
    )
    return graph_registry, interaction_registry


def _minimal_cfg():
    graph_cfg = {
        "nodes": [
            {
                "name": "student_encoder",
                "type_key": "encoder.baseline",
                "inputs": ["drug_seq", "protein_seq"],
            },
            {
                "name": "student_head",
                "type_key": "head.linear",
                "inputs": ["student_encoder.embedding"],
            },
        ]
    }
    interaction_cfg = {
        "order": ["noop"],
        "dependencies": {},
        "noop": {"type": "noop", "inputs": ["student.logits"]},
    }
    return {
        "graph": graph_cfg,
        "roles": {"student": {"outputs": ["student_head.logits"], "aggregation": "first"}},
        "interaction": interaction_cfg,
        "decision": {"type": "identity", "source_key": "student.logits"},
        "loss": {"type": "hard", "hard_weight": 1.0, "map": {}},
        "metrics": {"enabled": ["f1"], "by_scenario": False},
    }


def _teacher_student_cfg():
    cfg = _minimal_cfg()
    cfg["graph"]["nodes"] = [
        {
            "name": "student_encoder",
            "type_key": "encoder.baseline",
            "inputs": ["drug_seq", "protein_seq"],
        },
        {
            "name": "teacher_encoder",
            "type_key": "encoder.teacher",
            "inputs": ["drug_graph", "protein_seq"],
        },
        {
            "name": "student_head",
            "type_key": "head.linear",
            "inputs": ["student_encoder.embedding"],
        },
        {
            "name": "teacher_head",
            "type_key": "head.linear",
            "inputs": ["teacher_encoder.embedding"],
        },
    ]
    cfg["roles"] = {
        "student": {"outputs": ["student_head.logits"], "aggregation": "first"},
        "teacher": {"outputs": ["teacher_head.logits"], "aggregation": "first"},
    }
    cfg["interaction"]["noop"]["inputs"] = ["student.logits", "teacher.logits"]
    cfg["modalities"] = {
        "available": ["sequence", "structure"],
        "teacher": {"uses": ["sequence", "structure"]},
        "student": {"uses": ["sequence"]},
    }
    return cfg


def _teacher_student_kd_cfg():
    cfg = _teacher_student_cfg()
    cfg["interaction"] = {
        "order": ["kd"],
        "dependencies": {},
        "kd": {
            "type": "kd.standard",
            "inputs": ["teacher.logits", "student.logits"],
            "params": {"mode": "logits", "temperature": 2.0, "enabled": True},
        },
    }
    cfg["loss"] = {
        "type": "composite",
        "hard_weight": 1.0,
        "map": {"kd": {"from": "interaction.kd.loss_component", "weight": 0.3}},
    }
    return cfg


def _student_only_disabled_kd_cfg():
    cfg = _minimal_cfg()
    cfg["interaction"] = {
        "order": ["kd"],
        "dependencies": {},
        "kd": {
            "type": "kd.standard",
            "inputs": ["student.logits"],
            "params": {"mode": "logits", "enabled": False},
        },
    }
    return cfg


def _teacher_student_uncertainty_cfg():
    cfg = _teacher_student_cfg()
    cfg["interaction"] = {
        "order": ["uncertainty"],
        "dependencies": {},
        "uncertainty": {
            "type": "uncertainty.confidence_proxy",
            "inputs": ["teacher.logits", "student.logits"],
            "params": {
                "enabled": True,
                "targets": {"teacher": True, "student": True},
            },
        },
    }
    cfg["decision"] = {
        "type": "gate.uncertainty",
        "strategy": "soft",
        "use_uncertainty": True,
        "fallback": {"no_teacher": "student", "no_student": "teacher"},
    }
    return cfg


def test_minimal_baseline_pipeline_runs_end_to_end():
    torch = pytest.importorskip("torch")
    graph_registry, interaction_registry = _make_registries()
    cfg = _minimal_cfg()
    cfg["graph_plan"] = GraphPlanner(graph_registry).plan(GraphBuilder(graph_registry).build(cfg["graph"]))
    cfg["interaction_plan"] = InteractionPlanner(interaction_registry).plan(
        cfg["interaction"],
        available_inputs={"student.logits"},
    )

    batch = {
        "labels": torch.tensor([[1.0], [0.0]]),
        "scenario": ["s1", "s1"],
        "drug_seq": torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
        "protein_seq": torch.tensor([[0.0, 1.0], [0.2, 0.8]]),
    }
    executor = PipelineExecutor(
        graph_registry=graph_registry,
        interaction_registry=interaction_registry,
        debug=True,
    )
    state, trace = executor.run_until_decision(
        batch,
        cfg,
        ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False),
    )

    assert state.has("student.logits")
    assert state.has("logits")
    assert state.get("logits").shape == (2, 1)
    assert trace.graph_trace is not None
    assert trace.graph_trace.node_order == ["student_encoder", "student_head"]
    assert "decision" in trace.state_boundary_summaries

    losses = LossComposer().compose(cfg["loss"], state, batch["labels"])
    assert "loss.total" in losses
    assert "loss.hard" in losses
    assert losses["loss.total"].shape == ()


def test_teacher_student_pipeline_keeps_student_as_identity_decision_source():
    torch = pytest.importorskip("torch")
    graph_registry, interaction_registry = _make_registries()
    cfg = _teacher_student_cfg()
    cfg["graph_plan"] = GraphPlanner(graph_registry).plan(GraphBuilder(graph_registry).build(cfg["graph"]))
    cfg["interaction_plan"] = InteractionPlanner(interaction_registry).plan(
        cfg["interaction"],
        available_inputs={"student.logits", "teacher.logits"},
    )

    batch = {
        "labels": torch.tensor([[1.0], [0.0]]),
        "scenario": ["s1", "s1"],
        "drug_seq": torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
        "protein_seq": torch.tensor([[0.0, 1.0], [0.2, 0.8]]),
        "drug_graph": torch.tensor([[0.3, 0.7], [0.1, 0.9]]),
    }
    executor = PipelineExecutor(
        graph_registry=graph_registry,
        interaction_registry=interaction_registry,
        debug=True,
    )
    state, trace = executor.run_until_decision(
        batch,
        cfg,
        ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False),
    )

    assert state.has("student.logits")
    assert state.has("teacher.logits")
    assert state.has("logits")
    assert torch.equal(state.get("logits"), state.get("student.logits"))


def test_load_model_state_returns_restore_report_for_supported_runtime():
    torch = pytest.importorskip("torch")
    graph_registry = build_default_graph_registry()
    interaction_registry = build_default_interaction_registry()
    cfg = _minimal_cfg()
    cfg["graph"] = {
        "nodes": [
            {
                "name": "student_encoder",
                "type_key": "encoder.baseline",
                "inputs": ["drug_seq", "protein_seq"],
            },
            {
                "name": "student_head",
                "type_key": "head.linear",
                "inputs": ["student_encoder.embedding"],
            },
        ]
    }
    cfg["graph_plan"] = GraphPlanner(graph_registry).plan(GraphBuilder(graph_registry).build(cfg["graph"]))
    cfg["interaction_plan"] = InteractionPlanner(interaction_registry).plan(
        cfg["interaction"],
        available_inputs={"student.logits"},
    )

    executor = PipelineExecutor(graph_registry=graph_registry, interaction_registry=interaction_registry)
    report = executor.load_model_state(
        cfg,
        {"student_head": {"scale": torch.tensor([0.2]), "bias": torch.tensor([0.1])}},
    )

    assert report.restored_nodes == ["student_head"]


def test_load_model_state_fails_when_runtime_cannot_restore_checkpoint_state():
    torch = pytest.importorskip("torch")
    graph_registry, interaction_registry = _make_registries()
    cfg = _minimal_cfg()
    cfg["graph_plan"] = GraphPlanner(graph_registry).plan(GraphBuilder(graph_registry).build(cfg["graph"]))
    cfg["interaction_plan"] = InteractionPlanner(interaction_registry).plan(
        cfg["interaction"],
        available_inputs={"student.logits"},
    )

    executor = PipelineExecutor(graph_registry=graph_registry, interaction_registry=interaction_registry)

    with pytest.raises(CheckpointCorruptedError, match="does not support load_state_dict"):
        executor.load_model_state(cfg, {"student_head": {"scale": torch.tensor([0.2])}})


def test_teacher_student_kd_pipeline_maps_interaction_loss_explicitly():
    torch = pytest.importorskip("torch")
    graph_registry, interaction_registry = _make_registries()
    cfg = _teacher_student_kd_cfg()
    cfg["graph_plan"] = GraphPlanner(graph_registry).plan(GraphBuilder(graph_registry).build(cfg["graph"]))
    cfg["interaction_plan"] = InteractionPlanner(interaction_registry).plan(
        cfg["interaction"],
        available_inputs={"student.logits", "teacher.logits"},
    )

    batch = {
        "labels": torch.tensor([[1.0], [0.0]]),
        "scenario": ["s1", "s1"],
        "drug_seq": torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
        "protein_seq": torch.tensor([[0.0, 1.0], [0.2, 0.8]]),
        "drug_graph": torch.tensor([[0.3, 0.7], [0.1, 0.9]]),
    }
    executor = PipelineExecutor(
        graph_registry=graph_registry,
        interaction_registry=interaction_registry,
        debug=True,
    )
    state, trace = executor.run_batch(
        batch,
        cfg,
        ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False),
    )

    assert state.has("interaction.kd.loss_component")
    assert state.has("kd.teacher_target")
    assert state.has("kd.student_target")
    assert state.has("loss.kd")
    assert state.has("loss.total")
    expected_total = state.get("loss.hard") + 0.3 * state.get("interaction.kd.loss_component")
    assert torch.allclose(state.get("loss.kd"), state.get("interaction.kd.loss_component"))
    assert torch.allclose(state.get("loss.total"), expected_total)


def test_pipeline_executor_fails_closed_for_interaction_runtime_contract_violations():
    torch = pytest.importorskip("torch")
    graph_registry, interaction_registry = _make_registries()
    interaction_registry.register(
        InteractionPluginSpec(type_key="bad.runtime", output_keys_fn=lambda params: ["teacher.var"]),
        UnexpectedInteractionRuntime,
    )
    cfg = _teacher_student_cfg()
    cfg["interaction"] = {
        "order": ["bad"],
        "dependencies": {},
        "bad": {"type": "bad.runtime", "inputs": ["teacher.logits"]},
    }
    cfg["graph_plan"] = GraphPlanner(graph_registry).plan(GraphBuilder(graph_registry).build(cfg["graph"]))
    cfg["interaction_plan"] = InteractionPlanner(interaction_registry).plan(
        cfg["interaction"],
        available_inputs={"student.logits", "teacher.logits"},
    )
    batch = {
        "labels": torch.tensor([[1.0], [0.0]]),
        "scenario": ["s1", "s1"],
        "drug_seq": torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
        "protein_seq": torch.tensor([[0.0, 1.0], [0.2, 0.8]]),
        "drug_graph": torch.tensor([[0.3, 0.7], [0.1, 0.9]]),
    }
    executor = PipelineExecutor(graph_registry=graph_registry, interaction_registry=interaction_registry)

    with pytest.raises(InvalidInteractionGraphError, match="undeclared keys"):
        executor.run_until_decision(
            batch,
            cfg,
            ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False),
        )


def test_disabled_kd_pipeline_uses_explicit_noop_path_without_hidden_outputs():
    torch = pytest.importorskip("torch")
    graph_registry, interaction_registry = _make_registries()
    cfg = _student_only_disabled_kd_cfg()
    cfg["graph_plan"] = GraphPlanner(graph_registry).plan(GraphBuilder(graph_registry).build(cfg["graph"]))
    cfg["interaction_plan"] = InteractionPlanner(interaction_registry).plan(
        cfg["interaction"],
        available_inputs={"student.logits"},
    )

    batch = {
        "labels": torch.tensor([[1.0], [0.0]]),
        "scenario": ["s1", "s1"],
        "drug_seq": torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
        "protein_seq": torch.tensor([[0.0, 1.0], [0.2, 0.8]]),
    }
    executor = PipelineExecutor(
        graph_registry=graph_registry,
        interaction_registry=interaction_registry,
        debug=True,
    )
    state, trace = executor.run_batch(
        batch,
        cfg,
        ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False),
    )

    assert not state.has("interaction.kd.loss_component")
    assert state.has("loss.total")
    assert state.has("loss.hard")
    assert state.has("metrics.f1")
    assert trace.stage_order[-1] == "postprocess"


def test_uncertainty_driven_decision_pipeline_emits_gate_outputs():
    torch = pytest.importorskip("torch")
    graph_registry, interaction_registry = _make_registries()
    cfg = _teacher_student_uncertainty_cfg()
    cfg["graph_plan"] = GraphPlanner(graph_registry).plan(GraphBuilder(graph_registry).build(cfg["graph"]))
    cfg["interaction_plan"] = InteractionPlanner(interaction_registry).plan(
        cfg["interaction"],
        available_inputs={"student.logits", "teacher.logits"},
    )

    batch = {
        "labels": torch.tensor([[1.0], [0.0]]),
        "scenario": ["s1", "s1"],
        "drug_seq": torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
        "protein_seq": torch.tensor([[0.0, 1.0], [0.2, 0.8]]),
        "drug_graph": torch.tensor([[0.3, 0.7], [0.1, 0.9]]),
    }
    executor = PipelineExecutor(
        graph_registry=graph_registry,
        interaction_registry=interaction_registry,
        debug=True,
    )
    state, trace = executor.run_batch(
        batch,
        cfg,
        ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False),
    )

    assert state.has("teacher.var")
    assert state.has("student.var")
    assert state.has("gate.alpha")
    assert state.has("gate.uncertainty_source")
    assert state.has("logits")
    assert torch.all(torch.isfinite(state.get("teacher.var")))
    assert torch.all(torch.isfinite(state.get("student.var")))
    assert torch.all((state.get("gate.alpha") >= 0.0) & (state.get("gate.alpha") <= 1.0))
    assert state.get("gate.uncertainty_source") == "confidence_proxy"
    assert state.has("metrics.f1")
    assert trace.stage_order[-1] == "postprocess"
