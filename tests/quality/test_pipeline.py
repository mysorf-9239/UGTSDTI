"""Integration tests for the minimal baseline pipeline."""
from __future__ import annotations

import pytest

from ugtsdti.core.context import ExecutionContext
from ugtsdti.graph.builder import GraphBuilder
from ugtsdti.graph.planner import GraphPlanner
from ugtsdti.graph.registry import NodeRegistry
from ugtsdti.graph.specs import NodePluginSpec
from ugtsdti.interaction.base import InteractionPluginSpec
from ugtsdti.interaction.noop import NoOpInteraction
from ugtsdti.interaction.registry import InteractionPlanner, InteractionRegistry
from ugtsdti.nodes.base import NodeRuntime
from ugtsdti.postprocess.loss import LossComposer
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
    assert trace.graph_trace is not None
    assert trace.graph_trace.node_order == [
        "student_encoder",
        "teacher_encoder",
        "student_head",
        "teacher_head",
    ]
