"""Tests for trainer/evaluator orchestration."""
from __future__ import annotations

from pathlib import Path

import pytest

from ugtsdti.config.loader import ConfigLoader
from ugtsdti.config.normalize import ConfigNormalizer
from ugtsdti.config.validate import ConfigValidator
from ugtsdti.core.context import ExecutionContext
from ugtsdti.graph.builder import GraphBuilder
from ugtsdti.graph.planner import GraphPlanner
from ugtsdti.graph.registry import NodeRegistry
from ugtsdti.graph.specs import NodePluginSpec
from ugtsdti.interaction.base import InteractionPluginSpec
from ugtsdti.interaction.diagnostics import DiagnosticsInteraction, diagnostics_output_keys
from ugtsdti.interaction.kd import KDInteraction, kd_output_keys
from ugtsdti.interaction.noop import NoOpInteraction
from ugtsdti.interaction.registry import InteractionPlanner, InteractionRegistry
from ugtsdti.interaction.uncertainty import UncertaintyInteraction, uncertainty_output_keys
from ugtsdti.nodes.base import NodeRuntime
from ugtsdti.runtime import ArtifactWriter, build_experiment_identity
from ugtsdti.trainer import Evaluator, Trainer
from ugtsdti.trainer.trainer import PipelineExecutor


class EncoderRuntime(NodeRuntime):
    def forward(self, inputs, context):
        del context
        drug = inputs.get("drug_seq", inputs.get("drug_graph"))
        protein = inputs["protein_seq"]
        import torch

        return {"embedding": torch.cat([drug, protein], dim=1)}


class LearnableHeadRuntime(NodeRuntime):
    scale = None
    bias = None

    def forward(self, inputs, context):
        del context
        embedding = next(iter(inputs.values()))
        return {"logits": embedding.sum(dim=1, keepdim=True) * self.scale + self.bias}


def _make_registries():
    graph_registry = NodeRegistry()
    graph_registry.register(
        NodePluginSpec(type_key="encoder.student", output_attrs=["embedding"]),
        EncoderRuntime,
    )
    graph_registry.register(
        NodePluginSpec(type_key="encoder.teacher", output_attrs=["embedding"]),
        EncoderRuntime,
    )
    graph_registry.register(
        NodePluginSpec(type_key="head.linear", output_attrs=["logits"]),
        LearnableHeadRuntime,
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
        InteractionPluginSpec(type_key="uncertainty.mc_dropout", output_keys_fn=uncertainty_output_keys),
        UncertaintyInteraction,
    )
    interaction_registry.register(
        InteractionPluginSpec(type_key="diagnostics.basic", output_keys_fn=diagnostics_output_keys),
        DiagnosticsInteraction,
    )
    return graph_registry, interaction_registry


def _cfg():
    return {
        "graph": {
            "nodes": [
                {"name": "student_encoder", "type_key": "encoder.student", "inputs": ["drug_seq", "protein_seq"]},
                {"name": "teacher_encoder", "type_key": "encoder.teacher", "inputs": ["drug_graph", "protein_seq"]},
                {"name": "student_head", "type_key": "head.linear", "inputs": ["student_encoder.embedding"]},
                {"name": "teacher_head", "type_key": "head.linear", "inputs": ["teacher_encoder.embedding"]},
            ]
        },
        "roles": {
            "student": {"outputs": ["student_head.logits"], "aggregation": "first"},
            "teacher": {"outputs": ["teacher_head.logits"], "aggregation": "first"},
        },
        "interaction": {
            "order": ["kd", "uncertainty", "diagnostics"],
            "dependencies": {"diagnostics": ["kd", "uncertainty"]},
            "kd": {
                "type": "kd.standard",
                "inputs": ["teacher.logits", "student.logits"],
                "params": {"mode": "logits", "temperature": 2.0, "enabled": True},
            },
            "uncertainty": {
                "type": "uncertainty.mc_dropout",
                "inputs": ["teacher.logits", "student.logits"],
                "params": {"enabled": True, "targets": {"teacher": True, "student": True}},
            },
            "diagnostics": {
                "type": "diagnostics.basic",
                "inputs": ["teacher.logits", "student.logits"],
                "params": {"enabled": True},
            },
        },
        "decision": {
            "type": "gate.uncertainty",
            "strategy": "soft",
            "use_uncertainty": True,
            "fallback": {"no_teacher": "student", "no_student": "teacher"},
        },
        "training": {
            "teacher": {"freeze": True},
            "student": {"freeze": False},
            "kd": {"schedule": "warmup", "warmup_steps": 2},
            "gate": {"trainable": True},
        },
        "loss": {
            "type": "composite",
            "hard_weight": 1.0,
            "map": {"kd": {"from": "interaction.kd.loss_component", "weight": 0.4}},
        },
        "metrics": {"enabled": ["f1", "auroc"], "by_scenario": True},
    }


def _prepare(cfg):
    graph_registry, interaction_registry = _make_registries()
    cfg["graph_plan"] = GraphPlanner(graph_registry).plan(GraphBuilder(graph_registry).build(cfg["graph"]))
    cfg["interaction_plan"] = InteractionPlanner(interaction_registry).plan(
        cfg["interaction"],
        available_inputs={"student.logits", "teacher.logits"},
    )
    return graph_registry, interaction_registry


def _batch():
    torch = pytest.importorskip("torch")
    return {
        "labels": torch.tensor([[1.0], [0.0], [1.0], [0.0]]),
        "scenario": ["s1", "s1", "s4", "s4"],
        "drug_seq": torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.2, 0.8], [0.7, 0.3]]),
        "protein_seq": torch.tensor([[0.0, 1.0], [0.2, 0.8], [0.1, 0.9], [0.9, 0.1]]),
        "drug_graph": torch.tensor([[0.3, 0.7], [0.1, 0.9], [0.8, 0.2], [0.4, 0.6]]),
    }


def test_trainer_and_evaluator_share_pipeline_order_and_scenario_metrics():
    torch = pytest.importorskip("torch")
    LearnableHeadRuntime.scale = torch.nn.Parameter(torch.tensor([[0.2]], dtype=torch.float32))
    LearnableHeadRuntime.bias = torch.nn.Parameter(torch.tensor([[0.1]], dtype=torch.float32))
    cfg = _cfg()
    graph_registry, interaction_registry = _prepare(cfg)
    executor = PipelineExecutor(graph_registry=graph_registry, interaction_registry=interaction_registry)
    trainer = Trainer(executor)
    evaluator = Evaluator(executor)
    optimizer = torch.optim.SGD([LearnableHeadRuntime.scale, LearnableHeadRuntime.bias], lr=0.1)
    batch = _batch()

    train_result = trainer.step(
        batch,
        cfg,
        ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False),
        optimizer=optimizer,
        step_idx=0,
    )
    eval_result = evaluator.evaluate(
        [batch],
        cfg,
        ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False),
    )

    assert train_result.trace.stage_order == [
        "batch",
        "graph",
        "role_binding",
        "interaction",
        "decision",
        "postprocess",
    ]
    assert eval_result.traces[0].stage_order == ["batch", "graph", "role_binding", "interaction", "decision"]
    assert train_result.state.has("logits")
    assert eval_result.state.has("logits")
    assert "metrics.s1.f1" in eval_result.metrics
    assert "metrics.s4.f1" in eval_result.metrics
    assert eval_result.highlighted_scenarios == ["s4"]


def test_trainer_applies_kd_warmup_schedule():
    torch = pytest.importorskip("torch")
    LearnableHeadRuntime.scale = torch.nn.Parameter(torch.tensor([[0.2]], dtype=torch.float32))
    LearnableHeadRuntime.bias = torch.nn.Parameter(torch.tensor([[0.1]], dtype=torch.float32))
    teacher_param = torch.nn.Parameter(torch.tensor([[1.0]], dtype=torch.float32))
    student_param = torch.nn.Parameter(torch.tensor([[1.0]], dtype=torch.float32))
    gate_param = torch.nn.Parameter(torch.tensor([[1.0]], dtype=torch.float32))
    cfg = _cfg()
    graph_registry, interaction_registry = _prepare(cfg)
    executor = PipelineExecutor(graph_registry=graph_registry, interaction_registry=interaction_registry)
    trainer = Trainer(
        executor,
        parameter_groups={"teacher": [teacher_param], "student": [student_param], "gate": [gate_param]},
    )
    optimizer = torch.optim.SGD([LearnableHeadRuntime.scale, LearnableHeadRuntime.bias], lr=0.1)

    result = trainer.step(
        _batch(),
        cfg,
        ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False),
        optimizer=optimizer,
        step_idx=0,
    )

    assert result.kd_weight == pytest.approx(0.2)
    assert result.freeze_policy == {"teacher": True, "student": False, "gate": True}
    assert teacher_param.requires_grad is False
    assert student_param.requires_grad is True
    assert gate_param.requires_grad is True


def test_trainer_writes_artifact_bundle_with_provided_model_state(tmp_path):
    cfg = _cfg()
    graph_registry, interaction_registry = _prepare(cfg)
    executor = PipelineExecutor(graph_registry=graph_registry, interaction_registry=interaction_registry)
    artifact_writer = ArtifactWriter(tmp_path / "artifacts")
    trainer = Trainer(executor, artifact_writer=artifact_writer)
    identity = build_experiment_identity({"test": "train"})

    trainer.step(
        _batch(),
        cfg,
        ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False),
        step_idx=1,
        identity=identity,
        normalized_config={"version": "1.0"},
        split_manifest={"dataset": "davis", "split_version": "v1"},
        model_state={"student": {"weights": [1.0]}},
    )

    bundle_dir = tmp_path / "artifacts" / identity.run_id
    assert (bundle_dir / "model.pt").exists()

    torch = pytest.importorskip("torch")
    payload = torch.load(bundle_dir / "model.pt", map_location="cpu", weights_only=False)
    assert payload["student"]["weights"] == [1.0]


def test_evaluator_writes_artifact_bundle_when_configured(tmp_path):
    cfg = _cfg()
    graph_registry, interaction_registry = _prepare(cfg)
    executor = PipelineExecutor(graph_registry=graph_registry, interaction_registry=interaction_registry)
    artifact_writer = ArtifactWriter(tmp_path / "artifacts")
    evaluator = Evaluator(executor, artifact_writer=artifact_writer)
    identity = build_experiment_identity({"test": "eval"})

    result = evaluator.evaluate(
        [_batch()],
        cfg,
        ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False),
        identity=identity,
        normalized_config={"version": "1.0"},
        split_manifest={"dataset": "davis", "split_version": "v1"},
        model_state={"student": "ok"},
    )

    bundle_dir = tmp_path / "artifacts" / identity.run_id
    assert result.highlighted_scenarios == ["s4"]
    assert (bundle_dir / "metrics.json").exists()
    assert (bundle_dir / "diagnostics.json").exists()
    assert (bundle_dir / "execution_trace.json").exists()


def test_sample_ablation_and_sweep_configs_smoke_validate():
    base_dir = Path(__file__).resolve().parents[2] / "configs"
    loader = ConfigLoader()
    validator = ConfigValidator()
    normalizer = ConfigNormalizer()

    for relative_path in (
        "base_full.yaml",
        "ablations/no_kd.yaml",
        "ablations/no_uncertainty.yaml",
        "ablations/student_only.yaml",
        "profiles/local.yaml",
        "profiles/dev.yaml",
        "profiles/cpu.yaml",
        "profiles/gpu.yaml",
        "profiles/kaggle.yaml",
        "sweeps/full.yaml",
    ):
        raw = loader.load(base_dir / relative_path)
        validator.validate(raw)
        normalized = normalizer.normalize(raw).to_dict()
        assert normalized["version"] == "1.0"
