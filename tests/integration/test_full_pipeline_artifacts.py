"""Integration test for full-pipeline evaluation and artifact output."""
from __future__ import annotations

from dataclasses import asdict
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
from ugtsdti.interaction.registry import InteractionPlanner, InteractionRegistry
from ugtsdti.interaction.uncertainty import UncertaintyInteraction, uncertainty_output_keys
from ugtsdti.nodes.base import NodeRuntime
from ugtsdti.runtime import ArtifactWriter, build_experiment_identity
from ugtsdti.trainer import Evaluator, PipelineExecutor


class StudentEncoderRuntime(NodeRuntime):
    def forward(self, inputs, context):
        del context
        drug = inputs["drug_seq"]
        protein = inputs["protein_seq"]
        import torch

        return {"embedding": torch.cat([drug, protein], dim=1)}


class TeacherEncoderRuntime(NodeRuntime):
    def forward(self, inputs, context):
        del context
        graph = inputs["drug_graph"]
        protein = inputs["protein_seq"]
        import torch

        return {"embedding": torch.cat([graph, protein], dim=1)}


class HeadRuntime(NodeRuntime):
    def __init__(self, bias: float = 0.0) -> None:
        self._bias = bias

    def forward(self, inputs, context):
        del context
        embedding = next(iter(inputs.values()))
        import torch

        return {"logits": embedding.sum(dim=1, keepdim=True) * 0.15 + torch.tensor([[self._bias]])}


def _make_registries():
    graph_registry = NodeRegistry()
    graph_registry.register(
        NodePluginSpec(type_key="encoder.student", output_attrs=["embedding"]),
        StudentEncoderRuntime,
    )
    graph_registry.register(
        NodePluginSpec(type_key="encoder.teacher", output_attrs=["embedding"]),
        TeacherEncoderRuntime,
    )
    graph_registry.register(
        NodePluginSpec(type_key="head.student", output_attrs=["logits"]),
        HeadRuntime,
    )
    graph_registry.register(
        NodePluginSpec(type_key="head.teacher", output_attrs=["logits"]),
        HeadRuntime,
    )

    interaction_registry = InteractionRegistry()
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


def test_full_pipeline_evaluation_emits_s1_to_s4_metrics_and_artifacts(tmp_path):
    torch = pytest.importorskip("torch")
    config_path = Path(__file__).resolve().parents[2] / "configs" / "base_full.yaml"
    raw = ConfigLoader().load(config_path)
    ConfigValidator().validate(raw)
    cfg = ConfigNormalizer().normalize(raw).to_dict()
    cfg["graph"] = _graph_nodes_as_list(cfg["graph"])
    runtime_cfg = {key: value for key, value in cfg.items()}

    graph_registry, interaction_registry = _make_registries()
    runtime_cfg["graph_plan"] = GraphPlanner(graph_registry).plan(GraphBuilder(graph_registry).build(cfg["graph"]))
    runtime_cfg["interaction_plan"] = InteractionPlanner(interaction_registry).plan(
        cfg["interaction"],
        available_inputs={"student.logits", "teacher.logits"},
    )

    batch = {
        "labels": torch.tensor([[1.0], [0.0], [1.0], [0.0]]),
        "scenario": ["s1", "s2", "s3", "s4"],
        "drug_seq": torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.2, 0.8], [0.7, 0.3]]),
        "protein_seq": torch.tensor([[0.0, 1.0], [0.2, 0.8], [0.1, 0.9], [0.9, 0.1]]),
        "drug_graph": torch.tensor([[0.3, 0.7], [0.1, 0.9], [0.8, 0.2], [0.4, 0.6]]),
    }
    executor = PipelineExecutor(graph_registry=graph_registry, interaction_registry=interaction_registry)
    evaluator = Evaluator(executor)
    result = evaluator.evaluate(
        [batch],
        runtime_cfg,
        ExecutionContext(mode="eval", seed=7, device="cpu", deterministic=True),
    )

    assert "metrics.s1.f1" in result.metrics
    assert "metrics.s2.f1" in result.metrics
    assert "metrics.s3.f1" in result.metrics
    assert "metrics.s4.f1" in result.metrics
    assert result.highlighted_scenarios == ["s4"]

    diagnostics = {key: value for key, value in result.metrics.items() if key.startswith("diagnostics.")}
    metrics = {key: value for key, value in result.metrics.items() if key.startswith("metrics.")}
    identity = build_experiment_identity(cfg)
    bundle_dir = ArtifactWriter(tmp_path / "artifacts").write_bundle(
        identity=identity.to_dict(),
        config=cfg,
        metrics=_scalarize(metrics),
        diagnostics=_scalarize(diagnostics),
        split_manifest={
            "dataset": cfg["data"]["dataset"],
            "split_version": cfg["data"]["split_version"],
            "scenarios": cfg["scenario"]["eval"],
        },
        model_state={"student": "ok", "teacher": "ok"},
        execution_trace=asdict(result.traces[0]),
        state_boundary_summaries=result.traces[0].state_boundary_summaries,
    )

    assert (bundle_dir / "metrics.json").exists()
    assert (bundle_dir / "diagnostics.json").exists()
    assert (bundle_dir / "execution_trace.json").exists()
    assert (bundle_dir / "state_boundaries.json").exists()


def _scalarize(payload: dict[str, object]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in payload.items():
        result[key] = _to_float(value)
    return result


def _graph_nodes_as_list(graph_cfg: dict[str, object]) -> dict[str, object]:
    nodes = graph_cfg.get("nodes", {})
    if not isinstance(nodes, dict):
        return graph_cfg
    node_list = []
    for name, node_cfg in nodes.items():
        node = dict(node_cfg)
        node.setdefault("name", name)
        node_list.append(node)
    graph = dict(graph_cfg)
    graph["nodes"] = node_list
    return graph


def _to_float(value: object) -> float:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return float(value.detach().to(dtype=torch.float32).mean().cpu().item())
    except ImportError:
        pass
    return float(value)
