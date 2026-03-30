"""Integration tests for the shipped baseline reference model."""

from __future__ import annotations

from pathlib import Path

import pytest

from ugtsdti.config.loader import ConfigLoader
from ugtsdti.config.normalize import ConfigNormalizer
from ugtsdti.config.validate import ConfigValidator
from ugtsdti.core.context import ExecutionContext
from ugtsdti.graph.builder import GraphBuilder
from ugtsdti.graph.planner import GraphPlanner
from ugtsdti.interaction.registry import InteractionPlanner
from ugtsdti.runtime.defaults import build_default_graph_registry, build_default_interaction_registry
from ugtsdti.trainer import PipelineExecutor, Trainer


def _baseline_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "baseline_reference.yaml"


def _load_runtime_cfg() -> tuple[dict, object, object]:
    graph_registry = build_default_graph_registry()
    interaction_registry = build_default_interaction_registry()
    loader = ConfigLoader()
    validator = ConfigValidator(graph_registry=graph_registry, interaction_registry=interaction_registry)
    normalizer = ConfigNormalizer()
    raw_cfg = loader.load(_baseline_config_path())
    validator.validate(raw_cfg)
    cfg = normalizer.normalize(raw_cfg).to_dict()
    cfg["graph_plan"] = GraphPlanner(graph_registry).plan(GraphBuilder(graph_registry).build(cfg["graph"]))
    cfg["interaction_plan"] = InteractionPlanner(interaction_registry).plan(
        cfg["interaction"],
        available_inputs={"student.logits"},
    )
    return cfg, graph_registry, interaction_registry


def _baseline_batch():
    torch = pytest.importorskip("torch")
    return {
        "labels": torch.tensor([[1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0]], dtype=torch.float32),
        "scenario": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"],
        "drug_seq": torch.tensor(
            [
                [1, 1, 1, 1],
                [7, 7, 7, 7],
                [2, 2, 2, 2],
                [8, 8, 8, 8],
                [3, 3, 3, 3],
                [9, 9, 9, 9],
                [4, 4, 4, 4],
                [10, 10, 10, 10],
            ],
            dtype=torch.long,
        ),
        "protein_seq": torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [7, 7, 7, 7, 7],
                [2, 2, 2, 2, 2],
                [8, 8, 8, 8, 8],
                [3, 3, 3, 3, 3],
                [9, 9, 9, 9, 9],
                [4, 4, 4, 4, 4],
                [10, 10, 10, 10, 10],
            ],
            dtype=torch.long,
        ),
    }


def test_baseline_reference_config_runs_forward_loss_and_metrics():
    torch = pytest.importorskip("torch")
    cfg, graph_registry, interaction_registry = _load_runtime_cfg()
    executor = PipelineExecutor(graph_registry=graph_registry, interaction_registry=interaction_registry)

    state, trace = executor.run_batch(
        _baseline_batch(),
        cfg,
        ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False),
    )

    assert state.has("drug_encoder.embedding")
    assert state.has("protein_encoder.embedding")
    assert state.has("fusion.embedding")
    assert state.has("head.logits")
    assert state.has("student.logits")
    assert state.has("logits")
    assert state.has("loss.total")
    assert state.has("metrics.auroc")
    assert state.has("metrics.auprc")
    assert state.has("metrics.s1.auroc")
    assert trace.graph_trace is not None
    assert trace.graph_trace.node_order == ["drug_encoder", "protein_encoder", "fusion", "head"]
    assert trace.stage_order == ["batch", "graph", "role_binding", "interaction", "decision", "postprocess"]

    loss_total = state.get("loss.total")
    assert torch.isfinite(loss_total)
    assert 0.0 <= float(state.get("metrics.auroc").item()) <= 1.0
    assert 0.0 <= float(state.get("metrics.auprc").item()) <= 1.0


def test_baseline_reference_trains_and_restores_model_state():
    torch = pytest.importorskip("torch")
    cfg, graph_registry, interaction_registry = _load_runtime_cfg()
    executor = PipelineExecutor(graph_registry=graph_registry, interaction_registry=interaction_registry)
    trainer = Trainer(executor, parameter_groups=executor.parameter_groups(cfg))
    groups = executor.parameter_groups(cfg)
    assert groups["student"], "Expected baseline student branch to expose trainable parameters."

    optimizer = torch.optim.Adam(groups["student"], lr=0.05)
    batch = _baseline_batch()
    context = ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False)

    losses: list[float] = []
    for step_idx in range(25):
        result = trainer.step(
            batch,
            cfg,
            context,
            optimizer=optimizer,
            step_idx=step_idx,
        )
        losses.append(float(result.loss.detach().cpu().item()))

    assert losses[-1] < losses[0]

    evaluation_state, _ = executor.run_batch(
        batch,
        cfg,
        ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False),
    )
    auroc = float(evaluation_state.get("metrics.auroc").detach().cpu().item())
    assert auroc > 0.5

    model_state = executor.model_state(cfg)
    restored_executor = PipelineExecutor(
        graph_registry=build_default_graph_registry(),
        interaction_registry=build_default_interaction_registry(),
    )
    restored_cfg, _, _ = _load_runtime_cfg()
    restored_executor.load_model_state(restored_cfg, model_state)

    original_state, _ = executor.run_until_decision(
        batch,
        cfg,
        ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False),
    )
    restored_state, _ = restored_executor.run_until_decision(
        batch,
        restored_cfg,
        ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False),
    )
    assert torch.allclose(original_state.get("logits"), restored_state.get("logits"), atol=1e-5, rtol=1e-5)
