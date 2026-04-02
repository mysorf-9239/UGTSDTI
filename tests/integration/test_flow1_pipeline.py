"""Integration tests for Flow 1 Teacher-Student pipeline.

REQ-FLOW1-015 through REQ-FLOW1-020
"""

from __future__ import annotations

import pytest

from ugtsdti.config.loader import ConfigLoader
from ugtsdti.config.normalize import ConfigNormalizer
from ugtsdti.config.validate import ConfigValidator
from ugtsdti.core.context import ExecutionContext
from ugtsdti.graph.builder import GraphBuilder
from ugtsdti.graph.planner import GraphPlanner
from ugtsdti.interaction.registry import InteractionPlanner
from ugtsdti.runtime.defaults import build_default_graph_registry, build_default_interaction_registry
from ugtsdti.trainer.trainer import PipelineExecutor, _resolve_freeze_policy


def _ctx(mode: str = "train") -> ExecutionContext:
    from typing import Literal, cast

    return ExecutionContext(
        mode=cast("Literal['train', 'eval', 'infer']", mode), seed=42, device="cpu", deterministic=False
    )


def _build_flow1_cfg(graph_registry, interaction_registry):
    """Build runtime cfg dict for Flow 1 teacher-student pipeline."""
    graph_cfg = {
        "nodes": [
            # Teacher sequence branch
            {
                "name": "t_drug_enc",
                "type_key": "encoder.seq_bilstm",
                "inputs": ["drug_seq"],
                "params": {"vocab_size": 64, "embedding_dim": 16, "hidden_dim": 32, "lstm_dim": 16, "proj_dim": 32},
            },
            {
                "name": "t_prot_enc",
                "type_key": "encoder.seq_bilstm",
                "inputs": ["protein_seq"],
                "params": {"vocab_size": 32, "embedding_dim": 16, "hidden_dim": 32, "lstm_dim": 16, "proj_dim": 32},
            },
            # Teacher structure branch
            {
                "name": "t_graph_enc",
                "type_key": "encoder.gnn_drug",
                "inputs": ["drug_graph"],
                "params": {
                    "node_feat_dim": 4,
                    "hidden_dim": 32,
                    "proj_dim": 32,
                    "normalize_adj": True,
                    "allow_missing_input": True,
                },
            },
            # Teacher fusion + head
            {
                "name": "teacher_fusion",
                "type_key": "fusion.concat",
                "inputs": ["t_drug_enc.embedding", "t_prot_enc.embedding", "t_graph_enc.embedding"],
                "params": {"project_dim": 32},
            },
            {
                "name": "teacher_head",
                "type_key": "head.dense",
                "inputs": ["teacher_fusion.embedding"],
                "params": {"hidden_dim": 16, "output_dim": 1},
            },
            # Student sequence branch
            {
                "name": "s_drug_enc",
                "type_key": "encoder.seq_bilstm",
                "inputs": ["drug_seq"],
                "params": {"vocab_size": 64, "embedding_dim": 16, "hidden_dim": 32, "lstm_dim": 16, "proj_dim": 32},
            },
            {
                "name": "s_prot_enc",
                "type_key": "encoder.seq_bilstm",
                "inputs": ["protein_seq"],
                "params": {"vocab_size": 32, "embedding_dim": 16, "hidden_dim": 32, "lstm_dim": 16, "proj_dim": 32},
            },
            # Student fusion + head
            {
                "name": "student_fusion",
                "type_key": "fusion.concat",
                "inputs": ["s_drug_enc.embedding", "s_prot_enc.embedding"],
                "params": {"project_dim": 32},
            },
            {
                "name": "student_head",
                "type_key": "head.dense",
                "inputs": ["student_fusion.embedding"],
                "params": {"hidden_dim": 16, "output_dim": 1},
            },
        ]
    }
    interaction_cfg = {
        "order": ["kd", "uncertainty"],
        "dependencies": {},
        "kd": {
            "type": "kd.standard",
            "inputs": ["teacher.logits", "student.logits"],
            "params": {"enabled": True, "mode": "logits", "temperature": 4.0},
        },
        "uncertainty": {
            "type": "uncertainty.confidence_proxy",
            "inputs": ["teacher.logits", "student.logits"],
            "params": {"enabled": True, "targets": {"teacher": True, "student": True}},
        },
    }
    cfg = {
        "graph": graph_cfg,
        "roles": {
            "teacher": {"outputs": ["teacher_head.logits"], "aggregation": "first"},
            "student": {"outputs": ["student_head.logits"], "aggregation": "first"},
        },
        "interaction": interaction_cfg,
        "decision": {
            "type": "gate.uncertainty",
            "strategy": "soft",
            "use_uncertainty": True,
            "fallback": {"no_teacher": "student", "no_student": "teacher"},
        },
        "training": {
            "teacher": {"freeze": False},
            "student": {"freeze": False},
            "kd": {"schedule": "warmup", "warmup_steps": 100, "lambda_max": 0.5},
            "gate": {"trainable": False},
            "loop": {"fail_on_nonfinite_loss": False},
        },
        "loss": {
            "type": "composite",
            "hard_weight": 1.0,
            "map": {"kd": {"from": "interaction.kd.loss_component", "weight": 0.5}},
        },
        "metrics": {"enabled": ["auroc", "auprc"], "by_scenario": True},
    }
    cfg["graph_plan"] = GraphPlanner(graph_registry).plan(GraphBuilder(graph_registry).build(cfg["graph"]))
    cfg["interaction_plan"] = InteractionPlanner(interaction_registry).plan(
        cfg["interaction"],
        available_inputs={"teacher.logits", "student.logits"},
    )
    return cfg


def _s1_batch():
    torch = pytest.importorskip("torch")
    return {
        "labels": torch.tensor([[1.0], [0.0]]),
        "scenario": ["s1", "s1"],
        "drug_seq": torch.randint(0, 64, (2, 8)),
        "protein_seq": torch.randint(0, 32, (2, 12)),
        "drug_graph": {
            "adj": torch.rand(2, 5, 5).abs(),
            "node_feat": torch.rand(2, 5, 4),
        },
    }


def _s2_batch():
    """S2/S4 batch — drug_graph present but None (allow_missing_input handles it)."""
    torch = pytest.importorskip("torch")
    return {
        "labels": torch.tensor([[1.0], [0.0]]),
        "scenario": ["s2", "s2"],
        "drug_seq": torch.randint(0, 64, (2, 8)),
        "protein_seq": torch.randint(0, 32, (2, 12)),
        "drug_graph": None,  # present but None — GNN node handles via allow_missing_input
    }


# ---------------------------------------------------------------------------
# 6.1 Full pipeline S1
# ---------------------------------------------------------------------------


def test_flow1_s1_full_pipeline_state_keys():
    pytest.importorskip("torch")
    graph_registry = build_default_graph_registry()
    interaction_registry = build_default_interaction_registry()
    cfg = _build_flow1_cfg(graph_registry, interaction_registry)

    executor = PipelineExecutor(
        graph_registry=graph_registry,
        interaction_registry=interaction_registry,
    )
    state, _ = executor.run_batch(_s1_batch(), cfg, _ctx("train"))

    assert state.has("teacher.logits")
    assert state.has("student.logits")
    assert state.has("logits")
    assert state.has("gate.alpha")
    assert state.has("interaction.kd.loss_component")
    assert state.has("teacher.var")
    assert state.has("student.var")
    assert state.has("loss.total")


def test_flow1_s1_all_outputs_finite():
    torch = pytest.importorskip("torch")
    graph_registry = build_default_graph_registry()
    interaction_registry = build_default_interaction_registry()
    cfg = _build_flow1_cfg(graph_registry, interaction_registry)

    executor = PipelineExecutor(graph_registry=graph_registry, interaction_registry=interaction_registry)
    state, _ = executor.run_batch(_s1_batch(), cfg, _ctx("train"))

    for key in [
        "teacher.logits",
        "student.logits",
        "logits",
        "gate.alpha",
        "interaction.kd.loss_component",
        "teacher.var",
        "student.var",
        "loss.total",
    ]:
        val = state.get(key)
        assert torch.all(torch.isfinite(val)), f"{key} is not finite"


def test_flow1_s1_gate_alpha_in_range():
    torch = pytest.importorskip("torch")
    graph_registry = build_default_graph_registry()
    interaction_registry = build_default_interaction_registry()
    cfg = _build_flow1_cfg(graph_registry, interaction_registry)

    executor = PipelineExecutor(graph_registry=graph_registry, interaction_registry=interaction_registry)
    state, _ = executor.run_batch(_s1_batch(), cfg, _ctx("train"))

    alpha = state.get("gate.alpha")
    assert torch.all(alpha >= 0.0) and torch.all(alpha <= 1.0)


# ---------------------------------------------------------------------------
# 6.2 Pipeline S2/S4 — missing drug_graph
# ---------------------------------------------------------------------------


def test_flow1_s2_missing_drug_graph_does_not_crash():
    pytest.importorskip("torch")
    graph_registry = build_default_graph_registry()
    interaction_registry = build_default_interaction_registry()
    cfg = _build_flow1_cfg(graph_registry, interaction_registry)

    executor = PipelineExecutor(graph_registry=graph_registry, interaction_registry=interaction_registry)
    # Should not raise — t_graph_enc has allow_missing_input=True
    state, _ = executor.run_batch(_s2_batch(), cfg, _ctx("train"))
    assert state.has("logits")
    assert state.has("loss.total")


def test_flow1_s2_gnn_embedding_is_zeros_when_missing():
    torch = pytest.importorskip("torch")
    graph_registry = build_default_graph_registry()
    interaction_registry = build_default_interaction_registry()
    cfg = _build_flow1_cfg(graph_registry, interaction_registry)

    executor = PipelineExecutor(graph_registry=graph_registry, interaction_registry=interaction_registry)
    state, _ = executor.run_batch(_s2_batch(), cfg, _ctx("train"))

    graph_emb = state.get("t_graph_enc.embedding")
    assert torch.all(graph_emb == 0.0), "GNN embedding should be zeros when drug_graph is missing"


def test_flow1_student_pipeline_runs_without_drug_graph():
    """Student nodes (s_drug_enc, s_prot_enc) never need drug_graph."""
    pytest.importorskip("torch")
    graph_registry = build_default_graph_registry()
    interaction_registry = build_default_interaction_registry()
    cfg = _build_flow1_cfg(graph_registry, interaction_registry)

    executor = PipelineExecutor(graph_registry=graph_registry, interaction_registry=interaction_registry)
    state, _ = executor.run_batch(_s2_batch(), cfg, _ctx("train"))

    assert state.has("student.logits")
    assert state.has("s_drug_enc.embedding")
    assert state.has("s_prot_enc.embedding")


# ---------------------------------------------------------------------------
# 6.3 Decision fallback
# ---------------------------------------------------------------------------


def test_flow1_decision_fallback_no_teacher():
    """When teacher.logits is absent, decision falls back to student."""
    torch = pytest.importorskip("torch")
    graph_registry = build_default_graph_registry()
    interaction_registry = build_default_interaction_registry()

    # Student-only cfg (no teacher nodes)
    graph_cfg = {
        "nodes": [
            {
                "name": "s_drug_enc",
                "type_key": "encoder.seq_bilstm",
                "inputs": ["drug_seq"],
                "params": {"vocab_size": 64, "embedding_dim": 16, "hidden_dim": 32, "lstm_dim": 16, "proj_dim": 32},
            },
            {
                "name": "s_prot_enc",
                "type_key": "encoder.seq_bilstm",
                "inputs": ["protein_seq"],
                "params": {"vocab_size": 32, "embedding_dim": 16, "hidden_dim": 32, "lstm_dim": 16, "proj_dim": 32},
            },
            {
                "name": "student_fusion",
                "type_key": "fusion.concat",
                "inputs": ["s_drug_enc.embedding", "s_prot_enc.embedding"],
                "params": {"project_dim": 32},
            },
            {
                "name": "student_head",
                "type_key": "head.dense",
                "inputs": ["student_fusion.embedding"],
                "params": {"hidden_dim": 16, "output_dim": 1},
            },
        ]
    }
    from ugtsdti.interaction.base import InteractionPluginSpec
    from ugtsdti.interaction.noop import NoOpInteraction

    interaction_registry.register(
        InteractionPluginSpec(type_key="noop2", output_keys_fn=lambda p: []),
        NoOpInteraction,
    )
    cfg = {
        "graph": graph_cfg,
        "roles": {"student": {"outputs": ["student_head.logits"], "aggregation": "first"}},
        "interaction": {"order": ["noop2"], "dependencies": {}, "noop2": {"type": "noop2", "inputs": []}},
        "decision": {
            "type": "gate.uncertainty",
            "strategy": "soft",
            "use_uncertainty": True,
            "fallback": {"no_teacher": "student", "no_student": "teacher"},
        },
        "training": {
            "teacher": {"freeze": False},
            "student": {"freeze": False},
            "gate": {"trainable": False},
            "loop": {"fail_on_nonfinite_loss": False},
        },
        "loss": {"type": "hard", "hard_weight": 1.0, "map": {}},
        "metrics": {"enabled": ["auroc"], "by_scenario": False},
    }
    cfg["graph_plan"] = GraphPlanner(graph_registry).plan(GraphBuilder(graph_registry).build(cfg["graph"]))
    cfg["interaction_plan"] = InteractionPlanner(interaction_registry).plan(
        cfg["interaction"], available_inputs={"student.logits"}
    )

    batch = {
        "labels": torch.tensor([[1.0], [0.0]]),
        "scenario": ["s1", "s1"],
        "drug_seq": torch.randint(0, 64, (2, 8)),
        "protein_seq": torch.randint(0, 32, (2, 12)),
    }
    executor = PipelineExecutor(graph_registry=graph_registry, interaction_registry=interaction_registry)
    state, _ = executor.run_batch(batch, cfg, _ctx("train"))

    # Fallback: no teacher → logits == student.logits
    assert state.has("logits")
    assert torch.allclose(state.get("logits"), state.get("student.logits"))


# ---------------------------------------------------------------------------
# 6.4 Frozen teacher
# ---------------------------------------------------------------------------


def test_flow1_frozen_teacher_params_unchanged():
    torch = pytest.importorskip("torch")
    graph_registry = build_default_graph_registry()
    interaction_registry = build_default_interaction_registry()
    cfg = _build_flow1_cfg(graph_registry, interaction_registry)
    cfg["training"]["teacher"]["freeze"] = True

    executor = PipelineExecutor(graph_registry=graph_registry, interaction_registry=interaction_registry)

    # Run a forward pass first to initialize LazyLinear modules
    executor.run_batch(_s1_batch(), cfg, _ctx("train"))

    param_groups = executor.parameter_groups(cfg)

    # Freeze teacher params
    freeze_policy = _resolve_freeze_policy(cfg)
    from ugtsdti.trainer.trainer import _apply_freeze_policy

    _apply_freeze_policy(param_groups, freeze_policy)

    # Snapshot teacher params before step (after initialization)
    teacher_params_before = [p.clone().detach() for p in param_groups["teacher"]]

    # Run a training step
    import torch.optim as optim

    all_params = [p for group in param_groups.values() for p in group if p.requires_grad]
    optimizer = optim.Adam(all_params, lr=0.01) if all_params else None

    if optimizer:
        optimizer.zero_grad(set_to_none=True)
    state, _ = executor.run_batch(_s1_batch(), cfg, _ctx("train"))
    loss = state.get("loss.total")
    if loss is not None and loss.requires_grad:
        loss.backward()
        if optimizer:
            optimizer.step()

    # Teacher params must not have changed
    teacher_params_after = [p.clone().detach() for p in param_groups["teacher"]]
    for before, after in zip(teacher_params_before, teacher_params_after, strict=True):
        assert torch.allclose(before, after), "Teacher params changed despite freeze=True"


def test_flow1_student_params_update_when_not_frozen():
    torch = pytest.importorskip("torch")
    graph_registry = build_default_graph_registry()
    interaction_registry = build_default_interaction_registry()
    cfg = _build_flow1_cfg(graph_registry, interaction_registry)
    cfg["training"]["teacher"]["freeze"] = True  # freeze teacher, train student

    executor = PipelineExecutor(graph_registry=graph_registry, interaction_registry=interaction_registry)

    # Run a forward pass first to initialize LazyLinear modules
    executor.run_batch(_s1_batch(), cfg, _ctx("train"))

    param_groups = executor.parameter_groups(cfg)

    freeze_policy = _resolve_freeze_policy(cfg)
    from ugtsdti.trainer.trainer import _apply_freeze_policy

    _apply_freeze_policy(param_groups, freeze_policy)

    student_params_before = [p.clone().detach() for p in param_groups["student"]]

    import torch.optim as optim

    trainable = [p for p in param_groups["student"] if p.requires_grad]
    if not trainable:
        pytest.skip("No trainable student params found")

    optimizer = optim.Adam(trainable, lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    state, _ = executor.run_batch(_s1_batch(), cfg, _ctx("train"))
    loss = state.get("loss.total")
    if loss is not None and loss.requires_grad:
        loss.backward()
        optimizer.step()

    student_params_after = [p.clone().detach() for p in param_groups["student"]]
    changed = any(not torch.allclose(b, a) for b, a in zip(student_params_before, student_params_after, strict=True))
    assert changed, "Student params should have changed after training step"


# ---------------------------------------------------------------------------
# 6.5 KD warmup
# ---------------------------------------------------------------------------


def test_flow1_kd_warmup_step_0_near_zero():
    from ugtsdti.trainer.trainer import _scheduled_kd_weight

    cfg = {
        "training": {"kd": {"schedule": "warmup", "warmup_steps": 100, "lambda_max": 0.5}},
        "loss": {"map": {"kd": {"from": "interaction.kd.loss_component", "weight": 0.5}}},
    }
    result = _scheduled_kd_weight(cfg, step_idx=0)
    assert result is not None and result < 0.02  # near 0


def test_flow1_kd_warmup_step_at_max():
    from ugtsdti.trainer.trainer import _scheduled_kd_weight

    cfg = {
        "training": {"kd": {"schedule": "warmup", "warmup_steps": 100, "lambda_max": 0.5}},
        "loss": {"map": {"kd": {"from": "interaction.kd.loss_component", "weight": 0.5}}},
    }
    result = _scheduled_kd_weight(cfg, step_idx=99)
    assert result is not None and abs(result - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# 6.6 Config validation
# ---------------------------------------------------------------------------


def test_flow1_config_loads_and_validates():
    from pathlib import Path

    loader = ConfigLoader()
    validator = ConfigValidator()
    normalizer = ConfigNormalizer()

    cfg_path = Path("configs/flow1_teacher_student.yaml")
    if not cfg_path.exists():
        pytest.skip("configs/flow1_teacher_student.yaml not found")

    raw = loader.load(cfg_path)
    # Validate without registry (no plugin-aware check) — structural validation only
    validator.validate(raw)
    normalized = normalizer.normalize(raw)
    assert normalized.version == "v1"


def test_flow1_local_profile_resolves_extends():
    from pathlib import Path

    loader = ConfigLoader()
    ConfigValidator()

    cfg_path = Path("configs/profiles/flow1_local.yaml")
    if not cfg_path.exists():
        pytest.skip("configs/profiles/flow1_local.yaml not found")

    raw = loader.load(cfg_path)
    # After extends resolution, should have all required sections
    assert "graph" in raw
    assert "roles" in raw
    assert "interaction" in raw
    assert raw.get("runtime", {}).get("device") == "cpu"
    assert raw.get("runtime", {}).get("seed") == 42
