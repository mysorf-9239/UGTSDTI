"""Unit tests for Flow 1 node runtimes and related components.

REQ-FLOW1-001 through REQ-FLOW1-006
"""

from __future__ import annotations

import pytest

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.errors import InvalidConfigError, MissingDependencyError
from ugtsdti.nodes.baseline import ConcatFusionRuntime
from ugtsdti.nodes.flow1 import (
    DenseHeadRuntime,
    GNNDrugEncoderRuntime,
    SeqBiLSTMEncoderRuntime,
)
from ugtsdti.runtime.defaults import build_default_graph_registry
from ugtsdti.trainer.trainer import _scheduled_kd_weight


def _ctx() -> ExecutionContext:
    return ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False)


# ---------------------------------------------------------------------------
# 5.1 SeqBiLSTMEncoderRuntime
# ---------------------------------------------------------------------------


class TestSeqBiLSTMEncoderRuntime:
    def test_output_shape_drug_seq(self):
        torch = pytest.importorskip("torch")
        runtime = SeqBiLSTMEncoderRuntime(vocab_size=64, proj_dim=64)
        inputs = {"drug_seq": torch.randint(0, 64, (2, 10))}
        out = runtime.forward(inputs, _ctx())
        assert out["embedding"].shape == (2, 64)

    def test_output_shape_protein_seq(self):
        torch = pytest.importorskip("torch")
        runtime = SeqBiLSTMEncoderRuntime(vocab_size=32, proj_dim=32)
        inputs = {"protein_seq": torch.randint(0, 32, (2, 15))}
        out = runtime.forward(inputs, _ctx())
        assert out["embedding"].shape == (2, 32)

    def test_output_finite(self):
        torch = pytest.importorskip("torch")
        runtime = SeqBiLSTMEncoderRuntime(vocab_size=64, proj_dim=64)
        inputs = {"drug_seq": torch.randint(0, 64, (3, 8))}
        out = runtime.forward(inputs, _ctx())
        assert torch.all(torch.isfinite(out["embedding"]))

    def test_wrong_rank_raises(self):
        torch = pytest.importorskip("torch")
        runtime = SeqBiLSTMEncoderRuntime(vocab_size=64, proj_dim=64)
        inputs = {"drug_seq": torch.randint(0, 64, (10,))}  # rank 1
        with pytest.raises(InvalidConfigError):
            runtime.forward(inputs, _ctx())

    def test_state_dict_round_trip(self):
        torch = pytest.importorskip("torch")
        runtime = SeqBiLSTMEncoderRuntime(vocab_size=64, proj_dim=64)
        inputs = {"drug_seq": torch.randint(0, 64, (2, 8))}
        out1 = runtime.forward(inputs, _ctx())
        payload = runtime.state_dict()

        restored = SeqBiLSTMEncoderRuntime(vocab_size=64, proj_dim=64)
        restored.load_state_dict(payload)
        out2 = restored.forward(inputs, _ctx())
        assert torch.allclose(out1["embedding"], out2["embedding"])

    def test_gradient_flow(self):
        torch = pytest.importorskip("torch")
        runtime = SeqBiLSTMEncoderRuntime(vocab_size=64, proj_dim=64)
        inputs = {"drug_seq": torch.randint(0, 64, (2, 8))}
        out = runtime.forward(inputs, _ctx())
        loss = out["embedding"].sum()
        loss.backward()
        # At least one parameter should have a gradient
        grads = [p.grad for p in runtime.parameters() if p.grad is not None]
        assert len(grads) > 0


# ---------------------------------------------------------------------------
# 5.2 GNNDrugEncoderRuntime
# ---------------------------------------------------------------------------


class TestGNNDrugEncoderRuntime:
    def _make_graph(self, B: int = 2, N: int = 5, F: int = 9):
        torch = pytest.importorskip("torch")
        adj = torch.rand(B, N, N).abs()
        node_feat = torch.rand(B, N, F)
        return {"adj": adj, "node_feat": node_feat}

    def test_output_shape(self):
        pytest.importorskip("torch")
        runtime = GNNDrugEncoderRuntime(node_feat_dim=9, proj_dim=64)
        inputs = {"drug_graph": self._make_graph()}
        out = runtime.forward(inputs, _ctx())
        assert out["embedding"].shape == (2, 64)

    def test_output_finite(self):
        torch = pytest.importorskip("torch")
        runtime = GNNDrugEncoderRuntime(node_feat_dim=9, proj_dim=64)
        inputs = {"drug_graph": self._make_graph()}
        out = runtime.forward(inputs, _ctx())
        assert torch.all(torch.isfinite(out["embedding"]))

    def test_adj_normalization_produces_finite(self):
        torch = pytest.importorskip("torch")
        runtime = GNNDrugEncoderRuntime(node_feat_dim=9, proj_dim=64, normalize_adj=True)
        inputs = {"drug_graph": self._make_graph(B=2, N=4, F=9)}
        out = runtime.forward(inputs, _ctx())
        assert torch.all(torch.isfinite(out["embedding"]))

    def test_missing_modality_allow_returns_zeros(self):
        torch = pytest.importorskip("torch")
        runtime = GNNDrugEncoderRuntime(node_feat_dim=9, proj_dim=64, allow_missing_input=True)
        # Pass drug_graph=None explicitly
        inputs = {"drug_graph": None}
        out = runtime.forward(inputs, _ctx())
        assert out["embedding"].shape[1] == 64
        assert torch.all(out["embedding"] == 0.0)

    def test_missing_modality_disallow_raises(self):
        pytest.importorskip("torch")
        runtime = GNNDrugEncoderRuntime(node_feat_dim=9, proj_dim=64, allow_missing_input=False)
        inputs = {"drug_graph": None}
        with pytest.raises(MissingDependencyError):
            runtime.forward(inputs, _ctx())

    def test_state_dict_round_trip(self):
        torch = pytest.importorskip("torch")
        runtime = GNNDrugEncoderRuntime(node_feat_dim=9, proj_dim=64)
        inputs = {"drug_graph": self._make_graph()}
        out1 = runtime.forward(inputs, _ctx())
        payload = runtime.state_dict()

        restored = GNNDrugEncoderRuntime(node_feat_dim=9, proj_dim=64)
        restored.load_state_dict(payload)
        out2 = restored.forward(inputs, _ctx())
        assert torch.allclose(out1["embedding"], out2["embedding"])

    def test_gradient_flow(self):
        pytest.importorskip("torch")
        runtime = GNNDrugEncoderRuntime(node_feat_dim=9, proj_dim=64)
        inputs = {"drug_graph": self._make_graph()}
        out = runtime.forward(inputs, _ctx())
        loss = out["embedding"].sum()
        loss.backward()
        grads = [p.grad for p in runtime.parameters() if p.grad is not None]
        assert len(grads) > 0


# ---------------------------------------------------------------------------
# 5.3 DenseHeadRuntime
# ---------------------------------------------------------------------------


class TestDenseHeadRuntime:
    def test_output_shape_and_no_sigmoid(self):
        torch = pytest.importorskip("torch")
        runtime = DenseHeadRuntime(hidden_dim=32, output_dim=1)
        inputs = {"fusion.embedding": torch.randn(3, 64)}
        out = runtime.forward(inputs, _ctx())
        assert out["logits"].shape == (3, 1)
        # logits can be negative — no sigmoid applied
        assert out["logits"].min().item() < 1.0  # not clamped to [0,1]

    def test_wrong_rank_raises(self):
        torch = pytest.importorskip("torch")
        runtime = DenseHeadRuntime(hidden_dim=32, output_dim=1)
        inputs = {"fusion.embedding": torch.randn(64)}  # rank 1
        with pytest.raises(InvalidConfigError):
            runtime.forward(inputs, _ctx())

    def test_state_dict_round_trip(self):
        torch = pytest.importorskip("torch")
        runtime = DenseHeadRuntime(hidden_dim=32, output_dim=1)
        inputs = {"fusion.embedding": torch.randn(2, 64)}
        out1 = runtime.forward(inputs, _ctx())
        payload = runtime.state_dict()

        restored = DenseHeadRuntime(hidden_dim=32, output_dim=1)
        restored.forward(inputs, _ctx())  # materialize LazyLinear if any
        restored.load_state_dict(payload)
        out2 = restored.forward(inputs, _ctx())
        assert torch.allclose(out1["logits"], out2["logits"])

    def test_gradient_flow(self):
        torch = pytest.importorskip("torch")
        runtime = DenseHeadRuntime(hidden_dim=32, output_dim=1)
        inputs = {"fusion.embedding": torch.randn(2, 64)}
        out = runtime.forward(inputs, _ctx())
        loss = out["logits"].sum()
        loss.backward()
        grads = [p.grad for p in runtime.parameters() if p.grad is not None]
        assert len(grads) > 0


# ---------------------------------------------------------------------------
# 5.4 ConcatFusionRuntime — N >= 2
# ---------------------------------------------------------------------------


class TestConcatFusionNInput:
    def test_three_inputs_output_shape(self):
        torch = pytest.importorskip("torch")
        runtime = ConcatFusionRuntime(project_dim=128)
        inputs = {
            "a": torch.randn(2, 64),
            "b": torch.randn(2, 64),
            "c": torch.randn(2, 64),
        }
        out = runtime.forward(inputs, _ctx())
        assert out["embedding"].shape == (2, 128)

    def test_two_inputs_backward_compatible(self):
        torch = pytest.importorskip("torch")
        runtime = ConcatFusionRuntime(project_dim=64)
        inputs = {
            "a": torch.randn(2, 32),
            "b": torch.randn(2, 32),
        }
        out = runtime.forward(inputs, _ctx())
        assert out["embedding"].shape == (2, 64)

    def test_batch_mismatch_raises(self):
        torch = pytest.importorskip("torch")
        runtime = ConcatFusionRuntime(project_dim=64)
        inputs = {
            "a": torch.randn(2, 32),
            "b": torch.randn(3, 32),  # different batch
        }
        with pytest.raises(InvalidConfigError):
            runtime.forward(inputs, _ctx())

    def test_wrong_rank_raises(self):
        torch = pytest.importorskip("torch")
        runtime = ConcatFusionRuntime(project_dim=64)
        inputs = {
            "a": torch.randn(2, 32),
            "b": torch.randn(32),  # rank 1
        }
        with pytest.raises(InvalidConfigError):
            runtime.forward(inputs, _ctx())

    def test_concat_order_preserved(self):
        """Verify that concat order matches input dict insertion order."""
        torch = pytest.importorskip("torch")
        runtime = ConcatFusionRuntime(project_dim=None)  # no projection
        a = torch.ones(1, 2) * 1.0
        b = torch.ones(1, 3) * 2.0
        c = torch.ones(1, 4) * 3.0
        inputs = {"a": a, "b": b, "c": c}
        out = runtime.forward(inputs, _ctx())
        # Expected: [1,1, 2,2,2, 3,3,3,3]
        expected = torch.cat([a, b, c], dim=1)
        assert torch.allclose(out["embedding"], expected)


# ---------------------------------------------------------------------------
# 5.5 _scheduled_kd_weight
# ---------------------------------------------------------------------------


class TestScheduledKdWeight:
    def _cfg(self, schedule: str, warmup_steps: int = 100, lambda_max: float = 0.5, weight: float = 0.5):
        return {
            "training": {
                "kd": {
                    "schedule": schedule,
                    "warmup_steps": warmup_steps,
                    "lambda_max": lambda_max,
                }
            },
            "loss": {"map": {"kd": {"from": "interaction.kd.loss_component", "weight": weight}}},
        }

    def test_warmup_step_0(self):
        cfg = self._cfg("warmup", warmup_steps=100, lambda_max=0.5)
        result = _scheduled_kd_weight(cfg, step_idx=0)
        assert result is not None
        assert abs(result - 0.005) < 1e-6  # 0.5 * 1/100

    def test_warmup_step_at_warmup_steps(self):
        cfg = self._cfg("warmup", warmup_steps=100, lambda_max=0.5)
        result = _scheduled_kd_weight(cfg, step_idx=99)
        assert result is not None
        assert abs(result - 0.5) < 1e-6

    def test_warmup_step_beyond_warmup_steps(self):
        cfg = self._cfg("warmup", warmup_steps=100, lambda_max=0.5)
        result = _scheduled_kd_weight(cfg, step_idx=200)
        assert result is not None
        assert abs(result - 0.5) < 1e-6  # capped at lambda_max

    def test_constant_reads_from_loss_map(self):
        cfg = self._cfg("constant", weight=0.3)
        result = _scheduled_kd_weight(cfg, step_idx=0)
        assert result is not None
        assert abs(result - 0.3) < 1e-6

    def test_warmup_monotonically_increasing(self):
        cfg = self._cfg("warmup", warmup_steps=10, lambda_max=1.0)
        results = [_scheduled_kd_weight(cfg, step_idx=i) for i in range(15)]
        for i in range(1, len(results)):
            assert results[i] >= results[i - 1]


# ---------------------------------------------------------------------------
# 5.6 Registry
# ---------------------------------------------------------------------------


class TestFlow1Registry:
    def test_new_type_keys_registered(self):
        registry = build_default_graph_registry()
        keys = registry.registered_type_keys()
        assert "encoder.seq_bilstm" in keys
        assert "encoder.gnn_drug" in keys
        assert "head.dense" in keys

    def test_duplicate_registration_raises(self):
        from ugtsdti.graph.registry import NodeRegistry
        from ugtsdti.graph.specs import NodePluginSpec
        from ugtsdti.nodes.base import NodeRuntime

        class _Dummy(NodeRuntime):
            def forward(self, inputs, context):
                return {}

        registry = NodeRegistry()
        spec = NodePluginSpec(type_key="test.duplicate", output_attrs=["out"])
        registry.register(spec, _Dummy)
        with pytest.raises(InvalidConfigError):
            registry.register(spec, _Dummy)
