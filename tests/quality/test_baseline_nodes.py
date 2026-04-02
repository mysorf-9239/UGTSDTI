"""Unit tests for shipped baseline graph runtimes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.errors import InvalidConfigError
from ugtsdti.runtime.defaults import build_default_graph_registry


def _context() -> ExecutionContext:
    return ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False)


def _definition(type_key: str, **params: int | str) -> SimpleNamespace:
    return SimpleNamespace(type_key=type_key, params=params)


def test_default_registry_includes_baseline_type_keys():
    registry = build_default_graph_registry()
    type_keys = registry.registered_type_keys()
    assert "encoder.simple_drug" in type_keys
    assert "encoder.cnn_protein" in type_keys
    assert "fusion.concat" in type_keys
    assert "head.mlp" in type_keys


def test_simple_drug_encoder_outputs_embedding_and_restores_state():
    torch = pytest.importorskip("torch")
    registry = build_default_graph_registry()
    runtime = registry.build_runtime(_definition("encoder.simple_drug", vocab_size=16, embedding_dim=8, hidden_dim=12))
    inputs = {"drug_seq": torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.long)}
    outputs = runtime.forward(inputs, _context())
    assert outputs["embedding"].shape == (2, 12)
    payload = runtime.state_dict()
    restored = registry.build_runtime(_definition("encoder.simple_drug", vocab_size=16, embedding_dim=8, hidden_dim=12))
    restored.load_state_dict(payload)
    restored_outputs = restored.forward(inputs, _context())
    assert torch.allclose(outputs["embedding"], restored_outputs["embedding"])


def test_cnn_protein_encoder_supports_max_pooling():
    torch = pytest.importorskip("torch")
    registry = build_default_graph_registry()
    runtime = registry.build_runtime(
        _definition("encoder.cnn_protein", vocab_size=24, embedding_dim=10, hidden_dim=14, kernel_size=3, pooling="max")
    )
    outputs = runtime.forward({"protein_seq": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)}, _context())
    assert outputs["embedding"].shape == (2, 14)


def test_concat_fusion_projects_and_rejects_batch_mismatch():
    torch = pytest.importorskip("torch")
    registry = build_default_graph_registry()
    runtime = registry.build_runtime(_definition("fusion.concat", project_dim=16))
    outputs = runtime.forward(
        {
            "drug_encoder.embedding": torch.randn(3, 8),
            "protein_encoder.embedding": torch.randn(3, 12),
        },
        _context(),
    )
    assert outputs["embedding"].shape == (3, 16)
    with pytest.raises(InvalidConfigError, match="batch mismatch"):
        runtime.forward(
            {
                "drug_encoder.embedding": torch.randn(2, 8),
                "protein_encoder.embedding": torch.randn(3, 12),
            },
            _context(),
        )


def test_mlp_head_outputs_raw_logits_and_round_trips_state():
    torch = pytest.importorskip("torch")
    registry = build_default_graph_registry()
    definition = _definition("head.mlp", hidden_dim=10, output_dim=1)
    runtime = registry.build_runtime(definition)
    inputs = {"fusion.embedding": torch.randn(4, 6)}
    outputs = runtime.forward(inputs, _context())
    assert outputs["logits"].shape == (4, 1)
    payload = runtime.state_dict()
    restored = registry.build_runtime(definition)
    restored.load_state_dict(payload)
    restored_outputs = restored.forward(inputs, _context())
    assert torch.allclose(outputs["logits"], restored_outputs["logits"])


# --- Bug 6: LazyLinear state_dict safety ---


def test_concat_fusion_state_dict_before_forward_is_empty():
    """state_dict() before forward pass must not crash and must not serialize uninitialized weights."""
    pytest.importorskip("torch")
    registry = build_default_graph_registry()
    runtime = registry.build_runtime(_definition("fusion.concat", project_dim=16))
    payload = runtime.state_dict()
    # projection is LazyLinear — must be skipped entirely
    assert "projection" not in payload


def test_mlp_head_state_dict_before_forward_is_empty():
    """state_dict() before forward pass must not crash and must not serialize uninitialized weights."""
    pytest.importorskip("torch")
    registry = build_default_graph_registry()
    runtime = registry.build_runtime(_definition("head.mlp", hidden_dim=10, output_dim=1))
    payload = runtime.state_dict()
    # hidden is LazyLinear — must be skipped entirely
    assert "hidden" not in payload


def test_concat_fusion_state_dict_after_forward_is_non_empty():
    """state_dict() after forward pass must contain the projection weights."""
    torch = pytest.importorskip("torch")
    registry = build_default_graph_registry()
    runtime = registry.build_runtime(_definition("fusion.concat", project_dim=16))
    runtime.forward(
        {
            "drug_encoder.embedding": torch.randn(2, 8),
            "protein_encoder.embedding": torch.randn(2, 12),
        },
        _context(),
    )
    payload = runtime.state_dict()
    assert "projection" in payload


def test_mlp_head_state_dict_after_forward_is_non_empty():
    """state_dict() after forward pass must contain the hidden layer weights."""
    torch = pytest.importorskip("torch")
    registry = build_default_graph_registry()
    runtime = registry.build_runtime(_definition("head.mlp", hidden_dim=10, output_dim=1))
    inputs = {"fusion.embedding": torch.randn(4, 6)}
    runtime.forward(inputs, _context())
    payload = runtime.state_dict()
    assert "hidden" in payload
    assert "output" in payload


def test_mlp_head_load_state_dict_round_trip_after_forward():
    """load_state_dict() round-trip after forward pass must reproduce identical outputs."""
    torch = pytest.importorskip("torch")
    registry = build_default_graph_registry()
    definition = _definition("head.mlp", hidden_dim=10, output_dim=1)
    runtime = registry.build_runtime(definition)
    inputs = {"fusion.embedding": torch.randn(4, 6)}
    outputs = runtime.forward(inputs, _context())
    payload = runtime.state_dict()

    restored = registry.build_runtime(definition)
    # materialize lazy layers first, then load weights
    restored.forward(inputs, _context())
    restored.load_state_dict(payload)
    restored_outputs = restored.forward(inputs, _context())
    assert torch.allclose(outputs["logits"], restored_outputs["logits"])
