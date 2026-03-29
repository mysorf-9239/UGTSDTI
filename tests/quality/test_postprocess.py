"""Unit tests for composite loss and metrics reporting."""
from __future__ import annotations

import pytest

from ugtsdti.core.errors import InvalidLossMappingError
from ugtsdti.core.state import State, StateWriter
from ugtsdti.postprocess import LossComposer, MetricsReporter


def _make_state(**entries):
    state = State()
    StateWriter(state).commit("test", entries)
    return state


def test_loss_composer_rejects_invalid_mapping():
    torch = pytest.importorskip("torch")
    state = _make_state(logits=torch.tensor([[0.0], [1.0]]))

    with pytest.raises(InvalidLossMappingError):
        LossComposer().compose(
            {"hard_weight": 1.0, "map": {"kd": {"from": "interaction.kd.loss_component", "weight": 0.3}}},
            state,
            torch.tensor([[1.0], [0.0]]),
        )


def test_loss_composer_only_includes_explicit_mapped_components():
    torch = pytest.importorskip("torch")
    state = _make_state(
        logits=torch.tensor([[0.0], [1.0]]),
        **{
            "interaction.kd.loss_component": torch.tensor(0.25),
            "interaction.disagreement": torch.tensor(0.5),
        },
    )

    outputs = LossComposer().compose(
        {"hard_weight": 0.7, "map": {"kd": {"from": "interaction.kd.loss_component", "weight": 0.3}}},
        state,
        torch.tensor([[1.0], [0.0]]),
    )

    assert "loss.kd" in outputs
    assert "loss.disagreement" not in outputs
    expected_total = 0.7 * outputs["loss.hard"] + 0.3 * state.get("interaction.kd.loss_component")
    assert torch.allclose(outputs["loss.total"], expected_total)


def test_metrics_reporter_thresholds_probabilities_not_raw_logits():
    torch = pytest.importorskip("torch")
    state = _make_state(logits=torch.tensor([[0.4], [0.2], [-0.2], [-0.4]]))

    outputs = MetricsReporter().report(
        {"enabled": ["f1"]},
        state,
        torch.tensor([[1.0], [1.0], [0.0], [0.0]]),
    )

    assert torch.isclose(outputs["metrics.f1"], torch.tensor(1.0))


def test_metrics_reporter_emits_per_scenario_metrics():
    torch = pytest.importorskip("torch")
    state = _make_state(
        logits=torch.tensor([[0.1], [0.9], [0.2], [0.8]]),
        scenario=["s1", "s1", "s2", "s2"],
    )

    outputs = MetricsReporter().report(
        {"enabled": ["auroc", "f1"], "by_scenario": True},
        state,
        torch.tensor([[0.0], [1.0], [0.0], [1.0]]),
    )

    assert "metrics.auroc" in outputs
    assert "metrics.f1" in outputs
    assert "metrics.s1.auroc" in outputs
    assert "metrics.s2.f1" in outputs


def test_metrics_reporter_emits_diagnostics_with_stable_namespace():
    torch = pytest.importorskip("torch")
    state = _make_state(
        logits=torch.tensor([[0.1], [0.9]]),
        **{
            "interaction.disagreement": torch.tensor(0.2),
            "gate.alpha": torch.tensor([[0.7], [0.3]]),
            "teacher.var": torch.tensor([[0.2], [0.4]]),
            "student.var": torch.tensor([[0.1], [0.5]]),
        },
    )

    outputs = MetricsReporter().report(
        {"enabled": ["auprc"]},
        state,
        torch.tensor([[0.0], [1.0]]),
    )

    assert "diagnostics.disagreement" in outputs
    assert "diagnostics.gate_alpha" in outputs
    assert "diagnostics.uncertainty_error" in outputs
