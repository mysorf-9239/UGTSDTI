"""Unit tests for composite loss and metrics reporting."""

from __future__ import annotations

import pytest

from ugtsdti.core.errors import InvalidDecisionOutputError, InvalidLossMappingError
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


def test_metrics_reporter_matches_reference_metrics_on_known_fixture():
    torch = pytest.importorskip("torch")
    sklearn = pytest.importorskip("sklearn.metrics")
    state = _make_state(logits=torch.tensor([[2.0], [1.0], [-1.0], [0.3], [-0.8]], dtype=torch.float32))
    labels = torch.tensor([[1.0], [1.0], [0.0], [1.0], [0.0]], dtype=torch.float32)

    outputs = MetricsReporter().report(
        {"enabled": ["f1", "auroc", "auprc"]},
        state,
        labels,
    )

    probabilities = torch.sigmoid(state.get("logits").squeeze(1)).numpy()
    targets = labels.squeeze(1).numpy()
    expected_f1 = sklearn.f1_score(targets, probabilities >= 0.5, zero_division=0.0)
    expected_auroc = sklearn.roc_auc_score(targets, probabilities)
    expected_auprc = sklearn.average_precision_score(targets, probabilities)

    assert outputs["metrics.f1"].item() == pytest.approx(expected_f1)
    assert outputs["metrics.auroc"].item() == pytest.approx(expected_auroc)
    assert outputs["metrics.auprc"].item() == pytest.approx(expected_auprc)


def test_metrics_reporter_rejects_invalid_label_shape():
    torch = pytest.importorskip("torch")
    state = _make_state(logits=torch.tensor([[0.1], [0.9]], dtype=torch.float32))

    with pytest.raises(InvalidDecisionOutputError, match="binary labels shaped as"):
        MetricsReporter().report(
            {"enabled": ["f1"]},
            state,
            torch.tensor([[[1.0]], [[0.0]]], dtype=torch.float32),
        )


def test_metrics_reporter_rejects_non_binary_labels():
    torch = pytest.importorskip("torch")
    state = _make_state(logits=torch.tensor([[0.1], [0.9]], dtype=torch.float32))

    with pytest.raises(InvalidDecisionOutputError, match="binary labels encoded as 0/1"):
        MetricsReporter().report(
            {"enabled": ["f1"]},
            state,
            torch.tensor([[2.0], [0.0]], dtype=torch.float32),
        )


def test_metrics_reporter_uses_scenario_subsets_correctly():
    torch = pytest.importorskip("torch")
    sklearn = pytest.importorskip("sklearn.metrics")
    state = _make_state(
        logits=torch.tensor([[2.0], [-1.0], [0.4], [-0.3]], dtype=torch.float32),
        scenario=["s1", "s1", "s4", "s4"],
    )
    labels = torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32)

    outputs = MetricsReporter().report(
        {"enabled": ["auprc"], "by_scenario": True},
        state,
        labels,
    )

    probabilities = torch.sigmoid(state.get("logits").squeeze(1)).numpy()
    targets = labels.squeeze(1).numpy()
    expected_s1 = sklearn.average_precision_score(targets[:2], probabilities[:2])
    expected_s4 = sklearn.average_precision_score(targets[2:], probabilities[2:])

    assert outputs["metrics.s1.auprc"].item() == pytest.approx(expected_s1)
    assert outputs["metrics.s4.auprc"].item() == pytest.approx(expected_s4)


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
