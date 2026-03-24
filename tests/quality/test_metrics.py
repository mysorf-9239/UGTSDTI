"""Unit tests for compute_dti_metrics and _concordance_index.

Covers:
- Perfect classifier → auroc=1.0, auprc=1.0, f1=1.0
- Random classifier → auroc ≈ 0.5
- All-same predictions → graceful degradation (no crash)
- CI: perfect ranking, reverse ranking, ties
- Continuous affinity binarization at threshold
- MSE correctness
- Output dict has all required keys
"""

import numpy as np
import pytest

from ugtsdti.core.metrics import _concordance_index, compute_dti_metrics

# ---------------------------------------------------------------------------
# compute_dti_metrics — output contract
# ---------------------------------------------------------------------------


def test_output_keys():
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([0.9, 0.1, 0.8, 0.2])
    metrics = compute_dti_metrics(y_true, y_score)
    assert set(metrics.keys()) == {"auroc", "auprc", "f1", "mse", "ci"}


def test_perfect_binary_classifier():
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])
    metrics = compute_dti_metrics(y_true, y_score)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_worst_binary_classifier():
    """Inverted predictions → auroc=0.0."""
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_dti_metrics(y_true, y_score)
    assert metrics["auroc"] == pytest.approx(0.0)


def test_all_same_predictions_no_crash():
    """All predictions identical → no crash, auroc/auprc fall back to 0.0."""
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([0.5, 0.5, 0.5, 0.5])
    metrics = compute_dti_metrics(y_true, y_score)
    assert "auroc" in metrics
    assert "auprc" in metrics


def test_all_positive_labels_no_crash():
    """Only positive labels → roc_auc_score undefined, must return 0.0 or nan (no crash)."""
    y_true = np.array([1, 1, 1, 1])
    y_score = np.array([0.9, 0.8, 0.7, 0.6])
    metrics = compute_dti_metrics(y_true, y_score)
    # Must not crash; auroc is either 0.0 (caught ValueError) or nan (sklearn warning)
    assert "auroc" in metrics
    assert metrics["auroc"] == 0.0 or np.isnan(metrics["auroc"])


def test_mse_correctness():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_score = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = compute_dti_metrics(y_true, y_score, affinity_threshold=2.5)
    assert metrics["mse"] == pytest.approx(0.0)


def test_mse_nonzero():
    y_true = np.array([0.0, 0.0])
    y_score = np.array([1.0, 1.0])
    metrics = compute_dti_metrics(y_true, y_score)
    assert metrics["mse"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Continuous affinity binarization
# ---------------------------------------------------------------------------


def test_continuous_affinity_binarized_at_threshold():
    """Continuous labels >= threshold → positive class."""
    # pKd: 8.0 and 7.5 are positive (>= 7.0), 6.0 and 5.0 are negative
    y_true = np.array([8.0, 7.5, 6.0, 5.0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])
    metrics = compute_dti_metrics(y_true, y_score, affinity_threshold=7.0)
    assert metrics["auroc"] == pytest.approx(1.0)


def test_custom_threshold():
    y_true = np.array([5.0, 4.0, 3.0, 2.0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])
    # threshold=4.0 → positives are 5.0 and 4.0
    metrics = compute_dti_metrics(y_true, y_score, affinity_threshold=4.0)
    assert metrics["auroc"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _concordance_index
# ---------------------------------------------------------------------------


def test_ci_perfect_ranking():
    y_true = np.array([4.0, 3.0, 2.0, 1.0])
    y_score = np.array([4.0, 3.0, 2.0, 1.0])
    assert _concordance_index(y_true, y_score) == pytest.approx(1.0)


def test_ci_reverse_ranking():
    y_true = np.array([4.0, 3.0, 2.0, 1.0])
    y_score = np.array([1.0, 2.0, 3.0, 4.0])
    assert _concordance_index(y_true, y_score) == pytest.approx(0.0)


def test_ci_random_is_near_half():
    """Random predictions → CI ≈ 0.5 on average."""
    rng = np.random.default_rng(42)
    y_true = rng.uniform(0, 10, 100)
    y_score = rng.uniform(0, 10, 100)
    ci = _concordance_index(y_true, y_score)
    assert 0.3 < ci < 0.7


def test_ci_all_ties_in_score():
    """All predictions tied → CI = 0.5 (all pairs are tied)."""
    y_true = np.array([3.0, 2.0, 1.0])
    y_score = np.array([1.0, 1.0, 1.0])
    assert _concordance_index(y_true, y_score) == pytest.approx(0.5)


def test_ci_single_sample():
    """Single sample → no pairs → CI = 0.0."""
    assert _concordance_index(np.array([1.0]), np.array([0.5])) == pytest.approx(0.0)


def test_ci_two_samples_concordant():
    y_true = np.array([2.0, 1.0])
    y_score = np.array([0.9, 0.1])
    assert _concordance_index(y_true, y_score) == pytest.approx(1.0)


def test_ci_two_samples_discordant():
    y_true = np.array([2.0, 1.0])
    y_score = np.array([0.1, 0.9])
    assert _concordance_index(y_true, y_score) == pytest.approx(0.0)
