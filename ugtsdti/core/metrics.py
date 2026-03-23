import numpy as np
from sklearn.metrics import average_precision_score, f1_score, mean_squared_error, roc_auc_score


def compute_dti_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    affinity_threshold: float = 7.0,
) -> dict:
    """Compute standard DTI evaluation metrics.

    Supports both binary labels and continuous affinity values (e.g., pKd).
    Continuous labels are binarized at ``affinity_threshold`` for classification
    metrics (AUROC, AUPRC, F1), while regression metrics (MSE, CI) use raw values.

    Args:
        y_true: Ground-truth labels or affinity values. Shape ``(N,)``.
        y_score: Predicted scores or logits. Shape ``(N,)``.
        affinity_threshold: Threshold for binarizing continuous affinity values.
            Default 7.0 corresponds to pKd ≥ 7 (Kd ≤ 100 nM), the standard
            DAVIS/BindingDB convention.

    Returns:
        Dictionary with keys: ``auroc``, ``auprc``, ``f1``, ``mse``, ``ci``.
    """
    y_true = np.asarray(y_true).flatten()
    y_score = np.asarray(y_score).flatten()

    # Binarize continuous affinity values for classification metrics
    is_continuous = len(np.unique(y_true)) > 2
    y_true_binary = (y_true >= affinity_threshold).astype(int) if is_continuous else y_true.astype(int)
    y_pred_binary = (y_score >= 0.5).astype(int)  # threshold on predicted probability

    metrics: dict = {}

    try:
        metrics["auroc"] = roc_auc_score(y_true_binary, y_score)
    except ValueError:
        metrics["auroc"] = 0.0

    try:
        metrics["auprc"] = average_precision_score(y_true_binary, y_score)
    except ValueError:
        metrics["auprc"] = 0.0

    metrics["f1"] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    metrics["mse"] = mean_squared_error(y_true, y_score)
    metrics["ci"] = _concordance_index(y_true, y_score)

    return metrics


def _concordance_index(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute the Concordance Index (CI).

    CI measures the fraction of all pairs whose predicted scores are correctly
    ordered relative to their true affinity values. CI = 1.0 is perfect ranking;
    CI = 0.5 is random.

    Note:
        Naive O(N²) implementation. Acceptable for evaluation sets up to ~10k
        samples; use a vectorised approximation for larger sets.
    """
    order = np.argsort(y_true)[::-1]
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    n = len(y_true)
    concordant = 0
    discordant = 0
    tied = 0

    for i in range(n):
        for j in range(i + 1, n):
            if y_true_sorted[i] > y_true_sorted[j]:
                if y_score_sorted[i] > y_score_sorted[j]:
                    concordant += 1
                elif y_score_sorted[i] < y_score_sorted[j]:
                    discordant += 1
                else:
                    tied += 1

    total_pairs = concordant + discordant + tied
    if total_pairs == 0:
        return 0.0
    return (concordant + 0.5 * tied) / total_pairs
