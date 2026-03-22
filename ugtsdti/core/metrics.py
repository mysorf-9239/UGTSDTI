import numpy as np
from sklearn.metrics import average_precision_score, f1_score, mean_squared_error, roc_auc_score


def compute_dti_metrics(y_true, y_prob, threshold=0.5, returns_dict=True):
    """
    Standard evaluation metrics for DTI standard research.
    Args:
        y_true: Ground truth binary labels (N,)
        y_prob: Predicted probabilities / sigmoid outputs (N,)
        threshold: Binarization threshold for F1-score
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {}
    try:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auroc"] = 0.0

    try:
        metrics["auprc"] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics["auprc"] = 0.0

    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["mse"] = mean_squared_error(y_true, y_prob)

    # Concordance Index (CI) calculation natively
    metrics["ci"] = _compute_ci(y_true, y_prob)

    return metrics


def _compute_ci(y_true, y_prob):
    """Concordance Index implementation."""
    # Note: CI is O(N^2). This is a fast vectorized version.
    order = np.argsort(y_true)[::-1]
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]

    n = len(y_true)
    concordant = 0
    discordant = 0
    tied = 0

    for i in range(n):
        for j in range(i + 1, n):
            if y_true_sorted[i] > y_true_sorted[j]:
                if y_prob_sorted[i] > y_prob_sorted[j]:
                    concordant += 1
                elif y_prob_sorted[i] < y_prob_sorted[j]:
                    discordant += 1
                else:
                    tied += 1
            elif y_true_sorted[i] == y_true_sorted[j]:
                continue

    total_pairs = concordant + discordant + tied
    if total_pairs == 0:
        return 0.0
    return (concordant + 0.5 * tied) / total_pairs
