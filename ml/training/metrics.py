"""Shared evaluation metrics for credit scoring models."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve,
)
from scipy.stats import ks_2samp
import logging

logger = logging.getLogger(__name__)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = None) -> dict:
    """
    Compute full credit model evaluation suite.
    If threshold is None, Youden-optimal threshold is chosen.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # AUC
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc  = float(average_precision_score(y_true, y_prob))

    # Youden-optimal threshold
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
    youden_idx = np.argmax(tpr - fpr)
    opt_threshold = float(thresholds_roc[youden_idx]) if threshold is None else threshold

    y_pred = (y_prob >= opt_threshold).astype(int)

    acc       = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall    = float(recall_score(y_true, y_pred, zero_division=0))
    f1        = float(f1_score(y_true, y_pred, zero_division=0))
    brier     = float(brier_score_loss(y_true, y_prob))

    # G-Mean
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp + 1e-10)
    sensitivity = tp / (tp + fn + 1e-10)
    g_mean = float(np.sqrt(sensitivity * specificity))

    # KS statistic
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    ks_stat, ks_pval = ks_2samp(pos_probs, neg_probs) if len(pos_probs) > 0 and len(neg_probs) > 0 else (0.0, 1.0)

    # Gini coefficient
    gini = 2 * roc_auc - 1

    return {
        "roc_auc":       roc_auc,
        "pr_auc":        pr_auc,
        "gini":          gini,
        "ks_statistic":  float(ks_stat),
        "brier_score":   brier,
        "accuracy":      acc,
        "precision":     precision,
        "recall":        recall,
        "f1":            f1,
        "g_mean":        g_mean,
        "opt_threshold": opt_threshold,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index — measures score distribution drift.
    PSI < 0.1: stable, 0.1-0.25: slight shift, > 0.25: significant shift
    """
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    exp_pct = exp_counts / len(expected)
    act_pct = act_counts / len(actual)

    # Avoid log(0)
    exp_pct = np.where(exp_pct == 0, 0.0001, exp_pct)
    act_pct = np.where(act_pct == 0, 0.0001, act_pct)

    psi_val = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return psi_val


def calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        mean_pred = y_prob[mask].mean()
        mean_true = y_true[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(mean_pred - mean_true)
    return float(ece)


def log_metrics(metrics: dict, model_name: str, fold: str = ""):
    prefix = f"[{model_name}{'/' + fold if fold else ''}]"
    logger.info(
        f"{prefix} AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
        f"KS={metrics['ks_statistic']:.4f} F1={metrics['f1']:.4f} "
        f"Brier={metrics['brier_score']:.4f} G-Mean={metrics['g_mean']:.4f}"
    )
