"""Evaluation metrics and threshold analysis for fraud anomaly detection.

Metrics focus on severe class imbalance (~0.17% fraud):
- PR-AUC (Average Precision) as the PRIMARY performance metric.
- ROC-AUC as secondary metric.
- Recall at fixed FPR budgets: 0.1%, 0.5%, 1.0%.
- Precision at review capacity: Precision@K (K = 100, 500, 1000).
- Bootstrap 95% Confidence Intervals for PR-AUC.
- Threshold selection performed strictly on the VALIDATION set and applied unchanged to the TEST set.
- Trade-off metrics: False Positives per Fraud caught (review burden).
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute Precision-Recall Area Under the Curve (Average Precision)."""
    return float(average_precision_score(y_true, scores))


def compute_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute Receiver Operating Characteristic Area Under the Curve."""
    return float(roc_auc_score(y_true, scores))


def recall_at_fpr(
    y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01
) -> Tuple[float, float, float]:
    """Calculate Recall at a fixed False Positive Rate (FPR) budget.

    Threshold is computed on the negative class:
    threshold = (1 - target_fpr) quantile of negative scores.

    Returns:
        (recall, actual_fpr, threshold)
    """
    neg_scores = scores[y_true == 0]
    pos_scores = scores[y_true == 1]

    # Quantile threshold to achieve <= target_fpr on legitimate samples
    threshold = float(np.quantile(neg_scores, 1.0 - target_fpr))

    actual_fpr = float((neg_scores >= threshold).mean())
    actual_recall = float((pos_scores >= threshold).mean())

    return actual_recall, actual_fpr, threshold


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> Tuple[float, int, float]:
    """Calculate precision within the top K highest-scoring (flagged) transactions.

    Returns:
        (precision, k_count, threshold)
    """
    k = min(k, len(scores))
    top_indices = np.argsort(scores)[::-1][:k]
    prec = float(y_true[top_indices].mean())
    threshold = float(scores[top_indices[-1]])
    return prec, k, threshold


def bootstrap_pr_auc_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Calculate bootstrap confidence interval for PR-AUC.

    Returns:
        (mean_pr_auc, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    boot_scores = []

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    for _ in range(n_bootstraps):
        # Stratified resampling to guarantee both classes appear in every resample
        sample_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sample_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        sample_idx = np.concatenate([sample_pos, sample_neg])

        y_boot = y_true[sample_idx]
        scores_boot = scores[sample_idx]

        if len(np.unique(y_boot)) > 1:
            boot_scores.append(average_precision_score(y_boot, scores_boot))

    alpha = (1.0 - ci_level) / 2.0
    ci_lower = float(np.percentile(boot_scores, 100.0 * alpha))
    ci_upper = float(np.percentile(boot_scores, 100.0 * (1.0 - alpha)))
    mean_val = float(np.mean(boot_scores))

    return mean_val, ci_lower, ci_upper


def select_thresholds_on_val(
    y_val: np.ndarray,
    val_scores: np.ndarray,
    target_fprs: Optional[List[float]] = None,
    capacity_ks: Optional[List[int]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Select operating thresholds strictly using the VALIDATION set."""
    if target_fprs is None:
        target_fprs = [0.001, 0.005, 0.010]  # 0.1%, 0.5%, 1.0%

    if capacity_ks is None:
        capacity_ks = [100, 500, 1000]

    thresholds = {}

    # 1. Target FPR thresholds
    for fpr_budget in target_fprs:
        rec, actual_fpr, thresh = recall_at_fpr(y_val, val_scores, target_fpr=fpr_budget)
        thresholds[f"fpr_{fpr_budget*100:.1f}pct"] = {
            "type": "target_fpr",
            "target": fpr_budget,
            "threshold": thresh,
            "val_recall": rec,
            "val_fpr": actual_fpr,
            "label": f"Target FPR {fpr_budget*100:.1f}%",
        }

    # 2. Maximum F1 threshold sweep
    precisions, recalls, candidate_threshs = precision_recall_curve(y_val, val_scores)
    # Exclude division by zero
    f1_scores = np.zeros_like(candidate_threshs)
    denom = precisions[:-1] + recalls[:-1]
    valid_mask = denom > 0
    f1_scores[valid_mask] = (2 * precisions[:-1][valid_mask] * recalls[:-1][valid_mask]) / denom[valid_mask]

    best_idx = np.argmax(f1_scores)
    best_thresh = float(candidate_threshs[best_idx])
    thresholds["max_f1"] = {
        "type": "max_f1",
        "target": "optimal_f1",
        "threshold": best_thresh,
        "val_f1": float(f1_scores[best_idx]),
        "val_precision": float(precisions[best_idx]),
        "val_recall": float(recalls[best_idx]),
        "label": "Max F1 on Validation",
    }

    # 3. Capacity K thresholds
    for k in capacity_ks:
        prec, actual_k, thresh = precision_at_k(y_val, val_scores, k=k)
        thresholds[f"capacity_{k}"] = {
            "type": "capacity_k",
            "target": k,
            "threshold": thresh,
            "val_precision": prec,
            "label": f"Top-{k} Review Capacity",
        }

    return thresholds


def evaluate_threshold_on_test(
    y_test: np.ndarray, test_scores: np.ndarray, threshold: float
) -> Dict[str, Any]:
    """Apply a fixed threshold (chosen on validation) to the test set and report full metrics."""
    y_pred = (test_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fp_per_tp = fp / tp if tp > 0 else float("inf")

    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "total_flagged": int(tp + fp),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "fpr": float(fpr),
        "fp_per_fraud_caught": float(fp_per_tp),
    }


def compute_comprehensive_metrics(
    model_name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    val_y: Optional[np.ndarray] = None,
    val_scores: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute all required primary, secondary, budget, and capacity metrics for a model."""
    base_rate = float((y_true == 1).mean())
    pr_auc = compute_pr_auc(y_true, scores)
    roc_auc = compute_roc_auc(y_true, scores)

    # Bootstrap 95% CI for PR-AUC
    _, ci_lower, ci_upper = bootstrap_pr_auc_ci(y_true, scores, n_bootstraps=1000, seed=seed)

    # Fixed FPR recall budgets directly on test set
    rec_01, fpr_01, _ = recall_at_fpr(y_true, scores, target_fpr=0.001)
    rec_05, fpr_05, _ = recall_at_fpr(y_true, scores, target_fpr=0.005)
    rec_10, fpr_10, _ = recall_at_fpr(y_true, scores, target_fpr=0.010)

    # Precision at review capacities
    test_size = len(y_true)
    prec_100, _, _ = precision_at_k(y_true, scores, k=100)
    prec_500, _, _ = precision_at_k(y_true, scores, k=500)
    prec_1000, _, _ = precision_at_k(y_true, scores, k=1000)

    res = {
        "model_name": model_name,
        "base_rate": base_rate,
        "pr_auc": pr_auc,
        "pr_auc_ci_lower": ci_lower,
        "pr_auc_ci_upper": ci_upper,
        "roc_auc": roc_auc,
        "recall_at_fpr_0.1pct": rec_01,
        "recall_at_fpr_0.5pct": rec_05,
        "recall_at_fpr_1.0pct": rec_10,
        "precision_at_100": prec_100,
        "precision_at_100_pct_of_test": (100 / test_size) * 100,
        "precision_at_500": prec_500,
        "precision_at_500_pct_of_test": (500 / test_size) * 100,
        "precision_at_1000": prec_1000,
        "precision_at_1000_pct_of_test": (1000 / test_size) * 100,
    }

    # If validation set provided, evaluate threshold generalization from Val to Test
    if val_y is not None and val_scores is not None:
        val_thresholds = select_thresholds_on_val(val_y, val_scores)
        applied_eval = {}
        for k, v in val_thresholds.items():
            test_eval = evaluate_threshold_on_test(y_true, scores, v["threshold"])
            applied_eval[k] = {
                "label": v["label"],
                "val_threshold": v["threshold"],
                "test_eval": test_eval,
            }
        res["val_derived_thresholds"] = applied_eval

    return res
