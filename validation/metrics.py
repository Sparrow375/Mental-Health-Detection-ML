"""
Evaluation Metrics — Computes all clinical validation metrics:
  - Sensitivity (True Positive Rate / Recall)
  - Specificity (True Negative Rate)
  - Positive Predictive Value (Precision)
  - Negative Predictive Value
  - F1 Score
  - Balanced Accuracy
  - AUC-ROC
  - AUC-PR
  - Cohen's Kappa
  - Matthew's Correlation Coefficient
  - Youden's J Index
  - Anomaly Score vs PHQ-9 correlation
  - Per-student anomaly score distributions
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.metrics import (
        roc_auc_score, roc_curve, precision_recall_curve,
        auc as sklearn_auc, confusion_matrix, cohen_kappa_score,
        matthews_corrcoef, f1_score as sklearn_f1, balanced_accuracy_score,
    )
    from scipy import stats as scipy_stats
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def compute_binary_metrics(
    y_true: List[bool],
    y_pred: List[bool],
    y_scores: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive binary classification metrics.

    Args:
        y_true: Ground truth labels (True = depressed)
        y_pred: Predicted labels (True = predicted depressed)
        y_scores: Continuous anomaly scores for AUC computation

    Returns:
        Dictionary of all metrics.
    """
    y_true_int = [int(y) for y in y_true]
    y_pred_int = [int(y) for y in y_pred]

    n = len(y_true)
    if n == 0:
        return {}

    # Confusion matrix components
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)

    # Basic metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # TPR / Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # TNR
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0           # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    accuracy = (tp + tn) / n
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0.0
    balanced_acc = (sensitivity + specificity) / 2.0
    youdens_j = sensitivity + specificity - 1.0

    # False positive/negative rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics = {
        "n_total": n,
        "n_positive": tp + fn,
        "n_negative": tn + fp,
        "prevalence": (tp + fn) / n,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "ppv_precision": round(ppv, 4),
        "npv": round(npv, 4),
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "f1_score": round(f1, 4),
        "youdens_j": round(youdens_j, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
    }

    # Cohen's Kappa
    if HAS_SKLEARN:
        try:
            metrics["cohens_kappa"] = round(cohen_kappa_score(y_true_int, y_pred_int), 4)
        except:
            metrics["cohens_kappa"] = 0.0

        try:
            metrics["mcc"] = round(matthews_corrcoef(y_true_int, y_pred_int), 4)
        except:
            metrics["mcc"] = 0.0

    # AUC metrics (require continuous scores)
    if y_scores is not None and HAS_SKLEARN:
        try:
            # AUC-ROC
            if len(set(y_true_int)) > 1:
                metrics["auc_roc"] = round(roc_auc_score(y_true_int, y_scores), 4)

                # ROC curve data
                fpr_curve, tpr_curve, roc_thresholds = roc_curve(y_true_int, y_scores)
                metrics["_roc_fpr"] = fpr_curve.tolist()
                metrics["_roc_tpr"] = tpr_curve.tolist()
                metrics["_roc_thresholds"] = roc_thresholds.tolist()

                # Optimal threshold (Youden's J)
                j_scores = tpr_curve - fpr_curve
                optimal_idx = np.argmax(j_scores)
                metrics["optimal_threshold"] = round(float(roc_thresholds[optimal_idx]), 4)
                metrics["optimal_sensitivity"] = round(float(tpr_curve[optimal_idx]), 4)
                metrics["optimal_specificity"] = round(float(1 - fpr_curve[optimal_idx]), 4)

                # AUC-PR
                precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
                    y_true_int, y_scores
                )
                metrics["auc_pr"] = round(sklearn_auc(recall_curve, precision_curve), 4)
                metrics["_pr_precision"] = precision_curve.tolist()
                metrics["_pr_recall"] = recall_curve.tolist()
            else:
                metrics["auc_roc"] = float('nan')
                metrics["auc_pr"] = float('nan')
        except Exception as e:
            metrics["auc_roc"] = float('nan')
            metrics["auc_pr"] = float('nan')
            metrics["_auc_error"] = str(e)

    return metrics


def compute_correlation_metrics(
    phq9_scores: List[float],
    anomaly_scores: List[float],
    evidence_peaks: List[float],
) -> Dict[str, float]:
    """
    Compute correlation between continuous system outputs and PHQ-9 scores.
    """
    metrics = {}

    if len(phq9_scores) < 5:
        return metrics

    phq9_arr = np.array(phq9_scores)
    anomaly_arr = np.array(anomaly_scores)
    evidence_arr = np.array(evidence_peaks)

    # Pearson correlations
    try:
        r_anomaly, p_anomaly = scipy_stats.pearsonr(phq9_arr, anomaly_arr)
        metrics["pearson_anomaly_vs_phq9"] = round(r_anomaly, 4)
        metrics["pearson_anomaly_vs_phq9_pvalue"] = round(p_anomaly, 6)
    except:
        pass

    try:
        r_evidence, p_evidence = scipy_stats.pearsonr(phq9_arr, evidence_arr)
        metrics["pearson_evidence_vs_phq9"] = round(r_evidence, 4)
        metrics["pearson_evidence_vs_phq9_pvalue"] = round(p_evidence, 6)
    except:
        pass

    # Spearman rank correlations
    try:
        rho_anomaly, p_rho_anomaly = scipy_stats.spearmanr(phq9_arr, anomaly_arr)
        metrics["spearman_anomaly_vs_phq9"] = round(rho_anomaly, 4)
        metrics["spearman_anomaly_vs_phq9_pvalue"] = round(p_rho_anomaly, 6)
    except:
        pass

    try:
        rho_evidence, p_rho_evidence = scipy_stats.spearmanr(phq9_arr, evidence_arr)
        metrics["spearman_evidence_vs_phq9"] = round(rho_evidence, 4)
        metrics["spearman_evidence_vs_phq9_pvalue"] = round(p_rho_evidence, 6)
    except:
        pass

    # Point-biserial (PHQ9 binarized at 10 vs continuous scores)
    phq9_binary = (phq9_arr >= 10).astype(int)
    if len(set(phq9_binary)) > 1:
        try:
            rpb_anomaly, ppb_anomaly = scipy_stats.pointbiserialr(phq9_binary, anomaly_arr)
            metrics["pointbiserial_anomaly"] = round(rpb_anomaly, 4)
            metrics["pointbiserial_anomaly_pvalue"] = round(ppb_anomaly, 6)
        except:
            pass

        try:
            rpb_evidence, ppb_evidence = scipy_stats.pointbiserialr(phq9_binary, evidence_arr)
            metrics["pointbiserial_evidence"] = round(rpb_evidence, 4)
            metrics["pointbiserial_evidence_pvalue"] = round(ppb_evidence, 6)
        except:
            pass

    # Effect sizes (Cohen's d)
    depressed_mask = phq9_arr >= 10
    non_depressed_mask = phq9_arr < 10

    if sum(depressed_mask) >= 2 and sum(non_depressed_mask) >= 2:
        dep_scores = anomaly_arr[depressed_mask]
        non_dep_scores = anomaly_arr[non_depressed_mask]

        pooled_std = math.sqrt(
            (np.var(dep_scores) * (len(dep_scores) - 1) +
             np.var(non_dep_scores) * (len(non_dep_scores) - 1)) /
            (len(dep_scores) + len(non_dep_scores) - 2)
        )

        if pooled_std > 0:
            cohens_d = (np.mean(dep_scores) - np.mean(non_dep_scores)) / pooled_std
            metrics["cohens_d_anomaly"] = round(cohens_d, 4)

        dep_evidence = evidence_arr[depressed_mask]
        non_dep_evidence = evidence_arr[non_depressed_mask]
        pooled_std_ev = math.sqrt(
            (np.var(dep_evidence) * (len(dep_evidence) - 1) +
             np.var(non_dep_evidence) * (len(non_dep_evidence) - 1)) /
            (len(dep_evidence) + len(non_dep_evidence) - 2)
        )
        if pooled_std_ev > 0:
            cohens_d_ev = (np.mean(dep_evidence) - np.mean(non_dep_evidence)) / pooled_std_ev
            metrics["cohens_d_evidence"] = round(cohens_d_ev, 4)

        # Mann-Whitney U
        try:
            u_stat, u_pval = scipy_stats.mannwhitneyu(dep_scores, non_dep_scores, alternative='greater')
            metrics["mannwhitney_anomaly_u"] = round(u_stat, 2)
            metrics["mannwhitney_anomaly_pvalue"] = round(u_pval, 6)
        except:
            pass

    return metrics


def find_optimal_thresholds(
    y_true: List[bool],
    anomaly_scores: List[float],
    evidence_peaks: List[float],
    sustained_peaks: List[int],
    anomaly_ratios: List[float],
) -> Dict[str, Dict]:
    """
    Sweep thresholds on each continuous metric to find optimal operating points.

    Returns {metric_name: {threshold, sensitivity, specificity, f1, youdens_j}}
    """
    results = {}

    # For each metric, try a range of thresholds
    metric_sweeps = {
        "mean_anomaly_score": anomaly_scores,
        "peak_evidence": evidence_peaks,
        "peak_sustained_days": [float(x) for x in sustained_peaks],
        "anomaly_day_ratio": anomaly_ratios,
    }

    for metric_name, scores in metric_sweeps.items():
        if not scores or len(scores) < 5:
            continue

        # Generate thresholds
        sorted_scores = sorted(set(scores))
        if len(sorted_scores) < 3:
            continue

        best = {"youdens_j": -999}

        for thresh in np.linspace(min(sorted_scores), max(sorted_scores), 100):
            preds = [s >= thresh for s in scores]
            tp = sum(1 for t, p in zip(y_true, preds) if t and p)
            tn = sum(1 for t, p in zip(y_true, preds) if not t and not p)
            fp = sum(1 for t, p in zip(y_true, preds) if not t and p)
            fn = sum(1 for t, p in zip(y_true, preds) if t and not p)

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) > 0 else 0.0
            j = sens + spec - 1.0

            if j > best["youdens_j"]:
                best = {
                    "threshold": round(float(thresh), 4),
                    "sensitivity": round(sens, 4),
                    "specificity": round(spec, 4),
                    "ppv": round(ppv, 4),
                    "f1": round(f1, 4),
                    "youdens_j": round(j, 4),
                }

        results[metric_name] = best

    return results
