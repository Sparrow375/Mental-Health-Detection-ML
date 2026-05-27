"""
Report Generator — Creates comprehensive validation reports with charts.

Generates:
  1. ROC curves and AUC analysis
  2. Anomaly score distributions (depressed vs. non-depressed)
  3. Evidence accumulation timeseries
  4. Feature importance analysis
  5. Correlation scatter plots
  6. Parameter sensitivity heatmaps
  7. Per-student detail tables
  8. Summary markdown report
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

from studentlife_loader import load_phq9
from simulation_engine import SimulationConfig, SimulationEngine, StudentResult
from metrics import compute_binary_metrics, compute_correlation_metrics, find_optimal_thresholds


# ============================================================================
# Plot Styling
# ============================================================================

def setup_plot_style():
    """Apply premium dark plotting style."""
    if not HAS_PLOTTING:
        return
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#0d1117',
        'axes.facecolor': '#161b22',
        'axes.edgecolor': '#30363d',
        'text.color': '#c9d1d9',
        'axes.labelcolor': '#c9d1d9',
        'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',
        'grid.color': '#21262d',
        'grid.alpha': 0.5,
        'font.family': 'sans-serif',
        'font.size': 10,
        'figure.dpi': 150,
    })


# ============================================================================
# Individual Plots
# ============================================================================

def plot_roc_curve(
    metrics: Dict,
    output_path: str,
    title: str = "ROC Curve — MHealth System vs PHQ-9",
):
    """Plot ROC curve with AUC annotation."""
    if not HAS_PLOTTING:
        return
    setup_plot_style()

    fpr = metrics.get("_roc_fpr", [])
    tpr = metrics.get("_roc_tpr", [])
    auc_val = metrics.get("auc_roc", 0)

    if not fpr or not tpr:
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot ROC
    ax.plot(fpr, tpr, color='#58a6ff', linewidth=2.5, label=f'MHealth System (AUC = {auc_val:.3f})')
    ax.plot([0, 1], [0, 1], color='#484f58', linestyle='--', linewidth=1, label='Random Baseline')

    # Optimal point
    opt_sens = metrics.get("optimal_sensitivity", 0)
    opt_spec = metrics.get("optimal_specificity", 0)
    opt_fpr = 1 - opt_spec
    if opt_sens > 0:
        ax.scatter([opt_fpr], [opt_sens], color='#f0883e', s=150, zorder=5,
                   edgecolors='white', linewidths=2)
        ax.annotate(f'Optimal\nSens={opt_sens:.2f}\nSpec={opt_spec:.2f}',
                    xy=(opt_fpr, opt_sens), xytext=(opt_fpr + 0.15, opt_sens - 0.15),
                    fontsize=9, color='#f0883e',
                    arrowprops=dict(arrowstyle='->', color='#f0883e'))

    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='#0d1117')
    plt.close()


def plot_score_distributions(
    results: List[StudentResult],
    output_path: str,
    cutoff: int = 10,
):
    """Plot anomaly score distributions for depressed vs non-depressed students."""
    if not HAS_PLOTTING:
        return
    setup_plot_style()

    dep_scores = []
    non_dep_scores = []
    dep_evidence = []
    non_dep_evidence = []

    for r in results:
        phq = r.phq9_post if r.phq9_post is not None else r.phq9_pre
        if phq is None:
            continue
        if phq >= cutoff:
            dep_scores.append(r.mean_anomaly_score)
            dep_evidence.append(r.peak_evidence)
        else:
            non_dep_scores.append(r.mean_anomaly_score)
            non_dep_evidence.append(r.peak_evidence)

    if not dep_scores or not non_dep_scores:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Anomaly scores
    ax = axes[0]
    bins = np.linspace(0, max(max(dep_scores), max(non_dep_scores)) * 1.1, 20)
    ax.hist(non_dep_scores, bins=bins, alpha=0.7, color='#3fb950', label=f'Non-depressed (n={len(non_dep_scores)})')
    ax.hist(dep_scores, bins=bins, alpha=0.7, color='#f85149', label=f'Depressed (n={len(dep_scores)})')
    ax.axvline(x=np.mean(non_dep_scores), color='#3fb950', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axvline(x=np.mean(dep_scores), color='#f85149', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Mean Anomaly Score', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Anomaly Score Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Evidence peaks
    ax = axes[1]
    bins = np.linspace(0, max(max(dep_evidence), max(non_dep_evidence)) * 1.1, 20)
    ax.hist(non_dep_evidence, bins=bins, alpha=0.7, color='#3fb950', label=f'Non-depressed (n={len(non_dep_evidence)})')
    ax.hist(dep_evidence, bins=bins, alpha=0.7, color='#f85149', label=f'Depressed (n={len(dep_evidence)})')
    ax.axvline(x=np.mean(non_dep_evidence), color='#3fb950', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axvline(x=np.mean(dep_evidence), color='#f85149', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Peak Evidence Accumulated', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Evidence Accumulation Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Depressed vs Non-Depressed Students', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='#0d1117')
    plt.close()


def plot_correlation_scatter(
    results: List[StudentResult],
    output_path: str,
    cutoff: int = 10,
):
    """Scatter plot of PHQ-9 scores vs system outputs."""
    if not HAS_PLOTTING:
        return
    setup_plot_style()

    phq_scores = []
    anomaly_scores = []
    evidence_peaks = []
    is_depressed = []

    for r in results:
        phq = r.phq9_post if r.phq9_post is not None else r.phq9_pre
        if phq is None:
            continue
        phq_scores.append(phq)
        anomaly_scores.append(r.mean_anomaly_score)
        evidence_peaks.append(r.peak_evidence)
        is_depressed.append(phq >= cutoff)

    if len(phq_scores) < 5:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    colors = ['#f85149' if d else '#3fb950' for d in is_depressed]

    # PHQ-9 vs anomaly score
    ax = axes[0]
    ax.scatter(phq_scores, anomaly_scores, c=colors, s=80, alpha=0.8, edgecolors='white', linewidths=0.5)
    # Regression line
    z = np.polyfit(phq_scores, anomaly_scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(phq_scores), max(phq_scores), 100)
    ax.plot(x_line, p(x_line), color='#58a6ff', linewidth=2, alpha=0.7, linestyle='--')
    ax.axvline(x=cutoff, color='#f0883e', linestyle=':', linewidth=1.5, alpha=0.6, label=f'PHQ-9 cutoff ({cutoff})')
    ax.set_xlabel('PHQ-9 Score', fontsize=12)
    ax.set_ylabel('Mean Anomaly Score', fontsize=12)
    ax.set_title('PHQ-9 vs Mean Anomaly Score', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # PHQ-9 vs evidence
    ax = axes[1]
    ax.scatter(phq_scores, evidence_peaks, c=colors, s=80, alpha=0.8, edgecolors='white', linewidths=0.5)
    z = np.polyfit(phq_scores, evidence_peaks, 1)
    p = np.poly1d(z)
    ax.plot(x_line, p(x_line), color='#58a6ff', linewidth=2, alpha=0.7, linestyle='--')
    ax.axvline(x=cutoff, color='#f0883e', linestyle=':', linewidth=1.5, alpha=0.6, label=f'PHQ-9 cutoff ({cutoff})')
    ax.set_xlabel('PHQ-9 Score', fontsize=12)
    ax.set_ylabel('Peak Evidence', fontsize=12)
    ax.set_title('PHQ-9 vs Peak Evidence Accumulation', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Anomaly Score ↔ Depression Severity Relationship', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='#0d1117')
    plt.close()


def plot_student_timeseries(
    results: List[StudentResult],
    output_dir: str,
    n_examples: int = 6,
    cutoff: int = 10,
):
    """Plot anomaly score timeseries for a few example students."""
    if not HAS_PLOTTING:
        return
    setup_plot_style()

    # Select mix of depressed and non-depressed
    dep = [r for r in results if r.depressed_post]
    non_dep = [r for r in results if r.depressed_post is not None and not r.depressed_post]

    # Sort by data availability
    dep.sort(key=lambda r: len(r.dates), reverse=True)
    non_dep.sort(key=lambda r: len(r.dates), reverse=True)

    selected = dep[:n_examples // 2] + non_dep[:n_examples // 2]
    if not selected:
        selected = results[:n_examples]

    fig, axes = plt.subplots(len(selected), 1, figsize=(16, 4 * len(selected)), sharex=False)
    if len(selected) == 1:
        axes = [axes]

    for ax, r in zip(axes, selected):
        days = list(range(len(r.effective_scores)))
        scores = r.effective_scores

        # Color the plot based on depression status
        dep_status = "DEPRESSED" if r.depressed_post else "NON-DEPRESSED"
        color = '#f85149' if r.depressed_post else '#3fb950'

        ax.plot(days, scores, color=color, linewidth=1.5, alpha=0.9)
        ax.fill_between(days, 0, scores, color=color, alpha=0.15)

        # Baseline boundary
        ax.axvline(x=r.n_baseline_days, color='#f0883e', linestyle='--',
                   linewidth=1.5, alpha=0.7, label='Baseline→Monitoring')

        # Anomaly threshold
        threshold = r.config.anomaly_threshold
        ax.axhline(y=threshold, color='#d29922', linestyle=':', linewidth=1, alpha=0.5, label=f'Threshold ({threshold})')

        phq = r.phq9_post if r.phq9_post is not None else r.phq9_pre
        ax.set_title(f'{r.uid} — PHQ-9={phq} ({dep_status}) | '
                     f'Mean={r.mean_anomaly_score:.3f} | Peak Evidence={r.peak_evidence:.2f}',
                     fontsize=11, fontweight='bold')
        ax.set_ylabel('Anomaly Score', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.02, min(max(scores) * 1.3, 1.2))

    axes[-1].set_xlabel('Day Number', fontsize=11)
    plt.suptitle('Daily Anomaly Score Timeseries — Example Students',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'student_timeseries.png'),
                bbox_inches='tight', facecolor='#0d1117')
    plt.close()


def plot_parameter_sensitivity(
    trial_results: Dict,
    output_path: str,
):
    """Plot heatmap of how parameters affect sensitivity/specificity."""
    if not HAS_PLOTTING:
        return
    setup_plot_style()

    # Extract baseline_days vs anomaly_threshold grid
    bd_values = set()
    at_values = set()
    grid_data = {}

    for strat, data in trial_results.items():
        for trial in data.get("trials", []):
            config = trial.get("config", {})
            bd = config.get("baseline_days", 14)
            at = config.get("anomaly_threshold", 0.38)
            bm = trial.get("binary_metrics", {})
            sens = bm.get("sensitivity", 0)
            spec = bm.get("specificity", 0)
            comp = trial.get("composite_score", 0)

            bd_values.add(bd)
            at_values.add(at)
            grid_data[(bd, at, strat)] = {
                "sensitivity": sens,
                "specificity": spec,
                "composite": comp,
            }
        break  # Just use first strategy for the heatmap

    if not grid_data:
        return

    bd_sorted = sorted(bd_values)
    at_sorted = sorted(at_values)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    metrics_names = ["sensitivity", "specificity", "composite"]
    cmaps = ["YlOrRd", "YlGnBu", "magma"]

    for ax, metric, cmap in zip(axes, metrics_names, cmaps):
        matrix = np.zeros((len(bd_sorted), len(at_sorted)))
        for i, bd in enumerate(bd_sorted):
            for j, at in enumerate(at_sorted):
                key = (bd, at, list(trial_results.keys())[0])
                matrix[i, j] = grid_data.get(key, {}).get(metric, 0)

        sns.heatmap(matrix, ax=ax, annot=True, fmt=".2f",
                    xticklabels=[f"{at:.2f}" for at in at_sorted],
                    yticklabels=[str(bd) for bd in bd_sorted],
                    cmap=cmap, vmin=0, vmax=1)
        ax.set_xlabel('Anomaly Threshold', fontsize=11)
        ax.set_ylabel('Baseline Days', fontsize=11)
        ax.set_title(metric.capitalize(), fontsize=12, fontweight='bold')

    plt.suptitle('Parameter Sensitivity Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='#0d1117')
    plt.close()


def plot_confusion_matrix(
    metrics: Dict,
    output_path: str,
):
    """Plot confusion matrix."""
    if not HAS_PLOTTING:
        return
    setup_plot_style()

    tp = metrics.get("true_positives", 0)
    tn = metrics.get("true_negatives", 0)
    fp = metrics.get("false_positives", 0)
    fn = metrics.get("false_negatives", 0)

    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted\nNon-Depressed', 'Predicted\nDepressed'],
                yticklabels=['Actual\nNon-Depressed', 'Actual\nDepressed'],
                ax=ax, cbar_kws={'label': 'Count'},
                annot_kws={'size': 20, 'fontweight': 'bold'})
    ax.set_title('Confusion Matrix — Best Configuration', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='#0d1117')
    plt.close()


# ============================================================================
# Markdown Report Generator
# ============================================================================

def generate_markdown_report(
    best_result: Dict,
    all_results: Dict,
    output_path: str,
    plots_dir: str,
):
    """Generate comprehensive markdown validation report."""
    bm = best_result.get("binary_metrics", {})
    cm = best_result.get("correlation_metrics", {})
    ot = best_result.get("optimal_thresholds", {})
    config = best_result.get("config", {})

    report = []
    report.append("# MHealth System Validation Report")
    report.append(f"### StudentLife Dataset — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("")
    report.append("---")
    report.append("")

    # ── Executive Summary ──
    report.append("## Executive Summary")
    report.append("")
    report.append(f"The MHealth anomaly detection system was validated against the StudentLife dataset "
                  f"({best_result.get('n_students_evaluated', '?')} students, "
                  f"{best_result.get('n_students_labeled', '?')} with PHQ-9 ground truth).")
    report.append("")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| **Sensitivity (Recall)** | **{bm.get('sensitivity', '?')}** |")
    report.append(f"| **Specificity** | **{bm.get('specificity', '?')}** |")
    report.append(f"| **PPV (Precision)** | {bm.get('ppv_precision', '?')} |")
    report.append(f"| **NPV** | {bm.get('npv', '?')} |")
    report.append(f"| **F1 Score** | {bm.get('f1_score', '?')} |")
    report.append(f"| **Balanced Accuracy** | {bm.get('balanced_accuracy', '?')} |")
    report.append(f"| **AUC-ROC** | {bm.get('auc_roc', '?')} |")
    report.append(f"| **AUC-PR** | {bm.get('auc_pr', '?')} |")
    report.append(f"| **Cohen's Kappa** | {bm.get('cohens_kappa', '?')} |")
    report.append(f"| **MCC** | {bm.get('mcc', '?')} |")
    report.append(f"| **Youden's J** | {bm.get('youdens_j', '?')} |")
    report.append("")

    # ── Confusion Matrix ──
    report.append("## Confusion Matrix")
    report.append("")
    report.append(f"| | Predicted Non-Depressed | Predicted Depressed |")
    report.append(f"|---|---|---|")
    report.append(f"| **Actual Non-Depressed** | {bm.get('true_negatives', '?')} (TN) | {bm.get('false_positives', '?')} (FP) |")
    report.append(f"| **Actual Depressed** | {bm.get('false_negatives', '?')} (FN) | {bm.get('true_positives', '?')} (TP) |")
    report.append("")

    # ── Plots ──
    report.append("## Visual Analysis")
    report.append("")

    plot_files = ["roc_curve.png", "score_distributions.png", "correlation_scatter.png",
                  "confusion_matrix.png", "student_timeseries.png", "parameter_sensitivity.png"]
    plot_titles = ["ROC Curve", "Score Distributions", "PHQ-9 Correlation",
                   "Confusion Matrix", "Student Timeseries", "Parameter Sensitivity"]

    for pf, pt in zip(plot_files, plot_titles):
        pp = os.path.join(plots_dir, pf)
        if os.path.exists(pp):
            report.append(f"### {pt}")
            report.append(f"![{pt}]({pp})")
            report.append("")

    # ── Correlation Analysis ──
    report.append("## Anomaly Score ↔ Depression Correlation")
    report.append("")
    report.append("| Metric | Value | p-value |")
    report.append("|--------|-------|---------|")
    for key in ["pearson_anomaly_vs_phq9", "spearman_anomaly_vs_phq9",
                 "pearson_evidence_vs_phq9", "spearman_evidence_vs_phq9",
                 "pointbiserial_anomaly", "pointbiserial_evidence"]:
        val = cm.get(key, "N/A")
        pval = cm.get(f"{key}_pvalue", "N/A")
        name = key.replace("_vs_phq9", "").replace("_", " ").title()
        report.append(f"| {name} | {val} | {pval} |")
    report.append("")

    for key in ["cohens_d_anomaly", "cohens_d_evidence"]:
        val = cm.get(key, "N/A")
        name = key.replace("_", " ").title()
        report.append(f"- **{name}**: {val}")

    if cm.get("mannwhitney_anomaly_pvalue") is not None:
        report.append(f"- **Mann-Whitney U**: U={cm.get('mannwhitney_anomaly_u', '?')}, "
                      f"p={cm.get('mannwhitney_anomaly_pvalue', '?')}")
    report.append("")

    # ── Optimal Thresholds ──
    report.append("## Optimal Operating Points")
    report.append("")
    report.append("| Metric | Threshold | Sensitivity | Specificity | F1 | Youden's J |")
    report.append("|--------|-----------|-------------|-------------|-----|-----------|")
    for metric_name, vals in ot.items():
        report.append(f"| {metric_name} | {vals.get('threshold', '?')} | "
                      f"{vals.get('sensitivity', '?')} | {vals.get('specificity', '?')} | "
                      f"{vals.get('f1', '?')} | {vals.get('youdens_j', '?')} |")
    report.append("")

    # ── Best Configuration ──
    report.append("## Best Configuration")
    report.append("")
    report.append(f"**Config Name**: `{best_result.get('config_name', '?')}`")
    report.append(f"**Ground Truth Strategy**: `{best_result.get('gt_strategy', '?')}`")
    report.append(f"**Composite Score**: `{best_result.get('composite_score', '?')}`")
    report.append("")
    report.append("| Parameter | Value |")
    report.append("|-----------|-------|")
    for k, v in sorted(config.items()):
        report.append(f"| {k} | {v} |")
    report.append("")

    # ── Per-Student Detail Table ──
    report.append("## Per-Student Results")
    report.append("")
    report.append("| Student | PHQ-9 Pre | PHQ-9 Post | Depressed | Predicted | Mean Score | Peak Evidence | Days |")
    report.append("|---------|-----------|------------|-----------|-----------|------------|---------------|------|")
    for s in best_result.get("per_student", []):
        dep = "✓" if s.get("depressed_post") else ("✗" if s.get("depressed_post") is not None else "?")
        pred = "✓" if s.get("predicted_depressed") else "✗"
        correct = "✅" if s.get("depressed_post") == s.get("predicted_depressed") and s.get("depressed_post") is not None else (
            "❌" if s.get("depressed_post") is not None and s.get("depressed_post") != s.get("predicted_depressed") else "—"
        )
        report.append(f"| {s.get('uid', '?')} {correct} | {s.get('phq9_pre', '?')} | {s.get('phq9_post', '?')} | "
                      f"{dep} | {pred} | {s.get('mean_anomaly', '?')} | {s.get('peak_evidence', '?')} | "
                      f"{s.get('n_days', '?')} |")
    report.append("")

    # ── Trial Comparison ──
    report.append("## Trial Comparison Summary")
    report.append("")
    for strat, data in all_results.items():
        best = data.get("best", {})
        bm2 = best.get("binary_metrics", {})
        report.append(f"### Strategy: `{strat}`")
        report.append(f"- Best: `{best.get('config_name', '?')}`")
        report.append(f"- Sensitivity: {bm2.get('sensitivity', '?')}")
        report.append(f"- Specificity: {bm2.get('specificity', '?')}")
        report.append(f"- AUC-ROC: {bm2.get('auc_roc', '?')}")
        report.append(f"- F1: {bm2.get('f1_score', '?')}")
        report.append("")

    # ── Methodology ──
    report.append("## Methodology")
    report.append("")
    report.append("### Data Processing")
    report.append("- StudentLife raw sensor CSVs transformed into daily PersonalityVector-compatible features")
    report.append("- GPS → displacement, entropy, homeTimeRatio (Grid-Cell Transition Method)")
    report.append("- phonelock + dark → sleep duration, wake/sleep times (3-Signal Sleep Fusion)")
    report.append("- activity inference → step count, active minutes")
    report.append("- call_log → calls/day, duration, unique contacts, conversation frequency")
    report.append("- conversation → social ratio (face-to-face proxy)")
    report.append("- audio inference → ambient activity proxy (daylight exposure)")
    report.append("- app_usage → app launches, notification proxy")
    report.append("- phonecharge → charge duration, charge regularity")
    report.append("")
    report.append("### Simulation")
    report.append("- Data fed day-by-day through MHealth System 1 pipeline")
    report.append("- Bayesian baseline with rolling updates (hybrid mode)")
    report.append("- L1 Scorer: weighted z-scores + EWMA velocity")
    report.append("- Evidence Engine: compounding accumulation with decay")
    report.append("- Multi-signal prediction: evidence + sustained days + anomaly ratio + mean score")
    report.append("")
    report.append("### Ground Truth")
    report.append("- PHQ-9 questionnaire (pre and post study)")
    report.append("- Clinical depression cutoff: PHQ-9 ≥ 10 (moderate depression)")
    report.append("- Multiple strategies tested: post-only, pre-only, max(pre,post), any available")
    report.append("")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))


# ============================================================================
# Master Report Generator
# ============================================================================

def generate_full_report(
    best_result: Dict,
    all_results: Dict,
    sim_results: List[StudentResult],
    output_dir: str,
):
    """Generate all plots and the markdown report."""
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("\nGenerating validation report and visualizations...")

    bm = best_result.get("binary_metrics", {})

    # 1. ROC Curve
    if "_roc_fpr" in bm:
        plot_roc_curve(bm, os.path.join(plots_dir, "roc_curve.png"))
        print("  ✓ ROC curve")

    # 2. Score distributions
    plot_score_distributions(sim_results, os.path.join(plots_dir, "score_distributions.png"))
    print("  ✓ Score distributions")

    # 3. Correlation scatter
    plot_correlation_scatter(sim_results, os.path.join(plots_dir, "correlation_scatter.png"))
    print("  ✓ Correlation scatter")

    # 4. Confusion matrix
    plot_confusion_matrix(bm, os.path.join(plots_dir, "confusion_matrix.png"))
    print("  ✓ Confusion matrix")

    # 5. Student timeseries
    plot_student_timeseries(sim_results, plots_dir)
    print("  ✓ Student timeseries")

    # 6. Parameter sensitivity
    plot_parameter_sensitivity(all_results, os.path.join(plots_dir, "parameter_sensitivity.png"))
    print("  ✓ Parameter sensitivity")

    # 7. Markdown report
    report_path = os.path.join(output_dir, "validation_report.md")
    generate_markdown_report(best_result, all_results, report_path, plots_dir)
    print(f"  ✓ Markdown report: {report_path}")
