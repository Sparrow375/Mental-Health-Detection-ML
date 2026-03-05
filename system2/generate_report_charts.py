"""
System 2 — Visualization & Demo Script
========================================
Generates all charts and runs demo scenarios for the report.

Usage:
    python generate_report_charts.py

Outputs PNG charts to system2/charts/
"""

import sys
import os
import math
import numpy as np

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from system2.config import (
    BEHAVIORAL_FEATURES,
    POPULATION_NORMS,
    DISORDER_PROTOTYPES_FRAME1,
    DISORDER_PROTOTYPES_FRAME2,
    FEATURE_WEIGHTS,
    CONFIDENCE_THRESHOLDS,
    SHAPE_DISORDER_MATRIX,
    GATE_PARAMS,
)
from system2.baseline_screener import BaselineScreener
from system2.prototype_matcher import PrototypeMatcher
from system2.temporal_validator import TemporalValidator
from system2.life_event_filter import AnomalyReport, LifeEventFilter, FilterDecision
from system2.explainability import ExplainabilityEngine
from system2.pipeline import System2Pipeline, S1Input

# ── Setup ──────────────────────────────────────────────────────────────
CHART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system2", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

# Prettify feature names
def pretty(feat):
    return feat.replace("_", " ").replace("hours", "hrs").replace("minutes", "min").title()

SHORT_FEATURES = [
    "screen_time_hours", "social_app_ratio", "texts_per_day",
    "response_time_minutes", "daily_displacement_km", "location_entropy",
    "sleep_duration_hours", "conversation_frequency"
]

ALL_DISORDERS = ["healthy", "depression", "schizophrenia", "bpd",
                 "bipolar_depressive", "bipolar_manic", "anxiety"]

DISORDER_COLORS = {
    "healthy": "#2ecc71",
    "depression": "#3498db",
    "schizophrenia": "#9b59b6",
    "bpd": "#e67e22",
    "bipolar_depressive": "#f1c40f",
    "bipolar_manic": "#e74c3c",
    "anxiety": "#e74c3c",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 1: Population Norms Bar Chart
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def chart_population_norms():
    feats = BEHAVIORAL_FEATURES
    means = [POPULATION_NORMS[f]["mean"] for f in feats]
    stds  = [POPULATION_NORMS[f]["std"]  for f in feats]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(feats))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color="#3498db",
                  alpha=0.85, edgecolor="#2c3e50", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty(f) for f in feats], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Value (mean ± 1 SD)", fontsize=10)
    ax.set_title("Population Norms — Healthy Baseline Reference", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "01_population_norms.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 01_population_norms.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 2: Frame 1 Prototype Comparison (grouped bar)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def chart_frame1_prototypes():
    feats = SHORT_FEATURES
    disorders = ALL_DISORDERS
    n_d = len(disorders)
    n_f = len(feats)

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(n_f)
    width = 0.8 / n_d

    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#f1c40f", "#e74c3c", "#1abc9c"]
    for i, d in enumerate(disorders):
        vals = [DISORDER_PROTOTYPES_FRAME1[d].get(f, 0) for f in feats]
        ax.bar(x + i * width - 0.4 + width/2, vals, width,
               label=d.replace("_", " ").title(), color=colors[i], alpha=0.85,
               edgecolor="#2c3e50", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([pretty(f) for f in feats], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Absolute Value", fontsize=10)
    ax.set_title("Frame 1 — Disorder Prototypes (Absolute Values)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "02_frame1_prototypes.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 02_frame1_prototypes.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 3: Frame 2 Heatmap (Z-Scores)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def chart_frame2_heatmap():
    feats = BEHAVIORAL_FEATURES
    disorders = ALL_DISORDERS

    data = np.array([
        [DISORDER_PROTOTYPES_FRAME2[d].get(f, 0) for f in feats]
        for d in disorders
    ])

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(data, cmap="RdBu_r", aspect="auto", vmin=-3, vmax=3)

    ax.set_xticks(np.arange(len(feats)))
    ax.set_xticklabels([pretty(f) for f in feats], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(disorders)))
    ax.set_yticklabels([d.replace("_", " ").title() for d in disorders], fontsize=9)

    # Annotate cells
    for i in range(len(disorders)):
        for j in range(len(feats)):
            val = data[i, j]
            color = "white" if abs(val) > 1.5 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=6, color=color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, label="Z-Score from Personal Baseline")
    ax.set_title("Frame 2 — Disorder Prototype Z-Scores (Personal Baseline Anchored)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "03_frame2_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 03_frame2_heatmap.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 4: Feature Diagnostic Weights
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def chart_feature_weights():
    feats = sorted(FEATURE_WEIGHTS.keys(), key=lambda f: FEATURE_WEIGHTS[f], reverse=True)
    weights = [FEATURE_WEIGHTS[f] for f in feats]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#e74c3c" if w >= 0.8 else "#f39c12" if w >= 0.6 else "#3498db" for w in weights]
    bars = ax.barh(range(len(feats)), weights, color=colors, alpha=0.85, edgecolor="#2c3e50", linewidth=0.5)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([pretty(f) for f in feats], fontsize=9)
    ax.set_xlabel("Diagnostic Weight", fontsize=10)
    ax.set_title("Feature Diagnostic Reliability Weights", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.8, color="#e74c3c", linestyle="--", alpha=0.5, label="High (≥0.8)")
    ax.axvline(x=0.6, color="#f39c12", linestyle="--", alpha=0.5, label="Medium (≥0.6)")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "04_feature_weights.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 04_feature_weights.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 5: Radar Chart — Depression vs Healthy vs User
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def chart_radar_depression():
    feats = SHORT_FEATURES
    n = len(feats)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    healthy = DISORDER_PROTOTYPES_FRAME2["healthy"]
    dep     = DISORDER_PROTOTYPES_FRAME2["depression"]
    # Simulated depressed user
    user = {
        "screen_time_hours": 0.8, "social_app_ratio": -1.5,
        "texts_per_day": -1.2, "response_time_minutes": 1.5,
        "daily_displacement_km": -1.8, "location_entropy": -1.5,
        "sleep_duration_hours": 1.2, "conversation_frequency": -1.3,
    }

    def vals(profile):
        return [profile.get(f, 0) for f in feats] + [profile.get(feats[0], 0)]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    labels = [pretty(f) for f in feats]
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)

    # Healthy
    ax.fill(angles, vals(healthy), alpha=0.1, color="green")
    ax.plot(angles, vals(healthy), color="green", linewidth=1.5, linestyle="--", label="Healthy (0 SD)")

    # Depression prototype
    ax.plot(angles, vals(dep), color="blue", linewidth=2, linestyle="--", label="Depression Prototype")
    ax.fill(angles, vals(dep), alpha=0.05, color="blue")

    # User
    ax.plot(angles, vals(user), color="red", linewidth=2.5, label="User Profile")
    ax.fill(angles, vals(user), alpha=0.15, color="red")

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=9)
    ax.set_title("Radar: Depressed User vs Prototypes", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "05_radar_depression.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 05_radar_depression.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 6: Radar Chart — All Disorders Overlaid
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def chart_radar_all_disorders():
    feats = SHORT_FEATURES
    n = len(feats)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#f1c40f", "#e74c3c", "#1abc9c"]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    labels = [pretty(f) for f in feats]
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)

    for i, d in enumerate(ALL_DISORDERS):
        proto = DISORDER_PROTOTYPES_FRAME2[d]
        vals = [proto.get(f, 0) for f in feats] + [proto.get(feats[0], 0)]
        ax.plot(angles, vals, color=colors[i], linewidth=2, label=d.replace("_", " ").title())
        ax.fill(angles, vals, alpha=0.05, color=colors[i])

    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.12), fontsize=8)
    ax.set_title("All Disorder Prototypes — Frame 2 (Z-Scores)", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "06_radar_all_disorders.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 06_radar_all_disorders.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 7: Temporal Shape Examples
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def chart_temporal_shapes():
    rng = np.random.RandomState(42)
    x = np.arange(60)

    shapes = {
        "Monotonic Drift (Depression)":    np.linspace(0.5, -2.5, 60) + rng.normal(0, 0.15, 60),
        "Oscillating (BPD)":               1.5 * np.sin(2 * math.pi * x / 7) + rng.normal(0, 0.2, 60),
        "Chaotic (Schizophrenia)":         rng.normal(0, 2.0, 60),
        "Episodic Spike (Anxiety)":        np.concatenate([
            rng.normal(0, 0.3, 20),
            np.linspace(0, 3.0, 5),
            np.linspace(3.0, 0.2, 10),
            rng.normal(0, 0.3, 25),
        ]),
        "Phase Flip (Bipolar)":            np.concatenate([
            np.linspace(-1.5, -2.0, 30) + rng.normal(0, 0.2, 30),
            np.linspace(2.0, 2.5, 30) + rng.normal(0, 0.2, 30),
        ]),
    }

    colors = ["#3498db", "#e67e22", "#9b59b6", "#e74c3c", "#f1c40f"]
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    for ax, (title, ts), color in zip(axes, shapes.items(), colors):
        ax.plot(x, ts, color=color, linewidth=1.5, alpha=0.9)
        ax.fill_between(x, ts, alpha=0.15, color=color)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Anomaly\nScore", fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Days", fontsize=10)
    fig.suptitle("Temporal Shape Patterns — Anomaly Score Trajectories",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "07_temporal_shapes.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 07_temporal_shapes.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 8: Shape–Disorder Compatibility Heatmap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def chart_shape_disorder_matrix():
    shapes = list(SHAPE_DISORDER_MATRIX.keys())
    disorders = ALL_DISORDERS
    data = np.array([
        [SHAPE_DISORDER_MATRIX[s].get(d, 0) for d in disorders]
        for s in shapes
    ])

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=-1.5, vmax=1.5)

    ax.set_xticks(np.arange(len(disorders)))
    ax.set_xticklabels([d.replace("_", " ").title() for d in disorders], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(shapes)))
    ax.set_yticklabels([s.replace("_", " ").title() for s in shapes], fontsize=9)

    labels_map = {1: "✓ Supports", 0: "— Neutral", -1: "✗ Contradicts"}
    for i in range(len(shapes)):
        for j in range(len(disorders)):
            val = int(data[i, j])
            color = "white" if abs(val) == 1 else "black"
            ax.text(j, i, labels_map[val], ha="center", va="center", fontsize=7,
                    color=color, fontweight="bold")

    ax.set_title("Temporal Shape — Disorder Compatibility Matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "08_shape_disorder_matrix.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 08_shape_disorder_matrix.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 9: Gate Screening Decision Flowchart (simulated as bar chart)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def chart_gate_thresholds():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Gate 1
    ax = axes[0]
    ax.set_title("Gate 1: Population Anchor", fontsize=11, fontweight="bold")
    feats_example = BEHAVIORAL_FEATURES[:8]
    z_scores = [0.5, 1.2, 3.1, 0.8, 2.8, 3.5, 1.0, 0.3]
    colors = ["#e74c3c" if z > GATE_PARAMS["gate1_z_threshold"] else "#2ecc71" for z in z_scores]
    ax.barh(range(len(feats_example)), z_scores, color=colors, alpha=0.85)
    ax.axvline(GATE_PARAMS["gate1_z_threshold"], color="red", linestyle="--", label=f"Threshold: {GATE_PARAMS['gate1_z_threshold']} SD")
    ax.set_yticks(range(len(feats_example)))
    ax.set_yticklabels([pretty(f) for f in feats_example], fontsize=7)
    ax.set_xlabel("|Z-Score|")
    ax.legend(fontsize=7)
    ax.invert_yaxis()

    # Gate 2
    ax = axes[1]
    ax.set_title("Gate 2: Stability Check", fontsize=11, fontweight="bold")
    weeks = ["Week 1", "Week 2", "Week 3"]
    stable = [4.0, 4.1, 4.2]
    unstable = [4.0, 6.5, 3.0]
    ax.plot(weeks, stable, "o-", color="#2ecc71", linewidth=2, label="Stable Feature")
    ax.plot(weeks, unstable, "o-", color="#e74c3c", linewidth=2, label="Drifting Feature")
    ax.fill_between(weeks, [3.4]*3, [4.6]*3, alpha=0.1, color="green")
    ax.set_ylabel("Feature Value")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Gate 3
    ax = axes[2]
    ax.set_title("Gate 3: Prototype Proximity", fontsize=11, fontweight="bold")
    match_scores = {"Healthy": 0.45, "Depression": 0.78, "Schizophrenia": 0.30,
                    "BPD": 0.22, "Anxiety": 0.35}
    disorders_g3 = list(match_scores.keys())
    scores = list(match_scores.values())
    colors = ["#e74c3c" if s > GATE_PARAMS["gate3_healthy_threshold"] else "#2ecc71" for s in scores]
    ax.bar(disorders_g3, scores, color=colors, alpha=0.85, edgecolor="#2c3e50", linewidth=0.5)
    ax.axhline(GATE_PARAMS["gate3_healthy_threshold"], color="red", linestyle="--",
               label=f"Threshold: {GATE_PARAMS['gate3_healthy_threshold']}")
    ax.set_ylabel("Match Score")
    ax.legend(fontsize=8)
    ax.set_xticklabels(disorders_g3, rotation=20, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("3-Gate Baseline Screening — Example Scenarios", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "09_gate_screening.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 09_gate_screening.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 10: Pipeline Demo — Full Classification Output
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def chart_pipeline_demo():
    pipeline = System2Pipeline()

    # Simulated depressed user
    dep_proto = DISORDER_PROTOTYPES_FRAME2["depression"]
    report = AnomalyReport(
        feature_deviations=dep_proto.copy(),
        days_sustained=30,
        co_deviating_count=10,
        resolved=False,
        days_since_onset=30,
    )
    ts = list(np.linspace(0.5, -2.0, 60))
    baseline = {f: POPULATION_NORMS[f]["mean"] for f in BEHAVIORAL_FEATURES}
    s1_input = S1Input(
        baseline_data={
            "raw_7day": baseline.copy(),
            "weekly_windows": [baseline.copy()] * 3,
            "raw_28day": baseline.copy(),
        },
        anomaly_report=report,
        anomaly_timeseries=ts,
    )
    output = pipeline.classify(s1_input)

    # Plot match scores
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: all disorder scores
    ax = axes[0]
    if output.classification:
        scores = output.classification.all_scores
        disorders = sorted(scores, key=lambda d: scores[d], reverse=True)
        vals = [scores[d] for d in disorders]
        colors = ["#e74c3c" if d == output.disorder else "#95a5a6" for d in disorders]
        ax.barh(range(len(disorders)), vals, color=colors, alpha=0.85, edgecolor="#2c3e50")
        ax.set_yticks(range(len(disorders)))
        ax.set_yticklabels([d.replace("_", " ").title() for d in disorders], fontsize=9)
        ax.axvline(CONFIDENCE_THRESHOLDS["high"], color="green", linestyle="--", label="High Conf (0.75)")
        ax.axvline(CONFIDENCE_THRESHOLDS["low"], color="orange", linestyle="--", label="Low Conf (0.55)")
        ax.set_xlabel("Match Score")
        ax.set_title("Disorder Match Scores", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

    # Right: temporal adjustment
    ax = axes[1]
    if output.temporal_result:
        tr = output.temporal_result
        labels = ["Original Score", "Adjusted Score"]
        vals = [tr.original_score, tr.adjusted_score]
        colors = ["#3498db", "#2ecc71" if tr.shape_supports else "#e74c3c"]
        ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="#2c3e50", width=0.5)
        ax.axhline(CONFIDENCE_THRESHOLDS["high"], color="green", linestyle="--", alpha=0.5)
        ax.axhline(CONFIDENCE_THRESHOLDS["low"], color="orange", linestyle="--", alpha=0.5)
        ax.set_ylabel("Score")
        ax.set_title(f"Temporal Adjustment ({tr.temporal_shape})",
                     fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.2)
        ax.grid(axis="y", alpha=0.3)
        # Annotate
        factor = "×1.2 (boost)" if tr.shape_supports else "×0.6 (downgrade)"
        ax.annotate(f"Shape: {tr.temporal_shape}\n{factor}",
                    xy=(1, tr.adjusted_score), xytext=(1.3, tr.adjusted_score + 0.1),
                    fontsize=9, arrowprops=dict(arrowstyle="->", color="grey"))

    fig.suptitle(f"Pipeline Demo — Classification: {output.label}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "10_pipeline_demo.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 10_pipeline_demo.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 11: Confidence Thresholds Visualization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def chart_confidence_zones():
    fig, ax = plt.subplots(figsize=(10, 3))

    # Three zones
    ax.axhspan(0, 0.55, color="#e74c3c", alpha=0.2)
    ax.axhspan(0.55, 0.75, color="#f1c40f", alpha=0.2)
    ax.axhspan(0.75, 1.0, color="#2ecc71", alpha=0.2)

    # Labels
    ax.text(0.5, 0.28, "UNCLASSIFIED\n\"Escalate for review\"", ha="center", fontsize=11, fontweight="bold", color="#c0392b")
    ax.text(0.5, 0.65, "LOW CONFIDENCE\n\"Possible [Disorder] — monitor\"", ha="center", fontsize=11, fontweight="bold", color="#d68910")
    ax.text(0.5, 0.87, "HIGH CONFIDENCE\n\"Consistent with [Disorder]\"", ha="center", fontsize=11, fontweight="bold", color="#27ae60")

    # Thresholds
    ax.axhline(0.55, color="orange", linewidth=2, linestyle="--")
    ax.axhline(0.75, color="green", linewidth=2, linestyle="--")
    ax.text(1.02, 0.55, "0.55", fontsize=9, va="center", color="orange", fontweight="bold")
    ax.text(1.02, 0.75, "0.75", fontsize=9, va="center", color="green", fontweight="bold")

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Match Score", fontsize=10)
    ax.set_title("Confidence Threshold Zones", fontsize=13, fontweight="bold")
    ax.set_xticks([])
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "11_confidence_zones.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 11_confidence_zones.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    print(f"Generating charts to: {CHART_DIR}\n")
    chart_population_norms()
    chart_frame1_prototypes()
    chart_frame2_heatmap()
    chart_feature_weights()
    chart_radar_depression()
    chart_radar_all_disorders()
    chart_temporal_shapes()
    chart_shape_disorder_matrix()
    chart_gate_thresholds()
    chart_pipeline_demo()
    chart_confidence_zones()
    print(f"\n✅ All 11 charts generated in {CHART_DIR}")
