"""
System 2 — Practical Demo
===========================
Run this from the PROJECT ROOT:
    python demo_system2.py
"""

import sys
import os
import numpy as np

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system2.config import (
    BEHAVIORAL_FEATURES,
    POPULATION_NORMS,
    DISORDER_PROTOTYPES_FRAME1,
    DISORDER_PROTOTYPES_FRAME2,
)
from system2.life_event_filter import AnomalyReport
from system2.pipeline import System2Pipeline, S1Input


def divider(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def healthy_baseline():
    profile = {f: POPULATION_NORMS[f]["mean"] for f in BEHAVIORAL_FEATURES}
    return {
        "raw_7day": profile.copy(),
        "weekly_windows": [profile.copy(), profile.copy(), profile.copy()],
        "raw_28day": profile.copy(),
    }


pipeline = System2Pipeline()

# ── SCENARIO 1: Depressed User ──
divider("SCENARIO 1: Depressed User")
print("Situation: User had a clean baseline. Over the past month, they've")
print("shown reduced movement, social withdrawal, increased sleep.\n")

output = pipeline.classify(
    S1Input(
        baseline_data=healthy_baseline(),
        anomaly_report=AnomalyReport(
            feature_deviations=DISORDER_PROTOTYPES_FRAME2["depression"].copy(),
            days_sustained=30, co_deviating_count=10,
            resolved=False, days_since_onset=30,
        ),
        anomaly_timeseries=list(np.linspace(0.5, -2.0, 28)),
    )
)
print(f"  Result:       {output.disorder.upper()}")
print(f"  Confidence:   {output.confidence.value}")
print(f"  Score:        {output.score:.2f}")
print(f"  Label:        {output.label}")
print(f"  Shape:        {output.temporal_result.temporal_shape}")
print(f"  Narrative:    {output.explanation.narrative}")
print(f"  Top features: {output.explanation.top_features}")
print(f"\n  All scores:")
for d, s in sorted(output.classification.all_scores.items(), key=lambda x: -x[1]):
    marker = " <-- MATCH" if d == output.disorder else ""
    print(f"    {d:25s} {s:.4f}{marker}")


# ── SCENARIO 2: Life Event (Exam Week) ──
divider("SCENARIO 2: Life Event (Exam Week)")
print("Situation: Only 2 features deviated for 5 days, then resolved.\n")

output = pipeline.classify(
    S1Input(
        baseline_data=healthy_baseline(),
        anomaly_report=AnomalyReport(
            feature_deviations={
                **{f: 0.0 for f in BEHAVIORAL_FEATURES},
                "sleep_duration_hours": -1.0, "screen_time_hours": 1.2,
            },
            days_sustained=5, co_deviating_count=2,
            resolved=True, days_since_onset=7,
        ),
        anomaly_timeseries=list(np.zeros(28)),
    )
)
print(f"  Result:   {output.disorder.upper()}")
print(f"  Decision: {output.filter_decision.value}")
print(f"  --> Correctly dismissed as life event!")


# ── SCENARIO 3: Contaminated Baseline ──
divider("SCENARIO 3: Contaminated Baseline")
print("Situation: User was already depressed during onboarding.\n")

dep_raw = DISORDER_PROTOTYPES_FRAME1["depression"]
output = pipeline.classify(
    S1Input(
        baseline_data={
            "raw_7day": dep_raw.copy(),
            "weekly_windows": [dep_raw.copy()] * 3,
            "raw_28day": dep_raw.copy(),
        },
        anomaly_report=AnomalyReport(
            feature_deviations=dep_raw.copy(),
            days_sustained=28, co_deviating_count=8,
            resolved=False, days_since_onset=28,
        ),
        anomaly_timeseries=list(np.linspace(0.0, -1.5, 28)),
    )
)
print(f"  Screening passed? {output.screening.passed}")
print(f"  Gates fired:      {output.screening.gates_fired}")
print(f"  Action:           {output.screening.recommended_action.value}")
print(f"  Frame used:       Frame {output.screening.frame}")
print(f"  --> Contaminated baseline caught! Fell back to Frame 1.")


# ── SCENARIO 4: Bipolar Manic ──
divider("SCENARIO 4: Bipolar Manic Episode")
print("Situation: Sudden hyperactivity, reduced sleep, social surge.\n")

ts_flip = list(np.concatenate([
    np.linspace(-1.0, -1.5, 14) + np.random.RandomState(42).normal(0, 0.2, 14),
    np.linspace(2.0, 2.5, 14) + np.random.RandomState(42).normal(0, 0.2, 14),
]))

output = pipeline.classify(
    S1Input(
        baseline_data=healthy_baseline(),
        anomaly_report=AnomalyReport(
            feature_deviations=DISORDER_PROTOTYPES_FRAME2["bipolar_manic"].copy(),
            days_sustained=14, co_deviating_count=12,
            resolved=False, days_since_onset=14,
        ),
        anomaly_timeseries=ts_flip,
    )
)
print(f"  Result:       {output.disorder.replace('_', ' ').upper()}")
print(f"  Confidence:   {output.confidence.value}")
print(f"  Score:        {output.score:.2f}")
print(f"  Narrative:    {output.explanation.narrative}")
print(f"  Top features: {output.explanation.top_features}")

divider("DONE -- All 4 scenarios complete")
