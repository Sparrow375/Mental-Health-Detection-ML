"""
Run All StudentLife Students Through S1 + S2
==============================================

Batch-processes all 49 StudentLife students:
  1. Extract features (studentlife_extractor)
  2. Split into baseline (first 28 days) / monitoring (rest)
  3. Build PersonalityVector from baseline
  4. Run monitoring through System 1 (ImprovedAnomalyDetector)
  5. Pipe S1 output through S2 (System2Pipeline via s1_s2_adapter)
  6. Compare S2 classification against PHQ-9 ground truth

Usage
-----
    C:\\Users\\embar\\miniconda3\\envs\\Nirvana\\python.exe run_studentlife.py
"""

from __future__ import annotations

import sys
import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

# Project imports
from studentlife_extractor import StudentLifeExtractor
from system1 import PersonalityVector, ImprovedAnomalyDetector
from s1_s2_adapter import build_s1_input
from system2.pipeline import System2Pipeline

# ── Config ──────────────────────────────────────────────────────────────
BASELINE_DAYS = 28          # First N days → personal baseline
MIN_MONITORING_DAYS = 7     # Skip students with fewer monitoring days
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "system2", "data")

# Features in PersonalityVector order
FEATURES = [
    "screen_time_hours", "unlock_count", "social_app_ratio",
    "calls_per_day", "texts_per_day", "unique_contacts",
    "response_time_minutes", "daily_displacement_km",
    "location_entropy", "home_time_ratio", "places_visited",
    "wake_time_hour", "sleep_time_hour", "sleep_duration_hours",
    "dark_duration_hours", "charge_duration_hours",
    "conversation_duration_hours", "conversation_frequency",
]


def build_personality_vector(baseline_df: pd.DataFrame) -> PersonalityVector:
    """
    Build a PersonalityVector from baseline daily averages.
    Missing features get population-norm defaults from config.
    """
    from system2.config import POPULATION_NORMS

    means = {}
    variances = {}
    for feat in FEATURES:
        col = baseline_df[feat] if feat in baseline_df.columns else pd.Series()
        valid = col.dropna()
        if len(valid) >= 3:
            means[feat] = float(valid.mean())
            variances[feat] = float(valid.std()) if len(valid) > 1 else 0.1
        else:
            # Fall back to population norms
            means[feat] = POPULATION_NORMS[feat]["mean"]
            variances[feat] = POPULATION_NORMS[feat]["std"]

    return PersonalityVector(
        **means,
        variances=variances,
    )


def process_student(uid: str, student_df: pd.DataFrame, phq9: dict) -> dict:
    """
    Run one student through S1 → S2.

    Returns a result dict with S1 prediction, S2 classification, and ground truth.
    """
    n_days = len(student_df)
    if n_days < BASELINE_DAYS + MIN_MONITORING_DAYS:
        return {"uid": uid, "status": "skipped", "reason": f"only {n_days} days"}

    # Split baseline / monitoring
    baseline_df = student_df.iloc[:BASELINE_DAYS].copy()
    monitoring_df = student_df.iloc[BASELINE_DAYS:].copy()

    # Build PersonalityVector from baseline
    baseline_vec = build_personality_vector(baseline_df)

    # Initialize S1 detector
    detector = ImprovedAnomalyDetector(baseline_vec)

    # Run monitoring through S1
    reports = []
    deviations_history = []

    for idx, (_, row) in enumerate(monitoring_df.iterrows()):
        current_data = {}
        for feat in FEATURES:
            val = row.get(feat, np.nan)
            if pd.isna(val):
                # Use baseline mean for missing monitoring values
                current_data[feat] = baseline_vec.to_dict()[feat]
            else:
                current_data[feat] = float(val)

        report, daily_report = detector.analyze(
            current_data, deviations_history, idx + 1
        )
        reports.append(report)
        deviations_history.append(report.feature_deviations)

    # S1 final prediction
    s1_prediction = detector.generate_final_prediction(
        scenario="studentlife",
        patient_id=uid,
        monitoring_days=len(monitoring_df),
    )

    # ── S2 Classification ───────────────────────────────────────────
    latest_report = reports[-1] if reports else None

    s2_output = None
    s2_disorder = "unknown"
    s2_confidence = "unknown"
    s2_score = 0.0

    if latest_report is not None:
        try:
            s1_input = build_s1_input(
                detector=detector,
                baseline_df=baseline_df,
                s1_report=latest_report,
                timeseries_days=28,
            )
            pipeline = System2Pipeline()
            s2_output = pipeline.classify(s1_input)
            s2_disorder = s2_output.disorder
            s2_confidence = s2_output.confidence.value
            s2_score = s2_output.score
        except Exception as e:
            s2_disorder = f"error: {e}"

    # ── Ground truth ────────────────────────────────────────────────
    pre_score = phq9.get("pre_score", -1)
    post_score = phq9.get("post_score", -1)
    pre_severity = phq9.get("pre_severity", "unknown")
    post_severity = phq9.get("post_severity", "unknown")

    # PHQ-9 >= 10 = clinically depressed
    ground_truth = "depressed" if post_score >= 10 else (
        "mild" if post_score >= 5 else "healthy"
    )

    return {
        "uid": uid,
        "status": "processed",
        "total_days": n_days,
        "baseline_days": BASELINE_DAYS,
        "monitoring_days": len(monitoring_df),
        # S1 results
        "s1_anomaly_detected": s1_prediction.sustained_anomaly_detected,
        "s1_confidence": round(s1_prediction.confidence, 3),
        "s1_final_score": round(s1_prediction.final_anomaly_score, 3),
        "s1_peak_evidence": s1_prediction.evidence_summary["peak_evidence"],
        "s1_max_sustained_days": s1_prediction.evidence_summary["max_sustained_days"],
        "s1_pattern": s1_prediction.pattern_identified,
        "s1_recommendation": s1_prediction.recommendation.split(":")[0],
        # S2 results
        "s2_disorder": s2_disorder,
        "s2_confidence": s2_confidence,
        "s2_score": round(s2_score, 4) if isinstance(s2_score, float) else s2_score,
        "s2_classification": s2_output.classification if s2_output else None,
        # Ground truth
        "phq9_pre": pre_score,
        "phq9_post": post_score,
        "phq9_pre_severity": pre_severity,
        "phq9_post_severity": post_severity,
        "ground_truth": ground_truth,
        "feature_deviations": latest_report.feature_deviations if latest_report else {},
    }


def main():
    print("=" * 70)
    print("  StudentLife Full Pipeline Run (S1 + S2)")
    print("=" * 70)

    # Extract all students
    ext = StudentLifeExtractor()
    phq9_labels = ext.load_phq9_labels()
    student_ids = ext.get_student_ids()
    print(f"\nStudents found:   {len(student_ids)}")
    print(f"PHQ-9 labels:     {len(phq9_labels)}")
    print(f"Baseline period:  {BASELINE_DAYS} days")
    print(f"Min monitoring:   {MIN_MONITORING_DAYS} days")
    print()

    results = []
    for i, uid in enumerate(student_ids):
        print(f"  [{i+1:2d}/{len(student_ids)}] {uid}...", end=" ", flush=True)

        try:
            df = ext.extract_student(uid)
        except Exception as e:
            print(f"EXTRACT FAILED: {e}")
            results.append({"uid": uid, "status": "extract_failed", "reason": str(e)})
            continue

        if len(df) == 0:
            print("NO DATA")
            results.append({"uid": uid, "status": "no_data"})
            continue

        phq9 = phq9_labels.get(uid, {})
        result = process_student(uid, df, phq9)

        if result["status"] == "skipped":
            print(f"SKIPPED ({result['reason']})")
        else:
            s1_flag = "ANOMALY" if result["s1_anomaly_detected"] else "normal"
            gt = result["ground_truth"]
            s2d = result["s2_disorder"]
            print(f"S1={s1_flag:7s} | S2={s2d:12s} | GT={gt}")

        results.append(result)

    # ── Summary ─────────────────────────────────────────────────────
    processed = [r for r in results if r["status"] == "processed"]
    skipped = [r for r in results if r["status"] != "processed"]

    print(f"\n{'=' * 70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Processed: {len(processed)} / {len(results)}")
    print(f"  Skipped:   {len(skipped)}")

    if processed:
        # Ground truth distribution
        gt_counts = {}
        for r in processed:
            gt = r["ground_truth"]
            gt_counts[gt] = gt_counts.get(gt, 0) + 1
        print(f"\n  Ground Truth Distribution:")
        for gt, count in sorted(gt_counts.items()):
            print(f"    {gt:12s}: {count}")

        # S1 detection rates
        s1_detected = sum(1 for r in processed if r["s1_anomaly_detected"])
        print(f"\n  S1 Anomaly Detected: {s1_detected}/{len(processed)}")

        # S2 classification distribution
        s2_counts = {}
        for r in processed:
            d = r["s2_disorder"]
            s2_counts[d] = s2_counts.get(d, 0) + 1
        print(f"\n  S2 Classification Distribution:")
        for disorder, count in sorted(s2_counts.items(), key=lambda x: -x[1]):
            print(f"    {disorder:20s}: {count}")

        # Confusion analysis: depressed vs healthy
        depressed = [r for r in processed if r["ground_truth"] == "depressed"]
        healthy = [r for r in processed if r["ground_truth"] == "healthy"]

        if depressed:
            s1_tp = sum(1 for r in depressed if r["s1_anomaly_detected"])
            print(f"\n  S1 Sensitivity (depressed caught): {s1_tp}/{len(depressed)} = {s1_tp/len(depressed):.0%}")

        if healthy:
            s1_fp = sum(1 for r in healthy if r["s1_anomaly_detected"])
            s1_tn = len(healthy) - s1_fp
            print(f"  S1 Specificity (healthy correct):  {s1_tn}/{len(healthy)} = {s1_tn/len(healthy):.0%}")

        # S2 accuracy for depressed (Top-3 Classification)
        if depressed:
            s2_tp_top3 = 0
            for r in depressed:
                # Get top 3 predicted disorders
                if r.get("s2_classification"):
                    top3 = sorted(r["s2_classification"].all_scores.items(), key=lambda x: -x[1])[:3]
                    top3_names = [name for name, score in top3]
                    if any(name.startswith("depression") for name in top3_names):
                        s2_tp_top3 += 1
                    
            s2_tp_top1 = sum(1 for r in depressed if r["s2_disorder"].startswith("depression"))
            print(f"\n  S2 Depression Top-1 Sensitivity: {s2_tp_top1}/{len(depressed)} = {s2_tp_top1/len(depressed):.0%}")
            print(f"  S2 Depression Top-3 Sensitivity: {s2_tp_top3}/{len(depressed)} = {s2_tp_top3/len(depressed):.0%} (Clinical Standard)")

        if healthy:
            s2_correct = sum(1 for r in healthy if r["s2_disorder"].startswith("healthy") or r["s2_disorder"] in ("situational_stress", "life_event"))
            print(f"  S2 Healthy Specificity:    {s2_correct}/{len(healthy)} = {s2_correct/len(healthy):.0%}")

    # Save results to JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "studentlife_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")

    # Save as CSV for easy viewing
    csv_path = os.path.join(OUTPUT_DIR, "studentlife_results.csv")
    pd.DataFrame(processed).to_csv(csv_path, index=False)
    print(f"  CSV saved to:     {csv_path}")

    print(f"\n{'=' * 70}")
    print("  DONE")
    print(f"{'=' * 70}")

    return results


if __name__ == "__main__":
    results = main()
