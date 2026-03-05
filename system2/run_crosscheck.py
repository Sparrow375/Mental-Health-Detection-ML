"""
Run CrossCheck Patients Through S1 + S2
========================================

Same pipeline as run_studentlife.py but for the CrossCheck schizophrenia
dataset (90 patients).  Uses schz_extractor.py for feature extraction
and EMA labels (VOICES + SEEING_THINGS) for ground truth.
"""

from __future__ import annotations

import sys, os, json, warnings
import numpy as np
import pandas as pd

from schz_extractor import SchzExtractor
from system1 import PersonalityVector, ImprovedAnomalyDetector
from s1_s2_adapter import build_s1_input
from system2.pipeline import System2Pipeline
from system2.config import BEHAVIORAL_FEATURES, POPULATION_NORMS

BASELINE_DAYS = 28
MIN_MONITORING_DAYS = 7
FEATURES = BEHAVIORAL_FEATURES
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "system2", "data")


def build_personality_vector(baseline_df):
    means, variances = {}, {}
    for feat in FEATURES:
        col = baseline_df[feat] if feat in baseline_df.columns else pd.Series()
        valid = col.dropna()
        if len(valid) >= 3:
            means[feat] = float(valid.mean())
            variances[feat] = float(valid.std()) if len(valid) > 1 else 0.1
        else:
            means[feat] = POPULATION_NORMS[feat]["mean"]
            variances[feat] = POPULATION_NORMS[feat]["std"]
    return PersonalityVector(**means, variances=variances)


def process_patient(uid, patient_df, label):
    n_days = len(patient_df)
    if n_days < BASELINE_DAYS + MIN_MONITORING_DAYS:
        return {"uid": uid, "status": "skipped", "reason": f"only {n_days} days"}

    baseline_df = patient_df.iloc[:BASELINE_DAYS].copy()
    monitoring_df = patient_df.iloc[BASELINE_DAYS:].copy()
    baseline_vec = build_personality_vector(baseline_df)
    detector = ImprovedAnomalyDetector(baseline_vec)

    reports = []
    deviations_history = []
    for idx, (_, row) in enumerate(monitoring_df.iterrows()):
        current_data = {}
        for feat in FEATURES:
            val = row.get(feat, np.nan)
            current_data[feat] = baseline_vec.to_dict()[feat] if pd.isna(val) else float(val)
        report, _ = detector.analyze(current_data, deviations_history, idx + 1)
        reports.append(report)
        deviations_history.append(report.feature_deviations)

    s1_prediction = detector.generate_final_prediction(
        scenario="crosscheck", patient_id=uid, monitoring_days=len(monitoring_df)
    )

    latest_report = reports[-1] if reports else None
    s2_disorder = "unknown"
    s2_confidence = "unknown"
    s2_score = 0.0

    if latest_report:
        try:
            s1_input = build_s1_input(detector, baseline_df, latest_report, timeseries_days=28)
            pipeline = System2Pipeline()
            s2_output = pipeline.classify(s1_input)
            s2_disorder = s2_output.disorder
            s2_confidence = s2_output.confidence.value
            s2_score = s2_output.score
        except Exception as e:
            s2_disorder = f"error: {e}"

    # Ground truth
    schz_flag = label.get("schz_flag", False) if label else False
    dep_flag = label.get("depression_flag", False) if label else False
    if schz_flag:
        ground_truth = "schizophrenia"
    elif dep_flag:
        ground_truth = "depressed"
    else:
        ground_truth = "healthy"

    return {
        "uid": uid,
        "status": "processed",
        "total_days": n_days,
        "monitoring_days": len(monitoring_df),
        "s1_anomaly": s1_prediction.sustained_anomaly_detected,
        "s2_disorder": s2_disorder,
        "s2_confidence": s2_confidence,
        "s2_score": round(s2_score, 4) if isinstance(s2_score, float) else s2_score,
        "s2_classification": s2_output.classification if s2_output else None,
        "ground_truth": ground_truth,
        "schz_flag": schz_flag,
        "dep_flag": dep_flag,
        "feature_deviations": latest_report.feature_deviations if latest_report else {},
    }


def main():
    warnings.filterwarnings("ignore")
    print("=" * 70)
    print("  CrossCheck Full Pipeline Run (S1 + S2)")
    print("=" * 70)

    ext = SchzExtractor()
    labels = ext.load_ema_labels()
    patient_ids = ext.get_patient_ids()
    print(f"\nPatients found: {len(patient_ids)}")
    print(f"EMA labels:     {len(labels)}")

    schz_count = sum(1 for l in labels.values() if l.get("schz_flag"))
    dep_count = sum(1 for l in labels.values() if l.get("depression_flag"))
    print(f"Schz-positive:  {schz_count}")
    print(f"Dep-positive:   {dep_count}")
    print(f"Baseline:       {BASELINE_DAYS} days")
    print()

    results = []
    for i, uid in enumerate(patient_ids):
        print(f"  [{i+1:2d}/{len(patient_ids)}] {uid}...", end=" ", flush=True)
        try:
            df = ext.extract_patient(uid)
        except Exception as e:
            print(f"EXTRACT FAILED: {e}")
            results.append({"uid": uid, "status": "extract_failed"})
            continue

        if len(df) == 0:
            print("NO DATA")
            results.append({"uid": uid, "status": "no_data"})
            continue

        label = labels.get(uid, {})
        result = process_patient(uid, df, label)

        if result["status"] == "skipped":
            print(f"SKIPPED ({result['reason']})")
        else:
            s1_flag = "ANOMALY" if result["s1_anomaly"] else "normal"
            gt = result["ground_truth"]
            s2d = result["s2_disorder"]
            print(f"S1={s1_flag:7s} | S2={s2d:12s} | GT={gt}")

        results.append(result)

    # Summary
    processed = [r for r in results if r["status"] == "processed"]
    print(f"\n{'=' * 70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Processed: {len(processed)} / {len(results)}")

    gt_counts = {}
    for r in processed:
        gt = r["ground_truth"]
        gt_counts[gt] = gt_counts.get(gt, 0) + 1
    print(f"\n  Ground Truth:")
    for gt, c in sorted(gt_counts.items()):
        print(f"    {gt:15s}: {c}")

    s2_counts = {}
    for r in processed:
        d = r["s2_disorder"]
        s2_counts[d] = s2_counts.get(d, 0) + 1
    print(f"\n  S2 Classification Distribution:")
    for d, c in sorted(s2_counts.items(), key=lambda x: -x[1]):
        print(f"    {d:20s}: {c}")

    # Schizophrenia sensitivity
    schz_patients = [r for r in processed if r["ground_truth"] == "schizophrenia"]
    healthy_patients = [r for r in processed if r["ground_truth"] == "healthy"]

    if schz_patients:
        schz_tp_top3 = 0
        for r in schz_patients:
            if r.get("s2_classification"):
                top3 = sorted(r["s2_classification"].all_scores.items(), key=lambda x: -x[1])[:3]
                if any(name.startswith("schizo") for name, score in top3):
                    schz_tp_top3 += 1
        
        schz_tp_top1 = sum(1 for r in schz_patients if r["s2_disorder"].startswith("schizo"))
        print(f"\n  Schz Top-1 Sensitivity: {schz_tp_top1}/{len(schz_patients)} = {schz_tp_top1/len(schz_patients):.0%}")
        print(f"  Schz Top-3 Sensitivity: {schz_tp_top3}/{len(schz_patients)} = {schz_tp_top3/len(schz_patients):.0%} (Clinical Standard)")

    if healthy_patients:
        healthy_correct = sum(1 for r in healthy_patients
                             if r["s2_disorder"].startswith("healthy") or r["s2_disorder"] in ("life_event", "situational_stress"))
        print(f"  Healthy Specificity: {healthy_correct}/{len(healthy_patients)} = {healthy_correct/len(healthy_patients):.0%}")

    # Depression accuracy
    dep_patients = [r for r in processed if r["ground_truth"] == "depressed"]
    if dep_patients:
        dep_tp_top3 = 0
        for r in dep_patients:
            if r.get("s2_classification"):
                top3 = sorted(r["s2_classification"].all_scores.items(), key=lambda x: -x[1])[:3]
                if any(name.startswith("depression") for name, score in top3):
                    dep_tp_top3 += 1
                    
        dep_tp_top1 = sum(1 for r in dep_patients if r["s2_disorder"].startswith("depression"))
        print(f"  Depression Top-1 Sensitivity: {dep_tp_top1}/{len(dep_patients)} = {dep_tp_top1/len(dep_patients):.0%}")
        print(f"  Depression Top-3 Sensitivity: {dep_tp_top3}/{len(dep_patients)} = {dep_tp_top3/len(dep_patients):.0%} (Clinical Standard)")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "crosscheck_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {path}")

    csv_path = os.path.join(OUTPUT_DIR, "crosscheck_results.csv")
    pd.DataFrame(processed).to_csv(csv_path, index=False)
    print(f"  CSV saved to:     {csv_path}")
    print(f"\n{'=' * 70}")
    print("  DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
