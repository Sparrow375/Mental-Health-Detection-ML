"""Quick diagnostic: show match scores for each student to find confidence floor."""
import sys, os, warnings
import numpy as np
import pandas as pd

from studentlife_extractor import StudentLifeExtractor
from system1 import PersonalityVector, ImprovedAnomalyDetector
from s1_s2_adapter import build_s1_input
from system2.pipeline import System2Pipeline
from system2.config import BEHAVIORAL_FEATURES, POPULATION_NORMS

warnings.filterwarnings("ignore")

BASELINE_DAYS = 28
FEATURES = BEHAVIORAL_FEATURES

ext = StudentLifeExtractor()
labels = ext.load_phq9_labels()
depressed_ids = {uid for uid, info in labels.items() if info.get("phq9_score", 0) >= 10}

all_results = []
for uid in sorted(ext.get_student_ids()):
    if uid not in labels:
        continue
    df = ext.extract_student(uid)
    if len(df) < BASELINE_DAYS + 7:
        continue
    
    baseline_df = df.iloc[:BASELINE_DAYS].copy()
    monitoring_df = df.iloc[BASELINE_DAYS:].copy()
    
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
    
    baseline_vec = PersonalityVector(**means, variances=variances)
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
    
    latest = reports[-1] if reports else None
    if not latest:
        continue
    
    s1_input = build_s1_input(detector, baseline_df, latest, timeseries_days=28)
    pipeline = System2Pipeline()
    s2 = pipeline.classify(s1_input)
    
    gt = "DEPRESSED" if uid in depressed_ids else "healthy"
    phq = labels[uid].get("phq9_score", "?")
    
    # Get all scores
    if s2.classification:
        scores = s2.classification.all_scores
        dep_s = scores.get("depression", 0)
        sch_s = scores.get("schizophrenia", 0)
        anx_s = scores.get("anxiety", 0)
        bpm_s = scores.get("bipolar_manic", 0)
        bpd_s = scores.get("bipolar_depressive", 0)
        top_s = s2.classification.score
    else:
        dep_s = sch_s = anx_s = bpm_s = bpd_s = top_s = 0
    
    co_dev = s1_input.anomaly_report.co_deviating_count
    max_dev = max(abs(v) for v in latest.feature_deviations.values()) if latest.feature_deviations else 0
    
    # Count depression signals
    dep_signals = 0
    for feat, direction in [
        ("conversation_frequency", "negative"), ("conversation_duration_hours", "negative"),
        ("daily_displacement_km", "negative"), ("home_time_ratio", "positive"),
        ("dark_duration_hours", "positive"), ("calls_per_day", "negative"),
        ("texts_per_day", "negative"), ("screen_time_hours", "negative"),
        ("unlock_count", "negative"),
    ]:
        dev = latest.feature_deviations.get(feat, 0)
        if direction == "negative" and dev < -0.3:
            dep_signals += 1
        elif direction == "positive" and dev > 0.3:
            dep_signals += 1
    
    all_results.append({
        "uid": uid, "gt": gt, "phq": phq, "s2": s2.disorder,
        "dep_s": dep_s, "sch_s": sch_s, "top_s": top_s,
        "co_dev": co_dev, "max_dev": round(max_dev, 2),
        "dep_signals": dep_signals,
        "filter": s2.filter_decision.value,
    })

# Print table
print(f"{'UID':<6} {'GT':<10} {'PHQ':<5} {'S2':<18} {'DepS':<6} {'SchS':<6} {'TopS':<6} "
      f"{'CoDev':<6} {'MaxDev':<7} {'DepSig':<7} {'Filter':<8}")
print("-" * 100)
for r in all_results:
    gt_marker = "***" if r["gt"] == "DEPRESSED" else ""
    print(f"{r['uid']:<6} {r['gt']:<10} {r['phq']:<5} {r['s2']:<18} "
          f"{r['dep_s']:.3f} {r['sch_s']:.3f} {r['top_s']:.3f} "
          f"{r['co_dev']:<6} {r['max_dev']:<7} {r['dep_signals']:<7} {r['filter']:<8} {gt_marker}")

# Summary stats
dep_results = [r for r in all_results if r["gt"] == "DEPRESSED"]
healthy_results = [r for r in all_results if r["gt"] == "healthy"]
print(f"\nDEPRESSED students (n={len(dep_results)}):")
print(f"  Avg dep_signals: {np.mean([r['dep_signals'] for r in dep_results]):.1f}")
print(f"  Avg co_dev:      {np.mean([r['co_dev'] for r in dep_results]):.1f}")
print(f"  Avg max_dev:     {np.mean([r['max_dev'] for r in dep_results]):.2f}")
print(f"  Avg dep_score:   {np.mean([r['dep_s'] for r in dep_results]):.3f}")

print(f"\nHEALTHY students (n={len(healthy_results)}):")
print(f"  Avg dep_signals: {np.mean([r['dep_signals'] for r in healthy_results]):.1f}")
print(f"  Avg co_dev:      {np.mean([r['co_dev'] for r in healthy_results]):.1f}")
print(f"  Avg max_dev:     {np.mean([r['max_dev'] for r in healthy_results]):.2f}")
print(f"  Avg dep_score:   {np.mean([r['dep_s'] for r in healthy_results]):.3f}")
