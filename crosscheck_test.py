import sys, os, warnings
import numpy as np
import pandas as pd

# Use Android source code
sys.path.append(os.path.abspath("MHealth - Copy/app/src/main/python"))

from schz_extractor import SchzExtractor
from system1 import PersonalityVector, ImprovedAnomalyDetector
from s1_s2_adapter import build_s1_input
from pipeline import System2Pipeline
from config import BEHAVIORAL_FEATURES, POPULATION_NORMS

warnings.filterwarnings("ignore")

BASELINE_DAYS = 28
FEATURES = BEHAVIORAL_FEATURES

ext = SchzExtractor(data_path=os.path.abspath("schz/CrossCheck_Daily_Data.csv"))
labels = ext.load_ema_labels()

all_results = []
for uid in sorted(ext.get_patient_ids()):
    if uid not in labels:
        continue
    df = ext.extract_patient(uid)
    if len(df) < BASELINE_DAYS + 7:
        continue
    
    baseline_df = df.iloc[:BASELINE_DAYS].copy()
    monitoring_df = df.iloc[BASELINE_DAYS:].copy()
    
    means, variances = {}, {}
    for feat in FEATURES:
        col = baseline_df[feat] if feat in baseline_df.columns else pd.Series(dtype=float)
        valid = col.dropna()
        if len(valid) >= 3:
            means[feat] = float(valid.mean())
            variances[feat] = float(valid.std()) if len(valid) > 1 else 0.1
        else:
            means[feat] = POPULATION_NORMS.get(feat, {"mean": 0})["mean"]
            variances[feat] = POPULATION_NORMS.get(feat, {"std": 1})["std"]
            
    # Fix for missing features like texts_per_day
    for feat in ["screen_time_hours", "unlock_count", "social_app_ratio", "calls_per_day", 
                 "unique_contacts", "daily_displacement_km", "location_entropy", "home_time_ratio", 
                 "places_visited", "wake_time_hour", "sleep_time_hour", "sleep_duration_hours", 
                 "dark_duration_hours", "charge_duration_hours", "conversation_duration_hours", 
                 "conversation_frequency"]:
        if feat not in means:
            means[feat] = POPULATION_NORMS.get(feat, {"mean": 0})["mean"]
            variances[feat] = POPULATION_NORMS.get(feat, {"std": 1})["std"]
    
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
    
    gt = "schizophrenia" if labels[uid].get("schz_flag") else "negative"
    all_results.append({
        "uid": uid,
        "gt": gt,
        "s2": s2.disorder,
        "score": s2.score,
        "filter": s2.filter_decision.value
    })

print(f"{'UID':<10} {'GT':<15} {'S2':<25} {'Score':<10} {'Filter'}")
print("-" * 75)
schz_hits = 0
schz_total = 0

for r in all_results:
    match = r['gt'] == "schizophrenia" and r['s2'].startswith("schizophrenia")
    if r['gt'] == "schizophrenia":
        schz_total += 1
        if match:
            schz_hits += 1
    
    marker = " *** MATCH" if match else ""
    print(f"{r['uid']:<10} {r['gt']:<15} {r['s2']:<25} {r['score']:<10.3f} {r['filter']} {marker}")

print(f"\nTotal Ground Truth Schizophrenia: {schz_total}")
print(f"Top-1 Pipeline Sensitivity: {schz_hits}/{schz_total} ({schz_hits/schz_total*100:.1f}% if {schz_total} > 0)")
