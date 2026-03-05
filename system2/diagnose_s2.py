"""
S2 Diagnostic: trace why depressed students are misclassified.
Shows feature_deviations, co_deviating_count, Frame, and
prototype distances for each depressed student.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from studentlife_extractor import StudentLifeExtractor
from system1 import PersonalityVector, ImprovedAnomalyDetector
from s1_s2_adapter import build_s1_input
from system2.pipeline import System2Pipeline
from system2.config import BEHAVIORAL_FEATURES, DISORDER_PROTOTYPES_FRAME2, FEATURE_WEIGHTS
from system2.config import POPULATION_NORMS

BASELINE_DAYS = 28
MIN_MONITOR = 7
FEATURES = BEHAVIORAL_FEATURES

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

ext = StudentLifeExtractor()
labels = ext.load_phq9_labels()
ids = ext.get_student_ids()
depressed_ids = [u for u in ids if labels.get(u, {}).get("post_score", 0) >= 10]
print(f"Depressed students (PHQ >= 10): {depressed_ids}\n")

pipeline = System2Pipeline()

for uid in depressed_ids:
    try:
        df = ext.extract_student(uid)
        if len(df) < BASELINE_DAYS + MIN_MONITOR:
            print(f"{uid}: insufficient data\n"); continue

        baseline_df = df.iloc[:BASELINE_DAYS].copy()
        monitor_df  = df.iloc[BASELINE_DAYS:].copy()
        baseline_vec = build_personality_vector(baseline_df)
        detector = ImprovedAnomalyDetector(baseline_vec)

        reports = []
        deviations_history = []
        for idx, (_, row) in enumerate(monitor_df.iterrows()):
            current_data = {}
            for feat in FEATURES:
                val = row.get(feat, np.nan)
                current_data[feat] = baseline_vec.to_dict()[feat] if pd.isna(val) else float(val)
            report, _ = detector.analyze(current_data, deviations_history, idx + 1)
            reports.append(report)
            deviations_history.append(report.feature_deviations)

        last_report = reports[-1]
        s1_input = build_s1_input(detector, baseline_df, last_report, timeseries_days=28)
        output = pipeline.classify(s1_input)

        phq = labels.get(uid, {}).get("post_score", "?")
        print(f"{'='*72}")
        print(f"  {uid}  PHQ={phq}  GT=depressed  S2={output.disorder}")
        print(f"  Frame={output.screening.frame}  Filter={output.filter_decision.value}  co_dev={s1_input.anomaly_report.co_deviating_count}")
        print()

        devs = s1_input.anomaly_report.feature_deviations
        dep  = DISORDER_PROTOTYPES_FRAME2.get("depression", {})
        schz = DISORDER_PROTOTYPES_FRAME2.get("schizophrenia", {})

        print(f"  {'Feature':<30} {'Dev':>7} {'DepP':>7} {'SchzP':>7} {'dDep':>7} {'dSchz':>7} {'Win':>5}")
        print(f"  {'-'*66}")
        td, ts = 0.0, 0.0
        for f in FEATURES:
            d  = devs.get(f, 0)
            dp = dep.get(f, 0)
            sp = schz.get(f, 0)
            w  = FEATURE_WEIGHTS.get(f, 0.5)
            dd = w * (d - dp) ** 2
            ds = w * (d - sp) ** 2
            td += dd; ts += ds
            win = "SCHZ" if ds < dd else "DEP"
            print(f"  {f:<30} {d:>+7.2f} {dp:>+7.2f} {sp:>+7.2f} {dd:>7.3f} {ds:>7.3f} {win:>5}")

        print(f"  {'TOTAL':<30} {'':>7} {'':>7} {'':>7} {td:>7.3f} {ts:>7.3f}")
        winner = "DEPRESSION" if td < ts else "SCHIZOPHRENIA"
        print(f"\n  => Distance says: {winner}, S2 said: {output.disorder.upper()}")
        print()

    except Exception as e:
        import traceback
        print(f"{uid}: ERROR: {e}")
        traceback.print_exc()
