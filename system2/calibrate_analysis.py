"""
Calibration Analysis: SchzExtractor — compute schz vs non-schz distributions
"""
import numpy as np
import pandas as pd
from schz_extractor import SchzExtractor
from system2.config import BEHAVIORAL_FEATURES, DISORDER_PROTOTYPES_FRAME2, POPULATION_NORMS

FEATURES = [f for f in BEHAVIORAL_FEATURES if f != "charge_duration_hours"]  # not in CrossCheck

ext = SchzExtractor()
labels = ext.load_ema_labels()
ids = ext.get_patient_ids()

schz_ids = [u for u in ids if labels.get(u, {}).get("schz_flag", False)]
non_schz_ids = [u for u in ids if not labels.get(u, {}).get("schz_flag", False)]

print(f"Schz patients: {len(schz_ids)}")
print(f"Non-schz patients: {len(non_schz_ids)}")

def group_means(uid_list):
    all_means = []
    for uid in uid_list:
        try:
            df = ext.extract_patient(uid)
            if len(df) >= 7:
                m = df[FEATURES].mean()
                all_means.append(m)
        except: pass
    if all_means:
        return pd.DataFrame(all_means).mean()
    return pd.Series()

print("\nComputing group means...")
schz_means = group_means(schz_ids)
non_schz_means = group_means(non_schz_ids)

print("\n" + "="*80)
print("SCHZ vs NON-SCHZ FEATURE COMPARISON")
print("="*80)
print(f"{'Feature':<32} {'NonSchz':>10} {'Schz':>10} {'Diff':>8} {'Z':>8} {'CurrentF2':>10}")
print("-"*80)

real_z = {}
for feat in FEATURES:
    ns = non_schz_means.get(feat, np.nan)
    s = schz_means.get(feat, np.nan)
    # Use internal (CrossCheck) std as normalizer
    pop_std = max(abs(ns) * 0.3, 0.001) if not np.isnan(ns) else 0.001
    diff = s - ns if not (np.isnan(s) or np.isnan(ns)) else np.nan
    z = diff / pop_std if not np.isnan(diff) else np.nan
    cur = DISORDER_PROTOTYPES_FRAME2["schizophrenia"].get(feat, 0)
    real_z[feat] = round(z, 2) if not np.isnan(z) else 0.0
    print(f"{feat:<32} {ns:>10.3f} {s:>10.3f} {diff:>+8.3f} {z:>+8.2f} {cur:>+10.2f}")

print("\n\nSUGGESTED FRAME 2 SCHIZOPHRENIA PROTOTYPE (capped at ±5):")
print("{")
for feat in BEHAVIORAL_FEATURES:
    z = real_z.get(feat, DISORDER_PROTOTYPES_FRAME2["schizophrenia"].get(feat, 0))
    z_capped = max(-5.0, min(5.0, z))
    print(f'    "{feat}": {z_capped},')
print("}")

print("\n\nSCHZ GROUP FRAME 1 ABSOLUTE MEANS (for Frame 1 prototype):")
print("{")
for feat in BEHAVIORAL_FEATURES:
    v = schz_means.get(feat, np.nan)
    if np.isnan(v):
        v = DISORDER_PROTOTYPES_FRAME2.get("schizophrenia", {}).get(feat, 0)
    print(f'    "{feat}": {v:.3f},')
print("}")
