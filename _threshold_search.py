"""
Find the optimal PEAK_EVIDENCE_THRESHOLD to minimise False Positives
while preserving True Positives (depressed students, PHQ-9 > 9).
"""
import json
import numpy as np

data  = json.load(open('studentlife_full_results.json'))
valid = [r for r in data['results'] if r['phq9_post'] is not None]

# Collect (user, phq9, peak_evidence, max_sustained_days, is_depressed)
rows = []
for r in valid:
    pe   = r.get('peak_evidence', r.get('evidence', 0)) or 0
    msd  = r.get('max_sustained_days', 0) or 0
    dep  = r['phq9_post'] > 9
    rows.append((r['user_id'], r['phq9_post'], pe, msd, dep))

rows.sort(key=lambda x: x[2], reverse=True)   # sort by peak_evidence desc

print("=" * 75)
print(f"{'User':6s} {'PHQ-9':6s} {'PeakEvidence':13s} {'MaxDays':8s} {'Group':12s} {'Status'}")
print("=" * 75)
for uid, phq, pe, msd, dep in rows:
    group  = "DEPRESSED" if dep else "normal   "
    status = ""
    print(f"{uid:6s} {phq:6d} {pe:13.2f} {msd:8d} {group:12s} {status}")

print()

# ── Sweep PEAK_EVIDENCE_THRESHOLD ─────────────────────────────────────────────
print("=" * 75)
print(f"{'Threshold':12s} {'TP':4s} {'FP':4s} {'FN':4s} {'TN':4s} {'Spec':8s} {'Sens':8s} {'F1':6s}")
print("=" * 75)

thresholds = sorted(set([pe for *_, pe, __, ___ in rows] +
                        [pe + 0.01 for *_, pe, __, ___ in rows]))

best = None
for thresh in np.arange(0.5, 15.0, 0.25):
    tp = fp = fn = tn = 0
    for uid, phq, pe, msd, dep in rows:
        detected = pe >= thresh
        if dep  and     detected: tp += 1
        elif dep and not detected: fn += 1
        elif not dep and detected: fp += 1
        else:                      tn += 1
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    f1   = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0
    if fp <= 3:   # highlight promising rows
        marker = "  <<< FP<=3"
    elif fp <= 6:
        marker = ""
    else:
        continue   # skip worse-than-current rows
    print(f"{thresh:12.2f} {tp:4d} {fp:4d} {fn:4d} {tn:4d} {spec:8.3f} {sens:8.3f} {f1:6.3f}{marker}")
    if best is None or (fp <= best[1] and sens >= best[5] - 0.01):
        best = (thresh, fp, fn, tp, tn, sens, spec, f1)

print()
print("=" * 75)
print("ALSO SWEEP max_sustained_days threshold:")
print(f"{'Days':8s} {'TP':4s} {'FP':4s} {'FN':4s} {'TN':4s} {'Spec':8s} {'Sens':8s}")
for min_days in range(4, 25):
    tp = fp = fn = tn = 0
    for uid, phq, pe, msd, dep in rows:
        detected = msd >= min_days
        if dep  and     detected: tp += 1
        elif dep and not detected: fn += 1
        elif not dep and detected: fp += 1
        else:                      tn += 1
    spec = tn / (tn + fp) if (tn + fp) else 0
    sens = tp / (tp + fn) if (tp + fn) else 0
    if fp <= 3:
        marker = "  <<< FP<=3"
    elif fp > 6:
        continue
    else:
        marker = ""
    print(f"{min_days:8d} {tp:4d} {fp:4d} {fn:4d} {tn:4d} {spec:8.3f} {sens:8.3f}{marker}")
