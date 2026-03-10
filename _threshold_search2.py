"""
Find optimal PEAK_EVIDENCE_THRESHOLD to get FP <= 2-3 or zero.
"""
import json, sys
import numpy as np

data  = json.load(open('studentlife_full_results.json'))
valid = [r for r in data['results'] if r['phq9_post'] is not None]

rows = []
for r in valid:
    pe  = float(r.get('peak_evidence', r.get('evidence', 0)) or 0)
    msd = int(r.get('max_sustained_days', 0) or 0)
    dep = r['phq9_post'] > 9
    rows.append((r['user_id'], r['phq9_post'], pe, msd, dep))

rows.sort(key=lambda x: x[2], reverse=True)

lines = []
lines.append("=" * 70)
lines.append(f"{'User':6s} {'PHQ9':5s} {'PeakEv':9s} {'MaxDays':8s} {'Group'}")
lines.append("=" * 70)
for uid, phq, pe, msd, dep in rows:
    grp = "DEPRESSED" if dep else "normal"
    lines.append(f"{uid:6s} {phq:5d} {pe:9.3f} {msd:8d}   {grp}")

lines.append("")
lines.append("=== PEAK_EVIDENCE_THRESHOLD sweep ===")
lines.append(f"{'Thresh':8s} {'TP':3s} {'FP':3s} {'FN':3s} {'TN':3s} {'Spec':6s} {'Sens':6s} {'F1':5s}")
for thresh in np.arange(0.5, 20.0, 0.5):
    tp=fp=fn=tn=0
    for uid, phq, pe, msd, dep in rows:
        det = pe >= thresh
        if dep and det:       tp+=1
        elif dep and not det: fn+=1
        elif not dep and det: fp+=1
        else:                 tn+=1
    spec = tn/(tn+fp) if (tn+fp) else 0
    sens = tp/(tp+fn) if (tp+fn) else 0
    f1   = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0
    marker = "  <<<" if fp<=3 else ""
    lines.append(f"{thresh:8.1f} {tp:3d} {fp:3d} {fn:3d} {tn:3d} {spec:6.3f} {sens:6.3f} {f1:5.3f}{marker}")

lines.append("")
lines.append("=== MAX_SUSTAINED_DAYS sweep ===")
lines.append(f"{'MinDays':8s} {'TP':3s} {'FP':3s} {'FN':3s} {'TN':3s} {'Spec':6s} {'Sens':6s}")
for min_days in range(3, 30):
    tp=fp=fn=tn=0
    for uid, phq, pe, msd, dep in rows:
        det = msd >= min_days
        if dep and det:       tp+=1
        elif dep and not det: fn+=1
        elif not dep and det: fp+=1
        else:                 tn+=1
    spec = tn/(tn+fp) if (tn+fp) else 0
    sens = tp/(tp+fn) if (tp+fn) else 0
    marker = "  <<<" if fp<=3 else ""
    lines.append(f"{min_days:8d} {tp:3d} {fp:3d} {fn:3d} {tn:3d} {spec:6.3f} {sens:6.3f}{marker}")

with open('_thresh_results.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print("Written to _thresh_results.txt")
