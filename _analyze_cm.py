import json

data = json.load(open('studentlife_full_results.json'))
valid = [r for r in data['results'] if r['phq9_post'] is not None]

tp = fp = tn = fn = 0
fps = []
fns = []

for r in valid:
    phq = r['phq9_post']
    det = r['anomaly_detected']
    if phq > 9 and det:
        tp += 1
    elif phq <= 9 and det:
        fp += 1
        fps.append(r)
    elif phq > 9 and not det:
        fn += 1
        fns.append(r)
    else:
        tn += 1

print(f"Total valid: {len(valid)}")
print(f"TP={tp} | FP={fp} | FN={fn} | TN={tn}")
sens = tp/(tp+fn) if (tp+fn) else 0
spec = tn/(tn+fp) if (tn+fp) else 0
prec = tp/(tp+fp) if (tp+fp) else 0
acc  = (tp+tn)/(tp+fp+fn+tn)
f1   = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0
print(f"Sensitivity={sens:.3f} Specificity={spec:.3f} Precision={prec:.3f} Accuracy={acc:.3f} F1={f1:.3f}")

print("\n=== FALSE POSITIVES (PHQ<=9 but detected=True) ===")
for r in fps:
    print(f"  {r['user_id']:5s} PHQ={r['phq9_post']:2d} score={r['anomaly_score']:.3f} "
          f"peak_evidence={r.get('peak_evidence', r.get('evidence',0)):.2f} "
          f"max_days={r.get('max_sustained_days','-')}")

print("\n=== FALSE NEGATIVES (PHQ>9 but detected=False) ===")
for r in fns:
    print(f"  {r['user_id']:5s} PHQ={r['phq9_post']:2d} score={r['anomaly_score']:.3f} "
          f"peak_evidence={r.get('peak_evidence', r.get('evidence',0)):.2f} "
          f"max_days={r.get('max_sustained_days','-')}")

# Show all depressed students for context
print("\n=== ALL PHQ>9 STUDENTS ===")
depressed = [r for r in valid if r['phq9_post'] > 9]
for r in sorted(depressed, key=lambda x: x['phq9_post'], reverse=True):
    print(f"  {r['user_id']:5s} PHQ={r['phq9_post']:2d} detected={str(r['anomaly_detected']):5s} "
          f"score={r['anomaly_score']:.3f} "
          f"peak_ev={r.get('peak_evidence', r.get('evidence',0)):.2f} "
          f"max_days={r.get('max_sustained_days','-')}")
