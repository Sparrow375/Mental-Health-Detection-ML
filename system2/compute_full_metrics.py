"""
compute_full_metrics.py
========================
Runs BOTH datasets through S1 → S2 and produces exact metrics:

  Dataset 1: StudentLife  (C:\\Users\\SRIRAM\\Downloads\\dataset)
             Ground truth: PHQ-9 post score ≥ 10 = Depressed
  Dataset 2: CrossCheck   (C:\\Users\\SRIRAM\\Downloads\\schz)
             Ground truth: EMA schz_flag (avg voices+seeing > 0.5) = Schizophrenia

Outputs:
  • Console: confusion matrices, accuracy, precision, recall, F1
  • system2/data/studentlife_metrics.json
  • system2/data/crosscheck_metrics.json
  • system2/data/combined_metrics.csv

Run:
    .venv\\Scripts\\python.exe system2\\compute_full_metrics.py
"""

from __future__ import annotations
import os, sys, json, warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "system2"))

STUDENTLIFE_PATH = r"C:\Users\SRIRAM\Downloads\dataset"
CROSSCHECK_PATH  = os.path.join(r"C:\Users\SRIRAM\Downloads\schz", "CrossCheck_Daily_Data.csv")

from studentlife_extractor import StudentLifeExtractor
from schz_extractor         import SchzExtractor
from system1                import PersonalityVector, ImprovedAnomalyDetector
from s1_s2_adapter          import build_s1_input
from system2.pipeline       import System2Pipeline
from system2.config         import POPULATION_NORMS, BEHAVIORAL_FEATURES

BASELINE_DAYS       = 28
MIN_MONITORING_DAYS = 7
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)

W = 72

# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_pv(df: pd.DataFrame) -> PersonalityVector:
    means, variances = {}, {}
    for feat in BEHAVIORAL_FEATURES:
        col   = df[feat] if feat in df.columns else pd.Series(dtype=float)
        valid = col.dropna()
        if len(valid) >= 3:
            means[feat]     = float(valid.mean())
            variances[feat] = float(valid.std()) if len(valid) > 1 else 0.1
        else:
            means[feat]     = POPULATION_NORMS[feat]["mean"]
            variances[feat] = POPULATION_NORMS[feat]["std"]
    return PersonalityVector(**means, variances=variances)


def run_pipeline(uid: str, df: pd.DataFrame) -> dict:
    """Run S1→S2 on one patient/student. Returns result dict."""
    n = len(df)
    if n < BASELINE_DAYS + MIN_MONITORING_DAYS:
        return {"uid": uid, "status": "skipped", "reason": f"only {n} days"}

    baseline_df   = df.iloc[:BASELINE_DAYS].copy()
    monitoring_df = df.iloc[BASELINE_DAYS:].copy()

    pv       = build_pv(baseline_df)
    detector = ImprovedAnomalyDetector(pv)
    reports, dev_hist = [], []

    for idx, (_, row) in enumerate(monitoring_df.iterrows()):
        cur = {f: (float(row[f]) if f in row and not pd.isna(row[f])
                   else pv.to_dict()[f]) for f in BEHAVIORAL_FEATURES}
        rep, _ = detector.analyze(cur, dev_hist, idx + 1)
        reports.append(rep)
        dev_hist.append(rep.feature_deviations)

    s1 = detector.generate_final_prediction(
        scenario="batch", patient_id=uid,
        monitoring_days=len(monitoring_df),
    )

    latest = reports[-1] if reports else None
    s2_disorder, s2_conf, s2_score, s2_all, s2_label = \
        "unknown", "unknown", 0.0, {}, ""

    if latest:
        try:
            inp    = build_s1_input(detector=detector, baseline_df=baseline_df,
                                    s1_report=latest, timeseries_days=28)
            out    = System2Pipeline().classify(inp)
            s2_disorder = out.disorder
            s2_conf     = out.confidence.value
            s2_score    = float(out.score)
            s2_label    = out.label
            if out.classification:
                s2_all = out.classification.all_scores
        except Exception as e:
            s2_disorder = f"error:{e}"

    return {
        "uid": uid, "status": "processed", "n_days": n,
        "s1_anomaly":    s1.sustained_anomaly_detected,
        "s1_confidence": round(s1.confidence, 3),
        "s1_score":      round(s1.final_anomaly_score, 3),
        "s2_disorder":   s2_disorder,
        "s2_confidence": s2_conf,
        "s2_score":      round(s2_score, 4),
        "s2_all_scores": s2_all,
        "s2_label":      s2_label,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ─────────────────────────────────────────────────────────────────────────────

def s2_to_binary(disorder: str, positive_label: str) -> str:
    """Map S2 disorder name to binary class."""
    d = str(disorder).lower()
    if positive_label == "depressed":
        if d.startswith("depression"):
            return "depressed"
        return "not_depressed"
    else:  # schizophrenia
        if d.startswith("schizo"):
            return "schizophrenia"
        return "not_schizophrenia"


def compute_metrics(results: list, pos_gt: str, pos_pred: str) -> dict:
    tp = fp = tn = fn = 0
    top3_hits = top3_total = 0

    for r in results:
        gt   = r["ground_truth"]
        pred = s2_to_binary(r["s2_disorder"], pos_gt)
        pred_pos = pred == pos_pred

        if gt == pos_gt and pred_pos:      tp += 1
        elif gt != pos_gt and pred_pos:    fp += 1
        elif gt == pos_gt and not pred_pos: fn += 1
        else:                              tn += 1

        # Top-3
        if gt == pos_gt:
            top3_total += 1
            scores = r.get("s2_all_scores", {})
            if scores:
                top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
                pfx  = "depression" if pos_gt == "depressed" else "schizo"
                if any(k.startswith(pfx) for k, _ in top3):
                    top3_hits += 1
            elif r["s2_disorder"].startswith("depression" if pos_gt == "depressed" else "schizo"):
                top3_hits += 1

    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1    = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    acc   = (tp+tn)/(tp+fp+tn+fn) if (tp+fp+tn+fn) > 0 else 0
    spec  = tn/(tn+fp) if (tn+fp) > 0 else 0
    top3r = top3_hits/top3_total if top3_total > 0 else 0

    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                precision=round(prec,4), recall=round(rec,4),
                f1=round(f1,4), accuracy=round(acc,4),
                specificity=round(spec,4),
                top3_sensitivity=round(top3r,4),
                top3_hits=top3_hits, top3_total=top3_total)


def confusion_matrix_str(results, pos_gt, other_gt_labels):
    """Return formatted confusion matrix string."""
    all_labels = [pos_gt] + other_gt_labels
    preds = defaultdict(lambda: defaultdict(int))
    for r in results:
        gt   = r["ground_truth"]
        pred = s2_to_binary(r["s2_disorder"], pos_gt)
        preds[gt][pred] += 1

    pred_labels = list({s2_to_binary(r["s2_disorder"], pos_gt) for r in results})
    pred_labels = sorted(pred_labels)

    lines = []
    col_w = 16
    header = f"  {'GT \\ Pred':18s}" + "".join(f"{p:>{col_w}}" for p in pred_labels)
    lines.append(header)
    lines.append("  " + "─" * (18 + col_w * len(pred_labels)))
    for gt in all_labels:
        row = f"  {gt:18s}"
        for pred in pred_labels:
            row += f"{preds[gt][pred]:>{col_w}}"
        lines.append(row)
    return "\n".join(lines)


def print_section(title):
    print(f"\n{'─'*W}\n  {title}\n{'─'*W}")

def print_banner(title):
    print(f"\n{'═'*W}\n  {title}\n{'═'*W}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset processors
# ─────────────────────────────────────────────────────────────────────────────

def run_studentlife():
    print_banner("DATASET 1: StudentLife  (Depression vs. Healthy)")
    ext   = StudentLifeExtractor(STUDENTLIFE_PATH)
    phq9  = ext.load_phq9_labels()
    uids  = ext.get_student_ids()
    print(f"  Students found:   {len(uids)}")
    print(f"  PHQ-9 labels:     {len(phq9)}")

    results = []
    for i, uid in enumerate(uids):
        print(f"  [{i+1:2d}/{len(uids)}] {uid}...", end=" ", flush=True)
        try:
            df = ext.extract_student(uid)
        except Exception as e:
            print(f"FAILED: {e}"); results.append({"uid":uid,"status":"failed"}); continue
        if len(df) == 0:
            print("NO DATA"); results.append({"uid":uid,"status":"no_data"}); continue

        r = run_pipeline(uid, df)
        label = phq9.get(uid, {})
        post  = label.get("post_score", -1)
        r["phq9_post"] = post
        r["phq9_pre"]  = label.get("pre_score", -1)
        if post >= 10:   r["ground_truth"] = "depressed"
        elif post >= 5:  r["ground_truth"] = "mild"
        elif post >= 0:  r["ground_truth"] = "healthy"
        else:            r["ground_truth"] = "unknown"
        results.append(r)

        if r["status"] == "skipped":
            print(f"SKIPPED ({r.get('reason','')})")
        else:
            print(f"S1={'ANOM' if r['s1_anomaly'] else 'norm':4s} | "
                  f"S2={r['s2_disorder']:22s} | GT={r['ground_truth']}")

    processed = [r for r in results
                 if r["status"] == "processed" and r.get("ground_truth","unknown") != "unknown"]

    # Ground truth dist
    print_section("Ground Truth Distribution  (PHQ-9 post)")
    gt_dist = defaultdict(int)
    for r in processed: gt_dist[r["ground_truth"]] += 1
    for cls in ["depressed","mild","healthy"]:
        n = gt_dist[cls]
        print(f"  {cls:10s}  {n:3d}  {'█'*n}")

    # S1 metrics
    print_section("System 1  —  Binary Confusion Matrix  (Depressed vs. Not)")
    dep_gt  = [r for r in processed if r["ground_truth"] == "depressed"]
    hlth_gt = [r for r in processed if r["ground_truth"] == "healthy"]
    s1_tp = sum(1 for r in dep_gt  if r["s1_anomaly"])
    s1_fn = sum(1 for r in dep_gt  if not r["s1_anomaly"])
    s1_fp = sum(1 for r in hlth_gt if r["s1_anomaly"])
    s1_tn = sum(1 for r in hlth_gt if not r["s1_anomaly"])
    s1_sens  = s1_tp/(s1_tp+s1_fn)   if (s1_tp+s1_fn)>0 else 0
    s1_spec  = s1_tn/(s1_tn+s1_fp)   if (s1_tn+s1_fp)>0 else 0
    s1_prec  = s1_tp/(s1_tp+s1_fp)   if (s1_tp+s1_fp)>0 else 0
    s1_f1    = 2*s1_prec*s1_sens/(s1_prec+s1_sens) if (s1_prec+s1_sens)>0 else 0
    s1_acc   = (s1_tp+s1_tn)/len(dep_gt+hlth_gt) if (dep_gt+hlth_gt) else 0
    print(f"  {'':20s}  {'Pred: Anomaly':>14}  {'Pred: Normal':>12}")
    print(f"  {'GT: Depressed':20s}  {'TP = '+str(s1_tp):>14}  {'FN = '+str(s1_fn):>12}")
    print(f"  {'GT: Healthy':20s}  {'FP = '+str(s1_fp):>14}  {'TN = '+str(s1_tn):>12}")
    print(f"\n  Sensitivity (Recall): {s1_sens:.1%}  |  Specificity: {s1_spec:.1%}")
    print(f"  Precision:            {s1_prec:.1%}  |  F1-Score:    {s1_f1:.1%}")
    print(f"  Accuracy:             {s1_acc:.1%}")

    # S2 confusion
    print_section("System 2  —  Confusion Matrix  (Depressed | Mild | Healthy)")
    print(confusion_matrix_str(processed, "depressed", ["mild","healthy"]))

    # S2 binary metrics
    print_section("System 2  —  Binary Metrics  (Depressed vs. Not)")
    bm = compute_metrics(processed, "depressed", "depressed")
    print(f"  TP:{bm['tp']}  FP:{bm['fp']}  TN:{bm['tn']}  FN:{bm['fn']}")
    print(f"  Accuracy:              {bm['accuracy']:.1%}  ({bm['tp']+bm['tn']}/{len(processed)})")
    print(f"  Precision:             {bm['precision']:.1%}")
    print(f"  Recall (Sensitivity):  {bm['recall']:.1%}")
    print(f"  Specificity:           {bm['specificity']:.1%}")
    print(f"  F1-Score:              {bm['f1']:.1%}")
    print(f"  Top-3 Sensitivity:     {bm['top3_hits']}/{bm['top3_total']} = {bm['top3_sensitivity']:.1%}")

    # Per-student table
    print_section("Per-Student Results")
    print(f"  {'UID':6}  {'PHQ':>4}  {'GT':10}  {'S2 Disorder':24}  {'Conf':10}  {'OK':4}")
    print("  " + "─"*70)
    corr = 0
    for r in sorted(processed, key=lambda x: x["uid"]):
        pred = s2_to_binary(r["s2_disorder"], "depressed")
        ok   = "✅" if r["ground_truth"] == pred else "❌"
        if r["ground_truth"] == pred: corr += 1
        print(f"  {r['uid']:6}  {r['phq9_post']:4}  {r['ground_truth']:10}  "
              f"{r['s2_disorder']:24}  {r['s2_confidence']:10}  {ok}")
    print(f"\n  3-class student accuracy: {corr}/{len(processed)} = {corr/len(processed):.1%}")

    # Save
    out = {"dataset": "StudentLife", "n_processed": len(processed),
           "ground_truth_dist": dict(gt_dist),
           "s1": dict(tp=s1_tp,fp=s1_fp,tn=s1_tn,fn=s1_fn,
                      accuracy=round(s1_acc,4),sensitivity=round(s1_sens,4),
                      specificity=round(s1_spec,4),precision=round(s1_prec,4),f1=round(s1_f1,4)),
           "s2_binary": bm, "per_student": processed}
    with open(os.path.join(OUT_DIR,"studentlife_metrics.json"),"w") as f:
        json.dump(out, f, indent=2, default=str)
    pd.DataFrame(processed).to_csv(os.path.join(OUT_DIR,"studentlife_metrics.csv"), index=False)
    return processed


def run_crosscheck():
    print_banner("DATASET 2: CrossCheck  (Schizophrenia vs. Healthy)")
    ext   = SchzExtractor(CROSSCHECK_PATH)
    ema   = ext.load_ema_labels()
    uids  = ext.get_patient_ids()
    print(f"  Patients found:   {len(uids)}")

    schz_count = sum(1 for v in ema.values() if v["schz_flag"])
    print(f"  Schz flag True:   {schz_count}")
    print(f"  Non-schz:         {len(ema) - schz_count}")

    results = []
    for i, uid in enumerate(uids):
        print(f"  [{i+1:2d}/{len(uids)}] {uid}...", end=" ", flush=True)
        try:
            df = ext.extract_patient(uid)
        except Exception as e:
            print(f"FAILED:{e}"); results.append({"uid":uid,"status":"failed"}); continue
        if len(df) == 0:
            print("NO DATA"); results.append({"uid":uid,"status":"no_data"}); continue

        r = run_pipeline(uid, df)
        lbl = ema.get(uid, {})
        r["schz_flag"]        = lbl.get("schz_flag", False)
        r["depression_flag"]  = lbl.get("depression_flag", False)
        r["avg_voices"]       = lbl.get("avg_voices", 0)
        r["avg_ema_score"]    = lbl.get("avg_ema_score", 0)
        r["ground_truth"]     = "schizophrenia" if r["schz_flag"] else "healthy"
        results.append(r)

        if r["status"] == "skipped":
            print(f"SKIPPED ({r.get('reason','')})")
        else:
            print(f"S1={'ANOM' if r['s1_anomaly'] else 'norm':4s} | "
                  f"S2={r['s2_disorder']:22s} | GT={r['ground_truth']}")

    processed = [r for r in results if r["status"] == "processed"]

    # Ground truth dist
    print_section("Ground Truth Distribution  (EMA schz_flag)")
    gt_dist = defaultdict(int)
    for r in processed: gt_dist[r["ground_truth"]] += 1
    for cls in ["schizophrenia","healthy"]:
        n = gt_dist[cls]
        print(f"  {cls:15s}  {n:3d}  {'█'*n}")

    # S1 confusion
    schz_gt  = [r for r in processed if r["ground_truth"] == "schizophrenia"]
    hlth_gt  = [r for r in processed if r["ground_truth"] == "healthy"]
    print_section("System 1  —  Binary Confusion Matrix  (Schizophrenia vs. Not)")
    s1_tp = sum(1 for r in schz_gt if r["s1_anomaly"])
    s1_fn = sum(1 for r in schz_gt if not r["s1_anomaly"])
    s1_fp = sum(1 for r in hlth_gt if r["s1_anomaly"])
    s1_tn = sum(1 for r in hlth_gt if not r["s1_anomaly"])
    s1_sens  = s1_tp/(s1_tp+s1_fn)   if (s1_tp+s1_fn)>0 else 0
    s1_spec  = s1_tn/(s1_tn+s1_fp)   if (s1_tn+s1_fp)>0 else 0
    s1_prec  = s1_tp/(s1_tp+s1_fp)   if (s1_tp+s1_fp)>0 else 0
    s1_f1    = 2*s1_prec*s1_sens/(s1_prec+s1_sens) if (s1_prec+s1_sens)>0 else 0
    s1_acc   = (s1_tp+s1_tn)/len(processed) if processed else 0
    print(f"  {'':22s}  {'Pred: Anomaly':>14}  {'Pred: Normal':>12}")
    print(f"  {'GT: Schizophrenia':22s}  {'TP = '+str(s1_tp):>14}  {'FN = '+str(s1_fn):>12}")
    print(f"  {'GT: Healthy':22s}  {'FP = '+str(s1_fp):>14}  {'TN = '+str(s1_tn):>12}")
    print(f"\n  Sensitivity (Recall): {s1_sens:.1%}  |  Specificity: {s1_spec:.1%}")
    print(f"  Precision:            {s1_prec:.1%}  |  F1-Score:    {s1_f1:.1%}")
    print(f"  Accuracy:             {s1_acc:.1%}")

    # S2 confusion
    print_section("System 2  —  Confusion Matrix  (Schizophrenia | Healthy)")
    print(confusion_matrix_str(processed, "schizophrenia", ["healthy"]))

    # S2 binary metrics
    print_section("System 2  —  Binary Metrics  (Schizophrenia vs. Not)")
    bm = compute_metrics(processed, "schizophrenia", "schizophrenia")
    print(f"  TP:{bm['tp']}  FP:{bm['fp']}  TN:{bm['tn']}  FN:{bm['fn']}")
    print(f"  Accuracy:              {bm['accuracy']:.1%}  ({bm['tp']+bm['tn']}/{len(processed)})")
    print(f"  Precision:             {bm['precision']:.1%}")
    print(f"  Recall (Sensitivity):  {bm['recall']:.1%}")
    print(f"  Specificity:           {bm['specificity']:.1%}")
    print(f"  F1-Score:              {bm['f1']:.1%}")
    print(f"  Top-3 Sensitivity:     {bm['top3_hits']}/{bm['top3_total']} = {bm['top3_sensitivity']:.1%}")

    # Per-patient table
    print_section("Per-Patient Results")
    print(f"  {'UID':10}  {'GT':15}  {'S2 Disorder':24}  {'Conf':10}  {'OK':4}")
    print("  " + "─"*70)
    corr = 0
    for r in sorted(processed, key=lambda x: x["uid"]):
        pred = s2_to_binary(r["s2_disorder"], "schizophrenia")
        ok   = "✅" if r["ground_truth"] == pred else "❌"
        if r["ground_truth"] == pred: corr += 1
        print(f"  {r['uid']:10}  {r['ground_truth']:15}  "
              f"{r['s2_disorder']:24}  {r['s2_confidence']:10}  {ok}")
    print(f"\n  Patient-level accuracy: {corr}/{len(processed)} = {corr/len(processed):.1%}")

    # Save
    out = {"dataset": "CrossCheck", "n_processed": len(processed),
           "ground_truth_dist": dict(gt_dist),
           "s1": dict(tp=s1_tp,fp=s1_fp,tn=s1_tn,fn=s1_fn,
                      accuracy=round(s1_acc,4),sensitivity=round(s1_sens,4),
                      specificity=round(s1_spec,4),precision=round(s1_prec,4),f1=round(s1_f1,4)),
           "s2_binary": bm, "per_patient": processed}
    with open(os.path.join(OUT_DIR,"crosscheck_metrics.json"),"w") as f:
        json.dump(out, f, indent=2, default=str)
    pd.DataFrame(processed).to_csv(os.path.join(OUT_DIR,"crosscheck_metrics.csv"), index=False)
    return processed


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sl_results = run_studentlife()
    cc_results = run_crosscheck()

    print_banner("ALL DONE — Saved to system2/data/")
    print(f"  studentlife_metrics.json  |  studentlife_metrics.csv")
    print(f"  crosscheck_metrics.json   |  crosscheck_metrics.csv")
