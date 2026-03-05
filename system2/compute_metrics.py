"""
compute_metrics.py
==================
Runs all StudentLife students through the full S1 → S2 pipeline and
computes exact classification metrics:

  • Confusion matrix  (Healthy / Mild / Depressed  ×  S2 prediction)
  • Accuracy, Precision, Recall, F1  (binary: depressed vs. not)
  • Top-3 sensitivity (clinical standard)
  • Per-student breakdown table

Run from project root:
    .venv\\Scripts\\python.exe system2\\compute_metrics.py
"""

from __future__ import annotations

import os
import sys
import json
import warnings
import textwrap
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Make sure project root is on the path ──────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "system2"))

DATASET_PATH = r"C:\Users\SRIRAM\Downloads\dataset"

# ── Imports ────────────────────────────────────────────────────────────────
from studentlife_extractor import StudentLifeExtractor
from system1 import PersonalityVector, ImprovedAnomalyDetector
from s1_s2_adapter import build_s1_input
from system2.pipeline import System2Pipeline
from system2.config import POPULATION_NORMS, BEHAVIORAL_FEATURES

BASELINE_DAYS      = 28
MIN_MONITORING_DAYS = 7


# ── Helpers ────────────────────────────────────────────────────────────────

def build_personality_vector(df: pd.DataFrame) -> PersonalityVector:
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


def phq9_to_class(score: int) -> str:
    if score >= 10:
        return "depressed"
    if score >= 5:
        return "mild"
    return "healthy"


def process_student(uid: str, df: pd.DataFrame, phq9: dict) -> dict:
    n = len(df)
    if n < BASELINE_DAYS + MIN_MONITORING_DAYS:
        return {"uid": uid, "status": "skipped", "reason": f"only {n} days"}

    baseline_df  = df.iloc[:BASELINE_DAYS].copy()
    monitoring_df = df.iloc[BASELINE_DAYS:].copy()

    pv       = build_personality_vector(baseline_df)
    detector = ImprovedAnomalyDetector(pv)

    reports          = []
    dev_history      = []

    for idx, (_, row) in enumerate(monitoring_df.iterrows()):
        current = {}
        for feat in BEHAVIORAL_FEATURES:
            val = row.get(feat, np.nan)
            current[feat] = float(val) if not pd.isna(val) else pv.to_dict()[feat]

        report, _ = detector.analyze(current, dev_history, idx + 1)
        reports.append(report)
        dev_history.append(report.feature_deviations)

    s1_pred = detector.generate_final_prediction(
        scenario="studentlife", patient_id=uid,
        monitoring_days=len(monitoring_df),
    )

    latest = reports[-1] if reports else None
    s2_disorder    = "unknown"
    s2_confidence  = "unknown"
    s2_score       = 0.0
    s2_all_scores  = {}
    s2_label       = ""

    if latest:
        try:
            s1_input = build_s1_input(
                detector=detector,
                baseline_df=baseline_df,
                s1_report=latest,
                timeseries_days=28,
            )
            pipeline  = System2Pipeline()
            s2_output = pipeline.classify(s1_input)
            s2_disorder   = s2_output.disorder
            s2_confidence = s2_output.confidence.value
            s2_score      = s2_output.score
            s2_label      = s2_output.label
            if s2_output.classification:
                s2_all_scores = s2_output.classification.all_scores
        except Exception as e:
            s2_disorder = f"error:{e}"

    pre  = phq9.get("pre_score",  -1)
    post = phq9.get("post_score", -1)
    gt   = phq9_to_class(post)

    return {
        "uid":               uid,
        "status":            "processed",
        "n_days":            n,
        "monitoring_days":   len(monitoring_df),
        # S1
        "s1_anomaly":        s1_pred.sustained_anomaly_detected,
        "s1_confidence":     round(s1_pred.confidence, 3),
        "s1_score":          round(s1_pred.final_anomaly_score, 3),
        # S2
        "s2_disorder":       s2_disorder,
        "s2_confidence":     s2_confidence,
        "s2_score":          round(float(s2_score), 4),
        "s2_all_scores":     s2_all_scores,
        "s2_label":          s2_label,
        # Ground truth
        "phq9_pre":          pre,
        "phq9_post":         post,
        "ground_truth":      gt,
    }


# ── Metrics helpers ────────────────────────────────────────────────────────

def confusion_matrix_3class(results):
    """
    3-class confusion matrix: rows = ground truth, cols = S2 prediction.
    Classes: depressed, mild, healthy
    S2 mapping:
        depression* → depressed
        life_event / healthy* / situational* / unclassified → healthy
        everything else → mild  (bipolar, anxiety, bpd, schizo)
    """
    classes = ["depressed", "mild", "healthy"]

    def s2_to_class(d: str) -> str:
        d = str(d).lower()
        if d.startswith("depression") or d.startswith("schizo"):
            return "depressed"
        if d.startswith("healthy") or d in ("life_event", "situational_stress"):
            return "healthy"
        return "mild"

    cm = defaultdict(lambda: defaultdict(int))
    for r in results:
        gt  = r["ground_truth"]
        pred = s2_to_class(r["s2_disorder"])
        cm[gt][pred] += 1
    return cm, classes, s2_to_class


def binary_metrics(results, s2_to_class):
    """
    Binary: depressed (positive) vs. not-depressed (negative).
    Positive class = 'depressed' ground truth.
    """
    tp = fp = tn = fn = 0
    for r in results:
        gt   = r["ground_truth"]
        pred = s2_to_class(r["s2_disorder"])
        if gt == "depressed" and pred == "depressed":
            tp += 1
        elif gt != "depressed" and pred == "depressed":
            fp += 1
        elif gt == "depressed" and pred != "depressed":
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy  = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                precision=precision, recall=recall, f1=f1,
                accuracy=accuracy, specificity=specificity)


def top3_sensitivity(results):
    """
    Top-3 clinical standard: depression appears in top-3 scored disorders.
    """
    depressed = [r for r in results if r["ground_truth"] == "depressed"]
    if not depressed:
        return 0.0, 0, 0

    hits = 0
    for r in depressed:
        scores = r.get("s2_all_scores", {})
        if scores:
            top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
            if any(k.startswith("depression") for k, _ in top3):
                hits += 1
        elif r["s2_disorder"].startswith("depression"):
            hits += 1

    return hits / len(depressed), hits, len(depressed)


# ── Print helpers ──────────────────────────────────────────────────────────

W = 72

def banner(title: str):
    print("\n" + "═" * W)
    print(f"  {title}")
    print("═" * W)

def section(title: str):
    print(f"\n{'─' * W}")
    print(f"  {title}")
    print("─" * W)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    banner("StudentLife  —  Full S1 + S2 Metrics   (2026-03-05)")

    ext = StudentLifeExtractor(DATASET_PATH)
    phq9_labels  = ext.load_phq9_labels()
    student_ids  = ext.get_student_ids()

    print(f"  Dataset:          {DATASET_PATH}")
    print(f"  Students found:   {len(student_ids)}")
    print(f"  PHQ-9 labels:     {len(phq9_labels)}")
    print(f"  Baseline window:  {BASELINE_DAYS} days")
    print(f"  Min monitoring:   {MIN_MONITORING_DAYS} days")

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
        r    = process_student(uid, df, phq9)
        results.append(r)

        if r["status"] == "skipped":
            print(f"SKIPPED ({r['reason']})")
        else:
            s1f = "ANOMALY" if r["s1_anomaly"] else "normal"
            print(f"S1={s1f:7s} | S2={r['s2_disorder']:22s} | GT={r['ground_truth']}")

    # ── Filter to processed rows with valid PHQ-9 ───────────────────────
    processed = [r for r in results if r["status"] == "processed"
                 and r.get("phq9_post", -1) >= 0]

    skipped = len(results) - len(processed)

    banner("RESULTS")
    print(f"  Total students:   {len(results)}")
    print(f"  Processed:        {len(processed)}")
    print(f"  Skipped/failed:   {skipped}")

    if not processed:
        print("\n  No processed results — check dataset path.")
        return

    # ── Ground truth distribution ────────────────────────────────────────
    section("Ground Truth Distribution  (PHQ-9 post)")
    gt_counts = defaultdict(int)
    for r in processed:
        gt_counts[r["ground_truth"]] += 1
    for cls in ["depressed", "mild", "healthy"]:
        n = gt_counts[cls]
        bar = "█" * n
        print(f"  {cls:10s}  {n:3d}  {bar}")

    # ── S1 performance ───────────────────────────────────────────────────
    section("System 1  —  Anomaly Detection")
    depressed_gt = [r for r in processed if r["ground_truth"] == "depressed"]
    healthy_gt   = [r for r in processed if r["ground_truth"] == "healthy"]
    mild_gt      = [r for r in processed if r["ground_truth"] == "mild"]

    s1_tp = sum(1 for r in depressed_gt if r["s1_anomaly"])
    s1_fp = sum(1 for r in healthy_gt   if r["s1_anomaly"])
    s1_tn = sum(1 for r in healthy_gt   if not r["s1_anomaly"])
    s1_fn = sum(1 for r in depressed_gt if not r["s1_anomaly"])

    s1_sens = s1_tp / len(depressed_gt) if depressed_gt else 0
    s1_spec = s1_tn / len(healthy_gt)   if healthy_gt   else 0
    s1_acc  = (s1_tp + s1_tn) / len(processed)
    s1_prec = s1_tp / (s1_tp + s1_fp) if (s1_tp + s1_fp) > 0 else 0
    s1_f1   = 2 * s1_prec * s1_sens / (s1_prec + s1_sens) if (s1_prec + s1_sens) > 0 else 0

    print(f"  Sensitivity (depressed caught):  {s1_tp}/{len(depressed_gt)} = {s1_sens:.1%}")
    print(f"  Specificity (healthy correct):   {s1_tn}/{len(healthy_gt)}  = {s1_spec:.1%}")
    print(f"  Overall Accuracy:                {s1_acc:.1%}")
    print(f"  Precision:                       {s1_prec:.1%}")
    print(f"  F1-Score:                        {s1_f1:.1%}")
    print(f"\n  S1 Confusion Matrix  (positive = depressed):")
    print(f"  {'':20s}  Pred: Anomaly   Pred: Normal")
    print(f"  {'GT: Depressed':20s}  TP = {s1_tp:3d}        FN = {s1_fn:3d}")
    print(f"  {'GT: Healthy':20s}  FP = {s1_fp:3d}        TN = {s1_tn:3d}")

    # ── S2 3-class confusion matrix ───────────────────────────────────────
    section("System 2  —  3-Class Confusion Matrix")
    cm, classes, s2_to_class = confusion_matrix_3class(processed)

    # Header
    col_w = 12
    print(f"  {'GT \\ Pred':16s}", end="")
    for c in classes:
        print(f"  {c:>{col_w}}", end="")
    print()
    print("  " + "─" * (16 + (col_w + 2) * len(classes)))

    for gt_cls in classes:
        print(f"  {gt_cls:16s}", end="")
        for pred_cls in classes:
            val = cm[gt_cls][pred_cls]
            print(f"  {val:>{col_w}}", end="")
        print()

    # ── S2 binary metrics ─────────────────────────────────────────────────
    section("System 2  —  Binary Classification Metrics  (Depressed vs. Not)")
    bm = binary_metrics(processed, s2_to_class)

    print(f"  TP (depressed  → depressed):   {bm['tp']}")
    print(f"  FP (not-dep    → depressed):   {bm['fp']}")
    print(f"  TN (not-dep    → not-dep):     {bm['tn']}")
    print(f"  FN (depressed  → not-dep):     {bm['fn']}")
    print()
    print(f"  Accuracy:                      {bm['accuracy']:.1%}  ({bm['tp']+bm['tn']}/{len(processed)})")
    print(f"  Precision:                     {bm['precision']:.1%}")
    print(f"  Recall (Sensitivity):          {bm['recall']:.1%}")
    print(f"  Specificity:                   {bm['specificity']:.1%}")
    print(f"  F1-Score:                      {bm['f1']:.1%}")

    # Top-3 sensitivity
    top3_rate, top3_hits, top3_total = top3_sensitivity(processed)
    print(f"\n  Top-3 Sensitivity (clinical):  {top3_hits}/{top3_total} = {top3_rate:.1%}")

    # ── S2 disorder distribution ──────────────────────────────────────────
    section("System 2  —  Predicted Disorder Distribution")
    dis_counts = defaultdict(int)
    for r in processed:
        dis_counts[r["s2_disorder"]] += 1
    for dis, cnt in sorted(dis_counts.items(), key=lambda x: -x[1]):
        bar = "█" * cnt
        print(f"  {dis:30s}  {cnt:3d}  {bar}")

    # ── Per-student table ─────────────────────────────────────────────────
    section("Per-Student Breakdown")
    hdr = f"  {'UID':6s}  {'PHQ':>4s}  {'GT':10s}  {'S2 Disorder':22s}  {'S2 Conf':10s}  {'Correct':7s}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    correct_count = 0
    for r in sorted(processed, key=lambda x: x["uid"]):
        gt   = r["ground_truth"]
        pred = s2_to_class(r["s2_disorder"])
        ok   = "✅" if gt == pred else "❌"
        if gt == pred:
            correct_count += 1
        phq  = r["phq9_post"]
        print(f"  {r['uid']:6s}  {phq:4d}  {gt:10s}  {r['s2_disorder']:22s}  {r['s2_confidence']:10s}  {ok}")

    print(f"\n  Student-level accuracy: {correct_count}/{len(processed)} = {correct_count/len(processed):.1%}")

    # ── Save JSON ──────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "studentlife_metrics.json")
    summary = {
        "total_processed": len(processed),
        "ground_truth_distribution": dict(gt_counts),
        "s1": {
            "tp": s1_tp, "fp": s1_fp, "tn": s1_tn, "fn": s1_fn,
            "sensitivity": round(s1_sens, 4),
            "specificity": round(s1_spec, 4),
            "accuracy":    round(s1_acc,  4),
            "precision":   round(s1_prec, 4),
            "f1":          round(s1_f1,   4),
        },
        "s2_binary": {k: round(v, 4) if isinstance(v, float) else v
                      for k, v in bm.items()},
        "s2_top3_sensitivity": round(top3_rate, 4),
        "confusion_matrix_3class": {
            gt: dict(preds) for gt, preds in cm.items()
        },
        "per_student": processed,
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    csv_path = os.path.join(out_dir, "studentlife_metrics.csv")
    pd.DataFrame(processed).to_csv(csv_path, index=False)

    banner("DONE")
    print(f"  JSON saved: {json_path}")
    print(f"  CSV  saved: {csv_path}")
    print()


if __name__ == "__main__":
    main()
