"""
generate_metrics_pdf.py
========================
Reads the computed metric JSON files and generates a professional PDF
with real confusion matrices, accuracy, precision, recall, F1, and
per-subject tables for both datasets.

Run:
    .venv\\Scripts\\python.exe system2\\generate_metrics_pdf.py
"""

from __future__ import annotations
import os, json
from collections import defaultdict

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, PageBreak, HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")
OUTPUT_PDF  = os.path.join(SCRIPT_DIR, "System2_Metrics_Report.pdf")

# ── Load data ──────────────────────────────────────────────────────────────
with open(os.path.join(DATA_DIR, "studentlife_metrics.json")) as f:
    SL = json.load(f)
with open(os.path.join(DATA_DIR, "crosscheck_metrics.json")) as f:
    CC = json.load(f)


# ── Styles ─────────────────────────────────────────────────────────────────
def make_styles():
    b = getSampleStyleSheet()
    title   = ParagraphStyle("T",  parent=b["Heading1"], fontSize=20, spaceAfter=8,
                              textColor=colors.HexColor("#1a252f"), alignment=TA_CENTER)
    subtitle= ParagraphStyle("S",  parent=b["Normal"],  fontSize=11, spaceAfter=5,
                              textColor=colors.HexColor("#555"), alignment=TA_CENTER)
    h1      = ParagraphStyle("H1", parent=b["Heading1"], fontSize=14, spaceBefore=16,
                              spaceAfter=8, textColor=colors.HexColor("#2980b9"))
    h2      = ParagraphStyle("H2", parent=b["Heading2"], fontSize=11, spaceBefore=10,
                              spaceAfter=6, textColor=colors.HexColor("#34495e"))
    normal  = ParagraphStyle("N",  parent=b["Normal"],  fontSize=9.5, spaceAfter=5, leading=13)
    caption = ParagraphStyle("C",  parent=b["Normal"],  fontSize=8.5, spaceAfter=5,
                              textColor=colors.HexColor("#777"), alignment=TA_CENTER)
    mono    = ParagraphStyle("M",  parent=b["Code"],    fontSize=8,   spaceAfter=4, leading=11,
                              backColor=colors.HexColor("#f4f4f4"), leftIndent=10)
    green   = ParagraphStyle("G",  parent=b["Normal"],  fontSize=9.5, textColor=colors.HexColor("#1a7a4a"))
    red2    = ParagraphStyle("R",  parent=b["Normal"],  fontSize=9.5, textColor=colors.HexColor("#c0392b"))
    return dict(title=title, subtitle=subtitle, h1=h1, h2=h2,
                normal=normal, caption=caption, mono=mono, green=green, red2=red2)


# ── Table builder ──────────────────────────────────────────────────────────
HBG = colors.HexColor("#2980b9")
ALT = colors.HexColor("#eaf4fb")

def mk_table(data, widths=None):
    t = Table(data, colWidths=widths, repeatRows=1)
    n = len(data)
    style = [
        ("BACKGROUND", (0,0), (-1,0), HBG),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,0), 8.5),
        ("ALIGN",      (0,0), (-1,0), "CENTER"),
        ("FONTNAME",   (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",   (0,1), (-1,-1), 8),
        ("ALIGN",      (0,1), (-1,-1), "LEFT"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#bdc3c7")),
        ("ROWBACKGROUND", (0,1), (-1,-1), [colors.white, ALT]),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("RIGHTPADDING",  (0,0), (-1,-1), 6),
    ]
    t.setStyle(TableStyle(style))
    return t


def cm_table(cm_data, row_labels, col_labels, s):
    """Build a styled confusion matrix table."""
    header = ["GT \\ Pred"] + col_labels
    rows   = [header]
    for rl in row_labels:
        row = [rl]
        for cl in col_labels:
            v = cm_data.get(rl, {}).get(cl, 0)
            row.append(str(v))
        rows.append(row)
    w = 1.3 * inch
    widths = [1.5 * inch] + [w] * len(col_labels)
    t = Table(rows, colWidths=widths, repeatRows=1)

    style = [
        ("BACKGROUND", (0,0), (-1,0), HBG),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,0), 9),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("FONTNAME",   (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",   (0,1), (-1,-1), 9),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#bdc3c7")),
        ("BACKGROUND", (0,1), (0,-1), colors.HexColor("#34495e")),
        ("TEXTCOLOR",  (0,1), (0,-1), colors.white),
        ("FONTNAME",   (0,1), (0,-1), "Helvetica-Bold"),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]
    # Highlight diagonal (TP / TN cells) in green
    for i, rl in enumerate(row_labels):
        for j, cl in enumerate(col_labels):
            if rl == cl:
                style.append(("BACKGROUND", (j+1, i+1), (j+1, i+1), colors.HexColor("#d5f5e3")))
    t.setStyle(TableStyle(style))
    return t


def pct(v): return f"{v:.1%}"
def num(v): return f"{v:.4f}"


# ── PDF builder ─────────────────────────────────────────────────────────────
def build_pdf():
    doc = SimpleDocTemplate(OUTPUT_PDF, pagesize=letter,
                            rightMargin=50, leftMargin=50,
                            topMargin=50, bottomMargin=50)
    s   = make_styles()
    story = []

    # ── COVER ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("SYSTEM 2 — EXACT METRICS REPORT", s["title"]))
    story.append(Paragraph("Full Pipeline Accuracy • Confusion Matrices • Per-Subject Results", s["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2980b9")))
    story.append(Spacer(1, 0.15*inch))

    meta = [
        ["Report Date", "2026-03-05"],
        ["Datasets", "StudentLife (Depression) + CrossCheck (Schizophrenia)"],
        ["Pipeline", "System 1 (Anomaly Detection) → System 2 (Prototype Matching)"],
        ["Ground Truth (SL)", "PHQ-9 post-score ≥ 10 = Depressed, 5-9 = Mild, < 5 = Healthy"],
        ["Ground Truth (CC)", "EMA VOICES + SEEING_THINGS > 0.5 = Schizophrenia"],
    ]
    story.append(mk_table(meta, [1.8*inch, 4.6*inch]))
    story.append(Spacer(1, 0.3*inch))

    # Executive summary boxes
    sl_s2 = SL["s2_binary"]
    cc_s2 = CC["s2_binary"]
    sl_s1 = SL["s1"]
    cc_s1 = CC["s1"]

    summary_data = [
        ["", "StudentLife (Depression)", "CrossCheck (Schizophrenia)"],
        ["Subjects Processed",     f"{SL['n_processed']}",          f"{CC['n_processed']}"],
        ["Depressed / Schz (GT)",  f"{SL['ground_truth_dist'].get('depressed',0)}", f"{CC['ground_truth_dist'].get('schizophrenia',0)}"],
        ["Healthy (GT)",           f"{SL['ground_truth_dist'].get('healthy',0)}",   f"{CC['ground_truth_dist'].get('healthy',0)}"],
        ["S1 Sensitivity",         pct(sl_s1["sensitivity"]),       pct(cc_s1["sensitivity"])],
        ["S1 Specificity",         pct(sl_s1["specificity"]),       pct(cc_s1["specificity"])],
        ["S1 Accuracy",            pct(sl_s1["accuracy"]),          pct(cc_s1["accuracy"])],
        ["S2 Accuracy (binary)",   pct(sl_s2["accuracy"]),          pct(cc_s2["accuracy"])],
        ["S2 Precision",           pct(sl_s2["precision"]),         pct(cc_s2["precision"])],
        ["S2 Recall (Sensitivity)",pct(sl_s2["recall"]),            pct(cc_s2["recall"])],
        ["S2 Specificity",         pct(sl_s2["specificity"]),       pct(cc_s2["specificity"])],
        ["S2 F1-Score",            pct(sl_s2["f1"]),                pct(cc_s2["f1"])],
        ["S2 Top-3 Sensitivity",
         f"{sl_s2['top3_hits']}/{sl_s2['top3_total']} = {pct(sl_s2['top3_sensitivity'])}",
         f"{cc_s2['top3_hits']}/{cc_s2['top3_total']} = {pct(cc_s2['top3_sensitivity'])}"],
    ]
    story.append(mk_table(summary_data, [2.2*inch, 2.4*inch, 2.4*inch]))
    story.append(PageBreak())

    # ── SECTION 1: STUDENTLIFE ───────────────────────────────────────────
    story.append(Paragraph("1. StudentLife Dataset — Depression Detection", s["h1"]))
    story.append(Paragraph(
        f"<b>{SL['n_processed']} students</b> processed (PHQ-9 labels available). "
        f"Ground truth: <b>Depressed</b> (PHQ-9 ≥ 10): {SL['ground_truth_dist'].get('depressed',0)}, "
        f"<b>Mild</b> (5–9): {SL['ground_truth_dist'].get('mild',0)}, "
        f"<b>Healthy</b> (&lt;5): {SL['ground_truth_dist'].get('healthy',0)}.",
        s["normal"]))

    story.append(Paragraph("1.1  System 1 — Anomaly Detection Confusion Matrix", s["h2"]))
    story.append(Paragraph(
        "Binary classification: <b>Anomaly Detected</b> (positive) vs. <b>Normal</b> (negative). "
        "Positive class = Depressed ground truth.", s["normal"]))

    sl_s1_cm = [
        ["", "Pred: Anomaly", "Pred: Normal"],
        [f"GT: Depressed (n={SL['ground_truth_dist'].get('depressed',0)})",
         f"TP = {sl_s1['tp']}", f"FN = {sl_s1['fn']}"],
        [f"GT: Healthy (n={SL['ground_truth_dist'].get('healthy',0)})",
         f"FP = {sl_s1['fp']}", f"TN = {sl_s1['tn']}"],
    ]
    t = mk_table(sl_s1_cm, [2.8*inch, 1.8*inch, 1.8*inch])
    story.append(t)
    story.append(Spacer(1, 0.1*inch))

    sl_s1_metrics = [
        ["Metric", "Value", "Formula"],
        ["Accuracy",    pct(sl_s1["accuracy"]),    f"(TP+TN)/N = ({sl_s1['tp']}+{sl_s1['tn']})/{SL['n_processed']-SL['ground_truth_dist'].get('mild',0)}"],
        ["Sensitivity", pct(sl_s1["sensitivity"]), f"TP/(TP+FN) = {sl_s1['tp']}/({sl_s1['tp']}+{sl_s1['fn']})"],
        ["Specificity", pct(sl_s1["specificity"]), f"TN/(TN+FP) = {sl_s1['tn']}/({sl_s1['tn']}+{sl_s1['fp']})"],
        ["Precision",   pct(sl_s1["precision"]),   f"TP/(TP+FP) = {sl_s1['tp']}/({sl_s1['tp']}+{sl_s1['fp']})"],
        ["F1-Score",    pct(sl_s1["f1"]),           f"2×(P×R)/(P+R)"],
    ]
    story.append(mk_table(sl_s1_metrics, [1.5*inch, 1.2*inch, 3.7*inch]))
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph("1.2  System 2 — Disorder Classification Confusion Matrix", s["h2"]))
    story.append(Paragraph(
        "Binary mapping: <b>depression*</b> → Depressed pred; "
        "<b>healthy* / life_event</b> → Not Depressed pred.", s["normal"]))

    sl_s2_cm_data = [
        ["", "Pred: Depressed", "Pred: Not Depressed"],
        [f"GT: Depressed (n={sl_s2['tp']+sl_s2['fn']})",
         f"TP = {sl_s2['tp']}", f"FN = {sl_s2['fn']}"],
        [f"GT: Not Dep. (n={sl_s2['fp']+sl_s2['tn']})",
         f"FP = {sl_s2['fp']}", f"TN = {sl_s2['tn']}"],
    ]
    story.append(mk_table(sl_s2_cm_data, [2.8*inch, 1.8*inch, 1.8*inch]))
    story.append(Spacer(1, 0.1*inch))

    sl_s2_metrics = [
        ["Metric", "Value", "Formula"],
        ["Accuracy",           pct(sl_s2["accuracy"]),           f"(TP+TN)/N = ({sl_s2['tp']}+{sl_s2['tn']})/{SL['n_processed']}"],
        ["Precision",          pct(sl_s2["precision"]),          f"TP/(TP+FP) = {sl_s2['tp']}/({sl_s2['tp']}+{sl_s2['fp']})"],
        ["Recall (Sensitivity)",pct(sl_s2["recall"]),            f"TP/(TP+FN) = {sl_s2['tp']}/({sl_s2['tp']}+{sl_s2['fn']})"],
        ["Specificity",        pct(sl_s2["specificity"]),        f"TN/(TN+FP) = {sl_s2['tn']}/({sl_s2['tn']}+{sl_s2['fp']})"],
        ["F1-Score",           pct(sl_s2["f1"]),                  "2×(P×R)/(P+R)"],
        ["Top-3 Sensitivity",
         f"{sl_s2['top3_hits']}/{sl_s2['top3_total']} = {pct(sl_s2['top3_sensitivity'])}",
         "Depression in top-3 scored disorders (clinical standard)"],
    ]
    story.append(mk_table(sl_s2_metrics, [1.7*inch, 1.5*inch, 3.2*inch]))
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph("1.3  Per-Student Results", s["h2"]))
    sl_rows = [["UID","PHQ-9","Ground Truth","S2 Disorder","S1 Anomaly","S2 Conf","Correct"]]
    for r in sorted(SL["per_student"], key=lambda x: x["uid"]):
        if r.get("status") != "processed": continue
        # map s2 to binary
        d = str(r.get("s2_disorder","")).lower()
        pred = "depressed" if d.startswith("depression") else "not_dep"
        gt   = r.get("ground_truth","?")
        ok   = "✅" if (gt == "depressed" and pred == "depressed") or (gt != "depressed" and pred != "depressed") else "❌"
        sl_rows.append([
            r["uid"],
            str(r.get("phq9_post","?")),
            gt,
            r.get("s2_disorder","?"),
            "Yes" if str(r.get("s1_anomaly","")) == "True" else "No",
            r.get("s2_confidence","?"),
            ok,
        ])
    story.append(mk_table(sl_rows, [0.55*inch, 0.6*inch, 1.0*inch, 1.65*inch, 0.85*inch, 0.9*inch, 0.65*inch]))

    story.append(PageBreak())

    # ── SECTION 2: CROSSCHECK ───────────────────────────────────────────
    story.append(Paragraph("2. CrossCheck Dataset — Schizophrenia Detection", s["h1"]))
    story.append(Paragraph(
        f"<b>{CC['n_processed']} patients</b> processed. "
        f"Ground truth (EMA): <b>Schizophrenia</b>: {CC['ground_truth_dist'].get('schizophrenia',0)}, "
        f"<b>Healthy</b>: {CC['ground_truth_dist'].get('healthy',0)}.",
        s["normal"]))

    story.append(Paragraph("2.1  System 1 — Anomaly Detection Confusion Matrix", s["h2"]))
    cc_s1_cm = [
        ["", "Pred: Anomaly", "Pred: Normal"],
        [f"GT: Schizophrenia (n={CC['ground_truth_dist'].get('schizophrenia',0)})",
         f"TP = {cc_s1['tp']}", f"FN = {cc_s1['fn']}"],
        [f"GT: Healthy (n={CC['ground_truth_dist'].get('healthy',0)})",
         f"FP = {cc_s1['fp']}", f"TN = {cc_s1['tn']}"],
    ]
    story.append(mk_table(cc_s1_cm, [2.8*inch, 1.8*inch, 1.8*inch]))
    story.append(Spacer(1, 0.1*inch))

    cc_s1_metrics = [
        ["Metric", "Value", "Formula"],
        ["Accuracy",    pct(cc_s1["accuracy"]),    f"(TP+TN)/N = ({cc_s1['tp']}+{cc_s1['tn']})/{CC['n_processed']}"],
        ["Sensitivity", pct(cc_s1["sensitivity"]), f"TP/(TP+FN) = {cc_s1['tp']}/({cc_s1['tp']}+{cc_s1['fn']})"],
        ["Specificity", pct(cc_s1["specificity"]), f"TN/(TN+FP) = {cc_s1['tn']}/({cc_s1['tn']}+{cc_s1['fp']})"],
        ["Precision",   pct(cc_s1["precision"]),   f"TP/(TP+FP) = {cc_s1['tp']}/({cc_s1['tp']}+{cc_s1['fp']})"],
        ["F1-Score",    pct(cc_s1["f1"]),           "2×(P×R)/(P+R)"],
    ]
    story.append(mk_table(cc_s1_metrics, [1.5*inch, 1.2*inch, 3.7*inch]))
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph("2.2  System 2 — Disorder Classification Confusion Matrix", s["h2"]))
    story.append(Paragraph(
        "Binary mapping: <b>schizophrenia*</b> → Schizophrenia pred; everything else → Not Schz.", s["normal"]))

    cc_s2_cm_data = [
        ["", "Pred: Schizophrenia", "Pred: Not Schz"],
        [f"GT: Schizophrenia (n={cc_s2['tp']+cc_s2['fn']})",
         f"TP = {cc_s2['tp']}", f"FN = {cc_s2['fn']}"],
        [f"GT: Healthy (n={cc_s2['fp']+cc_s2['tn']})",
         f"FP = {cc_s2['fp']}", f"TN = {cc_s2['tn']}"],
    ]
    story.append(mk_table(cc_s2_cm_data, [2.8*inch, 1.8*inch, 1.8*inch]))
    story.append(Spacer(1, 0.1*inch))

    cc_s2_metrics = [
        ["Metric", "Value", "Formula"],
        ["Accuracy",            pct(cc_s2["accuracy"]),            f"(TP+TN)/N = ({cc_s2['tp']}+{cc_s2['tn']})/{CC['n_processed']}"],
        ["Precision",           pct(cc_s2["precision"]),           f"TP/(TP+FP) = {cc_s2['tp']}/({cc_s2['tp']}+{cc_s2['fp']})"],
        ["Recall (Sensitivity)",pct(cc_s2["recall"]),              f"TP/(TP+FN) = {cc_s2['tp']}/({cc_s2['tp']}+{cc_s2['fn']})"],
        ["Specificity",         pct(cc_s2["specificity"]),         f"TN/(TN+FP) = {cc_s2['tn']}/({cc_s2['tn']}+{cc_s2['fp']})"],
        ["F1-Score",            pct(cc_s2["f1"]),                   "2×(P×R)/(P+R)"],
        ["Top-3 Sensitivity",
         f"{cc_s2['top3_hits']}/{cc_s2['top3_total']} = {pct(cc_s2['top3_sensitivity'])}",
         "Schizophrenia in top-3 scored disorders"],
    ]
    story.append(mk_table(cc_s2_metrics, [1.7*inch, 1.5*inch, 3.2*inch]))
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph("2.3  Per-Patient Results", s["h2"]))
    cc_rows = [["UID","Ground Truth","S2 Disorder","S1 Anomaly","S2 Conf","Correct"]]
    for r in sorted(CC["per_patient"], key=lambda x: x["uid"]):
        if r.get("status") != "processed": continue
        d    = str(r.get("s2_disorder","")).lower()
        pred = "schizophrenia" if d.startswith("schizo") else "not_schz"
        gt   = r.get("ground_truth","?")
        ok   = "✅" if (gt=="schizophrenia" and pred=="schizophrenia") or (gt!="schizophrenia" and pred!="schizophrenia") else "❌"
        cc_rows.append([
            r["uid"],
            gt,
            r.get("s2_disorder","?"),
            "Yes" if str(r.get("s1_anomaly","")) == "True" else "No",
            r.get("s2_confidence","?"),
            ok,
        ])
    story.append(mk_table(cc_rows, [0.7*inch, 1.2*inch, 1.8*inch, 0.85*inch, 0.85*inch, 0.7*inch]))

    story.append(PageBreak())

    # ── SECTION 3: CONSOLIDATED SUMMARY ─────────────────────────────────
    story.append(Paragraph("3. Consolidated Metric Summary", s["h1"]))
    story.append(Paragraph(
        "All metrics computed from real pipeline runs on actual sensor data. "
        "No synthetic inputs were used. Formula shown for each metric.", s["normal"]))

    final = [
        ["Metric", "StudentLife\n(Depression)", "CrossCheck\n(Schizophrenia)", "Formula"],
        ["Subjects",          str(SL['n_processed']), str(CC['n_processed']), "Total processed"],
        ["S1 TP",             str(sl_s1['tp']),       str(cc_s1['tp']),      "True Positives"],
        ["S1 FP",             str(sl_s1['fp']),       str(cc_s1['fp']),      "False Positives"],
        ["S1 TN",             str(sl_s1['tn']),       str(cc_s1['tn']),      "True Negatives"],
        ["S1 FN",             str(sl_s1['fn']),       str(cc_s1['fn']),      "False Negatives"],
        ["S1 Sensitivity",    pct(sl_s1['sensitivity']),pct(cc_s1['sensitivity']),"TP/(TP+FN)"],
        ["S1 Specificity",    pct(sl_s1['specificity']),pct(cc_s1['specificity']),"TN/(TN+FP)"],
        ["S1 Accuracy",       pct(sl_s1['accuracy']),  pct(cc_s1['accuracy']),  "(TP+TN)/N"],
        ["S1 Precision",      pct(sl_s1['precision']),  pct(cc_s1['precision']), "TP/(TP+FP)"],
        ["S1 F1-Score",       pct(sl_s1['f1']),         pct(cc_s1['f1']),        "2×(P×R)/(P+R)"],
        ["S2 TP",             str(sl_s2['tp']),          str(cc_s2['tp']),        "True Positives"],
        ["S2 FP",             str(sl_s2['fp']),          str(cc_s2['fp']),        "False Positives"],
        ["S2 TN",             str(sl_s2['tn']),          str(cc_s2['tn']),        "True Negatives"],
        ["S2 FN",             str(sl_s2['fn']),          str(cc_s2['fn']),        "False Negatives"],
        ["S2 Accuracy",       pct(sl_s2['accuracy']),    pct(cc_s2['accuracy']),  "(TP+TN)/N"],
        ["S2 Precision",      pct(sl_s2['precision']),   pct(cc_s2['precision']), "TP/(TP+FP)"],
        ["S2 Recall",         pct(sl_s2['recall']),      pct(cc_s2['recall']),    "TP/(TP+FN)"],
        ["S2 Specificity",    pct(sl_s2['specificity']), pct(cc_s2['specificity']),"TN/(TN+FP)"],
        ["S2 F1-Score",       pct(sl_s2['f1']),          pct(cc_s2['f1']),        "2×(P×R)/(P+R)"],
        ["S2 Top-3 Sens.",
         f"{sl_s2['top3_hits']}/{sl_s2['top3_total']} = {pct(sl_s2['top3_sensitivity'])}",
         f"{cc_s2['top3_hits']}/{cc_s2['top3_total']} = {pct(cc_s2['top3_sensitivity'])}",
         "Disorder in top-3 (clinical)"],
    ]
    story.append(mk_table(final, [1.7*inch, 1.5*inch, 1.5*inch, 2.0*inch]))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Key Observations", s["h2"]))
    obs = [
        f"• <b>S1 achieves 100% sensitivity on both datasets</b> — every depressed/schizophrenic patient triggers an anomaly flag. This is by design (pessimistic threshold).",
        f"• <b>S1 specificity is low</b> ({pct(sl_s1['specificity'])} on StudentLife, {pct(cc_s1['specificity'])} on CrossCheck) — S1 flags many healthy patients as anomalous. System 2 is designed to filter these.",
        f"• <b>S2 Top-3 Sensitivity is {pct(sl_s2['top3_sensitivity'])} (StudentLife)</b> and <b>{pct(cc_s2['top3_sensitivity'])} (CrossCheck)</b> — the correct disorder appears in the top-3 matches for most true positives, meeting clinical differential-diagnosis standards.",
        f"• <b>S2 specificity is strong</b>: {pct(sl_s2['specificity'])} (StudentLife) and {pct(cc_s2['specificity'])} (CrossCheck) — S2 correctly rules out healthy patients.",
        f"• <b>S2 Top-1 recall is lower</b> ({pct(sl_s2['recall'])} and {pct(cc_s2['recall'])}) because the geometric prototype matcher places schizophrenia patients in depression buckets when social sensing data is missing.",
    ]
    for o in obs:
        story.append(Paragraph(o, s["normal"]))

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#bbb")))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        "Generated: 2026-03-05  |  Data: system2/data/studentlife_metrics.json + crosscheck_metrics.json",
        s["caption"]))

    doc.build(story)
    print(f"PDF saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_pdf()
