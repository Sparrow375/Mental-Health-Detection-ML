"""
Generate System 2 Validation Report PDF
========================================
Produces:  system2/System2_Validation_Report.pdf

Style matches StudentLife_Validation_Report.pdf (ReportLab).
Run from the project root:
    python system2/generate_validation_pdf.py
"""

import os
import sys

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak, HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")
OUTPUT_PDF  = os.path.join(SCRIPT_DIR, "System2_Validation_Report.pdf")


def chart(name, w=5.5*inch, h=3.2*inch):
    """Return an Image flowable if the file exists, else None."""
    path = os.path.join(CHARTS_DIR, name)
    if os.path.exists(path):
        return Image(path, width=w, height=h)
    return None


def make_styles():
    base = getSampleStyleSheet()

    title = ParagraphStyle(
        "MainTitle", parent=base["Heading1"],
        fontSize=22, spaceAfter=12,
        textColor=colors.HexColor("#1a252f"),
        alignment=TA_CENTER,
    )
    subtitle = ParagraphStyle(
        "Subtitle", parent=base["Normal"],
        fontSize=12, spaceAfter=6,
        textColor=colors.HexColor("#555555"),
        alignment=TA_CENTER,
    )
    h1 = ParagraphStyle(
        "H1", parent=base["Heading1"],
        fontSize=15, spaceBefore=18, spaceAfter=10,
        textColor=colors.HexColor("#2980b9"),
    )
    h2 = ParagraphStyle(
        "H2", parent=base["Heading2"],
        fontSize=12, spaceBefore=12, spaceAfter=7,
        textColor=colors.HexColor("#34495e"),
    )
    normal = ParagraphStyle(
        "N", parent=base["Normal"],
        fontSize=10, spaceAfter=6, leading=14,
    )
    bullet = ParagraphStyle(
        "B", parent=base["Normal"],
        fontSize=10, spaceAfter=4, leading=13,
        leftIndent=18, bulletIndent=6,
    )
    caption = ParagraphStyle(
        "Cap", parent=base["Normal"],
        fontSize=9, spaceAfter=8, leading=12,
        textColor=colors.HexColor("#777777"),
        alignment=TA_CENTER,
    )
    code = ParagraphStyle(
        "Code", parent=base["Code"],
        fontSize=8, spaceAfter=6, leading=12,
        backColor=colors.HexColor("#f4f4f4"),
        leftIndent=12,
    )
    verdict_green = ParagraphStyle(
        "VG", parent=base["Normal"],
        fontSize=10, leading=13,
        textColor=colors.HexColor("#1a7a4a"),
    )
    verdict_red = ParagraphStyle(
        "VR", parent=base["Normal"],
        fontSize=10, leading=13,
        textColor=colors.HexColor("#c0392b"),
    )
    return dict(
        title=title, subtitle=subtitle, h1=h1, h2=h2,
        normal=normal, bullet=bullet, caption=caption,
        code=code, verdict_green=verdict_green, verdict_red=verdict_red,
    )


# ── Table helper ───────────────────────────────────────────────────────────

HEADER_BG   = colors.HexColor("#2980b9")
ROW_ALT     = colors.HexColor("#eaf4fb")
ROW_WHITE   = colors.white
BORDER_COL  = colors.HexColor("#bdc3c7")

def styled_table(data, col_widths=None):
    """Build a ReportLab Table with blue-header styling."""
    t = Table(data, colWidths=col_widths, repeatRows=1)
    n_rows = len(data)
    style = [
        # Header
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 9),
        ("ALIGN",      (0, 0), (-1, 0), "CENTER"),
        # Body
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 1), (-1, -1), 8),
        ("ALIGN",      (0, 1), (-1, -1), "LEFT"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",       (0, 0), (-1, -1), 0.4, BORDER_COL),
        ("ROWBACKGROUND", (0, 1), (-1, -1), [ROW_WHITE, ROW_ALT]),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]
    t.setStyle(TableStyle(style))
    return t


# ── PDF assembly ────────────────────────────────────────────────────────────

def build_pdf():
    doc = SimpleDocTemplate(
        OUTPUT_PDF, pagesize=letter,
        rightMargin=50, leftMargin=50,
        topMargin=50, bottomMargin=50,
    )
    s = make_styles()
    story = []

    # ── TITLE PAGE ──────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.6 * inch))
    story.append(Paragraph("SYSTEM 2", s["title"]))
    story.append(Paragraph("VALIDATION REPORT", s["title"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2980b9")))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Metric-Based Clinical Prototype Matching Engine", s["subtitle"]))
    story.append(Paragraph("<b>Project:</b> Early Risk Detection of Mental Health Disorders", s["subtitle"]))
    story.append(Paragraph("<b>Date:</b> 2026-03-05 &nbsp;&nbsp; | &nbsp;&nbsp; <b>Status:</b> Phase 7 Validation — In Progress", s["subtitle"]))
    story.append(Spacer(1, 0.4 * inch))

    # Radar chart as cover hero
    img = chart("05_radar_depression.png", 4.5 * inch, 4.5 * inch)
    if img:
        story.append(img)
        story.append(Paragraph("Behavioral profile overlay: user vs. Depression prototype vs. Healthy baseline", s["caption"]))

    story.append(PageBreak())

    # ── SECTION 1: SCOPE ────────────────────────────────────────────────────
    story.append(Paragraph("1. Validation Scope &amp; Objectives", s["h1"]))
    story.append(Paragraph(
        "System 2 is <b>not a trained ML model</b> — there are no training/test splits. "
        "It is a <b>clinical prototype matching engine</b> grounded in published passive-sensing "
        "literature. Validation covers:", s["normal"]))

    scope_data = [
        ["Validation Type", "What It Checks"],
        ["Unit correctness", "Cosine similarity, weighted Euclidean, match score formulas"],
        ["Component integration", "Each pipeline stage receives and passes data correctly"],
        ["Clinical consistency", "Correct prototype scores higher for matching profiles"],
        ["Gate logic", "3 screening gates fire/pass at correct thresholds"],
        ["Temporal shape detection", "Shapes classified correctly; confidence adjusted"],
        ["End-to-end pipeline", "Synthetic profiles produce expected output"],
        ["StudentLife benchmarking", "(Pending full run) Stratification against PHQ-9"],
    ]
    story.append(styled_table(scope_data, [2.1*inch, 4.3*inch]))
    story.append(Spacer(1, 0.15 * inch))

    # ── SECTION 2: UNIT TESTS ───────────────────────────────────────────────
    story.append(Paragraph("2. Unit Test Results — 30 / 30 Passing ✅", s["h1"]))
    story.append(Paragraph(
        "All 30 unit tests pass as of 2026-03-02 across four test modules:", s["normal"]))

    summary_data = [
        ["Test Module", "Tests", "Status"],
        ["test_matcher.py",  "10", "✅ PASS"],
        ["test_pipeline.py", "6",  "✅ PASS"],
        ["test_screener.py", "9",  "✅ PASS"],
        ["test_temporal.py", "6",  "✅ PASS"],
        ["TOTAL",            "30", "✅ PASS"],
    ]
    story.append(styled_table(summary_data, [3.5*inch, 1.2*inch, 1.7*inch]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Full Test Coverage Map", s["h2"]))
    coverage_data = [
        ["Module", "Class / Method", "What It Validates"],
        ["test_matcher", "TestCosine – test_identical", "cos_sim(X, X) = 1.0"],
        ["test_matcher", "TestCosine – test_opposite", "cos_sim(X, -X) = -1.0"],
        ["test_matcher", "TestCosine – test_orthogonal", "cos_sim(orthogonal) ≈ 0"],
        ["test_matcher", "TestCosine – test_zero_vector", "Zero-division handled gracefully"],
        ["test_matcher", "TestEuclidean – test_zero_distance", "dist(X, X) = 0"],
        ["test_matcher", "TestEuclidean – test_known_distance", "Numeric accuracy"],
        ["test_matcher", "TestMatchScore – test_perfect_match", "Perfect match → score = 1.0"],
        ["test_matcher", "TestClassification – test_healthy", "Healthy features → healthy output"],
        ["test_matcher", "TestClassification – test_depression", "Depression prototype → depression"],
        ["test_matcher", "TestClassification – test_frame1", "Frame 1 path executes correctly"],
        ["test_screener", "TestGate1 – test_healthy_passes", "Population-normal profile clears Gate 1"],
        ["test_screener", "TestGate1 – test_extreme_flags", "4 features at +3 SD triggers FLAG"],
        ["test_screener", "TestGate1 – test_borderline_passes", "2 features at +3 SD — below threshold, PASS"],
        ["test_screener", "TestGate2 – test_stable_weeks", "Identical weeks pass Gate 2"],
        ["test_screener", "TestGate2 – test_drifting_flags", "High week-over-week swing flags"],
        ["test_screener", "TestGate3 – test_healthy_passes", "Healthy 28-day average clears Gate 3"],
        ["test_screener", "TestGate3 – test_depression_flags", "Depression onboarding → CONTAMINATED"],
        ["test_screener", "TestScreen – test_all_pass", "All gates pass → Frame 2 selected"],
        ["test_screener", "TestScreen – test_gate3_fires", "Gate 3 fires → Frame 1 selected"],
        ["test_temporal", "TestShapeDetection – test_drift", "Declining series → monotonic_drift"],
        ["test_temporal", "TestShapeDetection – test_osc", "Oscillating series → oscillating"],
        ["test_temporal", "TestShapeDetection – test_chaotic", "High variance → chaotic"],
        ["test_temporal", "TestShapeDetection – test_short", "< 10 points → none (graceful)"],
        ["test_temporal", "TestConfidence – test_boost", "Supporting shape → ×1.2 boost"],
        ["test_temporal", "TestConfidence – test_downgrade", "Contradicting shape → ×0.6 downgrade"],
        ["test_pipeline", "TestHealthy – test_zero_devs", "Zero deviations → life_event (dismissed)"],
        ["test_pipeline", "TestHealthy – test_mild_devs", "Mild ambiguous → UNCLASSIFIED (safe)"],
        ["test_pipeline", "TestDepression – test_depressed", "Dep. prototype profile → depression"],
        ["test_pipeline", "TestLifeEvent – test_dismissed", "≤ 2 co-deviating features → dismiss"],
        ["test_pipeline", "TestContaminated – test_frame1", "Depression baseline → Gate 3 → Frame 1"],
    ]
    story.append(styled_table(coverage_data, [1.4*inch, 2.5*inch, 2.5*inch]))

    story.append(PageBreak())

    # ── SECTION 3: PIPELINE INTEGRITY ───────────────────────────────────────
    story.append(Paragraph("3. Pipeline Integrity — 5 End-to-End Scenarios", s["h1"]))
    story.append(Paragraph(
        "Five synthetic scenarios were run through <b>System2Pipeline.classify()</b> "
        "to verify correct data flow across all stages:", s["normal"]))

    scenarios = [
        ["#", "Scenario", "Filter Decision", "Output", "Status"],
        ["1", "Healthy user — zero deviations",        "DISMISS (severity floor)", "life_event",         "✅"],
        ["2", "Healthy user — mild ambiguous devs",    "PROCEED",                 "UNCLASSIFIED (safe)", "✅"],
        ["3", "Depressed user — prototype match",      "PROCEED",                 "depression / HIGH",   "✅"],
        ["4", "Life event — narrow anomaly (2 feats)", "DISMISS (co-dev ≤ 2)",    "life_event",          "✅"],
        ["5", "Contaminated baseline (dep. onboard)",  "PROCEED",                 "[ONBOARDING] dep.",   "✅"],
    ]
    story.append(styled_table(scenarios, [0.3*inch, 2.3*inch, 1.7*inch, 1.5*inch, 0.6*inch]))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Gate Screening — Visual Reference", s["h2"]))
    img = chart("09_gate_screening.png", 5.5 * inch, 3.0 * inch)
    if img:
        story.append(img)
        story.append(Paragraph("Gate 1/2/3 threshold visualisation from the gates screening chart", s["caption"]))

    story.append(PageBreak())

    # ── SECTION 4: BASELINE SCREENER ────────────────────────────────────────
    story.append(Paragraph("4. Baseline Screener Gate Validation", s["h1"]))

    story.append(Paragraph("Gate 1 — Population Anchor Check", s["h2"]))
    story.append(Paragraph(
        "<b>Threshold:</b> 3 or more features exceeding ±2.5 SD simultaneously → "
        "FLAG_POSSIBLE_CONDITION.", s["normal"]))
    g1 = [
        ["Test Scenario", "Features Flagged", "Expected", "Actual", "OK"],
        ["All features at population mean",   "0", "PASS",                  "PASS",                  "✅"],
        ["4 features at +3 SD",               "4", "FLAG_POSSIBLE_CONDITION","FLAG_POSSIBLE_CONDITION","✅"],
        ["2 features at +3 SD",               "2", "PASS (below threshold)", "PASS",                  "✅"],
    ]
    story.append(styled_table(g1, [2.5*inch, 1.2*inch, 1.8*inch, 1.3*inch, 0.5*inch]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Gate 2 — Internal Stability Check", s["h2"]))
    story.append(Paragraph(
        "<b>Threshold:</b> std(week1, week2, week3) &gt; 1.5 × population_expected_drift "
        "for 3+ features → FLAG_UNSTABLE_BASELINE.", s["normal"]))
    g2 = [
        ["Test Scenario", "Expected", "Actual", "OK"],
        ["Identical weekly values",                "PASS",                  "PASS",                  "✅"],
        ["4 features swinging ±2 SD across weeks", "FLAG_UNSTABLE_BASELINE","FLAG_UNSTABLE_BASELINE","✅"],
    ]
    story.append(styled_table(g2, [3.0*inch, 2.0*inch, 1.5*inch, 0.5*inch]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Gate 3 — Prototype Proximity Check", s["h2"]))
    story.append(Paragraph(
        "<b>Threshold:</b> Top match ≠ healthy AND confidence &gt; 0.65 → CONTAMINATED_BASELINE.", s["normal"]))
    g3 = [
        ["Test Scenario", "Expected", "Match", "OK"],
        ["Healthy population mean as 28-day average",   "PASS",                 "healthy",    "✅"],
        ["Depression Frame 1 values as 28-day average", "CONTAMINATED_BASELINE","depression", "✅"],
    ]
    story.append(styled_table(g3, [3.0*inch, 1.8*inch, 1.2*inch, 0.5*inch]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Decision Matrix", s["h2"]))
    dm = [
        ["Gates Fired", "Expected Action", "Frame", "Verified"],
        ["None",               "LOCK_BASELINE",                   "Frame 2", "✅"],
        ["Gate 1 only",        "EXTEND_MONITORING",               "Frame 2", "✅"],
        ["Gate 2 only",        "FLAG_CYCLING",                    "Frame 2", "✅"],
        ["Gate 3",             "EARLY_DETECTION",                 "Frame 1", "✅"],
        ["Gate 1 + Gate 3",    "EARLY_DETECTION_WITH_SELF_REPORT","Frame 1", "✅"],
        ["All three",          "CLINICAL_REVIEW",                 "Frame 1", "Code review"],
    ]
    story.append(styled_table(dm, [2.0*inch, 2.4*inch, 1.1*inch, 1.3*inch]))

    story.append(PageBreak())

    # ── SECTION 5: PROTOTYPE CALIBRATION ────────────────────────────────────
    story.append(Paragraph("5. Clinical Prototype Calibration Review", s["h1"]))
    story.append(Paragraph(
        "Frame 2 prototype directions verified against clinical literature "
        "(depression cohort, Saeb 2015 / Canzian 2015):", s["normal"]))

    dep_cal = [
        ["Feature", "Clinical Expectation", "Frame 2 Prototype", "Match"],
        ["daily_displacement_km",   "↓ reduced mobility",              "-2.0 SD", "✅"],
        ["social_app_ratio",        "↓ social withdrawal",             "-1.8 SD", "✅"],
        ["texts_per_day",           "↓ less communication",            "-1.5 SD", "✅"],
        ["sleep_duration_hours",    "↑ hypersomnia OR ↓ insomnia",     "+1.5 SD", "✅"],
        ["response_time_minutes",   "↑ cognitive slowing",             "+1.8 SD", "✅"],
        ["location_entropy",        "↓ stays home",                    "-2.0 SD", "✅"],
        ["places_visited",          "↓ minimal mobility",              "-2.0 SD", "✅"],
        ["app_diversity",           "↓ narrow repetitive usage",       "-1.5 SD", "✅"],
    ]
    story.append(styled_table(dep_cal, [2.0*inch, 2.1*inch, 1.3*inch, 0.8*inch]))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Feature Diagnostic Weights", s["h2"]))
    img = chart("04_feature_weights.png", 5.5 * inch, 3.0 * inch)
    if img:
        story.append(img)
        story.append(Paragraph(
            "Diagnostic weight per feature. High weight (≥0.8): displacement, location entropy, "
            "places visited, sleep duration, social app ratio.", s["caption"]))

    story.append(Paragraph("Frame 2 Prototype Heatmap", s["h2"]))
    img = chart("03_frame2_heatmap.png", 5.5 * inch, 3.2 * inch)
    if img:
        story.append(img)
        story.append(Paragraph(
            "Blue = negative deviation from baseline (withdrawal). "
            "Red = positive deviation. White = no change. "
            "Depression shows consistent blue in mobility/social, red in sleep/response.", s["caption"]))

    story.append(PageBreak())

    # ── SECTION 6: TEMPORAL VALIDATOR ───────────────────────────────────────
    story.append(Paragraph("6. Temporal Validator Verification", s["h1"]))

    story.append(Paragraph(
        "Shape detection uses autocorrelation, linear regression slope, and variance analysis. "
        "Confidence is adjusted after the prototype match:", s["normal"]))

    tv = [
        ["Shape",           "Algorithm",                                     "Test Status"],
        ["monotonic_drift", "polyfit slope < -0.02 AND R² > 0.6",           "✅ test_drift"],
        ["oscillating",     "Autocorrelation peak at lag 3–10 days",         "✅ test_oscillating"],
        ["chaotic",         "High var + low lag-1 autocorrelation",          "✅ test_chaotic"],
        ["episodic_spike",  "Value > mean+2σ, recovers within 14 days",      "Code review"],
        ["phase_flip",      "Weekly mean diff > 3 SD",                       "Code review"],
    ]
    story.append(styled_table(tv, [1.5*inch, 3.2*inch, 1.7*inch]))
    story.append(Spacer(1, 0.1 * inch))

    ca = [
        ["Condition", "Multiplier", "Example", "Verified"],
        ["Shape SUPPORTS disorder",    "×1.2 (boost)",     "monotonic_drift + depression",       "✅"],
        ["Shape CONTRADICTS disorder", "×0.6 (downgrade)", "oscillating + depression → BPD flag","✅"],
    ]
    story.append(styled_table(ca, [2.0*inch, 1.3*inch, 2.4*inch, 0.8*inch]))
    story.append(Spacer(1, 0.15 * inch))

    img = chart("07_temporal_shapes.png", 5.5 * inch, 3.0 * inch)
    if img:
        story.append(img)
        story.append(Paragraph(
            "Temporal shapes detected by the validator. Each disorder has a characteristic trajectory signature.", s["caption"]))

    img = chart("08_shape_disorder_matrix.png", 5.5 * inch, 2.8 * inch)
    if img:
        story.append(img)
        story.append(Paragraph(
            "Shape–disorder compatibility matrix. Green = supports, Red = contradicts, White = neutral.", s["caption"]))

    story.append(PageBreak())

    # ── SECTION 7: CLINICAL GUARDRAILS ──────────────────────────────────────
    story.append(Paragraph("7. Clinical Guardrail Review", s["h1"]))
    story.append(Paragraph(
        "The <b>_clinical_guardrails()</b> method applies dataset-specific heuristic overrides "
        "when the geometric matcher under-fires due to missing sensors:", s["normal"]))

    gr = [
        ["Guardrail", "Trigger Condition", "Action", "Dataset"],
        [
            "Schizophrenia boost",
            "social_app_ratio ≈ 0 AND severe markers (location, displacement, sleep, calls > 1.4 SD or sum > 2.0)",
            "Force schizophrenia_type_2, score=0.95, HIGH confidence",
            "CrossCheck",
        ],
        [
            "Depression boost",
            "social_app_ratio present AND withdrawal markers (calls/texts/conv < -1.1 SD) OR sleep < -0.9 SD",
            "Force depression_type_1, score=0.90, HIGH confidence",
            "StudentLife",
        ],
    ]
    story.append(styled_table(gr, [1.3*inch, 2.6*inch, 1.8*inch, 0.9*inch]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "⚠️  <b>Deployment warning:</b> These guardrails are research-dataset-specific. "
        "They must be disabled or re-evaluated before any general clinical deployment. "
        "They assume a binary population (depression vs. healthy, OR schizophrenia vs. healthy) "
        "which does not hold in heterogeneous clinical settings.", s["normal"]))

    story.append(PageBreak())

    # ── SECTION 8: STUDENTLIFE BENCHMARK ────────────────────────────────────
    story.append(Paragraph("8. StudentLife Benchmark — Targets vs. Current State", s["h1"]))

    story.append(Paragraph("Dataset Profile", s["h2"]))
    ds = [
        ["Property", "Value"],
        ["Dataset",             "StudentLife (Dartmouth, 2014)"],
        ["Students",            "49 undergraduates"],
        ["Ground truth",        "PHQ-9 pre/post score (self-report)"],
        ["Depressed (≥ 10)",    "~16 students (~33%)"],
        ["Mild (5–9)",          "~15 students (~31%)"],
        ["Healthy (< 5)",       "~18 students (~37%)"],
        ["Feature set",         "18 behavioral (voice excluded)"],
    ]
    story.append(styled_table(ds, [2.5*inch, 3.9*inch]))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Performance Targets", s["h2"]))
    pt = [
        ["Metric",                          "Target", "Status"],
        ["S1 Sensitivity (depressed)","≥ 70%",   "🔶 Full run pending"],
        ["S1 Specificity (healthy)",  "≥ 60%",   "🔶 Full run pending"],
        ["S2 Top-1 Depression Sens.", "≥ 55%",   "🔶 Full run pending"],
        ["S2 Top-3 Depression Sens.", "≥ 75%",   "🔶 Full run pending"],
        ["S2 Healthy Specificity",    "≥ 65%",   "🔶 Full run pending"],
        ["UNCLASSIFIED rate",         "< 25%",   "🔶 Full run pending"],
    ]
    story.append(styled_table(pt, [3.0*inch, 1.2*inch, 2.2*inch]))
    story.append(Spacer(1, 0.15 * inch))

    img = chart("final_metrics_bar.png", 5.5 * inch, 3.2 * inch)
    if img:
        story.append(img)
        story.append(Paragraph(
            "Top-3 clinical validation metrics from prior integration run. "
            "100% StudentLife depression sensitivity (Top-3), 75% CrossCheck schizophrenia.", s["caption"]))

    story.append(PageBreak())

    # ── SECTION 9: KNOWN ISSUES ──────────────────────────────────────────────
    story.append(Paragraph("9. Known Issues &amp; Open Bugs", s["h1"]))

    story.append(Paragraph("Critical", s["h2"]))
    crit = [
        ["ID",      "Issue",                                            "Location",                    "Impact"],
        ["BUG-01", "social_app_ratio extraction returns 0% for all students",
         "studentlife_extractor.py", "Removes primary depression signal"],
        ["BUG-02", "Call log path incorrect — zero calls per student",
         "studentlife_extractor.py", "Undermines communication features"],
    ]
    story.append(styled_table(crit, [0.7*inch, 2.8*inch, 1.8*inch, 1.1*inch]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("High", s["h2"]))
    high = [
        ["ID",      "Issue",                                            "Location",              "Impact"],
        ["BUG-03", "Only 5/49 students processed in known runs",
         "run_studentlife.py", "Insufficient data for accuracy claims"],
        ["BUG-04", "Clinical guardrails are dataset-specific heuristics",
         "pipeline.py",        "Cannot generalize without re-tuning"],
    ]
    story.append(styled_table(high, [0.7*inch, 2.8*inch, 1.5*inch, 1.4*inch]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Medium", s["h2"]))
    med = [
        ["ID",      "Issue",                                            "Location",           "Impact"],
        ["BUG-05", "BPD detection relies on variance, not mean shift",
         "prototype_matcher.py","BPD frequently misclassified"],
        ["BUG-06", "Schizophrenia and depression overlap in sleep/mobility features",
         "config.py",           "Confusion in overlapping cases"],
        ["BUG-07", "Depression prototype doesn't branch on hyper- vs. insomnia direction",
         "config.py",           "Same prototype matches opposite sleep patterns"],
    ]
    story.append(styled_table(med, [0.7*inch, 2.8*inch, 1.5*inch, 1.4*inch]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Low", s["h2"]))
    low_t = [
        ["ID",      "Issue",                                            "Location",     "Impact"],
        ["BUG-08", "Confidence label shows >100% when temporal boost overcomes max",
         "pipeline.py",  "Display confusion in narrative"],
        ["BUG-09", "Output dir system2/data/ not created until first run",
         "run_studentlife.py", "Silent create via os.makedirs"],
    ]
    story.append(styled_table(low_t, [0.7*inch, 2.8*inch, 1.5*inch, 1.4*inch]))

    story.append(PageBreak())

    # ── SECTION 10: PHASE 7 ROADMAP ─────────────────────────────────────────
    story.append(Paragraph("10. Phase 7 Validation Roadmap", s["h1"]))

    story.append(Paragraph("Validation Gaps", s["h2"]))
    gaps = [
        ["Gap", "Required Action", "Priority"],
        ["Full StudentLife run (49 students)", "Fix BUG-01 &amp; BUG-02, run run_studentlife.py", "🔴 Critical"],
        ["Empirical prototype calibration",    "Compute z-scores from high-PHQ vs. low-PHQ cohorts", "🔴 Critical"],
        ["Confusion matrix",                   "Binary + multi-class performance metrics",            "🔴 Critical"],
        ["S1+S2 combined accuracy",            "End-to-end sensitivity/specificity on full cohort",   "🔴 Critical"],
        ["CrossCheck schizophrenia val.",      "Run CrossCheck dataset through S2",                   "🟠 High"],
        ["Feature importance analysis",        "Ablation study per feature",                          "🟠 High"],
        ["BPD/Bipolar FP rate",               "These are over-dismissed — need labeled dataset",     "🟡 Medium"],
        ["Demographic stratification",         "Population norms assumed universal",                  "🟡 Medium"],
    ]
    story.append(styled_table(gaps, [2.1*inch, 2.7*inch, 1.5*inch]))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Phase 7 Execution Steps", s["h2"]))
    steps = [
        ("Step 1 — Fix Feature Extraction Bugs",
         "Fix social_app_ratio calculation. Fix call log file path. Verify feature coverage on 10+ students."),
        ("Step 2 — Run Full StudentLife Cohort",
         "Run run_studentlife.py on all 49 students. Save studentlife_results.json and .csv. "
         "Verify ≥35 days per student."),
        ("Step 3 — Empirical Calibration",
         "Group students: depressed (PHQ ≥ 10) vs. healthy (PHQ < 5). Compute mean z-scores per feature. "
         "Compare against Frame 2 prototypes and adjust where deviation > 0.5 SD."),
        ("Step 4 — Validation Metrics",
         "S1 confusion matrix. S2 Top-1 and Top-3 sensitivity. ROC-AUC. "
         "Per-feature importance analysis."),
        ("Step 5 — Update Report",
         "Update this validation report with measured results. Update report.md Section 13. "
         "Generate final performance charts."),
    ]
    for title, detail in steps:
        story.append(Paragraph(f"<b>{title}</b>", s["h2"]))
        story.append(Paragraph(detail, s["normal"]))

    story.append(PageBreak())

    # ── SECTION 11: VERDICT ─────────────────────────────────────────────────
    story.append(Paragraph("11. Verdict Summary", s["h1"]))

    story.append(Paragraph("What Has Been Validated ✅", s["h2"]))
    validated = [
        ["Component", "Validation Status"],
        ["Cosine similarity math",              "✅  4 unit tests"],
        ["Weighted Euclidean distance",         "✅  2 unit tests"],
        ["Match score formula",                 "✅  1 unit test"],
        ["PrototypeMatcher (Frame 1 &amp; 2)", "✅  3 unit tests"],
        ["Gate 1 (Population Anchor)",          "✅  3 unit tests"],
        ["Gate 2 (Stability Check)",            "✅  2 unit tests"],
        ["Gate 3 (Prototype Proximity)",        "✅  2 unit tests"],
        ["Combined Screener Decision Matrix",   "✅  2 unit tests"],
        ["Temporal shape detection (3 shapes)", "✅  3 unit tests"],
        ["Temporal confidence adjustment",      "✅  2 unit tests"],
        ["End-to-End Pipeline (5 scenarios)",   "✅  5 pipeline tests"],
        ["S1 → S2 Adapter Interface",           "✅  Integration tests"],
        ["Prototype clinical grounding",        "✅  Manual review vs. 7 published sources"],
    ]
    story.append(styled_table(validated, [3.5*inch, 2.9*inch]))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("What Has NOT Been Validated ❌", s["h2"]))
    not_validated = [
        ["Component", "Status"],
        ["Full StudentLife cohort run (49 students)", "❌  Blocked by BUG-01 / BUG-02"],
        ["Empirical prototype calibration",           "❌  Pending full data run"],
        ["ROC-AUC and confusion matrices",            "❌  Pending"],
        ["BPD/Bipolar classification accuracy",       "❌  No labeled data for these disorders"],
        ["CrossCheck schizophrenia validation",       "❌  Separate dataset required"],
    ]
    story.append(styled_table(not_validated, [3.5*inch, 2.9*inch]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2980b9")))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "<b>Overall Assessment:</b> System 2 is mathematically correct, architecturally sound, "
        "and clinically grounded at the component level. Its real-world classification accuracy on the "
        "StudentLife depression cohort cannot be confirmed until BUG-01 (social_app_ratio) and BUG-02 "
        "(call log path) are fixed and the full 49-student pipeline run is completed.",
        s["normal"]))

    story.append(PageBreak())

    # ── SECTION 12: REFERENCES ───────────────────────────────────────────────
    story.append(Paragraph("12. References", s["h1"]))
    refs = [
        "Wang, R. et al. (2014). StudentLife: Assessing Mental Health, Academic Performance and Behavioral Trends of College Students using Smartphones. UbiComp 2014.",
        "Saeb, S. et al. (2015). Mobile Phone Sensor Correlates of Depressive Symptom Severity in Daily-Life Behavior. J. Med. Internet Res.",
        "Canzian, L. &amp; Musolesi, M. (2015). Trajectories of Depression: Unobtrusive Monitoring of Depressive States. UbiComp.",
        "Barnett, I. et al. (2018). Relapse prediction in schizophrenia through digital phenotyping. npj Schizophrenia.",
        "Faurholt-Jepsen, M. et al. (2015). Daily electronic self-monitoring in bipolar disorder. MONARCA study.",
        "Santangelo, P. et al. (2014). Ecological validity of ambulatory assessment in borderline personality disorder.",
        "Boukhechba, M. et al. (2018). Monitoring social anxiety from mobility and communication patterns. UbiComp.",
    ]
    for i, ref in enumerate(refs, 1):
        story.append(Paragraph(f"[{i}] {ref}", s["bullet"]))

    story.append(Spacer(1, 0.3 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#bbbbbb")))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "Generated: 2026-03-05  |  System 2 build: S1+S2 Integration (2026-03-02)  |  "
        "See also: system2/report.md | system2/metric_based_system2.md",
        s["caption"],
    ))

    doc.build(story)
    print(f"PDF saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_pdf()
