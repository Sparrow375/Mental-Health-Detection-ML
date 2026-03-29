import json
import logging
import traceback
import pandas as pd

from pipeline import System2Pipeline
from system1 import ImprovedAnomalyDetector, PersonalityVector
from s1_s2_adapter import build_s1_input


def run_analysis(json_string: str) -> str:
    """
    Main entry point for Kotlin/Chaquopy.
    Receives JSON per the Kotlin contract, passes it to System 1 + System 2,
    and returns a structured JSON result matching what Kotlin expects.

    Input JSON schema:
      {
        "current":   { featureName: value, ... },                    // today's 29 features
        "baseline":  { featureName: { "mean": x, "std": y }, ... },  // per-feature stats
        "history":   [ { featureName: value, ... }, ... ],           // last 14 days (oldest first)
        "day_number": int,
        "baseline_contaminated": bool,
        "gate_state": {}
      }
    """
    try:
        data = json.loads(json_string)

        current      = data.get("current", {})
        baseline     = data.get("baseline", {})
        history      = data.get("history", [])
        contaminated = data.get("baseline_contaminated", False)
        day_number   = data.get("day_number", 0)

        # ── Build PersonalityVector baseline from Android mean/std stats ──────
        # PersonalityVector.from_dict() accepts all 29 Android feature keys.
        # Missing keys default to 0; unknown extras are silently ignored.
        baseline_means: dict = {}
        baseline_stds: dict  = {}
        for feat, stats in baseline.items():
            if isinstance(stats, dict):
                baseline_means[feat] = float(stats.get("mean", 0.0))
                baseline_stds[feat]  = float(stats.get("std",  1.0))
            else:
                baseline_means[feat] = float(stats)
                baseline_stds[feat]  = 1.0

        s1_baseline = PersonalityVector.from_dict(baseline_means, variances=baseline_stds)

        # ── System 1 setup ─────────────────────────────────────────────────────
        s1 = ImprovedAnomalyDetector(baseline=s1_baseline)

        # Fast-forward state using history (oldest first)
        deviations_history = []
        history_start_day = max(0, day_number - len(history))
        for idx, h in enumerate(history):
            s1_report_h, _ = s1.analyze(
                h,
                deviations_history=list(deviations_history),
                day_number=history_start_day + idx,
            )
            deviations_history.append(s1_report_h.feature_deviations)

        # Analyze today
        s1_report, daily_report = s1.analyze(
            current,
            deviations_history=list(deviations_history),
            day_number=day_number,
        )

        # ── System 2 setup ─────────────────────────────────────────────────────
        # Synthesize a 28-row baseline DataFrame from aggregated means
        # (Room only stores per-feature stats, not raw daily rows).
        baseline_rows = [{"date": f"day_{d}", **baseline_means} for d in range(28)]
        baseline_df = pd.DataFrame(baseline_rows)

        pipeline = System2Pipeline()
        s1_input = build_s1_input(
            detector=s1,
            baseline_df=baseline_df,
            s1_report=s1_report,
            timeseries_days=60,
        )
        s2_output = pipeline.classify(s1_input)

        # Respect the Android-supplied contamination flag
        if contaminated and not s2_output.baseline_contaminated:
            s2_output.baseline_contaminated = True

        # ── Evidence score for UI ──────────────────────────────────────────────
        evidence_score = float(
            sum(abs(v) for v in s1_report.feature_deviations.values())
        ) if s1_report.feature_deviations else 0.0

        # ── Top 3 contributing features from System 2 explanation ─────────────
        top3_list = []
        if s2_output.explanation and hasattr(s2_output.explanation, "top_contributing_features"):
            top3_list = [
                [f, float(val)]
                for f, val in s2_output.explanation.top_contributing_features.items()
            ][:3]

        # ── Gate results ───────────────────────────────────────────────────────
        gate1 = "gate1" not in s2_output.screening.gates_fired
        gate2 = "gate2" not in s2_output.screening.gates_fired
        gate3 = "gate3" not in s2_output.screening.gates_fired

        # ── Map to Kotlin JSON contract ────────────────────────────────────────
        result_dict = {
            "status": "ok",
            "anomaly": {
                "detected":        s1_report.sustained_deviation_days >= 3,
                "anomaly_score":   float(s1_report.overall_anomaly_score),
                "alert_level":     s1_report.alert_level,
                "sustained_days":  int(s1_report.sustained_deviation_days),
                "evidence":        evidence_score,
                "flagged_features": list(s1_report.flagged_features),
                "pattern_type":    (
                    s2_output.temporal_result.temporal_shape
                    if s2_output.temporal_result else "stable"
                ),
                "message": daily_report.notes,
            },
            "prototype": {
                "match":            s2_output.disorder,
                "confidence":       float(s2_output.score),
                "confidence_label": (
                    s2_output.confidence.name if s2_output.confidence else "UNCERTAIN"
                ),
                "message":          s2_output.label,
                "top_3_features":   top3_list,
                "all_scores": (
                    s2_output.classification.all_scores
                    if s2_output.classification else {}
                ),
                "reference_frame": (
                    "frame1_population"
                    if (s2_output.baseline_contaminated or contaminated)
                    else "frame2_personal"
                ),
            },
            "gate": {
                "is_contaminated": s2_output.baseline_contaminated,
                "gate1_passed":    bool(gate1) if gate1 is not None else None,
                "gate2_passed":    bool(gate2) if gate2 is not None else None,
                "gate3_passed":    bool(gate3) if gate3 is not None else None,
                "action": (
                    s2_output.screening.recommended_action.name
                    if s2_output.screening.recommended_action else "CONTINUE"
                ),
                "message": (
                    "; ".join(
                        f"{k}: {v.value}"
                        for k, v in s2_output.screening.gate_details.items()
                    )
                    if s2_output.screening.gate_details else ""
                ),
            },
        }

        return json.dumps(result_dict)

    except Exception as e:
        err_msg = str(e) + "\n" + traceback.format_exc()
        logging.error(f"Python engine error: {err_msg}")
        return json.dumps({
            "status":        "error",
            "error_message": err_msg,
            "anomaly":       {},
            "prototype":     {},
            "gate":          {},
        })
