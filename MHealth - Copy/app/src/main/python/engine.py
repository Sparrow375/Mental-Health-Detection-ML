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
    Receives JSON per the Kotlin contract, passes it to the System2Pipeline,
    and returns a structured JSON result matching what Kotlin expects.
    """
    try:
        data = json.loads(json_string)

        current = data.get("current", {})
        baseline = data.get("baseline", {})
        history = data.get("history", [])
        contaminated = data.get("baseline_contaminated", False)
        day_number = data.get("day_number", 0)

        # Build pipeline
        pipeline = System2Pipeline()

        # Extract only the fields known to the Python PersonalityVector
        # Android provides 22 fields, Python System 2 only uses 16.
        valid_fields = set(PersonalityVector.__dataclass_fields__.keys()) if hasattr(PersonalityVector, '__dataclass_fields__') else set()
        
        pv_kwargs = {}
        variances = {}
        for feat, stats in baseline.items():
            if feat in valid_fields and feat != "variances":
                pv_kwargs[feat] = float(stats.get("mean", 0.0))
                variances[feat] = float(stats.get("std", 1.0)) ** 2
                
        pv_kwargs["variances"] = variances
        
        # Instantiate the proper dataclass baseline
        s1_baseline = PersonalityVector(**pv_kwargs)

        # Step 1: Run System 1 (Anomaly Detection)
        s1 = ImprovedAnomalyDetector(baseline=s1_baseline)
        
        deviations_history = []
        
        # Fast-forward state using history
        # History is ordered oldest first. If today is 'day_number', we count up to it.
        history_start_day = max(0, day_number - len(history))
        for idx, h in enumerate(history):
            s1.analyze(h, deviations_history=list(deviations_history), day_number=history_start_day + idx)
            devs = s1.calculate_deviation_magnitude(h)
            deviations_history.append(devs)
            
        # Process today
        s1_report, daily_report = s1.analyze(current, deviations_history=list(deviations_history), day_number=day_number)

        # Step 2: Build S1Input for Pipeline via the original adapter
        # In reality, baseline is passed as a dict of stats from Room.
        # But the adapter expects a 28-day DataFrame of raw values.
        # We synthesize a steady DataFrame matching the means since we only
        # have the aggregated baseline stats available in this nightly run context.
        baseline_rows = []
        for d in range(28):
            row = {"date": f"day_{d}"}
            for feat, stats in baseline.items():
                row[feat] = stats.get("mean", 0.0)
            baseline_rows.append(row)
        baseline_df = pd.DataFrame(baseline_rows)

        s1_input = build_s1_input(
            detector=s1,
            baseline_df=baseline_df,
            s1_report=s1_report,
            timeseries_days=60
        )

        # Step 3: Run full pipeline
        s2_output = pipeline.classify(s1_input)
        
        # Respect the Android-supplied contamination flag: if Android's Room DB already
        # determined the 28-day baseline was built during an active episode, override
        # the pipeline's own detection so the reference frame is correctly set to frame1.
        if contaminated and not s2_output.baseline_contaminated:
            s2_output.baseline_contaminated = True

        # Calculate evidence for UI
        evidence_score = 0.0
        if s1_report and s1_report.feature_deviations:
            evidence_score = float(sum(abs(v) for v in s1_report.feature_deviations.values()))

        # Extract top 3 features properly
        top3_list = []
        if s2_output.explanation and hasattr(s2_output.explanation, 'top_contributing_features'):
            top3_list = [
                [f, float(val)] 
                for f, val in s2_output.explanation.top_contributing_features.items()
            ][:3]

        # Ensure gate results are booleans
        gate1 = "gate1" not in s2_output.screening.gates_fired
        gate2 = "gate2" not in s2_output.screening.gates_fired
        gate3 = "gate3" not in s2_output.screening.gates_fired
        
        # Step 4: Map back to Kotlin JSON contract
        result_dict = {
            "status": "ok",
            "anomaly": {
                "detected": s1_report.sustained_deviation_days >= 3,
                "anomaly_score": float(s1_report.overall_anomaly_score),
                "alert_level": s1_report.alert_level,
                "sustained_days": int(s1_report.sustained_deviation_days),
                "evidence": evidence_score,
                "flagged_features": list(s1_report.flagged_features),
                "pattern_type": s2_output.temporal_result.temporal_shape if s2_output.temporal_result else "stable",
                "message": daily_report.notes
            },
            "prototype": {
                "match": s2_output.disorder,
                "confidence": float(s2_output.score),
                "confidence_label": s2_output.confidence.name if s2_output.confidence else "UNCERTAIN",
                "message": s2_output.label,
                "top_3_features": top3_list,
                "all_scores": s2_output.classification.all_scores if s2_output.classification else {},
                "reference_frame": "frame1_population" if (s2_output.baseline_contaminated or contaminated) else "frame2_personal"
            },
            "gate": {
                "is_contaminated": s2_output.baseline_contaminated,
                "gate1_passed": bool(gate1) if gate1 is not None else None,
                "gate2_passed": bool(gate2) if gate2 is not None else None,
                "gate3_passed": bool(gate3) if gate3 is not None else None,
                "action": s2_output.screening.recommended_action.name if s2_output.screening.recommended_action else "CONTINUE",
                "message": "; ".join(f"{k}: {v.value}" for k, v in s2_output.screening.gate_details.items()) if s2_output.screening.gate_details else ""
            }
        }
        
        return json.dumps(result_dict)

    except Exception as e:
        err_msg = str(e) + "\n" + traceback.format_exc()
        logging.error(f"Python engine error: {err_msg}")
        return json.dumps({
            "status": "error",
            "error_message": err_msg,
            "anomaly": {},
            "prototype": {},
            "gate": {}
        })
