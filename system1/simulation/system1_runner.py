"""
System 1 Runner — end-to-end simulation of the Behavioral Anomaly Detection pipeline.
"""

from __future__ import annotations

import datetime
import json
from collections import deque
from typing import Dict, List, Optional

import numpy as np

from ..data_structures import (
    PersonProfile, EvidenceState, CandidateState, DailyReport,
    AnomalyReport, FinalPrediction, ConfidenceTier,
)
from ..feature_meta import ALL_L1_FEATURES, THRESHOLDS
from ..baseline.baseline_builder import build_baseline
from ..scoring.l1_scorer import score_l1_day
from ..scoring.l2_scorer import score_l2_day
from ..engine.evidence_engine import update_evidence
from ..engine.candidate_cluster import (
    evaluate_candidate, open_candidate_window, promote_to_anchor_cluster,
)
from ..engine.alert_engine import determine_alert
from ..engine.prediction_engine import generate_final_prediction
from ..output.reporter import build_anomaly_report, build_daily_report
from .synthetic_data_generator import (
    generate_baseline_days, generate_depression_episode, generate_anxiety_episode,
    generate_healthy_monitoring, generate_session_events, generate_notification_events,
)


def run_full_simulation(
    scenario: str = "depression",
    baseline_days: int = 28,
    monitoring_days: int = 14,
    severity: float = 0.7,
    seed: int = 42,
) -> Dict:
    """
    Run a full end-to-end simulation.

    Scenarios: 'depression', 'anxiety', 'healthy'

    Returns dict with:
        - profile: PersonProfile summary
        - daily_reports: list of DailyReport dicts
        - anomaly_reports: list of AnomalyReport dicts
        - final_prediction: FinalPrediction dict
        - evidence_timeline: list of evidence values
    """
    print(f"\n{'='*60}")
    print(f"System 1 Behavioral Anomaly Detection Simulation")
    print(f"Scenario: {scenario} | Baseline: {baseline_days}d | Monitoring: {monitoring_days}d")
    print(f"{'='*60}\n")

    # ── Phase 0: Generate baseline data ──────────────────────────────────
    print("Phase 0: Generating baseline data...")
    base_features, base_dates = generate_baseline_days(n_days=baseline_days, seed=seed)
    base_sessions = generate_session_events(base_dates, seed=seed)
    base_notifications = generate_notification_events(base_dates, seed=seed)

    # ── Phase 1: Build baseline ──────────────────────────────────────────
    print("Phase 1: Building baseline profile...")
    tier = (
        ConfidenceTier.LOW if baseline_days < 28 else
        ConfidenceTier.MEDIUM if baseline_days < 60 else
        ConfidenceTier.HIGH
    )
    profile = build_baseline(
        patient_id="sim_patient_001",
        daily_features=base_features,
        dates=base_dates,
        sessions=base_sessions,
        notifications=base_notifications,
        confidence_tier=tier,
    )
    print(f"  PersonalityVector: {len(profile.personality_vector.means)} features")
    print(f"  AppDNAs: {len(profile.app_dnas)} apps")
    print(f"  Anchor clusters: {len(profile.anchor_clusters)}")
    print(f"  Texture profiles: {len(profile.texture_profiles)}")

    # ── Phase 2-3: Generate monitoring data ──────────────────────────────
    print(f"\nPhase 2-3: Generating monitoring data ({scenario})...")
    start_date = datetime.date(2025, 1, 1) + datetime.timedelta(days=baseline_days)
    monitor_dates = [
        (start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(monitoring_days)
    ]

    if scenario == "depression":
        monitor_features = generate_depression_episode(
            n_days=monitoring_days, severity=severity, seed=seed + 100
        )
        degraded = True
    elif scenario == "anxiety":
        monitor_features = generate_anxiety_episode(
            n_days=monitoring_days, severity=severity, seed=seed + 200
        )
        degraded = True
    else:
        monitor_features = generate_healthy_monitoring(
            n_days=monitoring_days, seed=seed + 300
        )
        degraded = False

    monitor_sessions = generate_session_events(
        monitor_dates, seed=seed + 400, degraded=degraded,
        degradation_factor=severity if degraded else 0.0,
    )
    monitor_notifications = generate_notification_events(
        monitor_dates, seed=seed + 500, degraded=degraded,
    )

    # ── Phase 2-6: Daily scoring loop ────────────────────────────────────
    print(f"\nPhase 2-6: Running daily scoring pipeline...")
    evidence_state = EvidenceState()
    candidate_state = CandidateState()
    recent_values: Dict[str, deque] = {f: deque(maxlen=7) for f in ALL_L1_FEATURES}
    daily_scores_history: List[float] = []
    daily_reports: List[Dict] = []
    anomaly_reports: List[Dict] = []
    evidence_timeline: List[float] = []
    candidate_day_counter = 0

    for day_idx in range(monitoring_days):
        date_str = monitor_dates[day_idx]
        today_features = monitor_features[day_idx]

        # Update rolling values
        for feat in ALL_L1_FEATURES:
            recent_values[feat].append(today_features.get(feat, 0.0))

        # Get today's sessions and notifications
        day_sessions = [
            s for s in monitor_sessions
            if datetime.datetime.fromtimestamp(s.open_ts / 1000.0).strftime("%Y-%m-%d") == date_str
        ]
        day_notifs = [
            n for n in monitor_notifications
            if datetime.datetime.fromtimestamp(n.arrival_ts / 1000.0).strftime("%Y-%m-%d") == date_str
        ]

        # L1 scoring
        l1_result = score_l1_day(today_features, profile.personality_vector, recent_values)

        # L2 scoring
        l2_result = score_l2_day(today_features, day_sessions, day_notifs, profile)

        # Candidate cluster evaluation
        if l2_result.candidate_flag and candidate_state.status == "CLOSED":
            candidate_state = open_candidate_window(date_str)
            candidate_day_counter = 0

        if candidate_state.status == "EVALUATING":
            candidate_day_counter += 1
            candidate_state, evidence_state, action = evaluate_candidate(
                candidate_state, l2_result, today_features,
                evidence_state, candidate_day_counter,
            )
            if action == "promote":
                profile = promote_to_anchor_cluster(candidate_state, profile)
                candidate_state = CandidateState()  # Reset
                print(f"  Day {day_idx+1}: Candidate cluster PROMOTED")
            elif action == "reject_clinical":
                print(f"  Day {day_idx+1}: Candidate cluster REJECTED (clinical onset)")
                candidate_state = CandidateState()  # Reset
            elif action == "hold":
                # Skip normal evidence accumulation during evaluation
                pass

        # Evidence update (skip if candidate is being evaluated)
        if candidate_state.status != "EVALUATING":
            evidence_state = update_evidence(
                evidence_state, l1_result.composite_score, l2_result.modifier, date_str
            )

        evidence_timeline.append(round(evidence_state.evidence_accumulated, 3))

        # Alert determination
        effective = l1_result.composite_score * l2_result.modifier
        alert_level, pattern_type, flagged, top_5 = determine_alert(
            evidence_state, effective, l1_result.weighted_z_scores, daily_scores_history
        )

        daily_scores_history.append(effective)

        # Build reports
        anomaly_report = build_anomaly_report(
            timestamp=date_str,
            l1_result=l1_result,
            l2_result=l2_result,
            evidence_state=evidence_state,
            alert_level=alert_level,
            pattern_type=pattern_type,
            flagged_features=flagged,
            top_5_deviations=top_5,
        )
        daily_report = build_daily_report(
            day_number=day_idx + 1,
            date_str=date_str,
            anomaly_report=anomaly_report,
        )

        anomaly_reports.append(anomaly_report.to_dict())
        daily_reports.append(daily_report.to_dict())

        # Print daily summary
        alert_icon = {"green": "🟢", "yellow": "🟡", "orange": "🟠", "red": "🔴"}.get(alert_level, "⚪")
        print(f"  Day {day_idx+1:2d} | {date_str} | L1={l1_result.composite_score:.3f} | "
              f"L2×{l2_result.modifier:.2f} | Eff={effective:.3f} | "
              f"Ev={evidence_state.evidence_accumulated:.2f} | "
              f"Sust={evidence_state.sustained_deviation_days}d | {alert_icon} {alert_level}")

    # ── Phase 8: Final prediction ────────────────────────────────────────
    print(f"\nPhase 8: Generating final prediction...")
    prediction = generate_final_prediction(
        patient_id="sim_patient_001",
        evidence_state=evidence_state,
        profile=profile,
        monitoring_days=monitoring_days,
        recent_daily_scores=daily_scores_history,
    )

    print(f"\n{'='*60}")
    print(f"FINAL PREDICTION")
    print(f"  Recommendation: {prediction.recommendation}")
    print(f"  Sustained Anomaly: {prediction.sustained_anomaly}")
    print(f"  Confidence: {prediction.confidence:.1%}")
    print(f"  Pattern: {prediction.pattern_identified}")
    print(f"  Max Evidence: {evidence_state.max_evidence:.3f}")
    print(f"  Max Sustained Days: {evidence_state.max_sustained_days}")
    print(f"{'='*60}\n")

    return {
        "scenario": scenario,
        "baseline_days": baseline_days,
        "monitoring_days": monitoring_days,
        "severity": severity,
        "profile_summary": {
            "patient_id": profile.patient_id,
            "n_features": len(profile.personality_vector.means),
            "n_app_dnas": len(profile.app_dnas),
            "n_clusters": len(profile.anchor_clusters),
            "confidence": profile.personality_vector.confidence.value,
        },
        "daily_reports": daily_reports,
        "anomaly_reports": anomaly_reports,
        "final_prediction": prediction.to_dict(),
        "evidence_timeline": evidence_timeline,
    }


if __name__ == "__main__":
    import sys
    scenario = sys.argv[1] if len(sys.argv) > 1 else "depression"
    result = run_full_simulation(scenario=scenario)
    print(f"\nJSON output written to stdout ({len(result['daily_reports'])} daily reports)")