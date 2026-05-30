import json
import logging
import traceback
import pandas as pd
import numpy as np

from system2.pipeline import System2Pipeline
from system1 import ImprovedAnomalyDetector, PersonalityVector
from system1.user_profile import UserProfile
from system2.s1_s2_adapter import build_s1_input
from dna import build_person_dna, build_daily_vector, PersonDNA
from dna_engine import (
    compute_context_coherence,
    compute_rhythm_integrity,
    compute_session_incoherence,
    compute_texture_quality,
    compute_l2_modifier,
    update_rolling_clusters,
    get_released_evidence,
    clear_rejected_candidates,
)
from s1_profile import build_full_profile, build_l1_profile, build_l2_texture_profile, compute_cluster_mismatch


def run_analysis(json_string: str) -> str:
    """
    Main entry point for Kotlin/Chaquopy.
    Receives JSON per the Kotlin contract, passes it to System 1 + System 2,
    and returns a structured JSON result matching what Kotlin expects.

    Input JSON schema:
      {
        "current":   { featureName: value, ... },
        "baseline":  { featureName: { "mean": x, "std": y }, ... },
        "history":   [ { featureName: value, ... }, ... ],
        "day_number": int,
        "baseline_contaminated": bool,
        "gate_state": {},
        "historical_anomaly_scores": [ float, ... ],
        "sessions":  [ { session fields }, ... ],   // Level 2: last 28 days of app sessions
        "sessions_today": [ { session fields }, ... ], // Level 2: today's sessions
        "dna": { ... PersonDNA as JSON ... } or null  // Level 2: persisted DNA or null if first run
      }
    """
    try:
        data = json.loads(json_string)
        
        print(f"  [Analysis] Starting run_analysis. userId={data.get('user_id')}, date={data.get('target_date')}")
        print(f"  [Analysis] history_len={len(data.get('history', []))}, sessions_len={len(data.get('sessions', []))}")

        current      = data.get("current", {})
        baseline     = data.get("baseline", {})
        history      = data.get("history", [])
        contaminated = data.get("baseline_contaminated", False)
        day_number   = data.get("day_number", 0)
        historical_scores = data.get("historical_anomaly_scores", [])
        is_provisional = data.get("is_provisional", False)

        # Onboarding self-report calibration metrics
        phq9_score = data.get("phq9_score", 0)
        gad7_score = data.get("gad7_score", 0)
        recent_life_events_count = data.get("recent_life_events_count", 0)

        py_user_profile = UserProfile(
            phq9_score=phq9_score,
            gad7_score=gad7_score,
            recent_life_event=(recent_life_events_count > 0)
        )

        # Level 2 Behavioral DNA inputs
        sessions_28day  = data.get("sessions", [])
        sessions_today  = data.get("sessions_today", [])
        dna_json        = data.get("dna", None)

        # Existing profile for cluster persistence (Phase 3) and app DNA persistence
        existing_profile = data.get("existing_profile", None)
        existing_clusters = None
        existing_app_profiles = None
        if existing_profile and isinstance(existing_profile, dict):
            existing_clusters = existing_profile.get("anchor_clusters", None)
            if existing_clusters:
                print(f"  [Analysis] Loaded {len(existing_clusters)} existing clusters for incremental growth")
            existing_app_profiles = existing_profile.get("app_dna_profiles", None)
            if existing_app_profiles:
                print(f"  [Analysis] Loaded {len(existing_app_profiles)} existing app DNA profiles")

        # ── Build PersonalityVector baseline from Android mean/std stats ──────
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

        # ── Level 2 Behavioral DNA ─────────────────────────────────────────────
        l2_modifier = 1.0  # graceful fallback default
        dna_result = {
            "coherence": 0.0,
            "rhythm_dissolution": 0.0,
            "session_incoherence": 0.0,
            "texture_quality": 0.0,
            "l2_modifier": 1.0,
            "matched_cluster": -1,
            "candidate_active": False,
            "candidate_days": 0,
            "dna_updated_json": None,
        }

        dna = None
        if dna_json is not None and isinstance(dna_json, dict) and dna_json.get("person_id"):
            try:
                dna = PersonDNA.from_dict(dna_json)
            except Exception as e:
                print(f"  [L2] Failed to deserialize DNA: {e}")
                dna = None

        cluster_promoted = False  # Track for evidence engine grace period

        # Build DNA if sessions available and no DNA yet
        if dna is None and sessions_28day:
            try:
                dna = build_person_dna(sessions_28day, person_id="user")
                print(f"  [L2] Built new PersonDNA: {len(dna.app_profiles)} apps, K={dna.anchor_k}")
            except Exception as e:
                print(f"  [L2] Failed to build DNA: {e}")

        # Compute L2 metrics if DNA exists and today's sessions available
        if dna is not None and sessions_today:
            try:
                today_vector = build_daily_vector(
                    sessions_today, dna.app_profiles,
                    vector_mean=dna.daily_vector_mean,
                    vector_std=dna.daily_vector_std,
                )
                coherence, matched = compute_context_coherence(today_vector, dna)
                rhythm_dissolution = compute_rhythm_integrity(sessions_today, dna)
                session_incoherence = compute_session_incoherence(sessions_today, dna)
                texture_quality = compute_texture_quality(coherence, rhythm_dissolution, session_incoherence)
                l2_modifier = compute_l2_modifier(coherence, rhythm_dissolution, session_incoherence)

                dna_result["coherence"] = round(coherence, 4)
                dna_result["rhythm_dissolution"] = round(rhythm_dissolution, 4)
                dna_result["session_incoherence"] = round(session_incoherence, 4)
                dna_result["texture_quality"] = round(texture_quality, 4)
                dna_result["l2_modifier"] = round(l2_modifier, 4)
                dna_result["matched_cluster"] = matched

                # Rolling cluster discovery (only when coherence < 0.3)
                if coherence < 0.3:
                    evidence_today = float(
                        sum(abs(v) for v in current.values())
                    ) * 0.1  # rough proxy for today's evidence
                    dna, action = update_rolling_clusters(
                        today_vector, texture_quality, coherence, dna, evidence_today,
                    )
                    if action == "rejected":
                        released = get_released_evidence(dna)
                        dna = clear_rejected_candidates(dna)
                        print(f"  [L2] Candidate rejected, released evidence: {released:.4f}")
                    elif action == "promoted":
                        cluster_promoted = True
                        print(f"  [L2] Candidate promoted to cluster!")
                    elif action == "candidate_opened":
                        print(f"  [L2] New candidate cluster opened")

                # Track candidate status
                active_candidates = [c for c in dna.candidate_clusters if c.status == "evaluating"]
                dna_result["candidate_active"] = len(active_candidates) > 0
                dna_result["candidate_days"] = (
                    active_candidates[0].days_observed if active_candidates else 0
                )

                # Serialize updated DNA for Room persistence
                dna_result["dna_updated_json"] = dna.to_dict()

                print(f"  [L2] coherence={coherence:.3f} rhythm={rhythm_dissolution:.3f} "
                      f"incoherence={session_incoherence:.3f} modifier={l2_modifier:.3f}")

            except Exception as e:
                print(f"  [L2] Metric computation failed: {e}")
                l2_modifier = 1.0
        elif dna is not None:
            # DNA exists but no today sessions — serialize for persistence
            dna_result["dna_updated_json"] = dna.to_dict()

        # ── System 1 setup ─────────────────────────────────────────────────────
        s1 = ImprovedAnomalyDetector(baseline=s1_baseline)
        s1.set_user_profile(py_user_profile)

        if historical_scores:
            s1.full_anomaly_history = list(historical_scores)
            for score in historical_scores[-14:]:
                s1.anomaly_score_history.append(score)
            print(f"  Loaded {len(historical_scores)} historical anomaly scores from Room")

        # Fast-forward state using history (oldest first)
        # IMPORTANT: Only replay actual monitoring days (post-baseline), not
        # baseline collection days. The number of historical_anomaly_scores
        # tells us how many previous analysis runs exist — those are the real
        # monitoring days. Replaying baseline days inflates sustained_deviation_days
        # because the evidence engine counts each replayed day as a monitoring day.
        deviations_history = []
        num_prior_monitoring_days = len(historical_scores)
        # Only use the most recent N history days that correspond to actual
        # monitoring (post-baseline) analysis runs.
        if num_prior_monitoring_days > 0 and len(history) > num_prior_monitoring_days:
            monitoring_history = history[-num_prior_monitoring_days:]
        else:
            monitoring_history = history if num_prior_monitoring_days > 0 else []

        history_start_day = max(0, day_number - len(monitoring_history))
        print(f"  History fast-forward: {len(history)} total history days, "
              f"{num_prior_monitoring_days} prior monitoring days, "
              f"replaying {len(monitoring_history)} days (skipping "
              f"{len(history) - len(monitoring_history)} baseline days)")
        for idx, h in enumerate(monitoring_history):
            s1_report_h, _ = s1.analyze(
                h,
                deviations_history=list(deviations_history),
                day_number=history_start_day + idx,
            )
            deviations_history.append(s1_report_h.feature_deviations)

        # Analyze today — with L2 modifier and cluster promotion flag
        s1_report, daily_report = s1.analyze(
            current,
            deviations_history=list(deviations_history),
            day_number=day_number,
            l2_modifier=l2_modifier,
            cluster_just_promoted=cluster_promoted,
            is_provisional=is_provisional,
        )

        # ── System 2 setup ─────────────────────────────────────────────────────
        baseline_rows = [{"date": f"day_{d}", **baseline_means} for d in range(28)]
        baseline_df = pd.DataFrame(baseline_rows)

        pipeline = System2Pipeline()
        s1_input = build_s1_input(
            detector=s1,
            baseline_df=baseline_df,
            s1_report=s1_report,
            timeseries_days=60,
        )
        s2_output = pipeline.classify(s1_input, user_profile=py_user_profile)

        if contaminated and not s2_output.baseline_contaminated:
            s2_output.baseline_contaminated = True

        # ── Evidence score for UI ──────────────────────────────────────────────
        evidence_score = float(s1_report.evidence_accumulated)

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
        gates_fired_list = s2_output.screening.gates_fired

        # ── Build System 1 Profile (L1 + L2 decoupled) ─────────────────────────
        profile_data = None
        l1_profile = None
        l2_profile = None
        try:
            # Inject date keys so build_anchor_clusters can label member_dates
            all_daily = []
            for i, h in enumerate(history):
                h_copy = dict(h)
                if "date" not in h_copy:
                    h_copy["date"] = f"day_{i}"
                all_daily.append(h_copy)
            cur_copy = dict(current)
            if "date" not in cur_copy:
                cur_copy["date"] = "today"
            all_daily.append(cur_copy)

            # Phase 1: Build L1 and L2 independently
            l1_profile = build_l1_profile(
                daily_features_list=all_daily,
                person_id=data.get("user_id", "user"),
                existing_clusters=existing_clusters,  # Phase 3: incremental growth
                user_profile=data.get("user_profile", None),
                existing_profile=existing_profile,
                cluster_just_promoted=cluster_promoted,
            )
            print(f"  [L1 Profile] Built: {l1_profile['days_of_data']} days, "
                  f"{len(l1_profile.get('anchor_clusters', []))} clusters")

            l2_profile = build_l2_texture_profile(
                daily_features_list=all_daily,
                sessions=sessions_28day,
                anchor_clusters=l1_profile.get("anchor_clusters", []),
                existing_app_profiles=existing_app_profiles,
            )
            print(f"  [L2 Texture] Built: {len(l2_profile.get('app_dna_profiles', {}))} apps, "
                  f"{len(l2_profile.get('texture_profiles', []))} textures")

            # Merge into profile_data for backward-compatible output
            profile_data = {**l1_profile, **l2_profile}
            profile_data["profile_version"] = "2.0"
            profile_data.pop("profile_layer", None)

        except Exception as e:
            import traceback as _tb
            print(f"  [Profile] Failed to build profile: {e}\n{_tb.format_exc()}")
            profile_data = {
                "person_id": data.get("user_id", "user"),
                "profile_version": "2.0",
                "days_of_data": len(history) + 1,
                "build_error": str(e),
                "personality_vector": {"means": {}, "variances": {}, "confidence": "LOW", "feature_count": 0},
                "app_dna_profiles": {},
                "phone_dna": {},
                "anchor_clusters": [],
                "texture_profiles": [],
                "feature_importance": {},
                "group_summaries": {},
            }

        # ── Cluster-Mismatch penalty ────────────────────────────────────────────
        # Weight constant: max +35 % amplification when today is a total archetype
        # outlier. Chosen conservatively so it only noticeably fires when the
        # behavioral pattern is genuinely unprecedented (mismatch ~ 1.0).
        # Phase 1: Cluster-Mismatch uses ONLY L1 profile (not L2 DNA data)
        MISMATCH_WEIGHT = 0.35
        cluster_mismatch = 0.0
        mismatch_source = l1_profile if l1_profile else profile_data
        if mismatch_source and mismatch_source.get("anchor_clusters"):
            try:
                cluster_mismatch = compute_cluster_mismatch(current, mismatch_source)
                if cluster_mismatch > 0.0:
                    l2_modifier = l2_modifier * (1.0 + MISMATCH_WEIGHT * cluster_mismatch)
                    l2_modifier = min(l2_modifier, 2.5)  # hard ceiling: never more than 2.5×
                    print(f"  [ClusterMismatch] mismatch={cluster_mismatch:.3f} "
                          f"l2_modifier_final={l2_modifier:.3f}")
            except Exception as e:
                print(f"  [ClusterMismatch] Error: {e}")

        # Update dna_result with the final (penalty-adjusted) modifier and mismatch score
        dna_result["l2_modifier"] = round(l2_modifier, 4)
        dna_result["cluster_mismatch"] = round(cluster_mismatch, 4)

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
                "gates_fired": gates_fired_list,
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
            "dna": dna_result,
            "profile": profile_data,
            "bayesian_baseline": {
                "means": s1.bayesian_state.effective_means if s1.bayesian_state else {},
                "stds": s1.bayesian_state.effective_stds if s1.bayesian_state else {}
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
            "dna":           {},
        })
