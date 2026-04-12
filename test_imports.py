"""Quick smoke test - verify all system1 modules import cleanly."""
import sys
errors = []

try:
    from system1.data_structures import PersonProfile, PersonalityVector, SessionEvent, NotificationEvent
    print("[OK] data_structures")
except Exception as e:
    errors.append(f"data_structures: {e}")
    print(f"[FAIL] data_structures: {e}")

try:
    from system1.feature_meta import FEATURE_META, CRITICAL_FEATURES, ALL_L1_FEATURES, THRESHOLDS
    print("[OK] feature_meta")
except Exception as e:
    errors.append(f"feature_meta: {e}")
    print(f"[FAIL] feature_meta: {e}")

try:
    from system1.baseline.baseline_builder import build_baseline, build_personality_vector
    print("[OK] baseline_builder")
except Exception as e:
    errors.append(f"baseline_builder: {e}")
    print(f"[FAIL] baseline_builder: {e}")

try:
    from system1.baseline.app_dna_builder import build_all_app_dnas
    print("[OK] app_dna_builder")
except Exception as e:
    errors.append(f"app_dna_builder: {e}")
    print(f"[FAIL] app_dna_builder: {e}")

try:
    from system1.baseline.phone_dna_builder import build_phone_dna
    print("[OK] phone_dna_builder")
except Exception as e:
    errors.append(f"phone_dna_builder: {e}")
    print(f"[FAIL] phone_dna_builder: {e}")

try:
    from system1.baseline.l1_clusterer import cluster_baseline_days
    print("[OK] l1_clusterer")
except Exception as e:
    errors.append(f"l1_clusterer: {e}")
    print(f"[FAIL] l1_clusterer: {e}")

try:
    from system1.baseline.l2_texture_builder import build_texture_profiles, compute_daily_texture_vector
    print("[OK] l2_texture_builder")
except Exception as e:
    errors.append(f"l2_texture_builder: {e}")
    print(f"[FAIL] l2_texture_builder: {e}")

try:
    from system1.baseline.detector_calibration import calibrate_detector
    print("[OK] detector_calibration")
except Exception as e:
    errors.append(f"detector_calibration: {e}")
    print(f"[FAIL] detector_calibration: {e}")

try:
    from system1.scoring.l1_scorer import score_l1_day
    print("[OK] l1_scorer")
except Exception as e:
    errors.append(f"l1_scorer: {e}")
    print(f"[FAIL] l1_scorer: {e}")

try:
    from system1.scoring.l2_scorer import score_l2_day
    print("[OK] l2_scorer")
except Exception as e:
    errors.append(f"l2_scorer: {e}")
    print(f"[FAIL] l2_scorer: {e}")

try:
    from system1.engine.evidence_engine import update_evidence
    print("[OK] evidence_engine")
except Exception as e:
    errors.append(f"evidence_engine: {e}")
    print(f"[FAIL] evidence_engine: {e}")

try:
    from system1.engine.candidate_cluster import evaluate_candidate, open_candidate_window, promote_to_anchor_cluster
    print("[OK] candidate_cluster")
except Exception as e:
    errors.append(f"candidate_cluster: {e}")
    print(f"[FAIL] candidate_cluster: {e}")

try:
    from system1.engine.alert_engine import determine_alert
    print("[OK] alert_engine")
except Exception as e:
    errors.append(f"alert_engine: {e}")
    print(f"[FAIL] alert_engine: {e}")

try:
    from system1.engine.prediction_engine import generate_final_prediction
    print("[OK] prediction_engine")
except Exception as e:
    errors.append(f"prediction_engine: {e}")
    print(f"[FAIL] prediction_engine: {e}")

try:
    from system1.output.reporter import build_anomaly_report, build_daily_report
    print("[OK] reporter")
except Exception as e:
    errors.append(f"reporter: {e}")
    print(f"[FAIL] reporter: {e}")

try:
    from system1.simulation.synthetic_data_generator import generate_baseline_days, generate_depression_episode
    print("[OK] synthetic_data_generator")
except Exception as e:
    errors.append(f"synthetic_data_generator: {e}")
    print(f"[FAIL] synthetic_data_generator: {e}")

try:
    from system1.simulation.system1_runner import run_full_simulation
    print("[OK] system1_runner")
except Exception as e:
    errors.append(f"system1_runner: {e}")
    print(f"[FAIL] system1_runner: {e}")

print()
if errors:
    print(f"FAILED: {len(errors)} module(s) had import errors")
    sys.exit(1)
else:
    print("ALL 17 system1 modules imported successfully!")
    sys.exit(0)