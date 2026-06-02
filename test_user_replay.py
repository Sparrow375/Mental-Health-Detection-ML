import json
import os
import sys
import numpy as np

sys.path.append(os.path.abspath("MHealth - Copy/app/src/main/python"))

import engine
from s1_profile import L1_CLUSTER_FEATURES, FEATURE_WEIGHTS

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_simulation(data):
    history = data.get("daily_history", [])
    print(f"Total days: {len(history)}")
    
    # Setup baseline statistics (mean and std) from history
    # Typically, baseline is built from onboarding days. Here we'll use the history means/stds
    # as the baseline reference, mimicking the device baseline.
    vectors = []
    key_mapping = {
        "sleepDurationHours": "sleepDurationHours",
        "wakeTimeHour": "wakeTimeHour",
        "sleepTimeHour": "sleepTimeHour",
        "dailyDisplacementKm": "displacementKm",
        "locationEntropy": "locationEntropy",
        "callsPerDay": "callsPerDay",
        "conversationFrequency": "conversationFrequency",
        "screenTimeHours": "screenTimeHours",
        "unlockCount": "unlockCount",
        "socialAppRatio": "socialRatio",
        "dailyStepCount": "dailyStepCount",
        "chargeRegularity": "chargeRegularity"
    }
    
    for day in history:
        metrics = day.get("metrics", {})
        vec = [float(metrics.get(key_mapping.get(feat, feat), 0.0)) for feat in L1_CLUSTER_FEATURES]
        vectors.append(vec)
        
    matrix = np.array(vectors)
    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0)
    stds_safe = np.where(stds > 1e-9, stds, 1.0)
    
    baseline_stats = {}
    for idx, feat in enumerate(L1_CLUSTER_FEATURES):
        baseline_stats[feat] = {
            "mean": float(means[idx]),
            "std": float(stds_safe[idx])
        }
        
    # We will simulate incremental nightly analysis starting from day 2
    # (since day 1 has no history to run analysis on yet)
    historical_anomaly_scores = []
    historical_l2_modifiers = []
    
    print("\nIncremental Simulation of run_analysis:")
    print(f"{'Date':<12} | {'Raw L1':<8} | {'L2 Mod':<8} | {'Effective':<10} | {'Evidence':<8} | {'Alert':<6} | {'Sustained':<10}")
    print("-" * 80)
    
    # Store results to print at the end
    sim_results = []
    
    for d in range(1, len(history)):
        today_data = history[d]
        today_date = today_data.get("date")
        
        # Build the payload
        payload = {
            "current": today_data.get("metrics", {}),
            "baseline": baseline_stats,
            "history": [h.get("metrics", {}) for h in history[:d]],
            "day_number": d + 1,
            "baseline_contaminated": False,
            "gate_state": {},
            "historical_anomaly_scores": list(historical_anomaly_scores),
            "historical_l2_modifiers": list(historical_l2_modifiers),
            "sessions": [],
            "sessions_today": [], # for simplicity, keep empty to trigger fallback/external modifier, or let L2 evaluate
            "existing_profile": None,
            "user_id": "test@user.com",
            "target_date": today_date
        }
        
        # Run analysis
        result_json_str = engine.run_analysis(json.dumps(payload))
        res = json.loads(result_json_str)
        
        if res.get("status") == "error":
            print(f"Error on day {today_date}: {res.get('error_message')}")
            break
            
        anomaly = res.get("anomaly", {})
        dna = res.get("dna", {})
        
        raw_l1 = anomaly.get("anomaly_score", 0.0)
        l2_mod = dna.get("l2_modifier", 1.0)
        eff_score = raw_l1 * l2_mod
        evidence = anomaly.get("evidence", 0.0)
        alert = anomaly.get("alert_level", "green")
        sustained = anomaly.get("sustained_days", 0)
        
        # Store for next iteration
        historical_anomaly_scores.append(eff_score)
        historical_l2_modifiers.append(l2_mod)
        
        print(f"{today_date:<12} | {raw_l1:8.4f} | {l2_mod:8.4f} | {eff_score:10.4f} | {evidence:8.4f} | {alert:<6} | {sustained:<10}")
        sim_results.append({
            "date": today_date,
            "raw_l1": raw_l1,
            "l2_mod": l2_mod,
            "effective": eff_score,
            "evidence": evidence,
            "alert": alert,
            "sustained": sustained
        })

    # Test Case 2: June 02 replay with actual historical modifiers from the reports in the JSON
    print("\n" + "=" * 50)
    print("Test Case: Replaying June 02 using actual historical modifiers")
    print("=" * 50)
    
    # Extract historical scores and modifiers from the first 6 reports
    reports = data.get("analysis_reports", [])
    hist_eff_scores = []
    hist_mods = []
    for r in reports[:-1]: # skip June 02 itself
        date = r.get("date")
        score = r.get("anomalyScore") # effective score in database
        msg = r.get("anomalyMessage", "")
        # Extract modifier from message, e.g., "modifier=0.15"
        mod = 1.0
        if "modifier=" in msg:
            try:
                mod = float(msg.split("modifier=")[1].split(")")[0])
            except Exception:
                mod = 1.0
        hist_eff_scores.append(score)
        hist_mods.append(mod)
        print(f"Report Date: {date} | Stored Effective Score: {score:.4f} | Extracted L2 Mod: {mod:.2f}")
        
    # Build payload for June 02
    june02_data = history[-1] # June 02 features
    payload_june02 = {
        "current": june02_data.get("metrics", {}),
        "baseline": baseline_stats,
        "history": [h.get("metrics", {}) for h in history[:-1]],
        "day_number": len(history),
        "baseline_contaminated": False,
        "gate_state": {},
        "historical_anomaly_scores": hist_eff_scores,
        "historical_l2_modifiers": hist_mods,
        "sessions": [],
        "sessions_today": [],
        "existing_profile": None,
        "user_id": "test@user.com",
        "target_date": "2026-06-02"
    }
    
    # 1. Run WITH our fix (passing historical_l2_modifiers)
    result_with_fix = json.loads(engine.run_analysis(json.dumps(payload_june02)))
    anom_with = result_with_fix.get("anomaly", {})
    
    # 2. Run WITHOUT our fix (passing empty historical_l2_modifiers to simulate the bug)
    payload_no_fix = dict(payload_june02)
    payload_no_fix["historical_l2_modifiers"] = []
    result_no_fix = json.loads(engine.run_analysis(json.dumps(payload_no_fix)))
    anom_no = result_no_fix.get("anomaly", {})
    
    print("\nResult on 2026-06-02:")
    print(f"With Fix (L2 Modifiers passed):    Sustained Days = {anom_with.get('sustained_days')}, Evidence = {anom_with.get('evidence'):.4f}, Message = '{anom_with.get('message')}'")
    print(f"Without Fix (Bug - No L2 Modifiers): Sustained Days = {anom_no.get('sustained_days')}, Evidence = {anom_no.get('evidence'):.4f}, Message = '{anom_no.get('message')}'")
        
if __name__ == "__main__":
    data_path = r"F:\Avaneesh\download\testData\26-2-1.json"
    data = load_data(data_path)
    run_simulation(data)
