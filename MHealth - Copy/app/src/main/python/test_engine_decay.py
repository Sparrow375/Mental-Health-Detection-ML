import json
from datetime import datetime, timedelta
import engine
from system1 import feature_meta
import random
from system2.config import POPULATION_NORMS

# We also need some baseline config to make baseline_ready = True
baseline_stats = {}
for k in feature_meta.ALL_L1_FEATURES:
    # Map camelCase to snake_case for population norms
    snake_k = "".join(["_" + c.lower() if c.isupper() else c for c in k])
    if snake_k in POPULATION_NORMS:
        norm = POPULATION_NORMS[snake_k]
        baseline_stats[k] = {"mean": norm["mean"], "std": norm["std"]}
    else:
        baseline_stats[k] = {"mean": 0.0, "std": 1.0}

def make_daily_features(date_str, is_anomalous=False):
    feat = {}
    for k in feature_meta.ALL_L1_FEATURES:
        mean = baseline_stats[k]["mean"]
        feat[k] = random.gauss(mean, 0.1)
    
    # specific high-variance features
    feat["sleepDurationHours"] = random.gauss(baseline_stats.get("sleepDurationHours", {}).get("mean", 8.0), 0.5)
    feat["screenTimeHours"] = random.gauss(baseline_stats.get("screenTimeHours", {}).get("mean", 4.0), 0.5)
    feat["dailyDisplacementKm"] = random.gauss(baseline_stats.get("dailyDisplacementKm", {}).get("mean", 10.0), 1.0)
    
    if is_anomalous:
        # Cause a large deviation
        feat["sleepDurationHours"] = 1.0
        feat["screenTimeHours"] = 16.0
        feat["dailyDisplacementKm"] = 0.5
        feat["callsPerDay"] = 0.0
        feat["conversationFrequency"] = 0.0
        feat["socialAppRatio"] = 0.8
        
    feat["date"] = date_str
    return feat

def main():
    print("Starting end-to-end synthetic test for Evidence Engine decay logic...")
    
    # 1. Create 30 days of normal baseline
    history = []
    start_date = datetime(2023, 1, 1)
    
    for i in range(30):
        d_str = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        history.append(make_daily_features(d_str, is_anomalous=False))
        
    print("\n--- Pushing anomalous days (Day 31-35) ---")
    
    historical_anomaly_scores = []
    gate_state = {}
    profile_json = {}
    
    for i in range(30, 45):
        d_str = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        is_anomalous = (30 <= i < 35) # 5 days of anomaly
        today_features = make_daily_features(d_str, is_anomalous=is_anomalous)
        
        # Prepare JSON payload mimicking NightlyAnalysisWorker
        payload = {
            "day_number": i + 1,
            "baseline_contaminated": False,
            "gate_state": gate_state,
            "historical_anomaly_scores": historical_anomaly_scores,
            "current": today_features,
            "baseline": baseline_stats,
            "history": history[-14:], # last 14 days
            "sessions": [],
            "sessions_today": [],
            "existing_profile": profile_json
        }
        
        result_json_str = engine.run_analysis(json.dumps(payload))
        res = json.loads(result_json_str)
        
        anomaly_dict = res.get("anomaly", {})
        ev = anomaly_dict.get("evidence", 0.0)
        score = anomaly_dict.get("anomaly_score", 0.0)
        flagged = anomaly_dict.get("flagged_features", [])
        gates = anomaly_dict.get("gates_fired", [])
        
        historical_anomaly_scores.append(score)
        
        print(f"Day {i+1} ({'Anomalous' if is_anomalous else 'Normal'}): Score={score:.2f}, Evidence={ev:.2f}, Flagged={flagged}, Gates={gates}")
        
        # Append to history for next day
        history.append(today_features)

if __name__ == "__main__":
    main()
