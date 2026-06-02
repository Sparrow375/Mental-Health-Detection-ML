import json
import numpy as np

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze(data):
    history = data.get("daily_history", [])
    print(f"User: {data.get('profile', {}).get('userId')}")
    print(f"Baseline Ready: {data.get('profile', {}).get('baselineReady')}")
    
    print("\nDaily History Summary:")
    print(f"{'Date':<12} | {'Anomaly Score':<13} | {'Steps':<6} | {'Displace (km)':<13} | {'Sleep (h)':<9} | {'Wake Time':<9} | {'Screen (h)':<10}")
    print("-" * 90)
    for day in history:
        date = day.get("date", "")
        score = day.get("anomaly_score", -1.0)
        metrics = day.get("metrics", {})
        steps = metrics.get("dailyStepCount", 0.0)
        displace = metrics.get("displacementKm", 0.0)
        sleep = metrics.get("sleepDurationHours", 0.0)
        wake = metrics.get("wakeTimeHour", 0.0)
        screen = metrics.get("screenTimeHours", 0.0)
        
        score_str = f"{score:.4f}" if score >= 0 else "BASELINE"
        print(f"{date:<12} | {score_str:<13} | {steps:<6.0f} | {displace:<13.2f} | {sleep:<9.2f} | {wake:<9.2f} | {screen:<10.2f}")

if __name__ == "__main__":
    analyze(load_data(r"F:\Avaneesh\download\testData\26-2-1.json"))
