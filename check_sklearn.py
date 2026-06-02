import json
import numpy as np
import os
import sys
from sklearn.cluster import estimate_bandwidth

sys.path.append(os.path.abspath("MHealth - Copy/app/src/main/python"))

from s1_profile import _meanshift, L1_CLUSTER_FEATURES, FEATURE_WEIGHTS, _clinical_weighted_pca

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_sklearn_check(data):
    history = data.get("daily_history", [])
    vectors = []
    dates = []
    
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
        date = day.get("date", "")
        metrics = day.get("metrics", {})
        vec = [float(metrics.get(key_mapping.get(feat, feat), 0.0)) for feat in L1_CLUSTER_FEATURES]
        vectors.append(vec)
        dates.append(date)
        
    matrix = np.array(vectors)
    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0)
    stds_safe = np.where(stds > 1e-9, stds, 1.0)
    matrix_norm = (matrix - means) / stds_safe
    weights = [FEATURE_WEIGHTS.get(f, 1.0) for f in L1_CLUSTER_FEATURES]
    
    projected, pca_components, pca_mean = _clinical_weighted_pca(matrix_norm, weights, target_variance=0.85)
    
    sklearn_bw = estimate_bandwidth(projected, quantile=0.3)
    print(f"Sklearn estimate_bandwidth = {sklearn_bw:.4f}")
    
    clusters = _meanshift(projected, bandwidth=sklearn_bw)
    print(f"Found {len(clusters)} clusters:")
    for cid, indices in clusters:
        cluster_dates = [dates[idx] for idx in indices]
        print(f"  Cluster {cid}: member_count={len(indices)}, dates={cluster_dates}")

if __name__ == "__main__":
    data_path = r"F:\Avaneesh\download\testData\26-2-1.json"
    data = load_data(data_path)
    run_sklearn_check(data)
