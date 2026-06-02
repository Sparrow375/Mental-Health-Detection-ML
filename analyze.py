import json
import numpy as np
import os
import sys

# Add the main python source folder to python path
sys.path.append(os.path.abspath("MHealth - Copy/app/src/main/python"))

from s1_profile import _meanshift, L1_CLUSTER_FEATURES, FEATURE_WEIGHTS, _clinical_weighted_pca

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_clustering_on_data(data):
    history = data.get("daily_history", [])
    print(f"Total days in daily_history: {len(history)}")
    
    # Extract features for each day
    vectors = []
    dates = []
    
    # We need to map database keys to the exact keys used in L1_CLUSTER_FEATURES
    # Let's check s1_s2_adapter.py or engine.py or just do a direct mapping here
    # L1_CLUSTER_FEATURES = [
    #     "sleepDurationHours", "wakeTimeHour", "sleepTimeHour",
    #     "dailyDisplacementKm", "locationEntropy",
    #     "callsPerDay", "conversationFrequency", "screenTimeHours",
    #     "unlockCount", "socialAppRatio", "dailyStepCount", "chargeRegularity"
    # ]
    
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
        
        vec = []
        for feat in L1_CLUSTER_FEATURES:
            mapped_key = key_mapping.get(feat, feat)
            val = metrics.get(mapped_key, 0.0)
            vec.append(float(val))
        vectors.append(vec)
        dates.append(date)
        
    matrix = np.array(vectors)
    
    # Let's print out the features for each day
    print("\nDaily Features:")
    print(f"{'Date':<12} " + " ".join([f"{feat[:8]:>8}" for feat in L1_CLUSTER_FEATURES]))
    for i, date in enumerate(dates):
        print(f"{date:<12} " + " ".join([f"{matrix[i, j]:8.2f}" for j in range(len(L1_CLUSTER_FEATURES))]))
    
    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0)
    stds_safe = np.where(stds > 1e-9, stds, 1.0)
    matrix_norm = (matrix - means) / stds_safe
    weights = [FEATURE_WEIGHTS.get(f, 1.0) for f in L1_CLUSTER_FEATURES]
    
    print("\nFeature Means:")
    for j, feat in enumerate(L1_CLUSTER_FEATURES):
        print(f"  {feat:<25}: mean={means[j]:.4f}, std={stds[j]:.4f}, weight={weights[j]:.1f}")
        
    # Run Clinical-Weighted PCA
    projected, pca_components, pca_mean = _clinical_weighted_pca(matrix_norm, weights, target_variance=0.85)
    print(f"\nProjected PCA coordinates (shape: {projected.shape}):")
    for i, date in enumerate(dates):
        print(f"  {date}: {projected[i]}")
        
    # Run Mean Shift clustering
    print("\nRunning Mean-Shift clustering...")
    clusters = _meanshift(projected)
    
    print(f"\nClustering result: Found {len(clusters)} clusters")
    for cid, indices in clusters:
        cluster_dates = [dates[idx] for idx in indices]
        print(f"  Cluster {cid}: member_count={len(indices)}, dates={cluster_dates}")
        
    # Pairwise distances
    print("\nPairwise distances between projected PCA points:")
    pairwise = np.linalg.norm(projected[:, None] - projected[None, :], axis=2)
    print("      " + " ".join([f"{d[-5:]:>8}" for d in dates]))
    for i, date in enumerate(dates):
        print(f"{date[-5:]}: " + " ".join([f"{pairwise[i, j]:8.3f}" for j in range(len(dates))]))

if __name__ == "__main__":
    data_path = r"F:\Avaneesh\download\testData\26-2-3.json"
    data = load_data(data_path)
    run_clustering_on_data(data)
