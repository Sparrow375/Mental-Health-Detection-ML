import json
import numpy as np
import os
import sys

sys.path.append(os.path.abspath("MHealth - Copy/app/src/main/python"))

from s1_profile import _meanshift, L1_CLUSTER_FEATURES, FEATURE_WEIGHTS, _clinical_weighted_pca

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_comparison(data):
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
        
        vec = []
        for feat in L1_CLUSTER_FEATURES:
            mapped_key = key_mapping.get(feat, feat)
            val = metrics.get(mapped_key, 0.0)
            vec.append(float(val))
        vectors.append(vec)
        dates.append(date)
        
    matrix = np.array(vectors)
    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0)
    stds_safe = np.where(stds > 1e-9, stds, 1.0)
    matrix_norm = (matrix - means) / stds_safe
    weights = [FEATURE_WEIGHTS.get(f, 1.0) for f in L1_CLUSTER_FEATURES]
    
    projected, pca_components, pca_mean = _clinical_weighted_pca(matrix_norm, weights, target_variance=0.85)
    
    # Method 1: Current implementation
    n = len(projected)
    n_neighbors = int(n * 0.3)
    if n_neighbors < 1:
        n_neighbors = 1
    pairwise = np.linalg.norm(projected[:, None] - projected[None, :], axis=2)
    sorted_dists = np.sort(pairwise, axis=1)
    kth_dists = sorted_dists[:, n_neighbors - 1]
    current_bw = float(np.median(kth_dists))
    
    # Method 2: Quantile of all pairwise distances (excluding self-distances)
    # Get all unique pairwise distances
    triu_indices = np.triu_indices(n, k=1)
    unique_dists = pairwise[triu_indices]
    sklearn_all_pairs_bw = float(np.percentile(unique_dists, 30)) # 30th percentile
    
    # Method 3: sklearn estimate_bandwidth quantile=0.3 on all pairs (including/excluding self)
    # sklearn's estimate_bandwidth is:
    # pairwise_distances = distance.pairwise_distances(X)
    # bandwidth = np.percentile(pairwise_distances, quantile * 100)
    # which includes the diagonal (zeros).
    sklearn_with_diag_bw = float(np.percentile(pairwise, 30))
    
    print(f"Bandwidths computed:")
    print(f"  Current Method (median of nearest-neighbor): {current_bw:.4f}")
    print(f"  Quantile of all pairwise (excl. self, 30%): {sklearn_all_pairs_bw:.4f}")
    print(f"  Quantile of all pairwise (incl. self, 30%): {sklearn_with_diag_bw:.4f}")
    
    for bw_name, bw in [
        ("Current Method", current_bw), 
        ("All-Pairs 30% (excl. self)", sklearn_all_pairs_bw),
        ("All-Pairs 30% (incl. self)", sklearn_with_diag_bw)
    ]:
        print(f"\n--- Running clustering with bandwidth {bw_name} = {bw:.4f} ---")
        clusters = _meanshift(projected, bandwidth=bw)
        print(f"Found {len(clusters)} clusters:")
        for cid, indices in clusters:
            cluster_dates = [dates[idx] for idx in indices]
            print(f"  Cluster {cid}: member_count={len(indices)}, dates={cluster_dates}")

if __name__ == "__main__":
    data_path = r"F:\Avaneesh\download\testData\26-2-1.json"
    data = load_data(data_path)
    run_comparison(data)
