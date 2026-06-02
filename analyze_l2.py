import json
import os
import sys

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_l2_analysis(data):
    dna = data.get("dna_profile") or data.get("profile") or {}
    print(f"DNA Keys: {list(dna.keys())}")
    
    l2_clusters = dna.get("l2_anchor_clusters", [])
    print(f"\nNumber of L2 Anchor Clusters: {len(l2_clusters)}")
    for c in l2_clusters:
        print(f"\nL2 Cluster {c.get('cluster_id')}:")
        print(f"  Member Count: {c.get('member_count')}")
        print(f"  Member Dates: {c.get('member_dates')}")
        print(f"  Radius: {c.get('radius')}")
        print(f"  Centroid PCA 2D: {c.get('centroid_pca_2d')}")
        print("  Centroid Features:")
        features = c.get("centroid_features", {})
        for k, v in sorted(features.items()):
            print(f"    {k:<22}: {v:.4f}")
            
if __name__ == "__main__":
    data_path = r"F:\Avaneesh\download\testData\26-2-3.json"
    data = load_data(data_path)
    run_l2_analysis(data)
