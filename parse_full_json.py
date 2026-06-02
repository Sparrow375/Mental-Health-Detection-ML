import json

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze(data):
    print("=== Profile ===")
    print(json.dumps(data.get("profile"), indent=2))
    
    print("\n=== Today Live ===")
    print(json.dumps(data.get("today_live"), indent=2)[:1000]) # Print first 1000 chars
    
    print("\n=== DNA Profile ===")
    dna = data.get("dna_profile")
    if dna:
        print(f"DNA keys: {list(dna.keys())}")
        print(f"Anchor clusters count: {len(dna.get('anchor_clusters', []))}")
        print(f"Texture profiles count: {len(dna.get('texture_profiles', []))}")
        print(f"App DNA profiles count: {len(dna.get('app_dna_profiles', {}))}")
        
        print("\n=== DNA Anchor Clusters ===")
        for idx, cluster in enumerate(dna.get("anchor_clusters", [])):
            print(f"  Cluster {idx}: id={cluster.get('cluster_id')}, member_days={cluster.get('member_days')}, member_dates={cluster.get('member_dates')}")
            
        print("\n=== DNA L2 Anchor Clusters ===")
        for idx, cluster in enumerate(dna.get("l2_anchor_clusters", [])):
            print(f"  L2 Cluster {idx}: id={cluster.get('cluster_id')}, member_days={cluster.get('member_days')}, member_dates={cluster.get('member_dates')}")
        
    print("\n=== Analysis Reports ===")
    reports = data.get("analysis_reports", [])
    print(f"Number of reports: {len(reports)}")
    for i, r in enumerate(reports):
        print(f"  Report {i}: date={r.get('date') or r.get('target_date')}, keys={list(r.keys())}")
        if 'anomaly' in r:
            print(f"    Anomaly: {r['anomaly'].get('anomaly_score')}, alert={r['anomaly'].get('alert_level')}")
        if 'prototype' in r:
            print(f"    Prototype: match={r['prototype'].get('match')}, conf={r['prototype'].get('confidence')}")

if __name__ == "__main__":
    analyze(load_data(r"F:\Avaneesh\download\testData\26-2-3.json"))
