import pandas as pd
import numpy as np

df = pd.read_csv(r"F:\Avaneesh\download\archive\CrossCheck_Daily_Data.csv")

def check_feature_stats(feature_name):
    print(f"\n--- {feature_name} ---")
    active = df[df[feature_name] > 0][feature_name]
    print(f"Active rows: {len(active)} / {len(df)}")
    if len(active) > 0:
        print(f"Stats (Active): Mean={active.mean():.2f}, Std={active.std():.2f}, Max={active.max():.2f}")
    
    # Check a specific user who might have baseline issues
    u = df.groupby('eureka_id')[feature_name].std().sort_values().head(10)
    print("Users with lowest std:")
    print(u)

check_feature_stats('call_in_num_ep_0')
check_feature_stats('loc_dist_ep_0')
check_feature_stats('unlock_duration_ep_0')
