import pandas as pd
import numpy as np

df = pd.read_csv(r"F:\Avaneesh\download\archive\CrossCheck_Daily_Data.csv")

for feat in ['call_in_num_ep_0', 'loc_dist_ep_0', 'unlock_duration_ep_0']:
    data = df[feat]
    print(f"{feat} | Mean: {data.mean():.2f} | Std: {data.std():.2f} | Max: {data.max():.2f}")
