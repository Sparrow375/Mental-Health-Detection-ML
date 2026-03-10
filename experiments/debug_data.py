import pandas as pd
import numpy as np

df = pd.read_csv(r"F:\Avaneesh\download\archive\CrossCheck_Daily_Data.csv")
u004 = df[df['eureka_id'] == 'u004'].copy()
u004['day'] = pd.to_datetime(u004['day'].astype(str), format='%Y%m%d')

# Filter for relevant columns
cols = ['day', 'unlock_duration_ep_4', 'unlock_num_ep_4']
data = u004[cols]

baseline = data.iloc[:56]
monitoring = data.iloc[56:]

b_mean = baseline['unlock_duration_ep_4'].mean()
b_std = baseline['unlock_duration_ep_4'].std()

print(f"Baseline Mean Unlock Duration (sec): {b_mean:.2f}")
print(f"Baseline Std Unlock Duration (sec):  {b_std:.2f}")

# Find days with massive z-scores
monitoring = monitoring.copy()
monitoring['z_score'] = (monitoring['unlock_duration_ep_4'] - b_mean) / b_std

print("\nTop Monitoring days for u004 by Unlock Duration Z-score:")
print(monitoring.sort_values('z_score', ascending=False).head(10))

print("\nCheck for outliers in raw data (> 24 hours):")
print(f"Count > 86400s: {len(df[df['unlock_duration_ep_4'] > 86400])}")
print(f"Max unlock duration: {df['unlock_duration_ep_4'].max()}")
