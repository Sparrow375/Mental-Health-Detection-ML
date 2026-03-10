import pandas as pd

df = pd.read_csv(r"F:\Avaneesh\download\archive\CrossCheck_Daily_Data.csv")
# Find a row where there is activity
active_rows = df[df['unlock_duration_ep_0'] > 0]
if not active_rows.empty:
    row = active_rows.iloc[0]
    print(f"User: {row['eureka_id']}, Day: {row['day']}")
    print(f"unlock_duration_ep_0: {row['unlock_duration_ep_0']}")
    print(f"unlock_duration_ep_1: {row['unlock_duration_ep_1']}")
    print(f"unlock_duration_ep_2: {row['unlock_duration_ep_2']}")
    print(f"unlock_duration_ep_3: {row['unlock_duration_ep_3']}")
    print(f"unlock_duration_ep_4: {row['unlock_duration_ep_4']}")
    total_1_4 = row['unlock_duration_ep_1'] + row['unlock_duration_ep_2'] + row['unlock_duration_ep_3'] + row['unlock_duration_ep_4']
    print(f"Sum of 1-4: {total_1_4}")
else:
    print("No rows with unlock activity found.")
