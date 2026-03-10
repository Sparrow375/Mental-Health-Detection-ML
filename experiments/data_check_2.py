import pandas as pd

df = pd.read_csv(r"F:\Avaneesh\download\archive\CrossCheck_Daily_Data.csv")
print("MAX loc_dist_ep_0:", df['loc_dist_ep_0'].max())
print("MEAN loc_dist_ep_0:", df['loc_dist_ep_0'].mean())

print("\nMAX call_in_num_ep_0:", df['call_in_num_ep_0'].max())
print("\nMAX call_out_num_ep_0:", df['call_out_num_ep_0'].max())
