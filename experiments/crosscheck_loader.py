"""
CrossCheck Dataset Loader
Loads consolidated daily data from CrossCheck dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

class CrossCheckLoader:
    def __init__(self, daily_csv_path):
        self.daily_csv_path = daily_csv_path
        self.df = None
        
    def load_data(self):
        print(f"Loading CrossCheck daily data from {self.daily_csv_path}...")
        self.df = pd.read_csv(self.daily_csv_path)
        
        # Convert date to datetime
        # format is YYYYMMDD
        self.df['date'] = pd.to_datetime(self.df['day'].astype(str), format='%Y%m%d')
        
        # Sort by user and date
        self.df = self.df.sort_values(['eureka_id', 'date'])
        
        print(f"✓ Loaded {len(self.df)} rows for {self.df['eureka_id'].nunique()} users")
        return self.df
    
    def get_users(self):
        if self.df is None:
            self.load_data()
        return sorted(self.df['eureka_id'].unique().tolist())
    
    def get_user_features(self, user_id):
        if self.df is None:
            self.load_data()
            
        user_df = self.df[self.df['eureka_id'] == user_id].copy()
        
        # Mapping CrossCheck Daily to System 1 Features
        # PersonalityVector features:
        # voice_pitch_mean, voice_pitch_std, voice_energy_mean, voice_speaking_rate
        # screen_time_hours, unlock_count, social_app_ratio, calls_per_day, texts_per_day, unique_contacts, response_time_minutes
        # daily_displacement_km, location_entropy, home_time_ratio, places_visited
        # wake_time_hour, sleep_time_hour, sleep_duration_hours, dark_duration_hours, charge_duration_hours
        # conversation_duration_hours, conversation_frequency
        
        # CrossCheck columns:
        # ep_4 is usually the Daily Total
        
        features_df = pd.DataFrame()
        features_df['date'] = user_df['date']
        
        # Voice (Proxy)
        # Using audio_amp_mean_ep_0 as voice energy proxy
        features_df['voice_energy_mean'] = user_df['audio_amp_mean_ep_0']
        features_df['voice_pitch_mean'] = np.nan
        features_df['voice_pitch_std'] = np.nan
        features_df['voice_speaking_rate'] = np.nan
        
        # Activity
        features_df['screen_time_hours'] = user_df['unlock_duration_ep_0'] / 3600.0
        features_df['unlock_count'] = user_df['unlock_num_ep_0']
        features_df['social_app_ratio'] = np.nan # Not in daily file
        features_df['calls_per_day'] = user_df['call_in_num_ep_0'] + user_df['call_out_num_ep_0']
        features_df['texts_per_day'] = user_df['sms_in_num_ep_0'] + user_df['sms_out_num_ep_0']
        features_df['unique_contacts'] = np.nan
        features_df['response_time_minutes'] = user_df['ema_resp_time_median']
        
        # Movement
        features_df['daily_displacement_km'] = user_df['loc_dist_ep_0'] / 1000.0
        features_df['location_entropy'] = np.nan # Maybe calculate from epochs if we had loc breakdown, but daily is summary
        features_df['home_time_ratio'] = np.nan
        features_df['places_visited'] = user_df['loc_visit_num_ep_0']
        
        # Sleep
        features_df['sleep_duration_hours'] = user_df['sleep_duration']
        # features_df['wake_time_hour'] = ... map from sleep_end?
        # sleep_end/start in CrossCheck are often minutes from midnight or similar
        features_df['wake_time_hour'] = user_df['sleep_end'] / 60.0 # Assuming minutes
        features_df['sleep_time_hour'] = user_df['sleep_start'] / 60.0
        features_df['dark_duration_hours'] = user_df['light_mean_ep_0'] # Actually light, not dark, but use as proxy
        features_df['charge_duration_hours'] = np.nan
        
        # Social
        features_df['conversation_duration_hours'] = user_df['audio_convo_duration_ep_0'] / 3600.0
        features_df['conversation_frequency'] = user_df['audio_convo_num_ep_0']
        
        # Ground Truth (EMA Depressed)
        features_df['ema_depressed'] = user_df['ema_DEPRESSED']
        
        return features_df

if __name__ == "__main__":
    loader = CrossCheckLoader(r"F:\Avaneesh\download\archive\CrossCheck_Daily_Data.csv")
    users = loader.get_users()
    print(f"Sample users: {users[:5]}")
    df = loader.get_user_features(users[0])
    print(df.head())
