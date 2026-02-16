"""
StudentLife Dataset Loader and Preprocessor
Extracts features from StudentLife dataset to match System 1's 18-feature PersonalityVector
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from glob import glob
from collections import defaultdict
import json

class StudentLifeLoader:
    """Load and preprocess StudentLife dataset"""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.users = []
        self.phq9_scores = {}
        
    def load_phq9_scores(self):
        """Load PHQ-9 depression scores"""
        phq9_path = os.path.join(self.dataset_path, 'survey', 'PHQ-9.csv')
        
        try:
            df = pd.read_csv(phq9_path)
            print(f"✓ Loaded PHQ-9 data: {len(df)} rows")
            print(f"  Columns: {list(df.columns[:3])}...")
            
            # Convert PHQ-9 responses to numeric scores
            # "Not at all" = 0, "Several days" = 1, "More than half the days" = 2, "Nearly every day" = 3
            score_map = {
                'Not at all': 0,
                'Several days': 1,
                'More than half the days': 2,
                'Nearly every day': 3
            }
            
            for idx, row in df.iterrows():
                uid = row['uid']
                survey_type = row['type']  # pre or post
                
                # Sum scores from questions 1-9 (skip question 10 which is functional impairment)
                total_score = 0
                for col_idx in range(2, 11):  # Columns 2-10 are the 9 PHQ questions
                    response = row.iloc[col_idx]
                    if pd.notna(response) and response in score_map:
                        total_score += score_map[response]
                
                key = f"{uid}_{survey_type}"
                self.phq9_scores[key] = total_score
                
            print(f"  PHQ-9 scores loaded for {len(self.phq9_scores)} user-assessments")
            print(f"  Example: {list(self.phq9_scores.items())[:3]}")
            
            return self.phq9_scores
            
        except Exception as e:
            print(f"✗ Error loading PHQ-9: {e}")
            return {}
    
    def get_available_users(self):
        """Find all users with phonelock data (proxy for activity)""" 
        phonelock_path = os.path.join(self.dataset_path, 'sensing', 'phonelock')
        
        if os.path.exists(phonelock_path):
            files = glob(os.path.join(phonelock_path, 'phonelock_u*.csv'))
            users = [os.path.basename(f).replace('phonelock_', '').replace('.csv', '') for f in files]
            self.users = sorted(users)
            print(f"✓ Found {len(self.users)} users: {self.users[:10]}...")
            return self.users
        else:
            print(f"✗ Path not found: {phonelock_path}")
            return []
    
    def load_phonelock_data(self, user_id):
        """Load phone lock/unlock events → screen_time, unlock_count"""
        filepath = os.path.join(self.dataset_path, 'sensing', 'phonelock', f'phonelock_{user_id}.csv')
        
        try:
            df = pd.read_csv(filepath)
            if len(df) > 0:
                print(f"  ✓ PhoneLock: {len(df)} events, columns: {list(df.columns)}")
                return df
        except Exception as e:
            print(f"  ✗ PhoneLock error: {e}")
        
        return pd.DataFrame()
    
    def load_gps_data(self, user_id):
        """Load GPS data → daily_displacement, location_entropy, places_visited"""
        filepath = os.path.join(self.dataset_path, 'sensing', 'gps', f'gps_{user_id}.csv')
        
        try:
            df = pd.read_csv(filepath)
            if len(df) > 0:
                print(f"  ✓ GPS: {len(df)} points")
                return df
        except Exception as e:
            print(f"  ✗ GPS error: {e}")
        
        return pd.DataFrame()
    
    def load_sms_data(self, user_id):
        """Load SMS data → texts_per_day"""
        filepath = os.path.join(self.dataset_path, 'sms', f'sms_{user_id}.csv')
        
        try:
            df = pd.read_csv(filepath)
            if len(df) > 0:
                print(f"  ✓ SMS: {len(df)} messages")
                return df
        except Exception as e:
            print(f"  ✗ SMS error: {e}")
        
        return pd.DataFrame()
    
    def load_call_data(self, user_id):
        """Load call logs → calls_per_day"""
        filepath = os.path.join(self.dataset_path, 'call_log', f'call_log_{user_id}.csv')
        
        try:
            df = pd.read_csv(filepath)
            if len(df) > 0:
                print(f"  ✓ Calls: {len(df)} calls")
                return df
        except Exception as e:
            print(f"  ✗ Calls error: {e}")
        
        return pd.DataFrame()
    
    def load_app_usage_data(self, user_id):
        """Load app usage → social_app_ratio"""
        filepath = os.path.join(self.dataset_path, 'app_usage', f'running_app_{user_id}.csv')
        
        try:
            df = pd.read_csv(filepath)
            if len(df) > 0:
                print(f"  ✓ AppUsage: {len(df)} events")
                return df
        except Exception as e:
            print(f"  ✗ AppUsage error: {e}")
        
        return pd.DataFrame()
        
    def load_phonecharge_data(self, user_id):
        """Load phone charge data → charge_duration"""
        filepath = os.path.join(self.dataset_path, 'sensing', 'phonecharge', f'phonecharge_{user_id}.csv')
        
        try:
            df = pd.read_csv(filepath)
            if len(df) > 0:
                print(f"  ✓ PhoneCharge: {len(df)} events")
                return df
        except Exception as e:
            # Try alt path if not in sensing
            print(f"  ✗ PhoneCharge error: {e}")
        
        return pd.DataFrame()
        
    def load_conversation_data(self, user_id):
        """Load conversation data → conversation_duration"""
        filepath = os.path.join(self.dataset_path, 'sensing', 'conversation', f'conversation_{user_id}.csv')
        
        try:
            df = pd.read_csv(filepath)
            if len(df) > 0:
                print(f"  ✓ Conversation: {len(df)} events")
                return df
        except Exception as e:
            print(f"  ✗ Conversation error: {e}")
        
        return pd.DataFrame()
        
    def load_dark_data(self, user_id):
        """Load dark environment data → sleep proxy"""
        filepath = os.path.join(self.dataset_path, 'sensing', 'dark', f'dark_{user_id}.csv')
        
        try:
            df = pd.read_csv(filepath)
            if len(df) > 0:
                print(f"  ✓ Dark: {len(df)} events")
                return df
        except Exception as e:
            print(f"  ✗ Dark error: {e}")
        
        return pd.DataFrame()
        
    def load_activity_data(self, user_id):
        """Load activity inference data → movement"""
        filepath = os.path.join(self.dataset_path, 'sensing', 'activity', f'activity_{user_id}.csv')
        
        try:
            df = pd.read_csv(filepath)
            if len(df) > 0:
                print(f"  ✓ Activity: {len(df)} points")
                return df
        except Exception as e:
            print(f"  ✗ Activity error: {e}")
        
        return pd.DataFrame()
    
    def load_user_data(self, user_id):
        """Load all available sensor data for a user"""
        print(f"\nLoading data for {user_id}...")
        
        data = {
            'phonelock': self.load_phonelock_data(user_id),
            'gps': self.load_gps_data(user_id),
            'sms': self.load_sms_data(user_id),
            'calls': self.load_call_data(user_id),
            'app_usage': self.load_app_usage_data(user_id),
            'phonecharge': self.load_phonecharge_data(user_id),
            'conversation': self.load_conversation_data(user_id),
            'dark': self.load_dark_data(user_id),
            'activity': self.load_activity_data(user_id)
        }
        
        return data


def main():
    """Test the loader"""
    print("="*80)
    print("STUDENTLIFE DATASET EXPLORATION")
    print("="*80)
    
    dataset_path = r'F:\Avaneesh\download\student\dataset'
    
    loader = StudentLifeLoader(dataset_path)
    
    # Load PHQ-9 scores
    phq9_scores = loader.load_phq9_scores()
    
    # Get available users
    users = loader.get_available_users()
    
    if len(users) > 0:
        # Test with first user
        test_user = users[0]
        print(f"\n{'='*80}")
        print(f"TESTING WITH {test_user}")
        print(f"{'='*80}")
        
        data = loader.load_user_data(test_user)
        
        print(f"\n{'='*80}")
        print("DATA AVAILABILITY SUMMARY")
        print(f"{'='*80}")
        for key, df in data.items():
            status = f"✓ {len(df)} rows" if len(df) > 0 else "✗ No data"
            print(f"  {key:15s}: {status}")
        
        # Get PHQ-9 score
        pre_key = f"{test_user}_pre"
        post_key = f"{test_user}_post"
        
        print(f"\n{'='*80}")
        print(f"PHQ-9 SCORES FOR {test_user}")
        print(f"{'='*80}")
        
        if pre_key in phq9_scores:
            score = phq9_scores[pre_key]
            severity = get_phq9_severity(score)
            print(f"  Pre-study:  {score}/27 ({severity})")
        
        if post_key in phq9_scores:
            score = phq9_scores[post_key]
            severity = get_phq9_severity(score)
            print(f"  Post-study: {score}/27 ({severity})")
    
    print(f"\n{'='*80}")
    print(f"NEXT STEPS:")
    print(f"{'='*80}")
    print(f"  1. Extract daily features from sensor data")
    print(f"  2. Build PersonalityVector baseline for each user")
    print(f"  3. Run System 1 detector on real data")
    print(f"  4. Compare anomaly scores to PHQ-9 scores")


def get_phq9_severity(score):
    """Convert PHQ-9 score to severity category"""
    if score <= 4:
        return "Minimal"
    elif score <= 9:
        return "Mild"
    elif score <= 14:
        return "Moderate"
    elif score <= 19:
        return "Moderately Severe"
    else:
        return "Severe"


if __name__ == "__main__":
    main()
