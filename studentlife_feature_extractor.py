"""
StudentLife Feature Extractor
Extract System 1-compatible features from Student Life dataset
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import sys

# Import the loader
from studentlife_loader import StudentLifeLoader, get_phq9_severity


class StudentLifeFeatureExtractor:
    """Extract features that match System 1's PersonalityVector"""
    
    def __init__(self, user_data, user_id):
        self.data = user_data
        self.user_id = user_id
        self.daily_features = {}
        
    def extract_phonelock_features(self):
        """Extract screen_time_hours and unlock_count from phonelock data"""
        df = self.data['phonelock']
        
        if len(df) == 0:
            return {}
        
        daily_stats = defaultdict(lambda: {'unlock_count': 0, 'screen_time_seconds': 0})
        
        for idx, row in df.iterrows():
            try:
                start = pd.to_datetime(row['start'], unit='s')
                end = pd.to_datetime(row['end'], unit='s')
                
                date_key = start.date()
                
                daily_stats[date_key]['unlock_count'] += 1
                duration = (end - start).total_seconds()
                if duration > 0 and duration < 86400:  # Max 24 hours
                    daily_stats[date_key]['screen_time_seconds'] += duration
                    
            except Exception as e:
                continue
        
        # Convert to hours
        features = {}
        for date, stats in daily_stats.items():
            features[date] = {
                'unlock_count': stats['unlock_count'],
                'screen_time_hours': stats['screen_time_seconds'] / 3600.0
            }
        
        return features
    
    def extract_gps_features(self):
        """Extract daily_displacement_km, location_entropy, places_visited from GPS"""
        df = self.data['gps']
        
        if len(df) == 0:
            return {}
        
        daily_stats = defaultdict(lambda: {'locations': [], 'total_distance': 0})
        
        for idx, row in df.iterrows():
            try:
                # StudentLife uses 'time' column (already in timestamp format)
                timestamp = pd.to_datetime(row.name, unit='s')  # Index is timestamp
                date_key = timestamp.date()
                
                lat = row['latitude']
                lon = row['longitude']
                
                daily_stats[date_key]['locations'].append((lat, lon))
                
            except Exception as e:
                continue
        
        # Calculate features
        features = {}
        for date, stats in daily_stats.items():
            locs = stats['locations']
            
            if len(locs) < 2:
                continue
            
            # Calculate total displacement (sum of distances between consecutive points)
            total_dist = 0
            for i in range(len(locs) - 1):
                lat1, lon1 = locs[i]
                lat2, lon2 = locs[i+1]
                dist = self.haversine_distance(lat1, lon1, lat2, lon2)
                if dist < 100:  # Filter out GPS errors > 100km jumps
                    total_dist += dist
            
            # Calculate unique places (cluster locations within 100m)
            unique_places = self.count_unique_places(locs)
            
            # Calculate location entropy (spread of locations)
            entropy = self.calculate_location_entropy(locs)
            
            features[date] = {
                'daily_displacement_km': total_dist,
                'places_visited': unique_places,
                'location_entropy': entropy
            }
        
        return features
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two GPS coordinates in km"""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Earth radius in km
        
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def count_unique_places(self, locations, threshold_km=0.1):
        """Count unique places (cluster locations within threshold)"""
        if len(locations) == 0:
            return 0
        
        clusters = []
        
        for lat, lon in locations:
            # Check if this location is close to any existing cluster
            matched = False
            for cluster_lat, cluster_lon in clusters:
                dist = self.haversine_distance(lat, lon, cluster_lat, cluster_lon)
                if dist < threshold_km:
                    matched = True
                    break
            
            if not matched:
                clusters.append((lat, lon))
        
        return len(clusters)
    
    def calculate_location_entropy(self, locations):
        """Calculate location entropy (measure of location diversity)"""
        if len(locations) < 2:
            return 0.0
        
        # Cluster locations
        clusters = defaultdict(int)
        
        for lat, lon in locations:
            # Round to ~100m precision
            key = (round(lat, 3), round(lon, 3))
            clusters[key] += 1
        
        # Calculate Shannon entropy
        total = len(locations)
        entropy = 0
        for count in clusters.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def extract_sms_features(self):
        """Extract texts_per_day from SMS data"""
        df = self.data['sms']
        
        if len(df) == 0:
            return {}
        
        daily_stats = defaultdict(int)
        
        for idx, row in df.iterrows():
            try:
                timestamp = pd.to_datetime(row['timestamp'], unit='s')
                date_key = timestamp.date()
                daily_stats[date_key] += 1
            except Exception as e:
                continue
        
        features = {}
        for date, count in daily_stats.items():
            features[date] = {'texts_per_day': count}
        
        return features
    
    def extract_app_usage_features(self):
        """Extract social_app_ratio from app usage data (running tasks sampling)"""
        df = self.data['app_usage']
        
        if len(df) == 0:
            return {}
        
        # Strip column names
        df.columns = df.columns.str.strip()
        
        # Define social apps
        social_apps = [
            'facebook', 'twitter', 'instagram', 'snapchat', 'messenger',
            'whatsapp', 'telegram', 'tiktok', 'reddit', 'discord', 'tumblr',
            'wechat', 'kik', 'pinterest', 'linkedin'
        ]
        
        daily_stats = defaultdict(lambda: {'total_count': 0, 'social_count': 0})
        
        for idx, row in df.iterrows():
            try:
                # Use timestamp for sampling
                ts = row.get('timestamp')
                if ts is None:
                    continue
                    
                timestamp = pd.to_datetime(ts, unit='s')
                date_key = timestamp.date()
                
                # Check package name
                pkg_name = str(row.get('RUNNING_TASKS_topActivity_mPackage', '')).lower()
                
                daily_stats[date_key]['total_count'] += 1
                
                if any(social in pkg_name for social in social_apps):
                    daily_stats[date_key]['social_count'] += 1
                    
            except Exception as e:
                continue
        
        features = {}
        for date, stats in daily_stats.items():
            if stats['total_count'] > 0:
                ratio = stats['social_count'] / stats['total_count']
                features[date] = {'social_app_ratio': ratio}
        
        return features

    def extract_call_features(self):
        """Extract calls_per_day from call logs"""
        df = self.data['calls']
        
        if len(df) == 0:
            return {}
        
        # Strip column names
        df.columns = df.columns.str.strip()
        
        daily_stats = defaultdict(int)
        
        for idx, row in df.iterrows():
            try:
                # Timestamp is likely in seconds, check column name
                ts = row.get('timestamp', row.get('time', None))
                if ts is None:
                    continue
                    
                timestamp = pd.to_datetime(ts, unit='s')
                date_key = timestamp.date()
                daily_stats[date_key] += 1
            except Exception as e:
                continue
        
        features = {}
        for date, count in daily_stats.items():
            features[date] = {'calls_per_day': count}
        
        return features

    def extract_phonecharge_features(self):
        """Extract charge_duration_hours from phonecharge data"""
        df = self.data['phonecharge']
        
        if len(df) == 0:
            return {}
            
        # Strip column names
        df.columns = df.columns.str.strip()
            
        daily_stats = defaultdict(float)
        
        for idx, row in df.iterrows():
            try:
                start = pd.to_datetime(row['start'], unit='s')
                end = pd.to_datetime(row['end'], unit='s')
                duration = (end - start).total_seconds()
                
                if duration > 0 and duration < 86400:
                    date_key = start.date()
                    daily_stats[date_key] += duration
            except Exception:
                continue
                
        features = {}
        for date, seconds in daily_stats.items():
            features[date] = {'charge_duration_hours': seconds / 3600.0}
            
        return features

    def extract_conversation_features(self):
        """Extract conversation_duration_hours and frequency"""
        df = self.data['conversation']
        
        if len(df) == 0:
            return {}
            
        # Strip column names (remove leading spaces)
        df.columns = df.columns.str.strip()
            
        daily_stats = defaultdict(lambda: {'duration': 0, 'count': 0})
        
        for idx, row in df.iterrows():
            try:
                start = pd.to_datetime(row['start_timestamp'], unit='s')
                end = pd.to_datetime(row['end_timestamp'], unit='s')
                duration = (end - start).total_seconds()
                
                if duration > 0 and duration < 86400:
                    date_key = start.date()
                    daily_stats[date_key]['duration'] += duration
                    daily_stats[date_key]['count'] += 1
            except Exception:
                continue
                
        features = {}
        for date, stats in daily_stats.items():
            features[date] = {
                'conversation_duration_hours': stats['duration'] / 3600.0,
                'conversation_frequency': stats['count']
            }
            
        return features

    def extract_dark_features(self):
        """Extract dark_duration_hours as sleep proxy"""
        df = self.data['dark']
        
        if len(df) == 0:
            return {}
            
        # Strip column names
        df.columns = df.columns.str.strip()
            
        daily_stats = defaultdict(float)
        
        for idx, row in df.iterrows():
            try:
                start = pd.to_datetime(row['start'], unit='s')
                end = pd.to_datetime(row['end'], unit='s')
                duration = (end - start).total_seconds()
                
                # Filter for likely sleep (long duration > 45 mins)
                # Or just sum it all up as "dark time"
                if duration > 0 and duration < 86400:
                    date_key = start.date()
                    daily_stats[date_key] += duration
            except Exception:
                continue
                
        features = {}
        for date, seconds in daily_stats.items():
            features[date] = {'dark_duration_hours': seconds / 3600.0}
            
        # Also map to sleep_duration_hours if it looks reasonable (e.g. > 3 hours)
        for date, feat in features.items():
            if feat['dark_duration_hours'] > 3.0:
                 feat['sleep_duration_hours'] = feat['dark_duration_hours']
            
        return features

    def extract_all_features(self):
        """Extract all available features and merge by date"""
        print(f"\nExtracting features for {self.user_id}...")
        
        # Extract from each data source
        phonelock_features = self.extract_phonelock_features()
        gps_features = self.extract_gps_features()
        sms_features = self.extract_sms_features()
        app_features = self.extract_app_usage_features()
        call_features = self.extract_call_features()
        charge_features = self.extract_phonecharge_features()
        conv_features = self.extract_conversation_features()
        dark_features = self.extract_dark_features()
        
        # Merge all features by date
        all_dates = set()
        all_dates.update(phonelock_features.keys())
        all_dates.update(gps_features.keys())
        all_dates.update(sms_features.keys())
        all_dates.update(app_features.keys())
        all_dates.update(call_features.keys())
        all_dates.update(charge_features.keys())
        all_dates.update(conv_features.keys())
        all_dates.update(dark_features.keys())
        
        all_dates = sorted(list(all_dates))
        
        # Build daily feature vectors
        daily_data = []
        
        for date in all_dates:
            features = {'date': pd.Timestamp(date)}
            
            # Helper to safely update
            def safe_update(source_dict, default_dict):
                if date in source_dict:
                    features.update(source_dict[date])
                else:
                    features.update(default_dict)

            safe_update(phonelock_features, {'unlock_count': np.nan, 'screen_time_hours': np.nan})
            safe_update(gps_features, {'daily_displacement_km': np.nan, 'places_visited': np.nan, 'location_entropy': np.nan})
            safe_update(sms_features, {'texts_per_day': np.nan})
            safe_update(app_features, {'social_app_ratio': np.nan})
            safe_update(call_features, {'calls_per_day': np.nan})
            safe_update(charge_features, {'charge_duration_hours': np.nan})
            safe_update(conv_features, {'conversation_duration_hours': np.nan, 'conversation_frequency': np.nan})
            safe_update(dark_features, {'dark_duration_hours': np.nan, 'sleep_duration_hours': np.nan})
            
            # Add stub features (still not available)
            features['voice_pitch_mean'] = np.nan
            features['voice_pitch_std'] = np.nan
            features['voice_energy_mean'] = np.nan
            features['voice_speaking_rate'] = np.nan # Could Map conversation duration here?
            features['unique_contacts'] = np.nan
            features['response_time_minutes'] = np.nan
            features['home_time_ratio'] = np.nan
            features['wake_time_hour'] = np.nan
            features['sleep_time_hour'] = np.nan
            
            # Heuristic: Map conversation duration to voice_energy/speaking_rate proxy if needed
            if pd.notna(features.get('conversation_duration_hours')):
                 features['voice_energy_mean'] = 0.5 # Default middle value if conversation exists
                 
            daily_data.append(features)
        
        df = pd.DataFrame(daily_data)
        
        print(f"\n  Total: {len(df)} days of data")
        if len(df) > 0:
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"\n  Available features coverage:")
            for col in df.columns:
                if col != 'date':
                    coverage = (1 - df[col].isna().sum() / len(df)) * 100
                    status = "✓" if coverage > 50 else "✗"
                    print(f"    {status} {col:30s}: {coverage:5.1f}%")
        
        return df


def main():
    """Test feature extraction"""
    print("="*80)
    print("STUDENTLIFE FEATURE EXTRACTION TEST")
    print("="*80)
    
    dataset_path = r'F:\Avaneesh\download\student\dataset'
    
    loader = StudentLifeLoader(dataset_path)
    loader.load_phq9_scores()
    users = loader.get_available_users()
    
    if len(users) == 0:
        print("No users found!")
        return
    
    # Test with first user
    test_user = users[0]
    user_data = loader.load_user_data(test_user)
    
    # Extract features
    extractor = StudentLifeFeatureExtractor(user_data, test_user)
    df = extractor.extract_all_features()
    
    # Save to CSV
    output_file = f'studentlife_features_{test_user}.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved features to: {output_file}")
    
    # Display sample
    print(f"\n{'='*80}")
    print("SAMPLE DATA (first 5 days):")
    print(f"{'='*80}")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df.head())
    
    print(f"\n{'='*80}")
    print("READY FOR SYSTEM 1 DETECTOR!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
