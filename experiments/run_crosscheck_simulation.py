"""
Run System 1 Anomaly Detection on CrossCheck Dataset
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from crosscheck_loader import CrossCheckLoader

# Add parent dir to path to import system1
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from system1 import ImprovedAnomalyDetector, PersonalityVector, PersonalityVector

def run_simulation(data_path, output_dir='experiments'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    loader = CrossCheckLoader(data_path)
    users = loader.get_users()
    
    all_results = []
    
    print(f"\nStarting simulation for {len(users)} users...")
    
    for i, user_id in enumerate(users):
        print(f"[{i+1}/{len(users)}] Analyzing {user_id}...")
        
        try:
            df = loader.get_user_features(user_id)
            
            # Filter days with minimum data coverage (e.g. have screen time or unlock)
            # CrossCheck has many empty days
            df = df.dropna(subset=['screen_time_hours', 'unlock_count'], how='all')
            
            if len(df) < 70: # Need at least 10 weeks (56 days for baseline + 14 for testing)
                print(f"  ! Insufficient data ({len(df)} days), skipping.")
                continue
                
            # Use first 56 valid days as baseline (8 weeks / 2 months)
            baseline_df = df.iloc[:56]
            monitoring_df = df.iloc[56:]
            
            # Create PersonalityVector for baseline
            baseline_params = {}
            variances = {}
            
            pv_fields = [
                'voice_pitch_mean', 'voice_pitch_std', 'voice_energy_mean', 'voice_speaking_rate',
                'screen_time_hours', 'unlock_count', 'social_app_ratio', 'calls_per_day', 
                'texts_per_day', 'unique_contacts', 'response_time_minutes',
                'daily_displacement_km', 'location_entropy', 'home_time_ratio', 'places_visited',
                'wake_time_hour', 'sleep_time_hour', 'sleep_duration_hours', 'dark_duration_hours', 'charge_duration_hours',
                'conversation_duration_hours', 'conversation_frequency'
            ]
            
            # Define sensible limits for human behavior to cap bad sensor data (GPS glitches, etc)
            LIMITS = {
                'calls_per_day': 50,
                'texts_per_day': 100,
                'screen_time_hours': 24,
                'daily_displacement_km': 500, # 500km is a long road trip, 12000 is a glitch
                'unlock_count': 300,
                'sleep_duration_hours': 24,
                'conversation_duration_hours': 24
            }

            # Behavioral variance floors (Standard Deviations we expect even in "stable" people)
            VAR_FLOORS = {
                'calls_per_day': 2.0,
                'texts_per_day': 5.0,
                'screen_time_hours': 1.0,
                'daily_displacement_km': 5.0,
                'unlock_count': 10.0,
                'sleep_duration_hours': 1.5,
                'wake_time_hour': 1.0,
                'sleep_time_hour': 1.0,
                'conversation_duration_hours': 1.0,
                'conversation_frequency': 2.0,
                'voice_energy_mean': 0.1,
                'response_time_minutes': 30.0
            }
            
            for field in pv_fields:
                if field in baseline_df.columns:
                    # Cap extreme outliers in baseline too
                    vals = baseline_df[field].clip(upper=LIMITS.get(field, 1e9))
                    val = vals.mean()
                    std = vals.std()
                    
                    # ENFORCE ROBUST VARIANCE FLOOR:
                    # Use a behavioral floor to prevent division by near-zero.
                    # This ensures that small changes (e.g. 0 to 1 call) are not flagged as massive anomalies.
                    floor = VAR_FLOORS.get(field, max(0.1, abs(val) * 0.1))
                    
                    baseline_params[field] = val if pd.notna(val) else 0.0
                    variances[field] = max(std, floor) if pd.notna(std) else floor
                else:
                    baseline_params[field] = 0.0
                    variances[field] = 1.0
            
            baseline_vector = PersonalityVector(**baseline_params)
            baseline_vector.variances = variances
            
            # Initialize detector
            detector = ImprovedAnomalyDetector(baseline_vector)
            
            # Run simulation
            daily_reports = []
            deviations_history = []
            
            for day_idx, (_, row) in enumerate(monitoring_df.iterrows()):
                # Cap monitoring data too
                current_data = {}
                for f in pv_fields:
                    raw_val = row[f] if pd.notna(row[f]) else baseline_params[f]
                    current_data[f] = min(raw_val, LIMITS.get(f, 1e9))
                
                report, daily_report = detector.analyze(current_data, deviations_history, day_idx)
                
                # Update history
                deviations_history.append(report.feature_deviations)
                
                # Convert daily_report to dict for JSON serialization
                daily_dict = {
                    'day_number': daily_report.day_number,
                    'date': daily_report.date.strftime('%Y-%m-%d'),
                    'anomaly_score': float(daily_report.anomaly_score),
                    'alert_level': daily_report.alert_level,
                    'flagged_features': daily_report.flagged_features,
                    'pattern_type': daily_report.pattern_type,
                    'sustained_deviation_days': daily_report.sustained_deviation_days,
                    'evidence_accumulated': float(daily_report.evidence_accumulated),
                    'top_deviations': {k: float(v) for k, v in daily_report.top_deviations.items()},
                    'notes': daily_report.notes
                }
                daily_reports.append(daily_dict)
                
            # Generate final prediction
            # We don't have a formal post-study PHQ-9, but we can average the ema_depressed
            avg_ema_depressed = monitoring_df['ema_depressed'].mean()
            
            final_pred = detector.generate_final_prediction("crosscheck", user_id, len(monitoring_df))
            
            result = {
                'user_id': user_id,
                'monitoring_days': len(monitoring_df),
                'anomaly_score': float(final_pred.final_anomaly_score),
                'anomaly_detected': bool(final_pred.sustained_anomaly_detected),
                'confidence': float(final_pred.confidence),
                'pattern': final_pred.pattern_identified,
                'ema_depressed_avg': float(avg_ema_depressed) if pd.notna(avg_ema_depressed) else None,
                'daily_reports': daily_reports,
                'recommendation': final_pred.recommendation
            }
            
            all_results.append(result)
            print(f"  ✓ Analysis complete. Score: {result['anomaly_score']:.3f}, Detected: {result['anomaly_detected']}")
            
        except Exception as e:
            print(f"  ✗ Error analyzing user {user_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            
    # Save results
    final_output = {
        'total_users': len(all_results),
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'results': all_results
    }
    
    results_file = os.path.join(output_dir, 'crosscheck_results.json')
    with open(results_file, 'w') as f:
        json.dump(final_output, f, indent=4)
        
    print(f"\nDONE! Results saved to {results_file}")
    return results_file

if __name__ == "__main__":
    data_path = r"F:\Avaneesh\download\archive\CrossCheck_Daily_Data.csv"
    run_simulation(data_path)
