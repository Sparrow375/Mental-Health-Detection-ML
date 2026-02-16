"""
Run System 1 Anomaly Detector on StudentLife Dataset
Real-world validation against PHQ-9 depression scores
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import System 1 components
from system1 import (
    PersonalityVector,
    ImprovedAnomalyDetector,
    SyntheticDataGenerator
)

# Import StudentLife components
from studentlife_loader import StudentLifeLoader, get_phq9_severity
from studentlife_feature_extractor import StudentLifeFeatureExtractor


def create_baseline_from_studentlife(df, baseline_days=28):
    """
    Create a PersonalityVector baseline from StudentLife data
    Uses first N days of data to establish baseline
    """
    print(f"\n  Creating baseline from first {baseline_days} days...")
    
    # Get baseline period
    baseline_df = df.head(baseline_days).copy()
    
    # Calculate mean and variance for each feature
    baseline_params = {}
    variances = {}
    
    feature_columns = [
        'voice_pitch_mean', 'voice_pitch_std', 'voice_energy_mean', 'voice_speaking_rate',
        'screen_time_hours', 'unlock_count', 'social_app_ratio',
        'calls_per_day', 'texts_per_day', 'unique_contacts', 'response_time_minutes',
        'daily_displacement_km', 'location_entropy', 'home_time_ratio', 'places_visited',
        'wake_time_hour', 'sleep_time_hour', 'sleep_duration_hours',
        'dark_duration_hours', 'charge_duration_hours', 
        'conversation_duration_hours', 'conversation_frequency'
    ]
    
    for feat in feature_columns:
        if feat in baseline_df.columns:
            # Calculate mean, handling NaN
            values = baseline_df[feat].dropna()
            
            if len(values) >= 3:  # Need at least 3 values
                baseline_params[feat] = float(values.mean())
                variances[feat] = float(values.std() + 0.01)  # Add small epsilon
            else:
                # Use a reasonable default if not enough data
                baseline_params[feat] = get_default_baseline_value(feat)
                variances[feat] = baseline_params[feat] * 0.15  # 15% variance
        else:
            baseline_params[feat] = get_default_baseline_value(feat)
            variances[feat] = baseline_params[feat] * 0.15
    
    # Create PersonalityVector
    baseline = PersonalityVector(**baseline_params)
    baseline.variances = variances
    
    # Debug info
    print(f"  ✓ Baseline established with {len(baseline_params)} features")
    print(f"  Sample values:")
    print(f"    Screen time: {baseline.screen_time_hours:.2f} ± {variances['screen_time_hours']:.2f} hours")
    print(f"    Texts/day: {baseline.texts_per_day:.2f} ± {variances['texts_per_day']:.2f}")
    if 'conversation_duration_hours' in baseline_params:
         print(f"    Conversation: {baseline.conversation_duration_hours:.2f} ± {variances['conversation_duration_hours']:.2f} hours")
    
    return baseline, baseline_df


def get_default_baseline_value(feature_name):
    """Get reasonable default values for features not in StudentLife"""
    defaults = {
        'voice_pitch_mean': 180.0,
        'voice_pitch_std': 15.0,
        'voice_energy_mean': 0.65,
        'voice_speaking_rate': 3.5,
        'screen_time_hours': 4.5,
        'unlock_count': 50.0,
        'social_app_ratio': 0.30,
        'calls_per_day': 2.0,
        'texts_per_day': 25.0,
        'unique_contacts': 8.0,
        'response_time_minutes': 15.0,
        'daily_displacement_km': 5.0,
        'location_entropy': 2.0,
        'home_time_ratio': 0.65,
        'places_visited': 4.0,
        'wake_time_hour': 8.0,
        'sleep_time_hour': 23.5,
        'sleep_duration_hours': 7.5,
        'dark_duration_hours': 8.0,
        'charge_duration_hours': 6.0,
        'conversation_duration_hours': 1.5,
        'conversation_frequency': 10.0,
    }
    return defaults.get(feature_name, 1.0)


def run_system1_on_studentlife_user(user_id, dataset_path):
    """Run System 1 detector on a single StudentLife user"""
    
    print(f"\n{'='*80}")
    print(f"ANALYZING STUDENT: {user_id}")
    print(f"{'='*80}")
    
    # Load user data
    loader = StudentLifeLoader(dataset_path)
    phq9_scores = loader.load_phq9_scores()
    user_data = loader.load_user_data(user_id)
    
    # Extract features
    extractor = StudentLifeFeatureExtractor(user_data, user_id)
    df = extractor.extract_all_features()
    
    if len(df) < 35:  # Need at least 35 days (28 baseline + 7 monitoring)
        print(f"  ✗ Insufficient data: only {len(df)} days available (need 35+)")
        return None
    
    # Create baseline from first 28 days
    baseline, baseline_df = create_baseline_from_studentlife(df, baseline_days=28)
    
    # Initialize detector
    detector = ImprovedAnomalyDetector(baseline)
    
    # Analyze monitoring period (days 29 onwards)
    monitoring_df = df.iloc[28:].copy().reset_index(drop=True)
    
    print(f"\n  Monitoring period: {len(monitoring_df)} days")
    print(f"  Date range: {monitoring_df['date'].min()} to {monitoring_df['date'].max()}")
    
    # Run detection
    reports = []
    daily_reports = []
    deviations_history = []
    
    for idx, row in monitoring_df.iterrows():
        # Prepare current data (fill NaN with baseline values)
        current_data = {}
        for feat in baseline.to_dict().keys():
            if pd.notna(row.get(feat)):
                current_data[feat] = float(row[feat])
            else:
                # Use baseline value for missing data
                current_data[feat] = baseline.to_dict()[feat]
        
        # Analyze
        report, daily_report = detector.analyze(current_data, deviations_history, idx + 1)
        reports.append(report)
        
        # Convert to dict for JSON serialization
        from dataclasses import asdict
        daily_reports.append(asdict(daily_report))
        deviations_history.append(report.feature_deviations)
    
    # Generate final prediction
    final_prediction = detector.generate_final_prediction(
        f"studentlife_{user_id}", 
        user_id,
        len(monitoring_df)
    )
    
    # Get PHQ-9 scores
    phq9_pre = phq9_scores.get(f"{user_id}_pre", None)
    phq9_post = phq9_scores.get(f"{user_id}_post", None)
    
    # Print results
    print(f"\n  {'='*76}")
    print(f"  SYSTEM 1 RESULTS:")
    print(f"  {'='*76}")
    print(f"  Status: {'ANOMALY DETECTED' if final_prediction.sustained_anomaly_detected else 'NORMAL'}")
    print(f"  Final Anomaly Score: {final_prediction.final_anomaly_score:.3f}")
    print(f"  Sustained Deviation Days: {final_prediction.evidence_summary['sustained_deviation_days']}")
    print(f"  Evidence Accumulated: {final_prediction.evidence_summary['evidence_accumulated']:.2f}")
    print(f"  Pattern: {final_prediction.pattern_identified}")
    
    print(f"\n  {'='*76}")
    print(f"  PHQ-9 CLINICAL SCORES:")
    print(f"  {'='*76}")
    
    if phq9_pre is not None:
        severity_pre = get_phq9_severity(phq9_pre)
        print(f"  Pre-study:  {phq9_pre}/27 ({severity_pre})")
    else:
        print(f"  Pre-study:  No data")
    
    if phq9_post is not None:
        severity_post = get_phq9_severity(phq9_post)
        print(f"  Post-study: {phq9_post}/27 ({severity_post})")
    else:
       print(f"  Post-study: No data")
    
    # Alert distribution
    alert_dist = {}
    for r in daily_reports:
        alert_dist[r['alert_level']] = alert_dist.get(r['alert_level'], 0) + 1
    
    print(f"\n  Alert Distribution:")
    for level in ['green', 'yellow', 'orange', 'red']:
        count = alert_dist.get(level, 0)
        pct = (count / len(daily_reports)) * 100 if len(daily_reports) > 0 else 0
        print(f"    {level.upper():6s}: {count:3d} days ({pct:5.1f}%)")
    
    return {
        'user_id': user_id,
        'phq9_pre': phq9_pre,
        'phq9_post': phq9_post,
        'anomaly_score': final_prediction.final_anomaly_score,
        'anomaly_detected': final_prediction.sustained_anomaly_detected,
        'sustained_days': final_prediction.evidence_summary['sustained_deviation_days'],
        'evidence': final_prediction.evidence_summary['evidence_accumulated'],
        'pattern': final_prediction.pattern_identified,
        'monitoring_days': len(monitoring_df),
        'daily_reports': daily_reports
    }


def main():
    """Run System 1 on multiple StudentLife users"""
    
    print("="*80)
    print("SYSTEM 1 VALIDATION ON REAL STUDENTLIFE DATA")
    print("="*80)
    print("\nObjective: Validate anomaly detection against PHQ-9 clinical scores")
    print("Expected: Higher anomaly scores should correlate with higher PHQ-9 scores\n")
    
    dataset_path = r'F:\Avaneesh\download\student\dataset'
    
    # Load available users
    loader = StudentLifeLoader(dataset_path)
    loader.load_phq9_scores()
    users = loader.get_available_users()
    
    print(f"Found {len(users)} students in dataset")
    print(f"Analyzing ALL {len(users)} students...\n")
    
    # Analyze ALL users
    test_users = users  # All 49 students
    results = []
    
    for user_id in test_users:
        try:
            result = run_system1_on_studentlife_user(user_id, dataset_path)
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"\n✗ Error processing {user_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary comparison
    if len(results) > 0:
        print(f"\n\n{'='*80}")
        print("COMPARATIVE ANALYSIS: SYSTEM 1 vs PHQ-9")
        print(f"{'='*80}\n")
        
        print(f"{'User':6s} {'PHQ-9 Pre':>10s} {'PHQ-9 Post':>11s} {'Anomaly Score':>14s} {'Detected?':>10s} {'Correlation?':>15s}")
        print("─" * 80)
        
        for r in results:
            phq9_display = f"{r['phq9_post']}/27" if r['phq9_post'] is not None else "N/A"
            phq9_pre_display = f"{r['phq9_pre']}/27" if r['phq9_pre'] is not None else "N/A"
            detected = "YES" if r['anomaly_detected'] else "NO"
            
            # Correlation check
            if r['phq9_post'] is not None:
                # PHQ-9 > 9 is clinically significant depression
                phq9_high = r['phq9_post'] > 9
                sys1_detected = r['anomaly_detected']
                
                if phq9_high == sys1_detected:
                    correlation = "✓ Match"
                elif phq9_high and not sys1_detected:
                    correlation = "✗ Missed"
                else:
                    correlation = "✗ False Alarm"
            else:
                correlation = "N/A"
            
            print(f"{r['user_id']:6s} {phq9_pre_display:>10s} {phq9_display:>11s} {r['anomaly_score']:>14.3f} {detected:>10s} {correlation:>15s}")
        
        print(f"\n{'='*80}")
        print("KEY INSIGHTS:")
        print(f"{'='*80}")
        
        # Calculate correlations
        valid_results = [r for r in results if r['phq9_post'] is not None]
        
        if len(valid_results) >= 2:
            phq9_scores = [r['phq9_post'] for r in valid_results]
            anomaly_scores = [r['anomaly_score'] for r in valid_results]
            
            # Pearson correlation
            correlation = np.corrcoef(phq9_scores, anomaly_scores)[0, 1]
            
            print(f"\n  Correlation (PHQ-9 vs Anomaly Score): {correlation:.3f}")
            
            if correlation > 0.5:
                print(f"  ✓ STRONG positive correlation - System 1 is detecting real depression!")
            elif correlation > 0.3:
                print(f"  ✓ MODERATE positive correlation - System 1 shows promise")
            elif correlation > 0:
                print(f"  ~ WEAK positive correlation - Needs tuning")
            else:
                print(f"  ✗ NO correlation - System 1 may need recalibration")
        
        print(f"\n  Note: This is a PRELIMINARY analysis with limited data.")
        print(f"  Full validation requires all {len(users)} students + cross-validation.\n")
    
    else:
        print("\n✗ No valid results obtained. Check data availability.")
    
    # Save results to JSON for PDF generation
    import json
    output_data = {
        'analysis_date': '2026-02-15',
        'total_students': len(test_users),
        'valid_results': len(results),
        'results': results
    }
    
    with open('studentlife_full_results.json', 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to: studentlife_full_results.json")
    print(f"  Total students analyzed: {len(results)}")
    print(f"\nReady for PDF generation!")
    
    return results


if __name__ == "__main__":
    main()
