"""
System 1: Improved Anomaly Detection
Detects sustained deviations from personalized baseline using synthetic data
Only flags after accumulating sufficient evidence over time
"""

import sys
import os

# Fix Windows console encoding issues
if sys.platform == 'win32':
    try:
        # Try to set UTF-8 for console output
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        # If reconfigure fails, continue with default
        pass

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from collections import deque
import json


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PersonalityVector:
    """Baseline personality profile"""
    # Voice features
    voice_pitch_mean: float
    voice_pitch_std: float
    voice_energy_mean: float
    voice_speaking_rate: float

    # Activity features
    screen_time_hours: float
    unlock_count: float
    social_app_ratio: float
    calls_per_day: float
    texts_per_day: float
    unique_contacts: float
    response_time_minutes: float

    # Movement features
    daily_displacement_km: float
    location_entropy: float
    home_time_ratio: float
    places_visited: float

    # Circadian features
    # Circadian & Environment features
    wake_time_hour: float
    sleep_time_hour: float
    sleep_duration_hours: float
    dark_duration_hours: float  # Proxy for sleep/pocket time
    charge_duration_hours: float # Phone charging pattern

    # Social & Audio features
    conversation_duration_hours: float # Voice activity
    conversation_frequency: float      # Number of conversations

    # Variance bounds (for each feature)
    variances: Dict[str, float] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easy manipulation"""
        return {
            'voice_pitch_mean': self.voice_pitch_mean,
            'voice_pitch_std': self.voice_pitch_std,
            'voice_energy_mean': self.voice_energy_mean,
            'voice_speaking_rate': self.voice_speaking_rate,
            'screen_time_hours': self.screen_time_hours,
            'unlock_count': self.unlock_count,
            'social_app_ratio': self.social_app_ratio,
            'calls_per_day': self.calls_per_day,
            'texts_per_day': self.texts_per_day,
            'unique_contacts': self.unique_contacts,
            'response_time_minutes': self.response_time_minutes,
            'daily_displacement_km': self.daily_displacement_km,
            'location_entropy': self.location_entropy,
            'home_time_ratio': self.home_time_ratio,
            'places_visited': self.places_visited,
            'wake_time_hour': self.wake_time_hour,
            'sleep_time_hour': self.sleep_time_hour,
            'sleep_duration_hours': self.sleep_duration_hours,
            'dark_duration_hours': self.dark_duration_hours,
            'charge_duration_hours': self.charge_duration_hours,
            'conversation_duration_hours': self.conversation_duration_hours,
            'conversation_frequency': self.conversation_frequency,
        }


@dataclass
class DailyReport:
    """Detailed daily report"""
    day_number: int
    date: datetime
    anomaly_score: float
    alert_level: str
    flagged_features: List[str]
    pattern_type: str
    sustained_deviation_days: int
    evidence_accumulated: float
    top_deviations: Dict[str, float]
    notes: str


@dataclass
class AnomalyReport:
    """Output from System 1"""
    timestamp: datetime
    overall_anomaly_score: float
    feature_deviations: Dict[str, float]
    deviation_velocity: Dict[str, float]
    alert_level: str
    flagged_features: List[str]
    pattern_type: str
    sustained_deviation_days: int
    evidence_accumulated: float


@dataclass
class FinalPrediction:
    """Final analysis after monitoring period"""
    patient_id: str
    scenario: str
    monitoring_days: int
    baseline_vector: PersonalityVector
    final_anomaly_score: float
    sustained_anomaly_detected: bool
    confidence: float
    pattern_identified: str
    evidence_summary: Dict[str, any]
    recommendation: str


# ============================================================================
# SYNTHETIC DATA GENERATOR
# ============================================================================

class SyntheticDataGenerator:
    """Generate realistic synthetic sensor data"""

    def __init__(self, seed=42):
        np.random.seed(seed)

    def generate_baseline(self, days=28) -> Tuple[PersonalityVector, pd.DataFrame]:
        """Generate clean baseline period data"""
        print(f"  Generating {days} days of baseline data...")

        # Define "true" baseline values for a hypothetical person
        baseline_params = {
            'voice_pitch_mean': 180.0,
            'voice_pitch_std': 15.0,
            'voice_energy_mean': 0.65,
            'voice_speaking_rate': 3.5,
            'screen_time_hours': 4.5,
            'unlock_count': 80.0,
            'social_app_ratio': 0.35,
            'calls_per_day': 3.0,
            'texts_per_day': 25.0,
            'unique_contacts': 8.0,
            'response_time_minutes': 15.0,
            'daily_displacement_km': 12.0,
            'location_entropy': 2.3,
            'home_time_ratio': 0.65,
            'places_visited': 4.0,
            'wake_time_hour': 7.5,
            'sleep_time_hour': 23.5,
            'sleep_duration_hours': 7.5,
            'dark_duration_hours': 8.5,
            'charge_duration_hours': 6.0,
            'conversation_duration_hours': 1.5,
            'conversation_frequency': 12.0,
        }

        dates = [datetime.now() - timedelta(days=days - i) for i in range(days)]
        data = []

        for date in dates:
            daily_data = {'date': date}
            for feature, mean_val in baseline_params.items():
                noise_factor = 0.12
                value = np.random.normal(mean_val, mean_val * noise_factor)
                daily_data[feature] = max(0, value)
            data.append(daily_data)

        df = pd.DataFrame(data)

        baseline_vector = PersonalityVector(
            **{k: df[k].mean() for k in baseline_params.keys()}
        )
        baseline_vector.variances = {k: df[k].std() for k in baseline_params.keys()}

        return baseline_vector, df

    def generate_monitoring_data(self, baseline: PersonalityVector,
                                 scenario: str, days=180) -> pd.DataFrame:
        """Generate monitoring period data with different patterns"""
        print(f"  Generating {days} days of '{scenario}' monitoring data...")

        baseline_dict = baseline.to_dict()
        dates = [datetime.now() + timedelta(days=i) for i in range(days)]
        data = []

        # For BPD scenario state tracking
        bpd_state = 'normal' # normal, impulsive, depressive
        days_in_state = 0
        state_threshold = np.random.randint(3, 10)

        for i, date in enumerate(dates):
            daily_data = {'date': date}

            if scenario == 'normal':
                # Just natural variance - healthy baseline
                for feature, mean_val in baseline_dict.items():
                    variance = baseline.variances[feature]
                    value = np.random.normal(mean_val, variance)
                    daily_data[feature] = max(0, value)

            elif scenario == 'bpd_rapid_cycling':
                # BPD Scenario: Rapid switching between states
                days_in_state += 1
                if days_in_state >= state_threshold:
                    # Switch state
                    choices = ['normal', 'impulsive', 'depressive']
                    choices.remove(bpd_state)
                    bpd_state = np.random.choice(choices)
                    days_in_state = 0
                    state_threshold = np.random.randint(3, 8) # Switch every 3-7 days

                for feature, mean_val in baseline_dict.items():
                    variance = baseline.variances[feature]
                    
                    if bpd_state == 'normal':
                        value = np.random.normal(mean_val, variance)
                    elif bpd_state == 'impulsive':
                        # High energy/social/screen time
                        if feature in ['screen_time_hours', 'social_app_ratio', 'texts_per_day', 'calls_per_day', 'unlock_count']:
                            value = mean_val * 1.6 + np.random.normal(0, variance)
                        elif feature in ['sleep_duration_hours']:
                            value = mean_val * 0.7 + np.random.normal(0, variance)
                        elif feature in ['voice_energy_mean', 'voice_speaking_rate']:
                            value = mean_val * 1.3 + np.random.normal(0, variance)
                        else:
                            value = np.random.normal(mean_val, variance)
                    elif bpd_state == 'depressive':
                        # Low energy/social/movement
                        if feature in ['screen_time_hours', 'social_app_ratio', 'texts_per_day', 'calls_per_day']:
                            # Actually screen time might INCREASE if they are isolating with phone
                            if feature == 'screen_time_hours':
                                value = mean_val * 1.4 + np.random.normal(0, variance)
                            else:
                                value = mean_val * 0.4 + np.random.normal(0, variance)
                        elif feature in ['daily_displacement_km', 'places_visited', 'location_entropy']:
                            value = mean_val * 0.3 + np.random.normal(0, variance)
                        elif feature in ['voice_energy_mean', 'voice_pitch_mean']:
                            value = mean_val * 0.8 + np.random.normal(0, variance)
                        elif feature in ['sleep_duration_hours']:
                            value = mean_val * 1.4 + np.random.normal(0, variance)
                        else:
                            value = np.random.normal(mean_val, variance)
                    
                    daily_data[feature] = max(0, value)

            elif scenario == 'anomaly_subtle_rapid':
                # Kept for compatibility but enhanced for longer period
                cycle_phase = np.sin(2 * np.pi * i / 5) 

                for feature, mean_val in baseline_dict.items():
                    variance = baseline.variances[feature]
                    if feature in ['voice_energy_mean', 'screen_time_hours',
                                   'social_app_ratio', 'texts_per_day']:
                        swing = cycle_phase * mean_val * 0.30 
                    else:
                        swing = cycle_phase * mean_val * 0.18

                    value = mean_val + swing + np.random.normal(0, variance * 0.5)
                    daily_data[feature] = max(0, value)

            elif scenario == 'anomaly_gradual_depression':
                # 6 months gradual decline is much more subtle per day
                # Total 50% decline over 180 days
                decline_factor = 1 - (i / days) * 0.50

                for feature, mean_val in baseline_dict.items():
                    variance = baseline.variances[feature]

                    if feature in ['voice_pitch_mean', 'voice_energy_mean',
                                   'voice_speaking_rate', 'screen_time_hours',
                                   'social_app_ratio', 'calls_per_day',
                                   'texts_per_day', 'daily_displacement_km',
                                   'places_visited']:
                        value = mean_val * decline_factor + np.random.normal(0, variance)
                    elif feature == 'sleep_duration_hours':
                        # Hypersomnia common in depression
                        value = mean_val * (1 + (i / days) * 0.40) + np.random.normal(0, variance)
                    else:
                        value = mean_val + np.random.normal(0, variance)

                    daily_data[feature] = max(0, value)

            elif scenario == 'normal_life_event':
                # Normal person with temporary dip around middle of 6 months
                # Days 0-70: normal
                # Days 71-100: dip (e.g. bereavement or job loss)
                # Days 101-180: gradual recovery

                if i < 70:
                    for feature, mean_val in baseline_dict.items():
                        variance = baseline.variances[feature]
                        value = np.random.normal(mean_val, variance)
                        daily_data[feature] = max(0, value)
                elif i < 100:
                    dip_progress = (i - 70) / 30 
                    dip_factor = 1 - (0.4 * np.sin(np.pi * dip_progress)) # Max 40% dip at middle

                    for feature, mean_val in baseline_dict.items():
                        variance = baseline.variances[feature]
                        if feature in ['voice_energy_mean', 'social_app_ratio',
                                       'calls_per_day', 'texts_per_day',
                                       'daily_displacement_km']:
                            value = mean_val * dip_factor + np.random.normal(0, variance)
                        elif feature == 'sleep_duration_hours':
                            value = mean_val * (1 + (1-dip_factor)*0.5) + np.random.normal(0, variance)
                        else:
                            value = mean_val + np.random.normal(0, variance)
                        daily_data[feature] = max(0, value)
                else:
                    # Recovery
                    for feature, mean_val in baseline_dict.items():
                        variance = baseline.variances[feature]
                        value = np.random.normal(mean_val, variance)
                        daily_data[feature] = max(0, value)

            elif scenario == 'mixed_signals':
                # Mixed signals over 6 months
                for feature, mean_val in baseline_dict.items():
                    variance = baseline.variances[feature]
                    if feature in ['voice_pitch_mean', 'voice_pitch_std',
                                   'wake_time_hour', 'sleep_time_hour']:
                        value = np.random.normal(mean_val, variance)
                    elif feature in ['voice_energy_mean', 'screen_time_hours']:
                        cycle = np.sin(2 * np.pi * i / 14) # 2-week cycle
                        swing = cycle * mean_val * 0.35 
                        value = mean_val + swing + np.random.normal(0, variance * 0.5)
                    elif feature in ['social_app_ratio', 'texts_per_day']:
                        decline = 1 - (i / days) * 0.3 
                        value = mean_val * decline + np.random.normal(0, variance)
                    else:
                        value = np.random.normal(mean_val, variance * 1.5)
                    daily_data[feature] = max(0, value)

            data.append(daily_data)

        return pd.DataFrame(data)


# ============================================================================
# IMPROVED SYSTEM 1: SUSTAINED ANOMALY DETECTION
# ============================================================================

class ImprovedAnomalyDetector:
    """System 1: Detects sustained deviations from baseline"""

    def __init__(self, baseline: PersonalityVector):
        self.baseline = baseline
        self.baseline_dict = baseline.to_dict()
        self.feature_names = list(self.baseline_dict.keys())

        # Track history for velocity and pattern detection
        self.history_window = 7
        self.feature_history = {feat: deque(maxlen=self.history_window)
                                for feat in self.feature_names}

        # SUSTAINED DEVIATION TRACKING
        self.anomaly_score_history = deque(maxlen=14)  # 2 weeks
        self.sustained_deviation_days = 0
        self.evidence_accumulated = 0.0

        # Thresholds for sustained detection (Real-time alerting)
        self.SUSTAINED_THRESHOLD_DAYS = 4  # Need 4+ days of deviation
        self.EVIDENCE_THRESHOLD = 2.0  # Daily alerting threshold
        self.ANOMALY_SCORE_THRESHOLD = 0.35  # Daily score to count as "deviant day"

        # PEAK Thresholds for clinical validation (Retrospective)
        self.PEAK_EVIDENCE_THRESHOLD = 2.7  # Much stricter for validation reports
        self.PEAK_SUSTAINED_THRESHOLD_DAYS = 5 # Require longer sustained periods

        # PEAK TRACKING (for retrospective validation)
        self.max_evidence = 0.0
        self.max_sustained_days = 0
        self.max_anomaly_score = 0.0

    def calculate_deviation_magnitude(self, current_data: Dict[str, float]) -> Dict[str, float]:
        """Calculate how many standard deviations from baseline"""
        deviations = {}

        for feature in self.feature_names:
            baseline_val = self.baseline_dict[feature]
            current_val = current_data[feature]
            variance = self.baseline.variances[feature]

            if variance > 0:
                deviation_sd = (current_val - baseline_val) / variance
            else:
                deviation_sd = 0

            deviations[feature] = deviation_sd

        return deviations

    def calculate_deviation_velocity(self, current_data: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate rate of change using EWMA (Exponential Weighted Moving Average)
        EWMA gives more weight to recent days - better for detecting acceleration
        """
        velocities = {}
        alpha = 0.4  # Smoothing factor (0.4 = recent days weighted ~2x older days)

        for feature in self.feature_names:
            self.feature_history[feature].append(current_data[feature])

        for feature in self.feature_names:
            history = list(self.feature_history[feature])

            if len(history) < 2:
                velocities[feature] = 0
            else:
                # Calculate EWMA
                ewma_values = []
                ewma = history[0]
                for val in history:
                    ewma = alpha * val + (1 - alpha) * ewma
                    ewma_values.append(ewma)
                
                # Velocity = change in EWMA over time
                slope = (ewma_values[-1] - ewma_values[0]) / len(ewma_values)
                
                baseline_val = self.baseline_dict[feature]
                if baseline_val > 0:
                    velocities[feature] = slope / baseline_val
                else:
                    velocities[feature] = 0

        return velocities

    def detect_pattern_type(self, deviations_history: List[Dict[str, float]]) -> str:
        """Identify temporal pattern"""
        if len(deviations_history) < 7:
            return 'insufficient_data'

        recent = deviations_history[-7:]

        avg_deviations = []
        for dev_dict in recent:
            avg_dev = np.mean([abs(v) for v in dev_dict.values()])
            avg_deviations.append(avg_dev)

        mean_dev = np.mean(avg_deviations)
        std_dev = np.std(avg_deviations)

        if mean_dev < 0.5:
            return 'stable'
        elif std_dev > 1.0 and mean_dev > 0.5:
            return 'rapid_cycling'
        elif mean_dev > 1.5 and std_dev < 0.8:
            return 'acute_spike'
        else:
            x = np.arange(len(avg_deviations))
            slope = np.polyfit(x, avg_deviations, 1)[0]
            if abs(slope) > 0.1:
                return 'gradual_drift'
            else:
                return 'mixed_pattern'

    def calculate_anomaly_score(self, deviations: Dict[str, float],
                                velocities: Dict[str, float]) -> float:
        """Overall anomaly score (0-1)"""
        magnitude_score = np.mean([abs(dev) for dev in deviations.values()])
        magnitude_score = min(magnitude_score / 3.0, 1.0)

        velocity_score = np.mean([abs(vel) for vel in velocities.values()])
        velocity_score = min(velocity_score * 10, 1.0)

        overall_score = 0.7 * magnitude_score + 0.3 * velocity_score

        return overall_score

    def update_sustained_tracking(self, anomaly_score: float):
        """Update tracking of sustained deviations"""
        self.anomaly_score_history.append(anomaly_score)
        
        # Track max anomaly score
        if anomaly_score > self.max_anomaly_score:
            self.max_anomaly_score = anomaly_score

        # Count consecutive days above threshold
        if anomaly_score > self.ANOMALY_SCORE_THRESHOLD:
            self.sustained_deviation_days += 1
            # Accumulate evidence (exponential growth for sustained patterns)
            self.evidence_accumulated += anomaly_score * (1 + self.sustained_deviation_days * 0.1)
        else:
            # Decay if we have a normal day (but more slowly)
            self.sustained_deviation_days = max(0, self.sustained_deviation_days - 1)
            self.evidence_accumulated *= 0.92  # Decay evidence more slowly

        # Track peaks (no decay)
        if self.evidence_accumulated > self.max_evidence:
            self.max_evidence = self.evidence_accumulated
        
        if self.sustained_deviation_days > self.max_sustained_days:
            self.max_sustained_days = self.sustained_deviation_days

    def should_alert_now(self) -> bool:
        """Real-time alerting - uses current state"""
        return (
            self.evidence_accumulated >= self.EVIDENCE_THRESHOLD or
            self.sustained_deviation_days >= self.SUSTAINED_THRESHOLD_DAYS
        )

    def had_episode(self) -> bool:
        """Retrospective detection for validation - uses peak state with stricter thresholds"""
        return (
            self.max_evidence >= self.PEAK_EVIDENCE_THRESHOLD or
            self.max_sustained_days >= self.PEAK_SUSTAINED_THRESHOLD_DAYS
        )

    def determine_alert_level(self, anomaly_score: float,
                              deviations: Dict[str, float]) -> str:
        """
        Improved alert level that considers SUSTAINED patterns
        Won't trigger high alerts for isolated odd days
        """

        # Critical features
        critical_features = ['voice_energy_mean', 'sleep_duration_hours',
                             'screen_time_hours', 'daily_displacement_km']
        critical_deviation = max([abs(deviations.get(f, 0)) for f in critical_features])

        # Check if we have sustained deviation
        has_sustained_deviation = (
            self.sustained_deviation_days >= self.SUSTAINED_THRESHOLD_DAYS or
            self.evidence_accumulated >= self.EVIDENCE_THRESHOLD
        )

        # Conservative alerting: require sustained patterns for yellow+
        if not has_sustained_deviation:
            # No sustained pattern yet - stay green even if today is high
            if anomaly_score < 0.6 and critical_deviation < 2.5:
                return 'green'
            else:
                # High single-day score, but not sustained - still green with note
                return 'green'

        # We have sustained deviation - now escalate based on severity
        if anomaly_score < 0.35 and critical_deviation < 2.0:
            return 'green'
        elif anomaly_score < 0.50 and critical_deviation < 2.5:
            return 'yellow'
        elif anomaly_score < 0.65 or critical_deviation < 3.0:
            return 'orange'
        else:
            return 'red'

    def identify_flagged_features(self, deviations: Dict[str, float],
                                  threshold=1.5) -> List[str]:
        """Identify significantly deviated features"""
        flagged = []
        for feature, deviation in deviations.items():
            if abs(deviation) > threshold:
                flagged.append(f"{feature} ({deviation:.2f} SD)")
        return flagged

    def get_top_deviations(self, deviations: Dict[str, float], top_n=5) -> Dict[str, float]:
        """Get top N deviated features"""
        sorted_devs = sorted(deviations.items(), key=lambda x: abs(x[1]), reverse=True)
        return dict(sorted_devs[:top_n])

    def analyze(self, current_data: Dict[str, float],
                deviations_history: List[Dict[str, float]],
                day_number: int) -> Tuple[AnomalyReport, DailyReport]:
        """Main analysis function - returns both summary and daily report"""

        deviations = self.calculate_deviation_magnitude(current_data)
        velocities = self.calculate_deviation_velocity(current_data)
        anomaly_score = self.calculate_anomaly_score(deviations, velocities)

        # Update sustained tracking BEFORE determining alert
        self.update_sustained_tracking(anomaly_score)

        alert_level = self.determine_alert_level(anomaly_score, deviations)
        flagged = self.identify_flagged_features(deviations)
        pattern_type = self.detect_pattern_type(deviations_history)
        top_devs = self.get_top_deviations(deviations)

        # Generate notes
        notes = self._generate_notes(anomaly_score, alert_level, pattern_type)

        report = AnomalyReport(
            timestamp=datetime.now(),
            overall_anomaly_score=anomaly_score,
            feature_deviations=deviations,
            deviation_velocity=velocities,
            alert_level=alert_level,
            flagged_features=flagged,
            pattern_type=pattern_type,
            sustained_deviation_days=self.sustained_deviation_days,
            evidence_accumulated=self.evidence_accumulated
        )

        daily_report = DailyReport(
            day_number=day_number,
            date=datetime.now() + timedelta(days=day_number),
            anomaly_score=anomaly_score,
            alert_level=alert_level,
            flagged_features=flagged,
            pattern_type=pattern_type,
            sustained_deviation_days=self.sustained_deviation_days,
            evidence_accumulated=self.evidence_accumulated,
            top_deviations=top_devs,
            notes=notes
        )

        return report, daily_report

    def _generate_notes(self, anomaly_score: float, alert_level: str, pattern_type: str) -> str:
        """Generate human-readable notes"""
        notes = []

        if self.sustained_deviation_days >= self.SUSTAINED_THRESHOLD_DAYS:
            notes.append(f"⚠ Sustained deviation detected ({self.sustained_deviation_days} consecutive days)")

        if self.evidence_accumulated >= self.EVIDENCE_THRESHOLD:
            notes.append(f"📊 Evidence accumulated: {self.evidence_accumulated:.2f}")

        if pattern_type in ['rapid_cycling', 'gradual_drift']:
            notes.append(f"📈 Pattern: {pattern_type}")

        if alert_level in ['orange', 'red']:
            notes.append(f"🚨 HIGH ALERT: {alert_level.upper()}")

        if anomaly_score > 0.6 and alert_level == 'green':
            notes.append("ℹ High single-day score but no sustained pattern yet")

        return " | ".join(notes) if notes else "Normal operation"

    def generate_final_prediction(self, scenario: str, patient_id: str,
                                  monitoring_days: int) -> FinalPrediction:
        """Generate final prediction after monitoring period"""

        # Calculate confidence based on data quality
        confidence = min(0.95, monitoring_days / 30 * 0.8 + 0.15)

        # Determine if sustained anomaly detected (Retrospective Detection Mode)
        sustained_anomaly = self.had_episode()

        # Calculate final anomaly score (average of recent history)
        if len(self.anomaly_score_history) > 0:
            final_score = np.mean(list(self.anomaly_score_history))
        else:
            final_score = 0.0

        # Pattern identification from recent history
        pattern = "stable"
        if len(self.anomaly_score_history) >= 7:
            recent_scores = list(self.anomaly_score_history)[-7:]
            if np.std(recent_scores) > 0.15:
                pattern = "unstable/cycling"
            elif np.mean(recent_scores) > 0.5:
                pattern = "persistent_elevation"
            else:
                pattern = "stable"

        # Generate recommendation using peak states
        if sustained_anomaly and self.max_evidence >= 4.0:
            recommendation = "REFER: Very strong evidence of sustained behavioral deviation (Critical Peak). Immediate clinical evaluation recommended."
        elif sustained_anomaly:
            recommendation = "MONITOR: Significant sustained deviation detected during study (Met Peak Threshold). Clinical follow-up recommended."
        elif self.max_evidence > 1.5:
            recommendation = "WATCH: Some periodic evidence of deviation. Suggest extending monitoring or additional check-ins."
        else:
            recommendation = "NORMAL: No significant sustained deviation detected during the study period."

        evidence_summary = {
            'sustained_deviation_days': self.sustained_deviation_days,
            'max_sustained_days': self.max_sustained_days,
            'evidence_accumulated_final': round(self.evidence_accumulated, 2),
            'peak_evidence': round(self.max_evidence, 2),
            'max_daily_anomaly_score': round(self.max_anomaly_score, 3),
            'avg_recent_anomaly_score': round(final_score, 3),
            'monitoring_days': monitoring_days,
            'days_above_threshold': sum(1 for s in self.anomaly_score_history if s > self.ANOMALY_SCORE_THRESHOLD)
        }

        prediction = FinalPrediction(
            patient_id=patient_id,
            scenario=scenario,
            monitoring_days=monitoring_days,
            baseline_vector=self.baseline,
            final_anomaly_score=final_score,
            sustained_anomaly_detected=sustained_anomaly,
            confidence=confidence,
            pattern_identified=pattern,
            evidence_summary=evidence_summary,
            recommendation=recommendation
        )

        return prediction


# ============================================================================
# VISUALIZATION
# ============================================================================

# ============================================================================
# COMPREHENSIVE REPORT GENERATOR (PDF)
# ============================================================================

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

class ReportGenerator:
    """Generate detailed clinical PDF reports for each scenario"""
    
    def __init__(self, scenario: str, patient_id: str):
        self.scenario = scenario
        self.patient_id = patient_id
        self.feature_groups = {
            'Voice Analysis': ['voice_pitch_mean', 'voice_pitch_std', 'voice_energy_mean', 'voice_speaking_rate'],
            'Digital Activity': ['screen_time_hours', 'unlock_count', 'social_app_ratio'],
            'Social Connection': ['calls_per_day', 'texts_per_day', 'unique_contacts', 'response_time_minutes'],
            'Movement & Mobility': ['daily_displacement_km', 'location_entropy', 'home_time_ratio', 'places_visited'],
            'Circadian & Sleep': ['wake_time_hour', 'sleep_time_hour', 'sleep_duration_hours']
        }
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    def generate_pdf(self, baseline_df, monitoring_df, daily_reports, final_prediction):
        filename = f"report_{self.scenario}_{self.patient_id}.pdf"
        print(f"  Generating PDF report: {filename}...")
        
        with PdfPages(filename) as pdf:
            # Page 1: Executive Summary
            self._plot_summary_page(pdf, final_prediction, daily_reports)
            
            # Page 2: Personality Vector Comparison (Current Snapshot)
            self._plot_personality_comparison_page(pdf, baseline_df, monitoring_df)

            # Page 3: Personality Vector Drift Timeline
            self._plot_drift_timeline_page(pdf, baseline_df, monitoring_df)
            
            # Pages 4+: Feature Group Details
            for group_name, features in self.feature_groups.items():
                self._plot_feature_group_page(pdf, group_name, features, monitoring_df, baseline_df, daily_reports)
                
        print(f"  ✓ Saved PDF Report: {filename}")
        return filename

    def _plot_summary_page(self, pdf, pred, daily_reports):
        fig = plt.figure(figsize=(11, 8.5))
        gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.3)
        
        # Header
        plt.figtext(0.5, 0.95, f"Mental Health Monitoring - Clinical Report", fontsize=18, fontweight='bold', ha='center')
        plt.figtext(0.5, 0.92, f"Scenario: {self.scenario.upper()} | Patient ID: {self.patient_id}", fontsize=12, ha='center')
        plt.figtext(0.5, 0.90, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=10, ha='center', alpha=0.7)
        
        # Status Box
        ax_status = fig.add_subplot(gs[0, 0])
        status_color = 'red' if pred.sustained_anomaly_detected else 'green'
        ax_status.set_facecolor(status_color)
        ax_status.patch.set_alpha(0.1)
        status_text = "ANOMALY DETECTED" if pred.sustained_anomaly_detected else "NORMAL RANGE"
        ax_status.text(0.5, 0.6, status_text, fontsize=16, fontweight='bold', color=status_color, ha='center')
        ax_status.text(0.5, 0.3, f"Confidence: {pred.confidence:.1%}", fontsize=12, ha='center')
        ax_status.set_title("Overall Status", fontweight='bold')
        ax_status.set_xticks([]); ax_status.set_yticks([])

        # Recommendation Box
        ax_rec = fig.add_subplot(gs[0, 1])
        ax_rec.text(0.05, 0.8, "Clinical Recommendation:", fontweight='bold')
        ax_rec.text(0.05, 0.4, pred.recommendation, wrap=True, fontsize=10)
        ax_rec.set_xticks([]); ax_rec.set_yticks([])
        ax_rec.set_facecolor('whitesmoke')

        # Anomaly Score Timeline
        ax_score = fig.add_subplot(gs[1, :])
        dates = [r.date for r in daily_reports]
        scores = [r.anomaly_score for r in daily_reports]
        ax_score.plot(dates, scores, color='black', alpha=0.3)
        
        colors_map = {'green': 'green', 'yellow': 'yellow', 'orange': 'orange', 'red': 'red'}
        ax_score.scatter(dates, scores, c=[colors_map[r.alert_level] for r in daily_reports], s=30)
        ax_score.axhline(0.4, color='orange', linestyle='--', alpha=0.5)
        ax_score.set_title("Long-term Anomaly Score Timeline (6 Months)", fontweight='bold')
        ax_score.set_ylabel("Score")
        ax_score.grid(True, alpha=0.2)

        # Alert Distribution
        ax_dist = fig.add_subplot(gs[2, 0])
        levels = ['green', 'yellow', 'orange', 'red']
        counts = [sum(1 for r in daily_reports if r.alert_level == l) for l in levels]
        ax_dist.bar(levels, counts, color=['green', 'yellow', 'orange', 'red'], alpha=0.7)
        ax_dist.set_title("Alert Distribution")

        # Evidence Accumulation
        ax_ev = fig.add_subplot(gs[2, 1])
        evidence = [r.evidence_accumulated for r in daily_reports]
        ax_ev.plot(dates, evidence, color='purple', linewidth=2)
        ax_ev.set_title("Evidence Accumulation")
        ax_ev.axhline(2.0, color='red', linestyle='--', alpha=0.5)

        pdf.savefig(fig)
        plt.close(fig)

    def _plot_personality_comparison_page(self, pdf, baseline_df, monitoring_df):
        fig = plt.figure(figsize=(11, 8.5))
        
        plt.figtext(0.5, 0.93, "Personality Vector Analysis: Baseline vs. Current", fontsize=16, fontweight='bold', ha='center')
        
        # Calculate recent average (last 30 days) vs initial baseline
        recent_df = monitoring_df.tail(30).mean(numeric_only=True)
        initial_baseline = baseline_df.mean(numeric_only=True)
        initial_std = baseline_df.std(numeric_only=True)

        features = list(initial_baseline.keys())
        # Drop 'date' if it exists
        if 'date' in features: features.remove('date')
        
        # Normalize deviations by baseline STD
        deviations = []
        for feat in features:
            dev = (recent_df[feat] - initial_baseline[feat]) / (initial_std[feat] + 1e-6)
            deviations.append(dev)

        ax = fig.add_axes([0.1, 0.1, 0.8, 0.7])
        y_pos = np.arange(len(features))
        
        colors = ['red' if abs(d) > 2 else 'orange' if abs(d) > 1 else 'green' for d in deviations]
        ax.barh(y_pos, deviations, color=colors, alpha=0.7)
        ax.axvline(0, color='black', linewidth=1)
        ax.axvline(2, color='red', linestyle='--', alpha=0.3)
        ax.axvline(-2, color='red', linestyle='--', alpha=0.3)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f.replace('_', ' ').title() for f in features])
        ax.set_xlabel("Deviation from Baseline (Standard Deviations)")
        ax.set_title("Recent Personality Vector Drift (Last 30 Days vs. Baseline)")
        ax.grid(True, alpha=0.3)

        pdf.savefig(fig)
        plt.close(fig)

    def _plot_drift_timeline_page(self, pdf, baseline_df, monitoring_df):
        fig = plt.figure(figsize=(11, 8.5))
        
        plt.figtext(0.5, 0.93, "Personality Vector Drift Timeline (Tracked Throughout)", fontsize=16, fontweight='bold', ha='center')
        
        # Calculate rolling 7-day Z-score for all features
        initial_baseline = baseline_df.mean(numeric_only=True)
        initial_std = baseline_df.std(numeric_only=True) + 1e-6
        
        dates = monitoring_df['date']
        
        # Select 8 most representative features to avoid clutter
        track_features = [
            'voice_energy_mean', 'screen_time_hours', 'sleep_duration_hours',
            'daily_displacement_km', 'social_app_ratio', 'texts_per_day',
            'voice_pitch_mean', 'calls_per_day'
        ]
        
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.75])
        
        for feat in track_features:
            if feat in monitoring_df.columns:
                # 7-day rolling mean
                rolling = monitoring_df[feat].rolling(window=7).mean()
                # Z-score relative to initial baseline
                z_score = (rolling - initial_baseline[feat]) / initial_std[feat]
                ax.plot(dates, z_score, label=feat.replace('_', ' ').title(), linewidth=1.5, alpha=0.8)

        ax.axhline(0, color='black', linewidth=1, linestyle='-')
        ax.axhline(2, color='red', linestyle='--', alpha=0.3)
        ax.axhline(-2, color='red', linestyle='--', alpha=0.3)
        ax.fill_between(dates, -1, 1, color='green', alpha=0.05, label='Normal Range (±1 SD)')
        
        ax.set_ylabel("Deviation (Z-Score)")
        ax.set_xlabel("Monitoring Duration")
        ax.set_title("How the Personality Vector Shifts over 180 Days", fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.2)

        pdf.savefig(fig)
        plt.close(fig)

    def _plot_feature_group_page(self, pdf, group_name, features, monitoring_df, baseline_df, daily_reports):
        n = len(features)
        fig = plt.figure(figsize=(11, 8.5))
        gs = gridspec.GridSpec(n, 1, hspace=0.4)
        
        plt.figtext(0.5, 0.95, f"Detailed Feature Analysis: {group_name}", fontsize=16, fontweight='bold', ha='center')
        
        dates = monitoring_df['date']
        
        for i, feat in enumerate(features):
            ax = fig.add_subplot(gs[i])
            
            b_mean = baseline_df[feat].mean()
            b_std = baseline_df[feat].std()
            
            # Baseline band
            ax.axhline(b_mean, color='green', linestyle='--', alpha=0.5)
            ax.axhspan(b_mean - 2*b_std, b_mean + 2*b_std, color='green', alpha=0.1, label='Normal Range (±2 SD)')
            
            # Monitoring data
            ax.plot(dates, monitoring_df[feat], color='blue', alpha=0.7, label='Observed')
            
            # Highlight anomaly days
            alert_colors = {'green': None, 'yellow': 'yellow', 'orange': 'orange', 'red': 'red'}
            for j, report in enumerate(daily_reports):
                if report.alert_level != 'green':
                    ax.axvspan(dates.iloc[j], dates.iloc[min(j+1, len(dates)-1)], 
                              color=alert_colors[report.alert_level], alpha=0.1)

            ax.set_title(feat.replace('_', ' ').title(), fontsize=10, fontweight='bold', loc='left')
            ax.grid(True, alpha=0.2)
            if i == n-1:
                ax.set_xlabel("Date")
            else:
                ax.set_xticklabels([])

        pdf.savefig(fig)
        plt.close(fig)


def plot_comprehensive_results(baseline_df: pd.DataFrame,
                               monitoring_df: pd.DataFrame,
                               daily_reports: List[DailyReport],
                               scenario: str,
                               final_prediction: FinalPrediction):
    """Create comprehensive visualization summary image"""

    # Generate PDF Report first
    rg = ReportGenerator(scenario, final_prediction.patient_id)
    pdf_path = rg.generate_pdf(baseline_df, monitoring_df, daily_reports, final_prediction)

    # Generate Summary PNG
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    fig.suptitle(f'System 1 Final Analysis Summary: {scenario.upper().replace("_", " ")}', 
                 fontsize=22, fontweight='bold')

    # Key features to plot in PNG summary
    key_features = [
        ('voice_energy_mean', 'Voice Energy'),
        ('screen_time_hours', 'Screen Time (hrs)'),
        ('sleep_duration_hours', 'Sleep Duration (hrs)'),
        ('daily_displacement_km', 'Daily Movement (km)'),
        ('texts_per_day', 'Texts per Day'),
        ('social_app_ratio', 'Social Activity Ratio')
    ]

    for idx, (feature, title) in enumerate(key_features):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])

        baseline_mean = baseline_df[feature].mean()
        baseline_std = baseline_df[feature].std()

        dates = monitoring_df['date']
        values = monitoring_df[feature]

        ax.axhline(baseline_mean, color='green', linestyle='--', alpha=0.7, linewidth=2)
        ax.axhspan(baseline_mean - 2*baseline_std, baseline_mean + 2*baseline_std,
                   alpha=0.1, color='green')

        ax.plot(dates, values, 'b-', linewidth=1.5, alpha=0.8)

        # Alert colors
        alert_colors = {'green': None, 'yellow': '#ffffcc', 'orange': '#ffcc99', 'red': '#ff9999'}
        for i, report in enumerate(daily_reports):
            if report.alert_level != 'green':
                ax.axvspan(dates.iloc[i], dates.iloc[min(i+1, len(dates)-1)],
                          alpha=0.3, color=alert_colors[report.alert_level])

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2)

    # Score Timeline
    ax7 = fig.add_subplot(gs[2, :])
    dates = [r.date for r in daily_reports]
    scores = [r.anomaly_score for r in daily_reports]
    
    colors = {'green': 'green', 'yellow': 'orange', 'orange': 'darkorange', 'red': 'red'}
    ax7.scatter(dates, scores, c=[colors[r.alert_level] for r in daily_reports], s=50, alpha=0.6)
    ax7.plot(dates, scores, 'k-', alpha=0.1)
    ax7.set_title('Anomaly Score Evolution (180 Days)', fontsize=14, fontweight='bold')
    ax7.set_ylabel('Score')
    ax7.grid(True, alpha=0.2)

    # Personality Radar/Bar Chart comparison in summary
    ax8 = fig.add_subplot(gs[3, 0])
    recent_avg = monitoring_df.tail(30).mean(numeric_only=True)
    base_avg = baseline_df.mean(numeric_only=True)
    base_std = baseline_df.std(numeric_only=True) + 1e-6
    
    devs = [(recent_avg[f] - base_avg[f])/base_std[f] for f in key_features[0][0:1] + key_features[1][0:1] + key_features[2][0:1] + key_features[3][0:1] + key_features[4][0:1] + key_features[5][0:1]]
    # Wait, the indexing above is wrong. Let's do it properly.
    feat_names = [kf[0] for kf in key_features]
    devs = [(recent_avg[fn] - base_avg[fn])/base_std[fn] for fn in feat_names]
    
    ax8.barh([kf[1] for kf in key_features], devs, color='skyblue')
    ax8.axvline(0, color='black', linewidth=1)
    ax8.set_title('Vector Deviation (Recent vs Base)')
    ax8.set_xlabel('SD Units')

    # Evidence
    ax9 = fig.add_subplot(gs[3, 1])
    evidence = [r.evidence_accumulated for r in daily_reports]
    ax9.plot(dates, evidence, 'purple', linewidth=2)
    ax9.set_title('Evidence Accumulation')
    ax9.axhline(2.0, color='red', linestyle='--')

    # Status
    ax10 = fig.add_subplot(gs[3, 2])
    ax10.axis('off')
    summary_text = f"RESULTS SUMMARY\n{'-'*20}\n" \
                   f"Scenario: {scenario}\n" \
                   f"Status: {'ANOMALY' if final_prediction.sustained_anomaly_detected else 'NORMAL'}\n" \
                   f"Confidence: {final_prediction.confidence:.1%}\n" \
                   f"Sustained Days: {final_prediction.evidence_summary['sustained_deviation_days']}\n\n" \
                   f"Recommendation:\n{final_prediction.recommendation}"
    ax10.text(0, 0.5, summary_text, fontweight='bold', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    plt.savefig(f'analysis_{scenario}.png', dpi=100, bbox_inches='tight')
    plt.close()

def generate_daily_report_summary(daily_reports: List[DailyReport], scenario: str):
    """Generate text summary of daily reports"""

    report_lines = []
    report_lines.append(f"\n{'='*80}")
    report_lines.append(f"DAILY MONITORING REPORT: {scenario.upper().replace('_', ' ')}")
    report_lines.append(f"{'='*80}\n")

    for report in daily_reports:
        report_lines.append(f"Day {report.day_number:02d} | {report.date.strftime('%Y-%m-%d')} | "
                           f"Alert: {report.alert_level.upper():6s} | "
                           f"Score: {report.anomaly_score:.3f} | "
                           f"Sustained: {report.sustained_deviation_days} days")

        if report.notes and report.notes != "Normal operation":
            report_lines.append(f"        {report.notes}")

        if report.alert_level in ['orange', 'red']:
            report_lines.append(f"        Top deviations: {list(report.top_deviations.keys())[:3]}")

        report_lines.append("")

    return "\n".join(report_lines)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_scenario(scenario: str, patient_id: str):
    """Run complete analysis for a scenario"""
    print(f"\n{'='*80}")
    print(f"PATIENT: {patient_id} | SCENARIO: {scenario.upper().replace('_', ' ')}")
    print(f"{'='*80}")

    generator = SyntheticDataGenerator(seed=hash(patient_id) % 1000)

    # Generate baseline (28 days is standard)
    baseline, baseline_df = generator.generate_baseline(days=28)
    print(f"  ✓ Baseline established")

    # Initialize detector
    detector = ImprovedAnomalyDetector(baseline)

    # Generate monitoring data (6 months = 180 days)
    monitoring_df = generator.generate_monitoring_data(baseline, scenario, days=180)

    # Analyze each day
    reports = []
    daily_reports = []
    deviations_history = []

    for idx, row in monitoring_df.iterrows():
        current_data = row.to_dict()
        del current_data['date']

        report, daily_report = detector.analyze(current_data, deviations_history, idx + 1)
        reports.append(report)
        daily_reports.append(daily_report)
        deviations_history.append(report.feature_deviations)

    # Generate final prediction
    final_prediction = detector.generate_final_prediction(scenario, patient_id, len(monitoring_df))

    # Print summary
    print(f"\n  📊 ANALYSIS COMPLETE (180 DAY SIMULATION)")
    print(f"  {'─'*76}")

    alert_dist = {}
    for r in daily_reports:
        alert_dist[r.alert_level] = alert_dist.get(r.alert_level, 0) + 1

    print(f"  Alert Distribution:")
    for level in ['green', 'yellow', 'orange', 'red']:
        count = alert_dist.get(level, 0)
        pct = (count / len(daily_reports)) * 100
        bar = '█' * (count // 4) # Adjust bar scale for 180 days
        print(f"    {level.upper():6s}: {bar:15s} {count:3d} days ({pct:5.1f}%)")

    print(f"\n  🎯 FINAL PREDICTION:")
    print(f"  {'─'*76}")
    print(f"  Status: {'ANOMALY' if final_prediction.sustained_anomaly_detected else 'NORMAL'}")
    print(f"  Confidence: {final_prediction.confidence:.1%}")
    print(f"  Pattern: {final_prediction.pattern_identified}")
    print(f"  Final Score: {final_prediction.final_anomaly_score:.3f}")
    print(f"  Sustained Days: {final_prediction.evidence_summary['sustained_deviation_days']}")
    print(f"  Evidence: {final_prediction.evidence_summary['evidence_accumulated']:.2f}")
    print(f"\n  💡 Recommendation:")
    print(f"  {final_prediction.recommendation}")

    # Generate visualizations (PNG and PDF combined in this call now)
    plot_comprehensive_results(baseline_df, monitoring_df, daily_reports, 
                               scenario, final_prediction)

    # Generate daily report text summary (brief)
    daily_summary = generate_daily_report_summary(daily_reports, scenario)

    return {
        'baseline': baseline,
        'monitoring': monitoring_df,
        'reports': reports,
        'daily_reports': daily_reports,
        'final_prediction': final_prediction,
        'daily_summary': daily_summary
    }


def main():
    """Run all scenarios"""

    print("\n" + "="*80)
    print("SYSTEM 1: 6-MONTH EXTENDED SIMULATION & BPD STRESS TESTING")
    print("="*80)
    print("\nObjective: Evaluate long-term patterns and complex cycling scenarios")
    print("         Generating detailed PDF reports for each patient profile.\n")

    scenarios = [
        ('normal', 'PT-001'),
        ('bpd_rapid_cycling', 'PT-002'),
        ('anomaly_gradual_depression', 'PT-003'),
        ('normal_life_event', 'PT-004'),
        ('mixed_signals', 'PT-005'),
    ]

    scenario_names = {
        'normal': 'Normal Baseline Patient',
        'bpd_rapid_cycling': 'BPD Patient (Rapid Cycling States)',
        'anomaly_gradual_depression': 'Depression Patient (Gradual Drift)',
        'normal_life_event': 'Normal Life Event (Bereavement)',
        'mixed_signals': 'Mixed Behavioral Signals'
    }

    all_results = {}

    for scenario, patient_id in scenarios:
        results = run_scenario(scenario, patient_id)
        all_results[scenario] = results

        # Save daily report to file
        report_file = f'daily_report_{scenario}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(results['daily_summary'])
        print(f"  ✓ Saved: {report_file}")

    # Generate comparison summary
    print(f"\n\n{'='*80}")
    print("6-MONTH COMPARATIVE ANALYSIS SUMMARY")
    print(f"{'='*80}\n")

    comparison_lines = []
    comparison_lines.append(f"{'Patient ID':<12} {'Scenario':<35} {'Anomaly?':<10} {'Confidence':<12} {'Final Score':<12}")
    comparison_lines.append("─" * 80)

    for scenario, patient_id in scenarios:
        pred = all_results[scenario]['final_prediction']
        comparison_lines.append(
            f"{pred.patient_id:<12} "
            f"{scenario_names[scenario]:<35} "
            f"{'YES' if pred.sustained_anomaly_detected else 'NO':<10} "
            f"{pred.confidence:<12.1%} "
            f"{pred.final_anomaly_score:<12.3f}"
        )

    comparison_text = "\n".join(comparison_lines)
    print(comparison_text)

    # Save comparison
    with open('comparison_summary.txt', 'w', encoding='utf-8') as f:
        f.write(comparison_text)

    print(f"\n{'='*80}")
    print("SIMULATION SUITE COMPLETE")
    print(f"{'='*80}")
    print(f"\nGenerated Artifacts:")
    print(f"  • Clinical PDF Reports: report_*.pdf")
    print(f"  • Overview PNG Charts: analysis_*.png")
    print(f"  • Comparative Summary: comparison_summary.txt")

    return all_results


if __name__ == "__main__":
    results = main()
