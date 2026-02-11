"""
System 1: Improved Anomaly Detection
Detects sustained deviations from personalized baseline using synthetic data
Only flags after accumulating sufficient evidence over time
"""

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
    wake_time_hour: float
    sleep_time_hour: float
    sleep_duration_hours: float

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
                                 scenario: str, days=30) -> pd.DataFrame:
        """Generate monitoring period data with different patterns"""
        print(f"  Generating {days} days of '{scenario}' monitoring data...")

        baseline_dict = baseline.to_dict()
        dates = [datetime.now() + timedelta(days=i) for i in range(days)]
        data = []

        for i, date in enumerate(dates):
            daily_data = {'date': date}

            if scenario == 'normal':
                # Just natural variance - healthy baseline
                for feature, mean_val in baseline_dict.items():
                    variance = baseline.variances[feature]
                    value = np.random.normal(mean_val, variance)
                    daily_data[feature] = max(0, value)

            elif scenario == 'anomaly_subtle_rapid':
                # Subtle but consistent rapid changes (NOT obvious single spikes)
                # Small amplitude, but high frequency oscillations
                cycle_phase = np.sin(2 * np.pi * i / 3)  # 3-day cycle

                for feature, mean_val in baseline_dict.items():
                    variance = baseline.variances[feature]

                    # More noticeable swings (25-35% instead of 12-20%)
                    if feature in ['voice_energy_mean', 'screen_time_hours',
                                   'social_app_ratio', 'texts_per_day']:
                        swing = cycle_phase * mean_val * 0.30  # 30% swings
                    else:
                        swing = cycle_phase * mean_val * 0.18  # 18% swings

                    value = mean_val + swing + np.random.normal(0, variance * 0.5)
                    daily_data[feature] = max(0, value)

            elif scenario == 'anomaly_gradual_depression':
                # Very gradual, subtle decline (depression pattern)
                # 35% decline over 30 days (was 25%, now more detectable)
                decline_factor = 1 - (i / days) * 0.35

                for feature, mean_val in baseline_dict.items():
                    variance = baseline.variances[feature]

                    if feature in ['voice_pitch_mean', 'voice_energy_mean',
                                   'voice_speaking_rate', 'screen_time_hours',
                                   'social_app_ratio', 'calls_per_day',
                                   'texts_per_day', 'daily_displacement_km',
                                   'places_visited']:
                        value = mean_val * decline_factor + np.random.normal(0, variance)
                    elif feature == 'sleep_duration_hours':
                        value = mean_val * (1 + (i / days) * 0.20) + np.random.normal(0, variance)
                    else:
                        value = mean_val + np.random.normal(0, variance)

                    daily_data[feature] = max(0, value)

            elif scenario == 'normal_life_event':
                # Normal person with temporary dip (breakup/job loss) then recovery
                # Days 0-10: normal
                # Days 11-20: dip
                # Days 21-30: recovery

                if i < 10:
                    # Normal baseline
                    for feature, mean_val in baseline_dict.items():
                        variance = baseline.variances[feature]
                        value = np.random.normal(mean_val, variance)
                        daily_data[feature] = max(0, value)

                elif i < 20:
                    # Temporary dip (30% decrease in some features)
                    dip_progress = (i - 10) / 10  # 0 to 1
                    dip_factor = 1 - dip_progress * 0.30

                    for feature, mean_val in baseline_dict.items():
                        variance = baseline.variances[feature]

                        if feature in ['voice_energy_mean', 'social_app_ratio',
                                       'calls_per_day', 'texts_per_day',
                                       'daily_displacement_km']:
                            value = mean_val * dip_factor + np.random.normal(0, variance)
                        elif feature == 'sleep_duration_hours':
                            value = mean_val * (1 + dip_progress * 0.15) + np.random.normal(0, variance)
                        else:
                            value = mean_val + np.random.normal(0, variance)

                        daily_data[feature] = max(0, value)

                else:
                    # Recovery (gradual return to baseline)
                    recovery_progress = (i - 20) / 10  # 0 to 1
                    recovery_factor = 0.70 + recovery_progress * 0.30  # 70% -> 100%

                    for feature, mean_val in baseline_dict.items():
                        variance = baseline.variances[feature]

                        if feature in ['voice_energy_mean', 'social_app_ratio',
                                       'calls_per_day', 'texts_per_day',
                                       'daily_displacement_km']:
                            value = mean_val * recovery_factor + np.random.normal(0, variance)
                        elif feature == 'sleep_duration_hours':
                            sleep_factor = 1 + (1 - recovery_progress) * 0.15
                            value = mean_val * sleep_factor + np.random.normal(0, variance)
                        else:
                            value = mean_val + np.random.normal(0, variance)

                        daily_data[feature] = max(0, value)

            elif scenario == 'mixed_signals':
                # Mixed signals: some features stable, others unstable
                # Hard to classify but should show some evidence

                for feature, mean_val in baseline_dict.items():
                    variance = baseline.variances[feature]

                    # Group 1: Stable features
                    if feature in ['voice_pitch_mean', 'voice_pitch_std',
                                   'wake_time_hour', 'sleep_time_hour']:
                        value = np.random.normal(mean_val, variance)

                    # Group 2: Oscillating features (stronger oscillations)
                    elif feature in ['voice_energy_mean', 'screen_time_hours']:
                        cycle = np.sin(2 * np.pi * i / 5)
                        swing = cycle * mean_val * 0.32  # Increased from 0.25
                        value = mean_val + swing + np.random.normal(0, variance * 0.5)

                    # Group 3: Gradually declining features (stronger decline)
                    elif feature in ['social_app_ratio', 'texts_per_day']:
                        decline = 1 - (i / days) * 0.28  # Increased from 0.20
                        value = mean_val * decline + np.random.normal(0, variance)

                    # Group 4: Normal with high noise
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

        # Thresholds for sustained detection
        self.SUSTAINED_THRESHOLD_DAYS = 4  # Need 4+ days of deviation
        self.EVIDENCE_THRESHOLD = 2.0  # Accumulated evidence score
        self.ANOMALY_SCORE_THRESHOLD = 0.35  # Daily score to count as "deviant day"

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
        """Calculate rate of change"""
        velocities = {}

        for feature in self.feature_names:
            self.feature_history[feature].append(current_data[feature])

        for feature in self.feature_names:
            history = list(self.feature_history[feature])

            if len(history) < 2:
                velocities[feature] = 0
            else:
                x = np.arange(len(history))
                y = np.array(history)
                slope = np.polyfit(x, y, 1)[0]

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

        # Count consecutive days above threshold
        if anomaly_score > self.ANOMALY_SCORE_THRESHOLD:
            self.sustained_deviation_days += 1
            # Accumulate evidence (exponential growth for sustained patterns)
            self.evidence_accumulated += anomaly_score * (1 + self.sustained_deviation_days * 0.1)
        else:
            # Decay if we have a normal day (but more slowly)
            self.sustained_deviation_days = max(0, self.sustained_deviation_days - 1)
            self.evidence_accumulated *= 0.92  # Decay evidence more slowly

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

        # Determine if sustained anomaly detected
        sustained_anomaly = (
            self.sustained_deviation_days >= self.SUSTAINED_THRESHOLD_DAYS or
            self.evidence_accumulated >= self.EVIDENCE_THRESHOLD
        )

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

        # Generate recommendation
        if sustained_anomaly and final_score > 0.55:
            recommendation = "REFER: Strong evidence of sustained behavioral deviation. Clinical evaluation recommended."
        elif sustained_anomaly and final_score > 0.40:
            recommendation = "MONITOR: Moderate sustained deviation detected. Continue close monitoring."
        elif self.evidence_accumulated > 1.5:
            recommendation = "WATCH: Some evidence of deviation. Extend monitoring period."
        else:
            recommendation = "NORMAL: No sustained deviation detected. Routine follow-up sufficient."

        evidence_summary = {
            'sustained_deviation_days': self.sustained_deviation_days,
            'evidence_accumulated': round(self.evidence_accumulated, 2),
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

def plot_comprehensive_results(baseline_df: pd.DataFrame,
                               monitoring_df: pd.DataFrame,
                               daily_reports: List[DailyReport],
                               scenario: str,
                               final_prediction: FinalPrediction):
    """Create comprehensive visualization"""

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    fig.suptitle(f'System 1 Analysis: {scenario.upper().replace("_", " ")}', 
                 fontsize=18, fontweight='bold')

    # Key features to plot
    key_features = [
        ('voice_energy_mean', 'Voice Energy'),
        ('screen_time_hours', 'Screen Time (hrs)'),
        ('sleep_duration_hours', 'Sleep Duration (hrs)'),
        ('daily_displacement_km', 'Daily Movement (km)'),
        ('texts_per_day', 'Texts per Day'),
        ('social_app_ratio', 'Social App Usage')
    ]

    # Plot 1-6: Feature time series
    for idx, (feature, title) in enumerate(key_features):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])

        baseline_mean = baseline_df[feature].mean()
        baseline_std = baseline_df[feature].std()

        dates = monitoring_df['date']
        values = monitoring_df[feature]

        # Baseline bands
        ax.axhline(baseline_mean, color='green', linestyle='--', 
                   label='Baseline', alpha=0.7, linewidth=2)
        ax.axhspan(baseline_mean - baseline_std, baseline_mean + baseline_std,
                   alpha=0.15, color='green', label='±1 SD')
        ax.axhspan(baseline_mean - 2*baseline_std, baseline_mean + 2*baseline_std,
                   alpha=0.1, color='yellow', label='±2 SD')

        # Data line
        ax.plot(dates, values, 'b-', linewidth=2, label='Observed', alpha=0.8)

        # Color regions by alert level
        alert_colors = {'green': 'lightgreen', 'yellow': 'yellow', 
                       'orange': 'orange', 'red': 'red'}
        for i, report in enumerate(daily_reports):
            if report.alert_level != 'green':
                ax.axvspan(dates.iloc[i], dates.iloc[min(i+1, len(dates)-1)],
                          alpha=0.2, color=alert_colors[report.alert_level])

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Date', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        ax.tick_params(labelsize=8)

    # Plot 7: Anomaly Score Timeline
    ax7 = fig.add_subplot(gs[2, :])
    dates = [r.date for r in daily_reports]
    scores = [r.anomaly_score for r in daily_reports]
    sustained_days = [r.sustained_deviation_days for r in daily_reports]

    # Color code points by alert level
    colors = {'green': 'green', 'yellow': 'yellow', 'orange': 'orange', 'red': 'red'}
    point_colors = [colors[r.alert_level] for r in daily_reports]

    ax7.scatter(dates, scores, c=point_colors, s=100, alpha=0.6, edgecolors='black', linewidth=1)
    ax7.plot(dates, scores, 'k-', alpha=0.3, linewidth=1)

    # Add sustained deviation counter
    ax7_twin = ax7.twinx()
    ax7_twin.plot(dates, sustained_days, 'r--', alpha=0.5, linewidth=2, label='Sustained Days')
    ax7_twin.set_ylabel('Sustained Deviation Days', color='r', fontsize=10)
    ax7_twin.tick_params(axis='y', labelcolor='r', labelsize=9)

    # Thresholds
    ax7.axhline(0.4, color='yellow', linestyle=':', alpha=0.5, label='Yellow threshold')
    ax7.axhline(0.5, color='orange', linestyle=':', alpha=0.5, label='Orange threshold')
    ax7.axhline(0.65, color='red', linestyle=':', alpha=0.5, label='Red threshold')

    ax7.set_title('Anomaly Score & Sustained Deviation Timeline', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Date', fontsize=10)
    ax7.set_ylabel('Anomaly Score', fontsize=10)
    ax7.legend(fontsize=9, loc='upper left')
    ax7.grid(True, alpha=0.3)
    ax7.tick_params(labelsize=9)

    # Plot 8: Alert Distribution
    ax8 = fig.add_subplot(gs[3, 0])
    alert_counts = {'green': 0, 'yellow': 0, 'orange': 0, 'red': 0}
    for r in daily_reports:
        alert_counts[r.alert_level] += 1

    colors_plot = ['green', 'yellow', 'orange', 'red']
    counts = [alert_counts[c] for c in colors_plot]
    ax8.bar(colors_plot, counts, color=colors_plot, alpha=0.7, edgecolor='black')
    ax8.set_title('Alert Level Distribution', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Days', fontsize=10)
    ax8.tick_params(labelsize=9)
    ax8.grid(True, alpha=0.3, axis='y')

    # Plot 9: Evidence Accumulation
    ax9 = fig.add_subplot(gs[3, 1])
    evidence = [r.evidence_accumulated for r in daily_reports]
    ax9.plot(dates, evidence, 'purple', linewidth=2, marker='o', markersize=4)
    ax9.axhline(2.0, color='red', linestyle='--', alpha=0.7, label='Alert Threshold')
    ax9.set_title('Evidence Accumulation', fontsize=11, fontweight='bold')
    ax9.set_xlabel('Date', fontsize=10)
    ax9.set_ylabel('Cumulative Evidence', fontsize=10)
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3)
    ax9.tick_params(labelsize=9)

    # Plot 10: Final Prediction Summary
    ax10 = fig.add_subplot(gs[3, 2])
    ax10.axis('off')

    summary_text = f"""
FINAL PREDICTION
{'='*30}

Status: {'ANOMALY DETECTED' if final_prediction.sustained_anomaly_detected else 'NORMAL'}
Confidence: {final_prediction.confidence:.1%}

Pattern: {final_prediction.pattern_identified}
Final Score: {final_prediction.final_anomaly_score:.3f}

Evidence Summary:
• Sustained days: {final_prediction.evidence_summary['sustained_deviation_days']}
• Evidence: {final_prediction.evidence_summary['evidence_accumulated']:.2f}
• Days above threshold: {final_prediction.evidence_summary['days_above_threshold']}

Recommendation:
{final_prediction.recommendation}
"""

    ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(f'/home/claude/analysis_{scenario}.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: analysis_{scenario}.png")
    plt.close()


# ============================================================================
# DAILY REPORT GENERATOR
# ============================================================================

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

    # Generate baseline
    baseline, baseline_df = generator.generate_baseline(days=28)
    print(f"  ✓ Baseline established")

    # Initialize detector
    detector = ImprovedAnomalyDetector(baseline)

    # Generate monitoring data
    monitoring_df = generator.generate_monitoring_data(baseline, scenario, days=30)

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
    print(f"\n  📊 ANALYSIS COMPLETE")
    print(f"  {'─'*76}")

    alert_dist = {}
    for r in daily_reports:
        alert_dist[r.alert_level] = alert_dist.get(r.alert_level, 0) + 1

    print(f"  Alert Distribution:")
    for level in ['green', 'yellow', 'orange', 'red']:
        count = alert_dist.get(level, 0)
        pct = (count / len(daily_reports)) * 100
        bar = '█' * (count // 2)
        print(f"    {level.upper():6s}: {bar:15s} {count:2d} days ({pct:5.1f}%)")

    print(f"\n  🎯 FINAL PREDICTION:")
    print(f"  {'─'*76}")
    print(f"  Status: {final_prediction.sustained_anomaly_detected}")
    print(f"  Confidence: {final_prediction.confidence:.1%}")
    print(f"  Pattern: {final_prediction.pattern_identified}")
    print(f"  Final Score: {final_prediction.final_anomaly_score:.3f}")
    print(f"  Sustained Days: {final_prediction.evidence_summary['sustained_deviation_days']}")
    print(f"  Evidence: {final_prediction.evidence_summary['evidence_accumulated']:.2f}")
    print(f"\n  💡 Recommendation:")
    print(f"  {final_prediction.recommendation}")

    # Generate visualizations
    plot_comprehensive_results(baseline_df, monitoring_df, daily_reports, 
                              scenario, final_prediction)

    # Generate daily report text
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
    print("SYSTEM 1: IMPROVED ANOMALY DETECTION WITH SUSTAINED PATTERN RECOGNITION")
    print("="*80)
    print("\nObjective: Only flag anomalies after accumulating sufficient evidence")
    print("         Individual odd days should NOT trigger high alerts\n")

    scenarios = [
        ('normal', 'PT-001'),
        ('anomaly_subtle_rapid', 'PT-002'),
        ('anomaly_gradual_depression', 'PT-003'),
        ('normal_life_event', 'PT-004'),
        ('mixed_signals', 'PT-005'),
    ]

    scenario_names = {
        'normal': 'Normal Patient',
        'anomaly_subtle_rapid': 'Anomaly Patient 1 (Rapid Subtle Changes)',
        'anomaly_gradual_depression': 'Anomaly Patient 2 (Gradual Depression)',
        'normal_life_event': 'Normal Patient 2 (Life Event Recovery)',
        'mixed_signals': 'Mixed Signals Patient'
    }

    all_results = {}

    for scenario, patient_id in scenarios:
        results = run_scenario(scenario, patient_id)
        all_results[scenario] = results

        # Save daily report to file
        with open(f'/home/claude/daily_report_{scenario}.txt', 'w') as f:
            f.write(results['daily_summary'])
        print(f"  ✓ Saved: daily_report_{scenario}.txt")

    # Generate comparison summary
    print(f"\n\n{'='*80}")
    print("COMPARATIVE ANALYSIS SUMMARY")
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
    with open('/home/claude/comparison_summary.txt', 'w') as f:
        f.write(comparison_text)

    print(f"\n{'='*80}")
    print("ALL ANALYSES COMPLETE")
    print(f"{'='*80}")
    print(f"\nGenerated files:")
    print(f"  • {len(scenarios)} analysis visualizations (analysis_*.png)")
    print(f"  • {len(scenarios)} daily reports (daily_report_*.txt)")
    print(f"  • 1 comparison summary (comparison_summary.txt)")

    return all_results


if __name__ == "__main__":
    results = main()
