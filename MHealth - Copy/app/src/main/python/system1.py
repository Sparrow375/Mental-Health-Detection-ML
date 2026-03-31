"""
System 1: Improved Anomaly Detection
Detects sustained deviations from personalized baseline.
Only flags after accumulating sufficient evidence over time.

Updated: Now covers all 29 features extracted by the Android DataCollector.
Feature groups:
  Group A — Screen & App Activity (5 features)
  Group B — Communication (4 features)
  Group C — Location & Movement (4 features)
  Group D — Sleep & Circadian (4 features)
  Group E — System Usage (5 features)
  Group F — Behavioural Signals (4 features, new)
  Group G — Calendar & Engagement (3 features)
"""

import sys
import os

# Fix Windows console encoding issues
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import deque
import json


# ============================================================================
# FEATURE METADATA — used for group-aware weighting and display
# ============================================================================

# Maps every feature key (matching Android's PersonalityVector.toMap() keys)
# to its semantic group and clinical weight.
# Weight > 1.0 = more sensitive to deviations of this feature (clinical priority)
# Weight < 1.0 = dampened (noisier sensor, less reliable signal)
FEATURE_META: Dict[str, Dict] = {
    # ── Group A: Screen & App Activity ────────────────────────────────────────
    "screenTimeHours":     {"group": "screen",        "weight": 1.4},
    "unlockCount":         {"group": "screen",        "weight": 1.2},
    "appLaunchCount":      {"group": "screen",        "weight": 0.9},
    "notificationsToday":  {"group": "screen",        "weight": 0.8},
    "socialAppRatio":      {"group": "screen",        "weight": 1.3},

    # ── Group B: Communication ─────────────────────────────────────────────
    "callsPerDay":               {"group": "communication", "weight": 1.3},
    "callDurationMinutes":       {"group": "communication", "weight": 1.2},
    "uniqueContacts":            {"group": "communication", "weight": 1.1},
    "conversationFrequency":     {"group": "communication", "weight": 0.9},

    # ── Group C: Location & Movement ──────────────────────────────────────
    "dailyDisplacementKm": {"group": "movement",      "weight": 1.5},
    "locationEntropy":     {"group": "movement",      "weight": 1.3},
    "homeTimeRatio":       {"group": "movement",      "weight": 1.2},
    "placesVisited":       {"group": "movement",      "weight": 1.1},

    # ── Group D: Sleep & Circadian ────────────────────────────────────────
    "wakeTimeHour":        {"group": "sleep",         "weight": 1.4},
    "sleepTimeHour":       {"group": "sleep",         "weight": 1.3},
    "sleepDurationHours":  {"group": "sleep",         "weight": 1.6},  # highest clinical weight
    "darkDurationHours":   {"group": "sleep",         "weight": 1.0},

    # ── Group E: System Usage ─────────────────────────────────────────────
    "chargeDurationHours": {"group": "system",        "weight": 0.8},
    "memoryUsagePercent":  {"group": "system",        "weight": 0.5},  # noisy, low value
    "networkWifiMB":       {"group": "system",        "weight": 0.6},
    "networkMobileMB":     {"group": "system",        "weight": 0.6},
    "storageUsedGB":       {"group": "system",        "weight": 0.4},  # quasi-static

    # ── Group F: Behavioural Signals (new features from Android) ──────────
    "totalAppsCount":      {"group": "behaviour",     "weight": 0.8},  # total installed apps
    "upiTransactionsToday":{"group": "behaviour",     "weight": 1.1},  # financial engagement proxy
    "appUninstallsToday":  {"group": "behaviour",     "weight": 0.9},
    "appInstallsToday":    {"group": "behaviour",     "weight": 0.8},

    # ── Group G: Calendar & Engagement ────────────────────────────────────
    "calendarEventsToday": {"group": "engagement",    "weight": 0.9},
    "mediaCountToday":     {"group": "engagement",    "weight": 0.7},
    "downloadsToday":      {"group": "engagement",    "weight": 0.6},
    "backgroundAudioHours":{"group": "engagement",    "weight": 0.9},
}

# Ordered list matches Android's PersonalityVector.toMap() key order
ALL_FEATURES: List[str] = list(FEATURE_META.keys())

# Critical features for alert-level escalation
CRITICAL_FEATURES = [
    "sleepDurationHours",
    "screenTimeHours",
    "dailyDisplacementKm",
    "socialAppRatio",
    "totalAppsCount",
    "upiTransactionsToday",
    "wakeTimeHour",
    "callsPerDay",
]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PersonalityVector:
    """
    Baseline personality profile.
    All 29 features mirror Android's PersonalityVector.toMap() keys exactly.
    """

    # ── Screen & App Activity ──────────────────────────────────────────────
    screenTimeHours: float = 0.0
    unlockCount: float = 0.0
    appLaunchCount: float = 0.0
    notificationsToday: float = 0.0
    socialAppRatio: float = 0.0

    # ── Communication ──────────────────────────────────────────────────────
    callsPerDay: float = 0.0
    callDurationMinutes: float = 0.0
    uniqueContacts: float = 0.0
    conversationFrequency: float = 0.0

    # ── Location & Movement ───────────────────────────────────────────────
    dailyDisplacementKm: float = 0.0
    locationEntropy: float = 0.0
    homeTimeRatio: float = 0.0
    placesVisited: float = 0.0

    # ── Sleep & Circadian ─────────────────────────────────────────────────
    wakeTimeHour: float = 0.0
    sleepTimeHour: float = 0.0
    sleepDurationHours: float = 0.0
    darkDurationHours: float = 0.0

    # ── System Usage ──────────────────────────────────────────────────────
    chargeDurationHours: float = 0.0
    memoryUsagePercent: float = 0.0
    networkWifiMB: float = 0.0
    networkMobileMB: float = 0.0
    storageUsedGB: float = 0.0

    # ── Behavioural Signals ───────────────────────────────────────────────
    totalAppsCount: float = 0.0
    upiTransactionsToday: float = 0.0
    appUninstallsToday: float = 0.0
    appInstallsToday: float = 0.0

    # ── Calendar & Engagement ─────────────────────────────────────────────
    calendarEventsToday: float = 0.0
    mediaCountToday: float = 0.0
    downloadsToday: float = 0.0
    backgroundAudioHours: float = 0.0

    # ── Internal: per-feature std deviation from baseline ─────────────────
    variances: Dict[str, float] = None   # actually std-dev, named for legacy compat
    
    # ── Individual App Usage Patterns ─────────────────────────────────────
    appBreakdown: Dict[str, float] = field(default_factory=dict)
    notificationBreakdown: Dict[str, float] = field(default_factory=dict)
    appLaunchesBreakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dict, matching Android PersonalityVector.toMap() key names."""
        return {
            "screenTimeHours":     self.screenTimeHours,
            "unlockCount":         self.unlockCount,
            "appLaunchCount":      self.appLaunchCount,
            "notificationsToday":  self.notificationsToday,
            "socialAppRatio":      self.socialAppRatio,
            "callsPerDay":         self.callsPerDay,
            "callDurationMinutes": self.callDurationMinutes,
            "uniqueContacts":      self.uniqueContacts,
            "conversationFrequency": self.conversationFrequency,
            "dailyDisplacementKm": self.dailyDisplacementKm,
            "locationEntropy":     self.locationEntropy,
            "homeTimeRatio":       self.homeTimeRatio,
            "placesVisited":       self.placesVisited,
            "wakeTimeHour":        self.wakeTimeHour,
            "sleepTimeHour":       self.sleepTimeHour,
            "sleepDurationHours":  self.sleepDurationHours,
            "darkDurationHours":   self.darkDurationHours,
            "chargeDurationHours": self.chargeDurationHours,
            "memoryUsagePercent":  self.memoryUsagePercent,
            "networkWifiMB":       self.networkWifiMB,
            "networkMobileMB":     self.networkMobileMB,
            "storageUsedGB":       self.storageUsedGB,
            "totalAppsCount":      self.totalAppsCount,
            "upiTransactionsToday": self.upiTransactionsToday,
            "appUninstallsToday":  self.appUninstallsToday,
            "appInstallsToday":    self.appInstallsToday,
            "calendarEventsToday": self.calendarEventsToday,
            "mediaCountToday":     self.mediaCountToday,
            "downloadsToday":      self.downloadsToday,
            "backgroundAudioHours": self.backgroundAudioHours,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float], variances: Dict[str, float] = None) -> "PersonalityVector":
        """Construct from a flat dict (e.g. from Android JSON baseline stats)."""
        return cls(
            screenTimeHours=     float(d.get("screenTimeHours", 0)),
            unlockCount=         float(d.get("unlockCount", 0)),
            appLaunchCount=      float(d.get("appLaunchCount", 0)),
            notificationsToday=  float(d.get("notificationsToday", 0)),
            socialAppRatio=      float(d.get("socialAppRatio", 0)),
            callsPerDay=         float(d.get("callsPerDay", 0)),
            callDurationMinutes= float(d.get("callDurationMinutes", 0)),
            uniqueContacts=      float(d.get("uniqueContacts", 0)),
            conversationFrequency= float(d.get("conversationFrequency", 0)),
            dailyDisplacementKm= float(d.get("dailyDisplacementKm", 0)),
            locationEntropy=     float(d.get("locationEntropy", 0)),
            homeTimeRatio=       float(d.get("homeTimeRatio", 0)),
            placesVisited=       float(d.get("placesVisited", 0)),
            wakeTimeHour=        float(d.get("wakeTimeHour", 0)),
            sleepTimeHour=       float(d.get("sleepTimeHour", 0)),
            sleepDurationHours=  float(d.get("sleepDurationHours", 0)),
            darkDurationHours=   float(d.get("darkDurationHours", 0)),
            chargeDurationHours= float(d.get("chargeDurationHours", 0)),
            memoryUsagePercent=  float(d.get("memoryUsagePercent", 0)),
            networkWifiMB=       float(d.get("networkWifiMB", 0)),
            networkMobileMB=     float(d.get("networkMobileMB", 0)),
            storageUsedGB=       float(d.get("storageUsedGB", 0)),
            totalAppsCount=      float(d.get("totalAppsCount", 0)),
            upiTransactionsToday= float(d.get("upiTransactionsToday", 0)),
            appUninstallsToday=  float(d.get("appUninstallsToday", 0)),
            appInstallsToday=    float(d.get("appInstallsToday", 0)),
            calendarEventsToday= float(d.get("calendarEventsToday", 0)),
            mediaCountToday=     float(d.get("mediaCountToday", 0)),
            downloadsToday=      float(d.get("downloadsToday", 0)),
            backgroundAudioHours= float(d.get("backgroundAudioHours", 0)),
            appBreakdown=        d.get("appBreakdown", {}),
            notificationBreakdown=d.get("notificationBreakdown", {}),
            appLaunchesBreakdown=d.get("appLaunchesBreakdown", {}),
            variances=variances,
        )


@dataclass
class DailyReport:
    """Daily analysis report."""
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
    """Output from System 1."""
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
    """Final analysis after monitoring period."""
    patient_id: str
    scenario: str
    monitoring_days: int
    baseline_vector: PersonalityVector
    final_anomaly_score: float
    sustained_anomaly_detected: bool
    confidence: float
    pattern_identified: str
    evidence_summary: Dict[str, object]
    recommendation: str


# ============================================================================
# SYNTHETIC DATA GENERATOR
# Generates realistic synthetic sensor data for testing and simulation.
# Uses the same feature keys as PersonalityVector.to_dict().
# ============================================================================

class SyntheticDataGenerator:
    """Generate realistic synthetic sensor data covering all 29 features."""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    def _baseline_params(self) -> Dict[str, float]:
        """Typical baseline values for a healthy working adult."""
        return {
            # Screen & App Activity
            "screenTimeHours":     4.5,
            "unlockCount":         80.0,
            "appLaunchCount":      55.0,
            "notificationsToday":  60.0,
            "socialAppRatio":      0.35,
            # Communication
            "callsPerDay":         3.0,
            "callDurationMinutes": 18.0,
            "uniqueContacts":      4.0,
            "conversationFrequency": 0.75,
            # Location & Movement
            "dailyDisplacementKm": 12.0,
            "locationEntropy":     2.3,
            "homeTimeRatio":       0.65,
            "placesVisited":       4.0,
            # Sleep & Circadian
            "wakeTimeHour":        7.5,
            "sleepTimeHour":       23.5,
            "sleepDurationHours":  7.5,
            "darkDurationHours":   8.5,
            # System Usage
            "chargeDurationHours": 6.0,
            "memoryUsagePercent":  65.0,
            "networkWifiMB":       450.0,
            "networkMobileMB":     120.0,
            "storageUsedGB":       22.0,
            # Behavioural Signals
            "nightInterruptions":  0.5,
            "upiTransactionsToday": 0.8,
            "appUninstallsToday":  0.05,
            "appInstallsToday":    0.1,
            # Calendar & Engagement
            "calendarEventsToday": 1.5,
            "mediaCountToday":     3.0,
            "downloadsToday":      1.0,
            "backgroundAudioHours": 1.5,
        }

    def generate_baseline(self, days: int = 28) -> Tuple[PersonalityVector, pd.DataFrame]:
        """Generate clean baseline period data."""
        print(f"  Generating {days} days of baseline data (29 features)...")

        bp = self._baseline_params()
        dates = [datetime.now() - timedelta(days=days - i) for i in range(days)]
        data = []

        noise_factors = {
            "screenTimeHours": 0.15,
            "unlockCount": 0.20,
            "appLaunchCount": 0.18,
            "notificationsToday": 0.25,
            "socialAppRatio": 0.12,
            "callsPerDay": 0.30,
            "callDurationMinutes": 0.35,
            "uniqueContacts": 0.25,
            "conversationFrequency": 0.30,
            "dailyDisplacementKm": 0.25,
            "locationEntropy": 0.15,
            "homeTimeRatio": 0.10,
            "placesVisited": 0.20,
            "wakeTimeHour": 0.08,
            "sleepTimeHour": 0.06,
            "sleepDurationHours": 0.10,
            "darkDurationHours": 0.12,
            "chargeDurationHours": 0.20,
            "memoryUsagePercent": 0.08,
            "networkWifiMB": 0.40,
            "networkMobileMB": 0.45,
            "storageUsedGB": 0.02,
            "nightInterruptions": 1.0,   # count, Poisson-like; std ≈ mean for low counts
            "upiTransactionsToday": 1.2,
            "appUninstallsToday": 2.0,
            "appInstallsToday": 2.0,
            "calendarEventsToday": 0.5,
            "mediaCountToday": 0.60,
            "downloadsToday": 0.80,
            "backgroundAudioHours": 0.50,
        }

        for date in dates:
            row = {"date": date}
            for feat, mean_val in bp.items():
                nf = noise_factors.get(feat, 0.15)
                std = max(mean_val * nf, 0.1)
                value = np.random.normal(mean_val, std)
                row[feat] = max(0.0, value)
            data.append(row)

        df = pd.DataFrame(data)
        feature_cols = [c for c in df.columns if c != "date"]

        variances = {f: float(df[f].std()) for f in feature_cols}
        # Ensure no zero std (avoid division-by-zero in detector)
        for f in feature_cols:
            if variances[f] < 0.01:
                variances[f] = 0.01

        baseline_means = {f: float(df[f].mean()) for f in feature_cols}
        baseline_vector = PersonalityVector.from_dict(baseline_means, variances=variances)

        return baseline_vector, df

    def generate_monitoring_data(
        self, baseline: PersonalityVector, scenario: str, days: int = 180
    ) -> pd.DataFrame:
        """Generate monitoring period data with different mental health patterns."""
        print(f"  Generating {days} days of '{scenario}' monitoring data (29 features)...")

        bp = baseline.to_dict()
        variances = baseline.variances or {}
        dates = [datetime.now() + timedelta(days=i) for i in range(days)]
        data = []

        bpd_state = "normal"
        days_in_state = 0
        state_threshold = np.random.randint(3, 10)

        for i, date in enumerate(dates):
            row = {"date": date}

            if scenario == "normal":
                for feat, mean_val in bp.items():
                    std = variances.get(feat, mean_val * 0.15)
                    row[feat] = max(0.0, np.random.normal(mean_val, std))

            elif scenario == "bpd_rapid_cycling":
                days_in_state += 1
                if days_in_state >= state_threshold:
                    choices = ["normal", "impulsive", "depressive"]
                    choices.remove(bpd_state)
                    bpd_state = np.random.choice(choices)
                    days_in_state = 0
                    state_threshold = np.random.randint(3, 8)

                for feat, mean_val in bp.items():
                    std = variances.get(feat, mean_val * 0.15)
                    if bpd_state == "normal":
                        val = np.random.normal(mean_val, std)
                    elif bpd_state == "impulsive":
                        if feat in ["screenTimeHours", "socialAppRatio", "callsPerDay",
                                    "unlockCount", "appLaunchCount", "upiTransactionsToday"]:
                            val = mean_val * 1.7 + np.random.normal(0, std)
                        elif feat == "sleepDurationHours":
                            val = mean_val * 0.65 + np.random.normal(0, std)
                        elif feat == "totalAppsCount":
                            val = mean_val + np.random.normal(0, std)
                        elif feat == "dailyDisplacementKm":
                            val = mean_val * 1.4 + np.random.normal(0, std)
                        else:
                            val = np.random.normal(mean_val, std)
                    else:  # depressive
                        if feat == "screenTimeHours":
                            val = mean_val * 1.5 + np.random.normal(0, std)  # doom-scrolling
                        elif feat in ["socialAppRatio", "callsPerDay", "callDurationMinutes",
                                      "uniqueContacts", "conversationFrequency"]:
                            val = mean_val * 0.35 + np.random.normal(0, std)
                        elif feat in ["dailyDisplacementKm", "placesVisited", "locationEntropy"]:
                            val = mean_val * 0.25 + np.random.normal(0, std)
                        elif feat == "sleepDurationHours":
                            val = mean_val * 1.5 + np.random.normal(0, std)
                        elif feat == "homeTimeRatio":
                            val = min(1.0, mean_val * 1.4 + np.random.normal(0, std))
                        elif feat == "upiTransactionsToday":
                            val = mean_val * 0.2 + np.random.normal(0, std)
                        else:
                            val = np.random.normal(mean_val, std)
                    row[feat] = max(0.0, val)

            elif scenario == "anomaly_gradual_depression":
                # 50% gradual decline over monitoring period
                decline_factor = 1.0 - (i / days) * 0.50

                for feat, mean_val in bp.items():
                    std = variances.get(feat, mean_val * 0.15)
                    if feat in ["socialAppRatio", "callsPerDay", "callDurationMinutes",
                                "uniqueContacts", "conversationFrequency",
                                "dailyDisplacementKm", "placesVisited", "locationEntropy",
                                "calendarEventsToday", "mediaCountToday", "backgroundAudioHours"]:
                        val = mean_val * decline_factor + np.random.normal(0, std)
                    elif feat == "sleepDurationHours":
                        # Hypersomnia
                        val = mean_val * (1.0 + (i / days) * 0.40) + np.random.normal(0, std)
                    elif feat == "screenTimeHours":
                        # Passive scrolling increases
                        val = mean_val * (1.0 + (i / days) * 0.30) + np.random.normal(0, std)
                    elif feat == "totalAppsCount":
                        val = mean_val + np.random.normal(0, std)
                    elif feat == "homeTimeRatio":
                        val = min(1.0, mean_val * (1.0 + (i / days) * 0.30) + np.random.normal(0, std))
                    else:
                        val = np.random.normal(mean_val, std)
                    row[feat] = max(0.0, val)

            elif scenario == "normal_life_event":
                # Days 0-70: normal; Days 71-100: temporary dip; Days 101+: recovery
                if i < 70 or i >= 100:
                    for feat, mean_val in bp.items():
                        std = variances.get(feat, mean_val * 0.15)
                        row[feat] = max(0.0, np.random.normal(mean_val, std))
                else:
                    dip_progress = (i - 70) / 30.0
                    dip_factor = 1.0 - (0.40 * np.sin(np.pi * dip_progress))
                    for feat, mean_val in bp.items():
                        std = variances.get(feat, mean_val * 0.15)
                        if feat in ["socialAppRatio", "callsPerDay", "dailyDisplacementKm",
                                    "calendarEventsToday", "uniqueContacts"]:
                            val = mean_val * dip_factor + np.random.normal(0, std)
                        elif feat == "sleepDurationHours":
                            val = mean_val * (1.0 + (1.0 - dip_factor) * 0.5) + np.random.normal(0, std)
                        else:
                            val = np.random.normal(mean_val, std)
                        row[feat] = max(0.0, val)

            elif scenario == "anxiety_pattern":
                # High arousal, fragmented sleep, erratic location, high phone usage
                for feat, mean_val in bp.items():
                    std = variances.get(feat, mean_val * 0.15)
                    if feat in ["unlockCount", "notificationsToday", "appLaunchCount"]:
                        val = mean_val * (1.0 + (i / days) * 0.60) + np.random.normal(0, std)
                    elif feat == "sleepDurationHours":
                        val = mean_val * (1.0 - (i / days) * 0.25) + np.random.normal(0, std)
                    elif feat in ["wakeTimeHour", "sleepTimeHour"]:
                        # Circadian shift — earlier wake, later sleep
                        shift = (i / days) * 1.5
                        val = mean_val + shift + np.random.normal(0, std)
                    elif feat == "totalAppsCount":
                        val = mean_val + np.random.normal(0, std)
                    elif feat in ["networkWifiMB", "networkMobileMB"]:
                        val = mean_val * (1.0 + (i / days) * 0.80) + np.random.normal(0, std)
                    else:
                        val = np.random.normal(mean_val, std)
                    row[feat] = max(0.0, val)

            elif scenario == "mixed_signals":
                for feat, mean_val in bp.items():
                    std = variances.get(feat, mean_val * 0.15)
                    if feat in ["screenTimeHours", "socialAppRatio"]:
                        cycle = np.sin(2 * np.pi * i / 14)
                        val = mean_val + cycle * mean_val * 0.35 + np.random.normal(0, std * 0.5)
                    elif feat in ["callsPerDay", "uniqueContacts"]:
                        decline = 1.0 - (i / days) * 0.30
                        val = mean_val * decline + np.random.normal(0, std)
                    elif feat == "totalAppsCount":
                        val = mean_val + abs(np.random.normal(0, std))
                    else:
                        val = np.random.normal(mean_val, std * 1.5)
                    row[feat] = max(0.0, val)

            else:
                # Unknown scenario — fallback to normal
                for feat, mean_val in bp.items():
                    std = variances.get(feat, mean_val * 0.15)
                    row[feat] = max(0.0, np.random.normal(mean_val, std))

            data.append(row)

        return pd.DataFrame(data)


# ============================================================================
# SYSTEM 1: IMPROVED ANOMALY DETECTOR
# ============================================================================

class ImprovedAnomalyDetector:
    """
    System 1: Detects sustained deviations from baseline across all 29 features.

    Scoring pipeline:
      1. Per-feature Z-score deviation (weighted by FEATURE_META weights)
      2. EWMA velocity — rate of change in 7-day window
      3. Composite score = 0.7 × magnitude + 0.3 × velocity
      4. Sustained evidence accumulation with exponential growth
      5. Alert escalation only after evidence threshold is met
    """

    def __init__(self, baseline: PersonalityVector):
        self.baseline = baseline
        self.baseline_dict = baseline.to_dict()
        self.feature_names = list(self.baseline_dict.keys())

        # History for velocity and pattern detection
        self.history_window = 7
        self.feature_history: Dict[str, deque] = {
            f: deque(maxlen=self.history_window) for f in self.feature_names
        }

        # Sustained deviation tracking
        self.anomaly_score_history: deque = deque(maxlen=14)   # 2-week rolling
        self.full_anomaly_history: List[float] = []           # full history for S2
        self.sustained_deviation_days: int = 0
        self.evidence_accumulated: float = 0.0

        # Real-time alerting thresholds
        self.ANOMALY_SCORE_THRESHOLD: float = 0.38   # tuned for 29-feature space
        self.SUSTAINED_THRESHOLD_DAYS: int = 5
        self.EVIDENCE_THRESHOLD: float = 2.0

        # Retrospective clinical thresholds
        self.PEAK_EVIDENCE_THRESHOLD: float = 7.0
        self.PEAK_SUSTAINED_THRESHOLD_DAYS: int = 10
        self.WATCH_EVIDENCE_THRESHOLD: float = 1.5

        # Peak tracking
        self.max_evidence: float = 0.0
        self.max_sustained_days: int = 0
        self.max_anomaly_score: float = 0.0

    # ── Public calibration ─────────────────────────────────────────────────────

    def calibrate_from_baseline(self, baseline_df: pd.DataFrame):
        """
        Calibrate PEAK thresholds from this user's own baseline noise.
        Only raises PEAK thresholds when the baseline is genuinely very noisy
        (mean score > 0.30), protecting against false positives.
        ANOMALY_SCORE_THRESHOLD is NOT changed here.
        """
        if baseline_df is None or len(baseline_df) < 7:
            return

        feature_cols = [c for c in self.feature_names if c in baseline_df.columns]
        if not feature_cols:
            return

        baseline_scores = []
        for _, row in baseline_df.iterrows():
            current = {}
            for feat in self.feature_names:
                if feat in row.index and pd.notna(row[feat]):
                    current[feat] = float(row[feat])
                else:
                    current[feat] = self.baseline_dict.get(feat, 0.0)
            deviations = self.calculate_deviation_magnitude(current)
            velocities = self.calculate_deviation_velocity(current)
            score = self.calculate_anomaly_score(deviations, velocities)
            baseline_scores.append(score)

        if not baseline_scores:
            return

        b_mean = float(np.mean(baseline_scores))
        b_std  = float(np.std(baseline_scores))

        if b_mean > 0.30:
            extra = (b_mean - 0.30) / 0.10
            self.PEAK_EVIDENCE_THRESHOLD = float(
                np.clip(2.71 + extra * 1.0, 2.71, 6.0))
            self.PEAK_SUSTAINED_THRESHOLD_DAYS = int(
                np.clip(round(5 + extra), 5, 12))

        print(f"  [Calibrate] baseline mean={b_mean:.3f} std={b_std:.3f}")
        print(f"  [Calibrate] ANOMALY_SCORE_THRESHOLD  = {self.ANOMALY_SCORE_THRESHOLD:.3f} (fixed)")
        print(f"  [Calibrate] PEAK_EVIDENCE_THRESHOLD  = {self.PEAK_EVIDENCE_THRESHOLD:.3f}")
        print(f"  [Calibrate] PEAK_SUSTAINED_DAYS      = {self.PEAK_SUSTAINED_THRESHOLD_DAYS}")

    # ── Core computation methods ──────────────────────────────────────────────

    def calculate_deviation_magnitude(
        self, current_data: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Z-score deviation from baseline, weighted by clinical feature importance.
        Features with weight > 1.0 are amplified — their deviations count more.
        Features with weight < 1.0 are damped — noisier sensors, less reliable.
        """
        deviations = {}
        variances = self.baseline.variances or {}

        for feature in self.feature_names:
            baseline_val = self.baseline_dict.get(feature, 0.0)
            current_val = current_data.get(feature, baseline_val)
            std = variances.get(feature, 1.0)
            std = std if std > 0 else 1.0

            z = (current_val - baseline_val) / std
            weight = FEATURE_META.get(feature, {}).get("weight", 1.0)
            deviations[feature] = z * weight

        return deviations

    def calculate_deviation_velocity(
        self, current_data: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Rate of change via EWMA (α=0.4).
        Higher α → more weight to recent days; detects accelerating trends.
        """
        velocities = {}
        alpha = 0.4

        for feature in self.feature_names:
            self.feature_history[feature].append(
                current_data.get(feature, self.baseline_dict.get(feature, 0.0))
            )

        for feature in self.feature_names:
            history = list(self.feature_history[feature])
            if len(history) < 2:
                velocities[feature] = 0.0
                continue

            ewma_values = []
            ewma = history[0]
            for val in history:
                ewma = alpha * val + (1 - alpha) * ewma
                ewma_values.append(ewma)

            slope = (ewma_values[-1] - ewma_values[0]) / len(ewma_values)
            baseline_val = self.baseline_dict.get(feature, 1.0)
            velocities[feature] = slope / baseline_val if baseline_val != 0 else 0.0

        return velocities

    def calculate_anomaly_score(
        self,
        deviations: Dict[str, float],
        velocities: Dict[str, float],
    ) -> float:
        """
        Composite anomaly score (0–1):
          score = 0.7 × magnitude_score + 0.3 × velocity_score

        Magnitude is the mean absolute weighted Z-score across all features,
        normalized to [0, 1] by dividing by 3.0 (≈ 3-SD threshold).
        Velocity is mean absolute EWMA slope × 10, capped at 1.0.
        """
        magnitude_score = np.mean([abs(d) for d in deviations.values()])
        magnitude_score = min(magnitude_score / 3.0, 1.0)

        velocity_score = np.mean([abs(v) for v in velocities.values()])
        velocity_score = min(velocity_score * 10.0, 1.0)

        return float(0.7 * magnitude_score + 0.3 * velocity_score)

    def update_sustained_tracking(self, anomaly_score: float):
        """
        Update evidence accumulation.
        Above threshold: evidence grows exponentially with consecutive days.
        Normal day: evidence decays at 8% per day (slow forgetting).
        """
        self.anomaly_score_history.append(anomaly_score)
        self.full_anomaly_history.append(anomaly_score)

        if anomaly_score > self.max_anomaly_score:
            self.max_anomaly_score = anomaly_score

        if anomaly_score > self.ANOMALY_SCORE_THRESHOLD:
            self.sustained_deviation_days += 1
            # Exponential growth: longer the streak, bigger the penalty
            self.evidence_accumulated += anomaly_score * (
                1.0 + self.sustained_deviation_days * 0.1
            )
        else:
            self.sustained_deviation_days = max(0, self.sustained_deviation_days - 1)
            self.evidence_accumulated *= 0.92

        if self.evidence_accumulated > self.max_evidence:
            self.max_evidence = self.evidence_accumulated
        if self.sustained_deviation_days > self.max_sustained_days:
            self.max_sustained_days = self.sustained_deviation_days

    def should_alert_now(self) -> bool:
        """Real-time alert gate — uses current state."""
        return (
            self.evidence_accumulated >= self.EVIDENCE_THRESHOLD
            or self.sustained_deviation_days >= self.SUSTAINED_THRESHOLD_DAYS
        )

    def had_episode(self) -> bool:
        """Retrospective detection — uses peak state with stricter thresholds."""
        return (
            self.max_evidence >= self.PEAK_EVIDENCE_THRESHOLD
            or self.max_sustained_days >= self.PEAK_SUSTAINED_THRESHOLD_DAYS
        )

    def determine_alert_level(
        self,
        anomaly_score: float,
        deviations: Dict[str, float],
    ) -> str:
        """
        Alert level: green / yellow / orange / red.
        Conservative: requires sustained pattern before escalating above green.
        Critical features (sleep, displacement, night interruptions etc.)
        amplify the alert when they deviate strongly.
        """
        critical_deviation = max(
            abs(deviations.get(f, 0.0)) for f in CRITICAL_FEATURES
        )

        has_sustained = (
            self.sustained_deviation_days >= self.SUSTAINED_THRESHOLD_DAYS
            or self.evidence_accumulated >= self.EVIDENCE_THRESHOLD
        )

        if not has_sustained:
            return "green"

        if anomaly_score < 0.35 and critical_deviation < 2.0:
            return "green"
        elif anomaly_score < 0.50 and critical_deviation < 2.5:
            return "yellow"
        elif anomaly_score < 0.65 or critical_deviation < 3.0:
            return "orange"
        else:
            return "red"

    def detect_pattern_type(
        self, deviations_history: List[Dict[str, float]]
    ) -> str:
        """Identify the temporal pattern of deviations over the last 7 days."""
        if len(deviations_history) < 7:
            return "insufficient_data"

        recent = deviations_history[-7:]
        avg_devs = [
            np.mean([abs(v) for v in d.values()]) for d in recent
        ]
        mean_dev = float(np.mean(avg_devs))
        std_dev  = float(np.std(avg_devs))

        if mean_dev < 0.5:
            return "stable"
        elif std_dev > 1.0 and mean_dev > 0.5:
            return "rapid_cycling"
        elif mean_dev > 1.5 and std_dev < 0.8:
            return "acute_spike"
        else:
            x = np.arange(len(avg_devs))
            slope = np.polyfit(x, avg_devs, 1)[0]
            return "gradual_drift" if abs(slope) > 0.1 else "mixed_pattern"

    def identify_flagged_features(
        self,
        deviations: Dict[str, float],
        threshold: float = 1.5,
    ) -> List[str]:
        """Return features deviating beyond threshold (in weighted SD units)."""
        return [
            f"{feat} ({dev:.2f} SD)"
            for feat, dev in deviations.items()
            if abs(dev) > threshold
        ]

    def get_top_deviations(
        self,
        deviations: Dict[str, float],
        top_n: int = 5,
    ) -> Dict[str, float]:
        """Return top N most deviated features."""
        return dict(
            sorted(deviations.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        )

    def analyze(
        self,
        current_data: Dict[str, float],
        deviations_history: List[Dict[str, float]],
        day_number: int,
    ) -> Tuple[AnomalyReport, DailyReport]:
        """
        Main analysis function.
        Accepts a dict of 29 feature values (matching PersonalityVector.to_dict() keys).
        Returns (AnomalyReport, DailyReport).
        """
        deviations = self.calculate_deviation_magnitude(current_data)
        velocities = self.calculate_deviation_velocity(current_data)
        anomaly_score = self.calculate_anomaly_score(deviations, velocities)

        # Update sustained tracking BEFORE determining alert level
        self.update_sustained_tracking(anomaly_score)

        alert_level = self.determine_alert_level(anomaly_score, deviations)
        pattern_type = self.detect_pattern_type(deviations_history)
        flagged = self.identify_flagged_features(deviations)
        top_devs = self.get_top_deviations(deviations)
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
            evidence_accumulated=self.evidence_accumulated,
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
            notes=notes,
        )

        return report, daily_report

    def generate_final_prediction(
        self, scenario: str, patient_id: str, monitoring_days: int
    ) -> FinalPrediction:
        """Generate final prediction after the full monitoring period."""
        confidence = min(0.95, monitoring_days / 30 * 0.8 + 0.15)
        sustained_anomaly = self.had_episode()

        final_score = float(np.mean(list(self.anomaly_score_history))) \
            if self.anomaly_score_history else 0.0

        pattern = "stable"
        if len(self.anomaly_score_history) >= 7:
            recent = list(self.anomaly_score_history)[-7:]
            if np.std(recent) > 0.15:
                pattern = "unstable/cycling"
            elif np.mean(recent) > 0.5:
                pattern = "persistent_elevation"

        if sustained_anomaly and self.max_evidence >= 4.0:
            recommendation = (
                "REFER: Very strong evidence of sustained behavioral deviation "
                "(Critical Peak). Immediate clinical evaluation recommended."
            )
        elif sustained_anomaly:
            recommendation = (
                "MONITOR: Significant sustained deviation detected "
                "(Met Peak Threshold). Clinical follow-up recommended."
            )
        elif self.max_evidence > 1.5:
            recommendation = (
                "WATCH: Some periodic evidence of deviation. "
                "Suggest extending monitoring or additional check-ins."
            )
        else:
            recommendation = "NORMAL: No significant sustained deviation detected."

        evidence_summary = {
            "sustained_deviation_days": self.sustained_deviation_days,
            "max_sustained_days": self.max_sustained_days,
            "evidence_accumulated_final": round(self.evidence_accumulated, 2),
            "peak_evidence": round(self.max_evidence, 2),
            "max_daily_anomaly_score": round(self.max_anomaly_score, 3),
            "avg_recent_anomaly_score": round(final_score, 3),
            "monitoring_days": monitoring_days,
            "days_above_threshold": sum(
                1 for s in self.full_anomaly_history if s > self.ANOMALY_SCORE_THRESHOLD
            ),
        }

        return FinalPrediction(
            patient_id=patient_id,
            scenario=scenario,
            monitoring_days=monitoring_days,
            baseline_vector=self.baseline,
            final_anomaly_score=final_score,
            sustained_anomaly_detected=sustained_anomaly,
            confidence=confidence,
            pattern_identified=pattern,
            evidence_summary=evidence_summary,
            recommendation=recommendation,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate_notes(
        self, anomaly_score: float, alert_level: str, pattern_type: str
    ) -> str:
        notes = []
        if self.sustained_deviation_days >= self.SUSTAINED_THRESHOLD_DAYS:
            notes.append(
                f"Sustained deviation ({self.sustained_deviation_days} consecutive days)"
            )
        if self.evidence_accumulated >= self.EVIDENCE_THRESHOLD:
            notes.append(f"Evidence: {self.evidence_accumulated:.2f}")
        if pattern_type in ("rapid_cycling", "gradual_drift"):
            notes.append(f"Pattern: {pattern_type}")
        if alert_level in ("orange", "red"):
            notes.append(f"HIGH ALERT: {alert_level.upper()}")
        if anomaly_score > 0.6 and alert_level == "green":
            notes.append("High single-day score but no sustained pattern yet")
        return " | ".join(notes) if notes else "Normal operation"


# ============================================================================
# SIMULATION ENTRY POINTS
# ============================================================================

def run_scenario(scenario: str, patient_id: str) -> dict:
    """Run a complete end-to-end simulation for one scenario."""
    print(f"\n{'='*80}")
    print(f"PATIENT: {patient_id} | SCENARIO: {scenario.upper().replace('_', ' ')}")
    print(f"{'='*80}")

    generator = SyntheticDataGenerator(seed=hash(patient_id) % 1000)

    baseline, baseline_df = generator.generate_baseline(days=28)
    print(f"  Baseline established ({len(baseline.to_dict())} features)")

    detector = ImprovedAnomalyDetector(baseline)
    detector.calibrate_from_baseline(baseline_df)

    monitoring_df = generator.generate_monitoring_data(baseline, scenario, days=180)

    reports = []
    daily_reports = []
    deviations_history = []

    for idx, row in monitoring_df.iterrows():
        current_data = {
            k: v for k, v in row.to_dict().items() if k != "date"
        }
        report, daily_report = detector.analyze(current_data, deviations_history, idx + 1)
        reports.append(report)
        daily_reports.append(daily_report)
        deviations_history.append(report.feature_deviations)

    final_prediction = detector.generate_final_prediction(
        scenario, patient_id, len(monitoring_df)
    )

    print(f"\n  ANALYSIS COMPLETE (180-day simulation)")
    alert_dist: Dict[str, int] = {}
    for r in daily_reports:
        alert_dist[r.alert_level] = alert_dist.get(r.alert_level, 0) + 1

    print("  Alert Distribution:")
    for level in ["green", "yellow", "orange", "red"]:
        count = alert_dist.get(level, 0)
        pct = count / len(daily_reports) * 100
        print(f"    {level.upper():6s}: {count:3d} days ({pct:5.1f}%)")

    print(f"\n  FINAL PREDICTION:")
    print(f"  Status:    {'ANOMALY' if final_prediction.sustained_anomaly_detected else 'NORMAL'}")
    print(f"  Confidence: {final_prediction.confidence:.1%}")
    print(f"  Pattern:   {final_prediction.pattern_identified}")
    print(f"  Evidence:  {final_prediction.evidence_summary['peak_evidence']:.2f}")
    print(f"  Recommendation: {final_prediction.recommendation}")

    return {
        "baseline": baseline,
        "monitoring": monitoring_df,
        "reports": reports,
        "daily_reports": daily_reports,
        "final_prediction": final_prediction,
    }


def main():
    """Run all scenarios."""
    print("\n" + "=" * 80)
    print("SYSTEM 1: 6-MONTH SIMULATION — 29-FEATURE PERSONALITY VECTOR")
    print("=" * 80)

    scenarios = [
        ("normal",                     "PT-001"),
        ("bpd_rapid_cycling",          "PT-002"),
        ("anomaly_gradual_depression",  "PT-003"),
        ("normal_life_event",           "PT-004"),
        ("mixed_signals",              "PT-005"),
        ("anxiety_pattern",            "PT-006"),
    ]

    all_results = {}
    for scenario, patient_id in scenarios:
        all_results[scenario] = run_scenario(scenario, patient_id)

    print(f"\n{'='*80}")
    print("COMPARATIVE SUMMARY")
    print(f"{'='*80}")
    header = f"{'Patient':<10} {'Scenario':<35} {'Anomaly':<9} {'Confidence':<12} {'Evidence':<10}"
    print(header)
    print("-" * 80)
    for scenario, patient_id in scenarios:
        pred = all_results[scenario]["final_prediction"]
        print(
            f"{pred.patient_id:<10} "
            f"{scenario:<35} "
            f"{'YES' if pred.sustained_anomaly_detected else 'NO':<9} "
            f"{pred.confidence:<12.1%} "
            f"{pred.evidence_summary['peak_evidence']:<10.2f}"
        )

    return all_results


if __name__ == "__main__":
    results = main()
