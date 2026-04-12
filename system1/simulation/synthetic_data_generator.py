"""
Synthetic Data Generator — generates realistic baseline and monitoring data for testing.
"""

from __future__ import annotations

import datetime
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..data_structures import SessionEvent, NotificationEvent
from ..feature_meta import ALL_L1_FEATURES


# ── Healthy baseline defaults ──────────────────────────────────────────────────
HEALTHY_DEFAULTS = {
    "screenTimeHours": 5.0,
    "unlockCount": 60.0,
    "appLaunchCount": 80.0,
    "notificationsToday": 50.0,
    "socialAppRatio": 0.25,
    "callsPerDay": 4.0,
    "callDurationMinutes": 30.0,
    "uniqueContacts": 6.0,
    "conversationFrequency": 3.0,
    "dailyDisplacementKm": 12.0,
    "locationEntropy": 2.5,
    "homeTimeRatio": 0.55,
    "placesVisited": 4.0,
    "wakeTimeHour": 7.5,
    "sleepTimeHour": 23.0,
    "sleepDurationHours": 7.5,
    "darkDurationHours": 8.0,
    "chargeDurationHours": 2.0,
    "memoryUsagePercent": 65.0,
    "networkWifiMB": 500.0,
    "networkMobileMB": 200.0,
    "storageUsedGB": 45.0,
    "totalAppsCount": 50.0,
    "upiTransactionsToday": 2.0,
    "appUninstallsToday": 0.2,
    "appInstallsToday": 0.3,
    "calendarEventsToday": 2.0,
    "mediaCountToday": 5.0,
    "downloadsToday": 1.0,
    "backgroundAudioHours": 1.0,
}

HEALTHY_STD = {k: v * 0.15 for k, v in HEALTHY_DEFAULTS.items()}


# ── Depression shift factors ───────────────────────────────────────────────────
DEPRESSION_SHIFTS = {
    "screenTimeHours": 1.4,
    "unlockCount": 0.7,
    "socialAppRatio": 1.6,
    "callsPerDay": 0.3,
    "callDurationMinutes": 0.3,
    "uniqueContacts": 0.4,
    "dailyDisplacementKm": 0.3,
    "locationEntropy": 0.5,
    "homeTimeRatio": 1.5,
    "placesVisited": 0.3,
    "wakeTimeHour": 1.3,  # Later wake
    "sleepTimeHour": 1.2,  # Later sleep
    "sleepDurationHours": 0.6,  # Less sleep
    "upiTransactionsToday": 0.2,
    "calendarEventsToday": 0.3,
    "mediaCountToday": 0.4,
}

# Anxiety shift factors
ANXIETY_SHIFTS = {
    "screenTimeHours": 1.3,
    "unlockCount": 1.8,  # More unlocks (compulsive checking)
    "notificationsToday": 1.5,
    "socialAppRatio": 1.4,
    "callsPerDay": 0.7,
    "sleepDurationHours": 0.7,
    "wakeTimeHour": 1.2,
    "dailyDisplacementKm": 0.5,
}

APP_IDS = [
    "com.whatsapp", "com.instagram.android", "com.google.android.apps.messaging",
    "com.twitter.android", "com.youtube", "com.spotify.music",
    "com.snapchat.android", "com.facebook.katana", "com.reddit.app",
    "com.google.android.gm",
]


def generate_baseline_days(
    n_days: int = 28,
    seed: int = 42,
    noise_level: float = 1.0,
) -> Tuple[List[Dict[str, float]], List[str]]:
    """Generate n_days of healthy baseline daily features."""
    np.random.seed(seed)
    random.seed(seed)

    daily_features = []
    dates = []
    start = datetime.date(2025, 1, 1)

    for i in range(n_days):
        date = start + datetime.timedelta(days=i)
        dates.append(date.strftime("%Y-%m-%d"))
        day = {}

        is_weekend = date.weekday() >= 5

        for feat in ALL_L1_FEATURES:
            base = HEALTHY_DEFAULTS.get(feat, 5.0)
            std = HEALTHY_STD.get(feat, 1.0) * noise_level

            # Weekend adjustments
            if is_weekend:
                if feat == "wakeTimeHour":
                    base += 1.5
                elif feat == "sleepTimeHour":
                    base += 1.0
                elif feat == "screenTimeHours":
                    base += 1.0
                elif feat == "dailyDisplacementKm":
                    base -= 3.0
                elif feat == "placesVisited":
                    base -= 1.0

            val = np.random.normal(base, std)
            # Ensure non-negative for most features
            if feat not in ["wakeTimeHour", "sleepTimeHour"]:
                val = max(0, val)
            day[feat] = round(val, 3)

        daily_features.append(day)

    return daily_features, dates


def generate_depression_episode(
    n_days: int = 14,
    severity: float = 0.7,
    onset: str = "gradual",
    seed: int = 123,
) -> List[Dict[str, float]]:
    """Generate monitoring days simulating a depressive episode."""
    np.random.seed(seed)
    random.seed(seed)

    days = []
    for i in range(n_days):
        day = {}

        if onset == "gradual":
            factor = min(1.0, i / (n_days * 0.6)) * severity
        elif onset == "sudden":
            factor = severity if i > 2 else 0.0
        else:
            factor = severity * (0.5 + 0.5 * np.sin(i * 0.5))

        for feat in ALL_L1_FEATURES:
            base = HEALTHY_DEFAULTS.get(feat, 5.0)
            std = HEALTHY_STD.get(feat, 1.0)

            shift = DEPRESSION_SHIFTS.get(feat, 1.0)
            shifted = base * (1.0 + (shift - 1.0) * factor)

            val = np.random.normal(shifted, std * (1.0 + factor * 0.5))
            if feat not in ["wakeTimeHour", "sleepTimeHour"]:
                val = max(0, val)
            day[feat] = round(val, 3)

        days.append(day)

    return days


def generate_anxiety_episode(
    n_days: int = 14,
    severity: float = 0.6,
    seed: int = 456,
) -> List[Dict[str, float]]:
    """Generate monitoring days simulating an anxiety episode."""
    np.random.seed(seed)

    days = []
    for i in range(n_days):
        day = {}
        factor = min(1.0, i / (n_days * 0.5)) * severity

        for feat in ALL_L1_FEATURES:
            base = HEALTHY_DEFAULTS.get(feat, 5.0)
            std = HEALTHY_STD.get(feat, 1.0)

            shift = ANXIETY_SHIFTS.get(feat, 1.0)
            shifted = base * (1.0 + (shift - 1.0) * factor)

            val = np.random.normal(shifted, std * (1.0 + factor * 0.3))
            if feat not in ["wakeTimeHour", "sleepTimeHour"]:
                val = max(0, val)
            day[feat] = round(val, 3)

        days.append(day)

    return days


def generate_healthy_monitoring(
    n_days: int = 14,
    seed: int = 789,
) -> List[Dict[str, float]]:
    """Generate healthy monitoring days (control)."""
    np.random.seed(seed)

    days = []
    for i in range(n_days):
        day = {}
        for feat in ALL_L1_FEATURES:
            base = HEALTHY_DEFAULTS.get(feat, 5.0)
            std = HEALTHY_STD.get(feat, 1.0)
            val = np.random.normal(base, std)
            if feat not in ["wakeTimeHour", "sleepTimeHour"]:
                val = max(0, val)
            day[feat] = round(val, 3)
        days.append(day)

    return days


def generate_session_events(
    dates: List[str],
    n_sessions_per_day: int = 40,
    seed: int = 42,
    degraded: bool = False,
    degradation_factor: float = 0.0,
) -> List[SessionEvent]:
    """Generate synthetic session events for given dates."""
    np.random.seed(seed)
    random.seed(seed)
    sessions = []

    for date_str in dates:
        dt_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")

        for _ in range(n_sessions_per_day):
            app = random.choice(APP_IDS)

            # Generate open time
            if degraded and random.random() < degradation_factor:
                # Late night sessions during depression
                hour = random.choice([1, 2, 3, 0, 23])
            else:
                hour = random.choices(
                    range(24),
                    weights=[1, 1, 1, 1, 1, 2, 5, 8, 10, 10, 10, 10,
                             10, 8, 8, 8, 10, 10, 10, 10, 8, 6, 3, 2],
                    k=1
                )[0]
            minute = random.randint(0, 59)

            open_dt = dt_date.replace(hour=hour, minute=minute)
            open_ts = open_dt.timestamp() * 1000

            # Session duration
            if degraded and random.random() < degradation_factor * 0.5:
                # Shorter sessions (abandon)
                duration_ms = random.uniform(10000, 120000)  # 10s-2min
            else:
                duration_ms = random.uniform(30000, 1800000)  # 30s-30min

            close_ts = open_ts + duration_ms

            trigger = random.choices(
                ["SELF", "NOTIFICATION", "SHORTCUT"],
                weights=[0.6, 0.3, 0.1],
                k=1
            )[0]

            if degraded:
                interactions = max(1, int(random.gauss(5, 3) * (1 - degradation_factor * 0.3)))
            else:
                interactions = max(1, int(random.gauss(8, 4)))

            sessions.append(SessionEvent(
                app_id=app,
                open_ts=open_ts,
                close_ts=close_ts,
                trigger=trigger,
                interaction_count=interactions,
            ))

    return sorted(sessions, key=lambda s: s.open_ts)


def generate_notification_events(
    dates: List[str],
    n_per_day: int = 30,
    seed: int = 42,
    degraded: bool = False,
) -> List[NotificationEvent]:
    """Generate synthetic notification events."""
    np.random.seed(seed)
    random.seed(seed)
    notifications = []

    for date_str in dates:
        dt_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")

        for _ in range(n_per_day):
            app = random.choice(APP_IDS)
            hour = random.randint(8, 22)
            minute = random.randint(0, 59)
            arrival_dt = dt_date.replace(hour=hour, minute=minute)
            arrival_ts = arrival_dt.timestamp() * 1000

            if degraded:
                action = random.choices(
                    ["TAP", "DISMISS", "IGNORE"],
                    weights=[0.2, 0.3, 0.5],
                    k=1
                )[0]
            else:
                action = random.choices(
                    ["TAP", "DISMISS", "IGNORE"],
                    weights=[0.4, 0.35, 0.25],
                    k=1
                )[0]

            tap_latency = None
            if action == "TAP":
                tap_latency = random.uniform(0.5, 30.0)

            notifications.append(NotificationEvent(
                app_id=app,
                arrival_ts=arrival_ts,
                action=action,
                tap_latency_min=tap_latency,
            ))

    return notifications