"""
S1 → S2 Adapter
================

Bridges System 1's output format to System 2's S1Input contract.
This module is the ONLY coupling point between the two systems.

Handles feature name translation:
    System 1 (Android) uses camelCase: screenTimeHours, callsPerDay, ...
    System 2 (Literature) uses snake_case: screen_time_hours, calls_per_day, ...

Usage
-----
    from system2.s1_s2_adapter import build_s1_input

    s1_input = build_s1_input(detector, baseline_df, latest_s1_report)
    s2_output = pipeline.classify(s1_input)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .life_event_filter import AnomalyReport as S2AnomalyReport
from .pipeline import S1Input


# ── Feature name mapping: camelCase (S1/Android) ↔ snake_case (S2/Literature) ──

CAMEL_TO_SNAKE: Dict[str, str] = {
    "screenTimeHours": "screen_time_hours",
    "unlockCount": "unlock_count",
    "appLaunchCount": "app_launch_count",
    "notificationsToday": "notifications_today",
    "socialAppRatio": "social_app_ratio",
    "callsPerDay": "calls_per_day",
    "callDurationMinutes": "call_duration_minutes",
    "uniqueContacts": "unique_contacts",
    "conversationFrequency": "conversation_frequency",
    "dailyDisplacementKm": "daily_displacement_km",
    "locationEntropy": "location_entropy",
    "homeTimeRatio": "home_time_ratio",
    "wakeTimeHour": "wake_time_hour",
    "sleepTimeHour": "sleep_time_hour",
    "sleepDurationHours": "sleep_duration_hours",
    "darkDurationHours": "dark_duration_hours",
    "chargeDurationHours": "charge_duration_hours",
    "memoryUsagePercent": "memory_usage_percent",
    "networkWifiMB": "network_wifi_mb",
    "networkMobileMB": "network_mobile_mb",
    "storageUsedGB": "storage_used_gb",
    "totalAppsCount": "total_apps_count",
    "upiTransactionsToday": "upi_transactions_today",
    "appUninstallsToday": "app_uninstalls_today",
    "appInstallsToday": "app_installs_today",
    "calendarEventsToday": "calendar_events_today",
    "mediaCountToday": "media_count_today",
    "downloadsToday": "downloads_today",
    "backgroundAudioHours": "background_audio_hours",
}

SNAKE_TO_CAMEL: Dict[str, str] = {v: k for k, v in CAMEL_TO_SNAKE.items()}


def translate_dict(d: Dict[str, float], mapping: Dict[str, str]) -> Dict[str, float]:
    """Translate dict keys using mapping. Keys not in mapping are passed through."""
    return {mapping.get(k, k): v for k, v in d.items()}


def translate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Translate DataFrame column names from camelCase to snake_case."""
    return df.rename(columns=CAMEL_TO_SNAKE)


# ── Baseline Data Builder ──────────────────────────────────────────────

def build_baseline_data(baseline_df: pd.DataFrame) -> Dict:
    """
    Convert S1's baseline DataFrame (28 days) into S2's expected format.
    Translates camelCase columns to snake_case for System 2 compatibility.

    Parameters
    ----------
    baseline_df : pd.DataFrame
        28-day baseline period data from Kotlin (camelCase columns).

    Returns
    -------
    dict with keys: "raw_7day", "weekly_windows" (3x), "raw_28day"
    """
    # Translate column names camelCase → snake_case
    translated = translate_dataframe(baseline_df)

    # Drop non-feature columns
    feature_cols = [c for c in translated.columns if c != 'date']
    df = translated[feature_cols]

    # raw_7day: first 7 days average
    raw_7day = df.iloc[:7].mean().to_dict()

    # weekly_windows: 3 weekly averages (days 0-6, 7-13, 14-20)
    weekly_windows = []
    for week_start in [0, 7, 14]:
        week_end = min(week_start + 7, len(df))
        if week_start < len(df):
            window = df.iloc[week_start:week_end].mean().to_dict()
            weekly_windows.append(window)

    # raw_28day: full 28-day average
    raw_28day = df.mean().to_dict()

    return {
        "raw_7day": raw_7day,
        "weekly_windows": weekly_windows,
        "raw_28day": raw_28day,
    }


# ── Anomaly Report Mapper ─────────────────────────────────────────────

def build_anomaly_report(
    s1_report,
    co_deviating_threshold: float = 1.0,
) -> S2AnomalyReport:
    """
    Map System 1's AnomalyReport to System 2's AnomalyReport format.
    Translates camelCase feature names to snake_case.

    Parameters
    ----------
    s1_report : system1.AnomalyReport
        The latest report from ImprovedAnomalyDetector.analyze().
    co_deviating_threshold : float
        SD threshold for counting co-deviating features.

    Returns
    -------
    S2AnomalyReport
        Ready for System 2's LifeEventFilter and PrototypeMatcher.
    """
    # Translate feature_deviations keys from camelCase to snake_case
    snake_deviations = translate_dict(s1_report.feature_deviations, CAMEL_TO_SNAKE)

    # Count features deviating beyond threshold
    co_deviating_count = sum(
        1 for dev in snake_deviations.values()
        if abs(dev) > co_deviating_threshold
    )

    return S2AnomalyReport(
        feature_deviations=snake_deviations,
        days_sustained=s1_report.sustained_deviation_days,
        co_deviating_count=co_deviating_count,
        resolved=False,
        days_since_onset=s1_report.sustained_deviation_days,
        s1_alert_level=getattr(s1_report, 'alert_level', 'green'),
        s1_evidence=getattr(s1_report, 'evidence_accumulated', 0.0),
    )


# ── Full S1Input Builder ──────────────────────────────────────────────

def build_s1_input(
    detector,
    baseline_df: pd.DataFrame,
    s1_report,
    timeseries_days: int = 28,
) -> S1Input:
    """
    Build the complete S1Input bundle from System 1 components.
    Handles camelCase → snake_case translation at the boundary.

    Parameters
    ----------
    detector : ImprovedAnomalyDetector
        The active detector instance (has full_anomaly_history).
    baseline_df : pd.DataFrame
        28-day baseline DataFrame (camelCase columns from Kotlin).
    s1_report : system1.AnomalyReport
        Latest anomaly report from detector.analyze().
    timeseries_days : int
        Number of recent days for the anomaly timeseries (default 28).

    Returns
    -------
    S1Input
        Ready for System2Pipeline.classify().
    """
    # 1. Build baseline data (with translation)
    baseline_data = build_baseline_data(baseline_df)

    # 2. Map anomaly report (with translation)
    anomaly_report = build_anomaly_report(s1_report)

    # 3. Get anomaly timeseries (last N days, pad with 0 if shorter)
    history = detector.full_anomaly_history
    if len(history) >= timeseries_days:
        timeseries = history[-timeseries_days:]
    else:
        padding = [0.0] * (timeseries_days - len(history))
        timeseries = padding + list(history)

    return S1Input(
        baseline_data=baseline_data,
        anomaly_report=anomaly_report,
        anomaly_timeseries=timeseries,
    )
