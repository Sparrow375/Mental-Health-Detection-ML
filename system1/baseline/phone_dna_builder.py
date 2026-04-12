"""
PhoneDNA Builder — constructs device-level behavioral DNA from all session events.
"""

from __future__ import annotations

import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..data_structures import PhoneDNA, SessionEvent, NotificationEvent
from ..feature_meta import THRESHOLDS


def build_phone_dna(
    sessions: List[SessionEvent],
    notifications: List[NotificationEvent],
    baseline_dates: List[str],
) -> PhoneDNA:
    """Build device-level PhoneDNA from all baseline sessions and notifications."""
    dna = PhoneDNA()

    if not sessions:
        return dna

    # Sort sessions by open timestamp
    sorted_sessions = sorted(sessions, key=lambda s: s.open_ts)

    # ── Group sessions by date ────────────────────────────────────────────
    daily_sessions: Dict[str, List[SessionEvent]] = defaultdict(list)
    for sess in sorted_sessions:
        dt = datetime.datetime.fromtimestamp(sess.open_ts / 1000.0)
        date_str = dt.strftime("%Y-%m-%d")
        daily_sessions[date_str].append(sess)

    # ── First pickup hour distribution ────────────────────────────────────
    first_pickups = []
    last_pickups = []
    for date_str in sorted(daily_sessions.keys()):
        day_sess = daily_sessions[date_str]
        if day_sess:
            first_dt = datetime.datetime.fromtimestamp(day_sess[0].open_ts / 1000.0)
            first_pickups.append(first_dt.hour + first_dt.minute / 60.0)
            last_dt = datetime.datetime.fromtimestamp(day_sess[-1].open_ts / 1000.0)
            last_pickups.append(last_dt.hour + last_dt.minute / 60.0)

    if first_pickups:
        dna.first_pickup_hour_mean = float(np.mean(first_pickups))
        dna.first_pickup_hour_std = float(np.std(first_pickups)) if len(first_pickups) > 1 else 0.0

    # ── Active window duration ────────────────────────────────────────────
    if first_pickups and last_pickups:
        windows = []
        for f, l in zip(first_pickups, last_pickups):
            windows.append(max(0, l - f))
        dna.active_window_duration_mean = float(np.mean(windows))
        dna.active_window_duration_std = float(np.std(windows)) if len(windows) > 1 else 0.0

    # ── Pickups per hour by hour (24-element array) ───────────────────────
    hourly_pickups = np.zeros(24)
    for sess in sorted_sessions:
        dt = datetime.datetime.fromtimestamp(sess.open_ts / 1000.0)
        hourly_pickups[dt.hour] += 1
    n_days = max(1, len(baseline_dates))
    hourly_pickups /= n_days
    dna.pickups_per_hour_by_hour = hourly_pickups

    # ── Pickup burst rate (fraction within 5 min of previous) ─────────────
    if len(sorted_sessions) > 1:
        burst_count = 0
        for i in range(1, len(sorted_sessions)):
            gap_min = (sorted_sessions[i].open_ts - sorted_sessions[i - 1].open_ts) / 60000.0
            if gap_min < 5:
                burst_count += 1
        dna.pickup_burst_rate = burst_count / (len(sorted_sessions) - 1)

    # ── Inter-pickup interval ─────────────────────────────────────────────
    if len(sorted_sessions) > 1:
        gaps = []
        for i in range(1, len(sorted_sessions)):
            gap_min = (sorted_sessions[i].open_ts - sorted_sessions[i - 1].open_ts) / 60000.0
            gaps.append(gap_min)
        dna.inter_pickup_interval_mean = float(np.mean(gaps))
        dna.inter_pickup_interval_std = float(np.std(gaps)) if len(gaps) > 1 else 0.0

    # ── Session duration distribution (5-bin histogram) ───────────────────
    durations = [s.duration_minutes for s in sorted_sessions]
    bins = [0, 2, 15, 30, 60, float('inf')]
    hist, _ = np.histogram(durations, bins=bins)
    hist = hist.astype(float)
    total_sess = max(1, hist.sum())
    hist /= total_sess  # Normalize to ratios
    dna.session_duration_distribution = hist

    # ── Deep / micro session ratios ───────────────────────────────────────
    deep_count = sum(1 for d in durations if d > 20)
    micro_count = sum(1 for d in durations if d < 2)
    dna.deep_session_ratio = deep_count / max(1, len(durations))
    dna.micro_session_ratio = micro_count / max(1, len(durations))

    # ── Notification relationship ─────────────────────────────────────────
    if notifications:
        tap_count = sum(1 for n in notifications if n.action == "TAP")
        dismiss_count = sum(1 for n in notifications if n.action == "DISMISS")
        ignore_count = sum(1 for n in notifications if n.action == "IGNORE")
        total_notif = max(1, len(notifications))
        dna.notification_open_rate = tap_count / total_notif
        dna.notification_dismiss_rate = dismiss_count / total_notif
        dna.notification_ignore_rate = ignore_count / total_notif

    # ── Daily rhythm regularity (autocorrelation of hourly pickup pattern) ──
    if len(daily_sessions) > 2:
        daily_patterns = []
        for date_str in sorted(daily_sessions.keys()):
            pattern = np.zeros(24)
            for sess in daily_sessions[date_str]:
                dt = datetime.datetime.fromtimestamp(sess.open_ts / 1000.0)
                pattern[dt.hour] += 1
            # Normalize
            pmax = pattern.max()
            if pmax > 0:
                pattern /= pmax
            daily_patterns.append(pattern)

        if len(daily_patterns) > 2:
            # Mean pairwise correlation as regularity measure
            correlations = []
            for i in range(len(daily_patterns) - 1):
                corr = np.corrcoef(daily_patterns[i], daily_patterns[i + 1])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            if correlations:
                dna.daily_rhythm_regularity = float(np.mean(correlations))

    # ── Weekday/weekend delta ─────────────────────────────────────────────
    weekday_features = defaultdict(list)
    weekend_features = defaultdict(list)
    feature_keys = ["screenTimeHours", "unlockCount"]

    weekday_pickups = []
    weekend_pickups = []
    for date_str, day_sess in daily_sessions.items():
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        n_pickups = len(day_sess)
        if dt.weekday() < 5:
            weekday_pickups.append(n_pickups)
        else:
            weekend_pickups.append(n_pickups)

    if weekday_pickups and weekend_pickups:
        mean_wd = np.mean(weekday_pickups)
        mean_we = np.mean(weekend_pickups)
        dna.weekday_weekend_delta = float(abs(mean_wd - mean_we))

    # ── Historically active hours ─────────────────────────────────────────
    threshold = hourly_pickups.mean() + 0.5 * hourly_pickups.std() if len(hourly_pickups) > 0 else 0
    dna.historically_active_hours = [int(h) for h in range(24) if hourly_pickups[h] > threshold]

    # ── App co-occurrence matrix ──────────────────────────────────────────
    # Apps appearing in the same 30-minute window
    app_list = sorted(set(s.app_id for s in sorted_sessions))
    app_idx = {a: i for i, a in enumerate(app_list)}
    n_apps = len(app_list)
    if n_apps > 0:
        cooccurrence = np.zeros((n_apps, n_apps))
        # Group sessions into 30-min windows
        for i, s1 in enumerate(sorted_sessions):
            for j in range(i + 1, len(sorted_sessions)):
                s2 = sorted_sessions[j]
                gap_min = (s2.open_ts - s1.open_ts) / 60000.0
                if gap_min > 30:
                    break
                if s1.app_id != s2.app_id:
                    idx1 = app_idx[s1.app_id]
                    idx2 = app_idx[s2.app_id]
                    cooccurrence[idx1, idx2] += 1
                    cooccurrence[idx2, idx1] += 1
        # Normalize
        total_co = cooccurrence.sum()
        if total_co > 0:
            cooccurrence /= total_co
        dna.app_cooccurrence_matrix = cooccurrence

    return dna