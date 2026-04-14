"""
PhoneDNA Builder: device-level behavioural fingerprint from session events.

Aggregates across all apps to build:
    - Pickup patterns (first pickup hour, burst rate, inter-pickup intervals)
    - Session duration distribution (5-bin histogram)
    - App co-occurrence matrix
    - Notification relationship metrics
    - Rhythm regularity and weekday/weekend delta
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional

from system1.data_structures import PhoneDNA


class PhoneDNABuilder:
    """
    Builds the device-level PhoneDNA from all session and notification events.

    Returns a default PhoneDNA if no session data is available.
    """

    def build(
        self,
        baseline_session_events: Optional[List[List[Dict]]] = None,
        baseline_notification_events: Optional[List[List[Dict]]] = None,
        baseline_days: int = 28,
    ) -> PhoneDNA:
        """
        Build PhoneDNA from baseline events.

        Parameters
        ----------
        baseline_session_events : list of daily session event lists
        baseline_notification_events : list of daily notification event lists

        Returns
        -------
        PhoneDNA instance (default values if no data)
        """
        dna = PhoneDNA()

        if not baseline_session_events:
            return dna

        # Flatten all sessions
        all_sessions = []
        daily_sessions: Dict[int, List[Dict]] = defaultdict(list)
        for day_idx, day_events in enumerate(baseline_session_events):
            if day_events:
                all_sessions.extend(day_events)
                daily_sessions[day_idx] = day_events

        if not all_sessions:
            return dna

        # --- First pickup hour ---
        daily_first_hours = []
        daily_last_hours = []
        for day_idx, sessions in daily_sessions.items():
            hours = [float(s.get('hour', 12)) for s in sessions]
            if hours:
                daily_first_hours.append(min(hours))
                daily_last_hours.append(max(hours))

        if daily_first_hours:
            dna.first_pickup_hour_mean = float(np.mean(daily_first_hours))
            dna.first_pickup_hour_std = float(np.std(daily_first_hours))

        # --- Active window duration ---
        active_windows = [last - first for first, last in zip(daily_first_hours, daily_last_hours)]
        if active_windows:
            dna.active_window_duration_mean = float(np.mean(active_windows))
            dna.active_window_duration_std = float(np.std(active_windows))

        # --- Pickups per hour ---
        hourly_pickups = np.zeros(24)
        for s in all_sessions:
            hour = int(s.get('hour', 0)) % 24
            hourly_pickups[hour] += 1
        # Normalise by number of days
        if baseline_days > 0:
            hourly_pickups /= baseline_days
        dna.pickups_per_hour_by_hour = hourly_pickups

        # --- Pickup burst rate ---
        # Fraction of sessions within 5 minutes of a previous session
        timestamps = sorted(float(s.get('open_timestamp_ms', 0)) for s in all_sessions)
        burst_count = 0
        for i in range(1, len(timestamps)):
            gap_minutes = (timestamps[i] - timestamps[i - 1]) / 60000.0
            if gap_minutes < 5:
                burst_count += 1
        dna.pickup_burst_rate = burst_count / max(len(timestamps) - 1, 1)

        # --- Inter-pickup interval ---
        intervals = []
        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i - 1]) / 60000.0
            if 0 < gap < 1440:  # exclude gaps > 24h
                intervals.append(gap)
        if intervals:
            dna.inter_pickup_interval_mean = float(np.mean(intervals))
            dna.inter_pickup_interval_std = float(np.std(intervals))

        # --- Session duration distribution (5 bins) ---
        durations = [float(s.get('duration_minutes', 0)) for s in all_sessions]
        bins = [0, 2, 15, 30, 60, float('inf')]
        hist = np.histogram(durations, bins=bins)[0]
        total = max(hist.sum(), 1)
        dna.session_duration_distribution = hist / total

        # Deep and micro session ratios
        dna.deep_session_ratio = sum(1 for d in durations if d > 20) / max(len(durations), 1)
        dna.micro_session_ratio = sum(1 for d in durations if d < 2) / max(len(durations), 1)

        # --- Notification relationship ---
        if baseline_notification_events:
            all_notifs = []
            for day_notifs in baseline_notification_events:
                if day_notifs:
                    all_notifs.extend(day_notifs)

            if all_notifs:
                actions = [n.get('action', '') for n in all_notifs]
                n_total = max(len(actions), 1)
                dna.notification_open_rate = sum(1 for a in actions if a == 'TAP') / n_total
                dna.notification_dismiss_rate = sum(1 for a in actions if a == 'DISMISS') / n_total
                dna.notification_ignore_rate = sum(1 for a in actions if a == 'IGNORE') / n_total

        # --- Rhythm regularity ---
        # Autocorrelation of hourly pickup pattern across days
        if len(daily_sessions) >= 3:
            daily_patterns = []
            for day_idx in sorted(daily_sessions.keys()):
                pattern = np.zeros(24)
                for s in daily_sessions[day_idx]:
                    hour = int(s.get('hour', 0)) % 24
                    pattern[hour] += 1
                daily_patterns.append(pattern)

            if len(daily_patterns) >= 2:
                correlations = []
                for i in range(1, len(daily_patterns)):
                    p1 = daily_patterns[i - 1]
                    p2 = daily_patterns[i]
                    if np.std(p1) > 0 and np.std(p2) > 0:
                        corr = float(np.corrcoef(p1, p2)[0, 1])
                        if not np.isnan(corr):
                            correlations.append(corr)
                if correlations:
                    dna.daily_rhythm_regularity = float(np.mean(correlations))

        # --- Weekday/weekend delta ---
        weekday_features = []
        weekend_features = []
        for day_idx, sessions in daily_sessions.items():
            dow = day_idx % 7
            n_sessions = len(sessions)
            total_dur = sum(float(s.get('duration_minutes', 0)) for s in sessions)
            vec = [n_sessions, total_dur]
            if dow < 5:
                weekday_features.append(vec)
            else:
                weekend_features.append(vec)

        if weekday_features and weekend_features:
            wd_mean = np.mean(weekday_features, axis=0)
            we_mean = np.mean(weekend_features, axis=0)
            dna.weekday_weekend_delta = float(np.sum(np.abs(wd_mean - we_mean)))

        # --- Historically active hours ---
        threshold = np.mean(hourly_pickups) * 0.5
        dna.historically_active_hours = [
            h for h in range(24) if hourly_pickups[h] > threshold
        ]

        print(f"  [PhoneDNA] Built: {len(all_sessions)} sessions, "
              f"burst_rate={dna.pickup_burst_rate:.2f}, "
              f"rhythm_reg={dna.daily_rhythm_regularity:.2f}")

        return dna
