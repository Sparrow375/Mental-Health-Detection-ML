"""
StudentLife Dataset Loader — Transforms raw sensor CSVs into daily
PersonalityVector-compatible feature dicts for each student.

Maps available StudentLife sensors to our 30-feature clinical vector:
  - GPS → dailyDisplacementKm, locationEntropy, homeTimeRatio, placesVisited
  - phonelock/dark → sleepDurationHours, wakeTimeHour, sleepTimeHour, unlockCount, screenTimeHours
  - activity → dailyStepCount, activeMinutes
  - call_log → callsPerDay, callDurationMinutes, uniqueContacts, conversationFrequency
  - conversation → socialAppRatio (as proxy for face-to-face social)
  - audio → daylightExposureMinutes (ambient audio as activity proxy)
  - app_usage → appLaunchCount, notificationsToday
  - phonecharge → chargeDurationHours, chargeRegularity
  - EMA/Stress → stress level ground truth
  - survey/PHQ-9 → depression ground truth (pre/post)

Features not available in StudentLife are imputed with sensible defaults
or derived from related proxies.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Constants
# ============================================================================

DATASET_ROOT = r"F:\Avaneesh\download\student\dataset"

# StudentLife study epoch: March 2013 – June 2013 (Dartmouth Spring term)
# Timestamps are Unix epoch seconds

# PHQ-9 scoring: "Not at all"=0, "Several days"=1, "More than half the days"=2, "Nearly every day"=3
PHQ9_SCORE_MAP = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3,
}

# Activity inference codes from StudentLife
# 0=stationary, 1=walking, 2=running, 3=unknown
ACTIVITY_STATIONARY = 0
ACTIVITY_WALKING = 1
ACTIVITY_RUNNING = 2

# Audio inference codes
# 0=silence, 1=voice, 2=noise, 3=unknown
AUDIO_SILENCE = 0
AUDIO_VOICE = 1
AUDIO_NOISE = 2

# Earth radius for distance calculations
EARTH_RADIUS_KM = 6371.0

# Grid cell size for location (~110m)
GRID_CELL_DEG = 0.001


def _ts_to_date(ts: float) -> str:
    """Convert unix timestamp to date string YYYY-MM-DD (US Eastern)."""
    dt = datetime.fromtimestamp(ts, tz=timezone(timedelta(hours=-5)))
    return dt.strftime("%Y-%m-%d")


def _ts_to_hour(ts: float) -> float:
    """Convert unix timestamp to hour of day (0-23.99)."""
    dt = datetime.fromtimestamp(ts, tz=timezone(timedelta(hours=-5)))
    return dt.hour + dt.minute / 60.0


def _haversine(lat1, lon1, lat2, lon2) -> float:
    """Haversine distance in km."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _grid_cell(lat: float, lon: float) -> Tuple[int, int]:
    """Map lat/lon to ~110m grid cell."""
    return (int(lat / GRID_CELL_DEG), int(lon / GRID_CELL_DEG))


def _shannon_entropy(counts: Dict) -> float:
    """Shannon entropy from frequency dict."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def _get_student_ids() -> List[str]:
    """Discover all student IDs from the GPS directory."""
    gps_dir = os.path.join(DATASET_ROOT, "sensing", "gps")
    ids = []
    for f in os.listdir(gps_dir):
        if f.startswith("gps_u") and f.endswith(".csv"):
            uid = f.replace("gps_", "").replace(".csv", "")
            ids.append(uid)
    return sorted(ids)


# ============================================================================
# Sensor Parsers
# ============================================================================

def _load_gps(uid: str) -> Dict[str, List[Tuple[float, float, float]]]:
    """Load GPS data grouped by date → [(timestamp, lat, lon), ...]."""
    path = os.path.join(DATASET_ROOT, "sensing", "gps", f"gps_{uid}.csv")
    daily = defaultdict(list)
    if not os.path.exists(path):
        return daily
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = float(row['time'])
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                acc = float(row.get('accuracy', 100))
                # Filter out low-accuracy cell tower fixes
                if acc > 500:
                    continue
                date = _ts_to_date(ts)
                daily[date].append((ts, lat, lon))
            except (ValueError, KeyError):
                continue
    return daily


def _compute_gps_features(points: List[Tuple[float, float, float]]) -> Dict[str, float]:
    """Compute displacement, entropy, homeTimeRatio, placesVisited from day's GPS."""
    if len(points) < 2:
        return {
            "dailyDisplacementKm": 0.0,
            "locationEntropy": 0.0,
            "homeTimeRatio": 1.0,
        }

    points.sort(key=lambda x: x[0])

    # Grid-cell transition method for displacement
    total_km = 0.0
    prev_cell = _grid_cell(points[0][1], points[0][2])
    prev_lat, prev_lon = points[0][1], points[0][2]

    # Time spent in each grid cell (for time-based entropy)
    cell_time = defaultdict(float)
    places = set()
    places.add(prev_cell)

    for i in range(1, len(points)):
        ts, lat, lon = points[i]
        cell = _grid_cell(lat, lon)
        dt = ts - points[i - 1][0]
        if dt > 0 and dt < 7200:  # max 2-hour gap
            cell_time[prev_cell] += dt

        if cell != prev_cell:
            dist = _haversine(prev_lat, prev_lon, lat, lon)
            if dist > 0.05:  # >50m to count as real movement
                total_km += dist
            prev_cell = cell
            prev_lat, prev_lon = lat, lon
            places.add(cell)

    # Entropy from time-spent distribution
    entropy = _shannon_entropy(cell_time)

    # Home cell = most time spent
    if cell_time:
        home_cell = max(cell_time, key=cell_time.get)
        total_time = sum(cell_time.values())
        home_ratio = cell_time[home_cell] / total_time if total_time > 0 else 1.0
    else:
        home_ratio = 1.0

    return {
        "dailyDisplacementKm": round(total_km, 3),
        "locationEntropy": round(entropy, 4),
        "homeTimeRatio": round(home_ratio, 4),
    }


def _load_phonelock(uid: str) -> Dict[str, List[Tuple[float, float]]]:
    """Load phone lock data as date → [(start, end), ...]."""
    path = os.path.join(DATASET_ROOT, "sensing", "phonelock", f"phonelock_{uid}.csv")
    daily = defaultdict(list)
    if not os.path.exists(path):
        return daily
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = float(row['start'])
                end = float(row['end'])
                date = _ts_to_date(start)
                daily[date].append((start, end))
            except (ValueError, KeyError):
                continue
    return daily


def _load_dark(uid: str) -> Dict[str, List[Tuple[float, float]]]:
    """Load screen dark/off periods as date → [(start, end), ...]."""
    path = os.path.join(DATASET_ROOT, "sensing", "dark", f"dark_{uid}.csv")
    daily = defaultdict(list)
    if not os.path.exists(path):
        return daily
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = float(row['start'])
                end = float(row['end'])
                date = _ts_to_date(start)
                daily[date].append((start, end))
            except (ValueError, KeyError):
                continue
    return daily


def _compute_sleep_features(
    dark_periods: List[Tuple[float, float]],
    lock_periods: List[Tuple[float, float]],
    date_str: str,
) -> Dict[str, float]:
    """
    3-Signal Sleep Fusion:
    Find the longest screen-off gap in the overnight window (6PM-12PM next day).
    Merge micro-wakes < 5min. Use lock periods as confirmation.
    """
    if not dark_periods and not lock_periods:
        return {
            "sleepDurationHours": 7.0,  # fallback
            "wakeTimeHour": 8.0,
            "sleepTimeHour": 23.0,
        }

    # Combine dark and lock periods, use the union
    all_off = []
    for periods in [dark_periods, lock_periods]:
        for start, end in periods:
            if end > start:
                all_off.append((start, end))

    if not all_off:
        return {
            "sleepDurationHours": 7.0,
            "wakeTimeHour": 8.0,
            "sleepTimeHour": 23.0,
        }

    # Sort and merge overlapping intervals
    all_off.sort()
    merged = [all_off[0]]
    for start, end in all_off[1:]:
        if start <= merged[-1][1] + 300:  # merge if gap < 5 min
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Find longest off period (sleep candidate)
    longest = max(merged, key=lambda x: x[1] - x[0])
    duration_hours = (longest[1] - longest[0]) / 3600.0

    # Cap at reasonable values
    duration_hours = min(duration_hours, 14.0)
    duration_hours = max(duration_hours, 0.0)

    sleep_hour = _ts_to_hour(longest[0])
    wake_hour = _ts_to_hour(longest[1])

    return {
        "sleepDurationHours": round(duration_hours, 2),
        "wakeTimeHour": round(wake_hour, 2),
        "sleepTimeHour": round(sleep_hour, 2),
    }


def _compute_unlock_screen_features(
    lock_periods: List[Tuple[float, float]],
) -> Dict[str, float]:
    """Compute unlock count and approximate screen time from lock events."""
    if not lock_periods:
        return {"unlockCount": 0.0, "screenTimeHours": 0.0}

    # Each lock period represents phone being locked → the gaps are screen-on time
    # Number of lock periods ≈ number of unlocks
    unlock_count = len(lock_periods)

    # Screen-on time = 24h minus total locked time
    total_lock_seconds = sum(end - start for start, end in lock_periods if end > start)
    screen_on_hours = max(0.0, 24.0 - total_lock_seconds / 3600.0)
    screen_on_hours = min(screen_on_hours, 18.0)  # cap

    return {
        "unlockCount": float(unlock_count),
        "screenTimeHours": round(screen_on_hours, 2),
    }


def _load_activity(uid: str) -> Dict[str, List[Tuple[float, int]]]:
    """Load activity data as date → [(timestamp, activity_code), ...].
    Subsamples every 10th row for performance (original is ~3s resolution)."""
    path = os.path.join(DATASET_ROOT, "sensing", "activity", f"activity_{uid}.csv")
    daily = defaultdict(list)
    if not os.path.exists(path):
        return daily
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        row_idx = 0
        for row in reader:
            row_idx += 1
            if row_idx % 10 != 0:  # subsample every 10th row (~30s)
                continue
            try:
                ts = float(row['timestamp'].strip())
                activity = int(row[' activity inference'].strip())
                date = _ts_to_date(ts)
                daily[date].append((ts, activity))
            except (ValueError, KeyError):
                continue
    return daily


def _compute_activity_features(entries: List[Tuple[float, int]]) -> Dict[str, float]:
    """Compute step count and active minutes from activity inference."""
    if not entries:
        return {"dailyStepCount": 0.0, "activeMinutes": 0.0}

    entries.sort()
    active_seconds = 0.0
    estimated_steps = 0.0

    for i in range(1, len(entries)):
        dt = entries[i][0] - entries[i - 1][0]
        if dt > 300:  # gap > 5 min = no data
            continue
        activity = entries[i - 1][1]
        if activity == ACTIVITY_WALKING:
            active_seconds += dt
            estimated_steps += dt * 1.67  # ~100 steps/min
        elif activity == ACTIVITY_RUNNING:
            active_seconds += dt
            estimated_steps += dt * 2.5  # ~150 steps/min

    return {
        "dailyStepCount": round(estimated_steps),
        "activeMinutes": round(active_seconds / 60.0, 1),
    }


def _load_calls(uid: str) -> Dict[str, List[Dict]]:
    """Load call log data grouped by date."""
    path = os.path.join(DATASET_ROOT, "call_log", f"call_log_{uid}.csv")
    daily = defaultdict(list)
    if not os.path.exists(path):
        return daily
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = float(row.get('CALLS_date', row.get('timestamp', 0)))
                if ts > 1e12:  # milliseconds
                    ts /= 1000.0
                duration = float(row.get('CALLS_duration', 0))
                call_type = row.get('CALLS_type', '')
                number_hash = row.get('CALLS_number', '')
                date = _ts_to_date(ts)
                daily[date].append({
                    'duration': duration,
                    'type': call_type,
                    'number': number_hash,
                })
            except (ValueError, KeyError):
                continue
    return daily


def _compute_call_features(calls: List[Dict]) -> Dict[str, float]:
    """Compute call features from daily call log."""
    if not calls:
        return {
            "callsPerDay": 0.0,
            "callDurationMinutes": 0.0,
            "uniqueContacts": 0.0,
            "conversationFrequency": 0.0,
        }

    n_calls = len(calls)
    total_duration = sum(c.get('duration', 0) for c in calls) / 60.0
    unique = len(set(c.get('number', '') for c in calls if c.get('number', '')))
    conv_freq = n_calls / max(unique, 1)

    return {
        "callsPerDay": float(n_calls),
        "callDurationMinutes": round(total_duration, 2),
        "uniqueContacts": float(unique),
        "conversationFrequency": round(conv_freq, 2),
    }


def _load_conversation(uid: str) -> Dict[str, List[Tuple[float, float]]]:
    """Load conversation (face-to-face) data as date → [(start, end), ...]."""
    path = os.path.join(DATASET_ROOT, "sensing", "conversation", f"conversation_{uid}.csv")
    daily = defaultdict(list)
    if not os.path.exists(path):
        return daily
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = float(row['start_timestamp'].strip())
                end = float(row[' end_timestamp'].strip())
                date = _ts_to_date(start)
                daily[date].append((start, end))
            except (ValueError, KeyError):
                continue
    return daily


def _compute_conversation_features(convos: List[Tuple[float, float]]) -> Dict[str, float]:
    """Social time from face-to-face conversations → socialAppRatio proxy."""
    if not convos:
        return {"socialAppRatio": 0.0}

    total_seconds = sum(end - start for start, end in convos if end > start)
    # Normalize to fraction of waking hours (~16h)
    ratio = total_seconds / (16 * 3600.0)
    return {"socialAppRatio": round(min(ratio, 1.0), 4)}


def _load_audio(uid: str) -> Dict[str, List[Tuple[float, int]]]:
    """Load audio inference data as date → [(timestamp, inference), ...].
    Subsamples every 60th row for performance (original is ~1s resolution)."""
    path = os.path.join(DATASET_ROOT, "sensing", "audio", f"audio_{uid}.csv")
    daily = defaultdict(list)
    if not os.path.exists(path):
        return daily
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        row_idx = 0
        for row in reader:
            row_idx += 1
            if row_idx % 60 != 0:  # subsample every 60th row (~1/min)
                continue
            try:
                ts = float(row['timestamp'].strip())
                audio = int(row[' audio inference'].strip())
                date = _ts_to_date(ts)
                daily[date].append((ts, audio))
            except (ValueError, KeyError):
                continue
    return daily


def _compute_audio_features(entries: List[Tuple[float, int]]) -> Dict[str, float]:
    """Use audio inference as proxy for ambient activity/daylight exposure."""
    if not entries:
        return {"daylightExposureMinutes": 0.0}

    entries.sort()
    # Voice + noise = human-present/active environment (proxy for daylight/exposure)
    active_seconds = 0.0
    for i in range(1, len(entries)):
        dt = entries[i][0] - entries[i - 1][0]
        if dt > 300:
            continue
        if entries[i - 1][1] in (AUDIO_VOICE, AUDIO_NOISE):
            active_seconds += dt

    return {"daylightExposureMinutes": round(active_seconds / 60.0, 1)}


def _load_app_usage(uid: str) -> Dict[str, List[Dict]]:
    """Load running app data grouped by date. Only tracks unique timestamp snapshots."""
    path = os.path.join(DATASET_ROOT, "app_usage", f"running_app_{uid}.csv")
    daily = defaultdict(list)
    if not os.path.exists(path):
        return daily
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        last_ts = None
        daily_packages = defaultdict(set)
        daily_timestamps = defaultdict(set)
        for row in reader:
            try:
                ts = float(row['timestamp'])
                # Only process first row of each timestamp snapshot
                if ts == last_ts:
                    continue
                last_ts = ts
                pkg = row.get('RUNNING_TASKS_topActivity_mPackage', '')
                if not pkg:
                    continue
                date = _ts_to_date(ts)
                daily_packages[date].add(pkg)
                daily_timestamps[date].add(ts)
            except (ValueError, KeyError):
                continue
    # Convert to expected format
    for date in daily_timestamps:
        for ts in daily_timestamps[date]:
            daily[date].append({'ts': ts, 'package': ''})
        # Store unique packages count separately
        daily[date] = list(daily_timestamps[date]), daily_packages[date]
    return daily


def _compute_app_features(entries) -> Dict[str, float]:
    """Compute app launch count and notification proxy from app usage."""
    if not entries:
        return {
            "appLaunchCount": 0.0,
            "notificationsToday": 0.0,
        }

    # Handle new tuple format (timestamps_set, packages_set)
    if isinstance(entries, tuple) and len(entries) == 2:
        timestamps, unique_packages = entries
        return {
            "appLaunchCount": float(len(timestamps)),
            "notificationsToday": float(len(unique_packages)),
        }

    # Fallback for old format
    unique_packages = set(e['package'] for e in entries if isinstance(e, dict))
    timestamps = set(e['ts'] for e in entries if isinstance(e, dict))

    return {
        "appLaunchCount": float(len(timestamps)),
        "notificationsToday": float(len(unique_packages)),
    }


def _load_phonecharge(uid: str) -> Dict[str, List[Tuple[float, float]]]:
    """Load charging data as date → [(start, end), ...]."""
    path = os.path.join(DATASET_ROOT, "sensing", "phonecharge", f"phonecharge_{uid}.csv")
    daily = defaultdict(list)
    if not os.path.exists(path):
        return daily
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = float(row['start'])
                end = float(row['end'])
                date = _ts_to_date(start)
                daily[date].append((start, end))
            except (ValueError, KeyError):
                continue
    return daily


def _compute_charge_features(
    charges: List[Tuple[float, float]],
    all_charges_for_regularity: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """Compute charge duration and regularity."""
    if not charges:
        return {
            "chargeDurationHours": 0.0,
            "chargeRegularity": 0.5,
        }

    total = sum(end - start for start, end in charges if end > start) / 3600.0

    # Charge regularity: consistency of charging start times
    if all_charges_for_regularity and len(all_charges_for_regularity) >= 3:
        start_hours = [_ts_to_hour(s) for s, _ in all_charges_for_regularity[-14:]]
        if len(start_hours) >= 3:
            # Convert to circular mean for regularity
            sin_sum = sum(math.sin(2 * math.pi * h / 24) for h in start_hours)
            cos_sum = sum(math.cos(2 * math.pi * h / 24) for h in start_hours)
            r = math.sqrt(sin_sum ** 2 + cos_sum ** 2) / len(start_hours)
            regularity = r
        else:
            regularity = 0.5
    else:
        regularity = 0.5

    return {
        "chargeDurationHours": round(total, 2),
        "chargeRegularity": round(regularity, 4),
    }


# ============================================================================
# PHQ-9 Ground Truth Parser
# ============================================================================

def load_phq9() -> Dict[str, Dict[str, int]]:
    """
    Parse PHQ-9 survey and compute total scores.

    Returns:
        {uid: {"pre": score, "post": score}} where score is 0-27.
        PHQ-9 >= 10 is clinical depression cutoff.
    """
    path = os.path.join(DATASET_ROOT, "survey", "PHQ-9.csv")
    results = defaultdict(dict)

    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row['uid']
            phase = row['type']  # 'pre' or 'post'

            total = 0
            for col_name, value in row.items():
                if col_name in ('uid', 'type', 'Response'):
                    continue
                score = PHQ9_SCORE_MAP.get(value.strip(), 0) if value.strip() else 0
                total += score

            results[uid][phase] = total

    return dict(results)


def get_depression_labels(phq9_scores: Dict[str, Dict[str, int]], cutoff: int = 10) -> Dict[str, Dict[str, bool]]:
    """
    Convert PHQ-9 scores to binary depression labels.

    PHQ-9 >= cutoff → depressed=True.
    Standard clinical cutoff is 10 (moderate depression).
    """
    labels = {}
    for uid, scores in phq9_scores.items():
        labels[uid] = {}
        for phase, score in scores.items():
            labels[uid][phase] = score >= cutoff
    return labels


# ============================================================================
# Stress EMA Loader
# ============================================================================

def load_stress_ema() -> Dict[str, List[Tuple[str, float]]]:
    """Load daily stress EMA responses → {uid: [(date, level), ...]}."""
    stress_dir = os.path.join(DATASET_ROOT, "EMA", "response", "Stress")
    results = {}

    if not os.path.exists(stress_dir):
        return results

    for fname in os.listdir(stress_dir):
        if not fname.startswith("Stress_") or not fname.endswith(".json"):
            continue
        uid = fname.replace("Stress_", "").replace(".json", "")
        path = os.path.join(stress_dir, fname)

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        entries = []
        for item in data:
            try:
                ts = float(item.get('resp_time', 0))
                level = float(item.get('level', item.get('null', -1)))
                if level < 0 or level > 5:
                    continue
                date = _ts_to_date(ts)
                entries.append((date, level))
            except (ValueError, TypeError):
                continue

        if entries:
            results[uid] = entries

    return results


# ============================================================================
# Main Data Assembly Pipeline
# ============================================================================

def load_student_daily_features(uid: str) -> Dict[str, Dict[str, float]]:
    """
    Assemble complete daily feature vectors for one student.

    Returns:
        {date_str: {feature_name: value, ...}} sorted by date.
    """
    print(f"  Loading sensors for {uid}...")

    # Load all sensor streams
    gps_data = _load_gps(uid)
    lock_data = _load_phonelock(uid)
    dark_data = _load_dark(uid)
    activity_data = _load_activity(uid)
    call_data = _load_calls(uid)
    conv_data = _load_conversation(uid)
    audio_data = _load_audio(uid)
    app_data = _load_app_usage(uid)
    charge_data = _load_phonecharge(uid)

    # Collect all charge events for regularity calculation
    all_charge_events = []
    for periods in charge_data.values():
        all_charge_events.extend(periods)
    all_charge_events.sort()

    # Find all unique dates across sensors
    all_dates = set()
    for src in [gps_data, lock_data, dark_data, activity_data,
                call_data, conv_data, audio_data, app_data, charge_data]:
        all_dates.update(src.keys())

    daily_features = {}

    for date in sorted(all_dates):
        features = {}

        # GPS features
        gps_feats = _compute_gps_features(gps_data.get(date, []))
        features.update(gps_feats)

        # Sleep features (from dark + lock)
        sleep_feats = _compute_sleep_features(
            dark_data.get(date, []),
            lock_data.get(date, []),
            date,
        )
        features.update(sleep_feats)

        # Screen/unlock features
        unlock_feats = _compute_unlock_screen_features(lock_data.get(date, []))
        features.update(unlock_feats)

        # Activity features
        act_feats = _compute_activity_features(activity_data.get(date, []))
        features.update(act_feats)

        # Call features
        call_feats = _compute_call_features(call_data.get(date, []))
        features.update(call_feats)

        # Conversation (social) features
        conv_feats = _compute_conversation_features(conv_data.get(date, []))
        features.update(conv_feats)

        # Audio-derived features
        audio_feats = _compute_audio_features(audio_data.get(date, []))
        features.update(audio_feats)

        # App usage features
        app_feats = _compute_app_features(app_data.get(date, []))
        features.update(app_feats)

        # Charging features
        charge_feats = _compute_charge_features(
            charge_data.get(date, []),
            all_charge_events,
        )
        features.update(charge_feats)

        # Features not available in StudentLife — sensible defaults
        features.setdefault("keystrokeSpeed", 3.5)
        features.setdefault("backspaceRatio", 0.08)
        features.setdefault("scrollVelocity", 250.0)
        features.setdefault("upiTransactionsToday", 0.0)
        features.setdefault("appUninstallsToday", 0.0)
        features.setdefault("appInstallsToday", 0.0)
        features.setdefault("calendarEventsToday", 0.0)
        features.setdefault("mediaCountToday", 0.0)
        features.setdefault("downloadsToday", 0.0)
        features.setdefault("musicTimeMinutes", 0.0)

        daily_features[date] = features

    return daily_features


def load_all_students() -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Load daily features for ALL students.

    Returns:
        {uid: {date: {feature: value}}}
    """
    student_ids = _get_student_ids()
    all_data = {}

    print(f"Loading data for {len(student_ids)} students...")
    for uid in student_ids:
        daily = load_student_daily_features(uid)
        if daily:
            all_data[uid] = daily
            print(f"    {uid}: {len(daily)} days loaded")

    return all_data


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    # Test with one student
    uid = "u00"
    daily = load_student_daily_features(uid)
    print(f"\n{uid}: {len(daily)} days")
    if daily:
        sample_date = sorted(daily.keys())[0]
        print(f"Sample day ({sample_date}):")
        for k, v in sorted(daily[sample_date].items()):
            print(f"  {k}: {v}")

    # Load PHQ-9
    phq9 = load_phq9()
    labels = get_depression_labels(phq9)
    print(f"\nPHQ-9 scores loaded for {len(phq9)} students")
    for uid, scores in sorted(phq9.items()):
        pre = scores.get('pre', '?')
        post = scores.get('post', '?')
        dep_pre = labels.get(uid, {}).get('pre', '?')
        dep_post = labels.get(uid, {}).get('post', '?')
        print(f"  {uid}: pre={pre} (dep={dep_pre}), post={post} (dep={dep_post})")
