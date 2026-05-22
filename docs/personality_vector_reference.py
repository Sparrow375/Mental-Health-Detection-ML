"""
Personality Vector — MHealth App
=================================
Canonical reference of every feature currently extracted by the Android
DataCollector and processed by System 1 (Python).

All key names match Android's  PersonalityVector.toMap()  exactly,
and  system1.py's  PersonalityVector.to_dict()  exactly.

Sources per feature are the same OS APIs that Android's Digital Wellbeing
dashboard uses internally — no accessibility services, no root required.

Groups
------
  A  Screen & App Activity      (5 features)
  B  Communication              (4 features)
  C  Location & Movement        (4 features)
  D  Sleep & Circadian          (4 features)
  E  System Usage               (5 features)
  F  Behavioural Signals        (4 features)   ← new, from expanded Android collector
  G  Calendar & Engagement      (3 features)   ← new, from expanded Android collector

Total: 31 real-valued features per day.
"""

personality_vector = {

    # ════════════════════════════════════════════════════════════════════════
    # GROUP A — Screen & App Activity
    # Source: UsageStatsManager.queryEvents()  (same as Digital Wellbeing)
    # ════════════════════════════════════════════════════════════════════════
    "screenTimeHours": {
        "type":   float,
        "unit":   "hours",
        "source": "UsageEvents MOVE_TO_FOREGROUND / MOVE_TO_BACKGROUND pairs",
        "notes":  "Total foreground app time today. Computed by replaying raw "
                  "events — identical method to Digital Wellbeing.",
        "s1_weight": 1.4,
    },
    "unlockCount": {
        "type":   float,
        "unit":   "count",
        "source": "UsageEvents type=18 (KEYGUARD_HIDDEN)",
        "notes":  "Number of times the phone was unlocked today.",
        "s1_weight": 1.2,
    },
    "appLaunchCount": {
        "type":   float,
        "unit":   "count",
        "source": "UsageEvents ACTIVITY_RESUMED, debounced >1.5 s since last background",
        "notes":  "Distinct app launch events. Debounce removes accidental double-counts.",
        "s1_weight": 0.9,
    },
    "notificationsToday": {
        "type":   float,
        "unit":   "count",
        "source": "UsageEvents type=12 (NOTIFICATION_INTERRUPTION)",
        "notes":  "Total notification interruptions across all non-system apps.",
        "s1_weight": 0.8,
    },
    "socialAppRatio": {
        "type":   float,
        "unit":   "ratio [0–1]",
        "source": "ApplicationInfo.CATEGORY_SOCIAL + package-name heuristics",
        "notes":  "Social/messaging foreground time ÷ total screen time. "
                  "Covers Instagram, WhatsApp, Twitter, Telegram, Discord, etc.",
        "s1_weight": 1.3,
    },

    # ════════════════════════════════════════════════════════════════════════
    # GROUP B — Communication
    # Source: ContentResolver → CallLog.Calls
    # ════════════════════════════════════════════════════════════════════════
    "callsPerDay": {
        "type":   float,
        "unit":   "count",
        "source": "CallLog.Calls — all rows since midnight",
        "notes":  "Total calls today (incoming + outgoing + missed).",
        "s1_weight": 1.3,
    },
    "callDurationMinutes": {
        "type":   float,
        "unit":   "minutes",
        "source": "CallLog.Calls.DURATION (seconds) summed and converted",
        "notes":  "Total voice call time today in minutes.",
        "s1_weight": 1.2,
    },
    "uniqueContacts": {
        "type":   float,
        "unit":   "count",
        "source": "CallLog.Calls.NUMBER — distinct, whitespace-stripped",
        "notes":  "Number of unique phone numbers called or received today.",
        "s1_weight": 1.1,
    },
    "conversationFrequency": {
        "type":   float,
        "unit":   "ratio",
        "source": "callsPerDay / uniqueContacts",
        "notes":  "Average number of calls per unique contact today. "
                  "Measures contact depth vs breadth.",
        "s1_weight": 0.9,
    },

    # ════════════════════════════════════════════════════════════════════════
    # GROUP C — Location & Movement
    # Source: FusedLocationProviderClient (GPS snapshots every 15 min)
    # ════════════════════════════════════════════════════════════════════════
    "dailyDisplacementKm": {
        "type":   float,
        "unit":   "km",
        "source": "Haversine polyline over GPS track (15-min snapshots)",
        "notes":  "Total physical distance traveled today.",
        "s1_weight": 1.5,
    },
    "locationEntropy": {
        "type":   float,
        "unit":   "nats (Shannon entropy)",
        "source": "Shannon entropy over 0.001° grid cells (~110 m resolution)",
        "notes":  "Spatial diversity of locations visited. Higher = more varied movement.",
        "s1_weight": 1.3,
    },
    "homeTimeRatio": {
        "type":   float,
        "unit":   "ratio [0–1]",
        "source": "Most-visited GPS grid cell count ÷ total GPS fix count",
        "notes":  "Fraction of day spent at the most-frequent location (proxy for home).",
        "s1_weight": 1.2,
    },
    "placesVisited": {
        "type":   float,
        "unit":   "count",
        "source": "Distinct 0.001° GPS grid cells visited today",
        "notes":  "Number of meaningfully different locations visited.",
        "s1_weight": 1.1,
    },

    # ════════════════════════════════════════════════════════════════════════
    # GROUP D — Sleep & Circadian
    # Source: UsageEvents screen on/off gap analysis (window: prev 20:00 → today 12:00)
    # ════════════════════════════════════════════════════════════════════════
    "wakeTimeHour": {
        "type":   float,
        "unit":   "hour of day [0–24]",
        "source": "End timestamp of longest screen-off gap in sleep window",
        "notes":  "Proxy for wake time — hour when phone was first meaningfully used. "
                  "Micro-wakes < 10 min are merged into the sleep episode.",
        "s1_weight": 1.4,
    },
    "sleepTimeHour": {
        "type":   float,
        "unit":   "hour of day [0–24]",
        "source": "Start timestamp of longest screen-off gap in sleep window",
        "notes":  "Proxy for sleep onset — hour phone went silent before the main sleep gap.",
        "s1_weight": 1.3,
    },
    "sleepDurationHours": {
        "type":   float,
        "unit":   "hours",
        "source": "Duration of longest contiguous screen-off gap (20:00 prev → 12:00 today)",
        "notes":  "Estimated sleep duration. Highest clinical weight in the model. "
                  "Contiguous gaps separated by < 10-min screen-on events are merged.",
        "s1_weight": 1.6,   # highest clinical priority
    },
    "darkDurationHours": {
        "type":   float,
        "unit":   "hours",
        "source": "Accumulated SCREEN_NON_INTERACTIVE (type=16) event time today",
        "notes":  "Total time the screen was completely off during the day. "
                  "Broader than sleep — includes pocket time and idle periods.",
        "s1_weight": 1.0,
    },

    # ════════════════════════════════════════════════════════════════════════
    # GROUP E — System Usage
    # Source: BatteryManager, ActivityManager, NetworkStatsManager, StatFs
    # ════════════════════════════════════════════════════════════════════════
    "chargeDurationHours": {
        "type":   float,
        "unit":   "hours",
        "source": "BatteryManager.EXTRA_STATUS accumulated across 15-min ticks",
        "notes":  "Total hours the phone spent charging today. "
                  "Irregular charging patterns can reflect disrupted routines.",
        "s1_weight": 0.8,
    },
    "memoryUsagePercent": {
        "type":   float,
        "unit":   "percent [0–100]",
        "source": "ActivityManager.MemoryInfo — (totalMem - availMem) / totalMem × 100",
        "notes":  "RAM usage % — identical to Settings → Memory. "
                  "Low clinical signal; noisy sensor (weight 0.5).",
        "s1_weight": 0.5,
    },
    "networkWifiMB": {
        "type":   float,
        "unit":   "MB",
        "source": "NetworkStatsManager.querySummaryForDevice(TRANSPORT_WIFI)",
        "notes":  "Wi-Fi data (rx + tx) used today. Same source as Settings → Data Usage.",
        "s1_weight": 0.6,
    },
    "networkMobileMB": {
        "type":   float,
        "unit":   "MB",
        "source": "NetworkStatsManager.querySummaryForDevice(TRANSPORT_CELLULAR)",
        "notes":  "Cellular data (rx + tx) used today.",
        "s1_weight": 0.6,
    },
    "storageUsedGB": {
        "type":   float,
        "unit":   "GB",
        "source": "StatFs(dataDir) — (totalBlocks - availBlocks) × blockSize",
        "notes":  "Internal storage currently used. Quasi-static; low clinical signal.",
        "s1_weight": 0.4,
    },

    # ════════════════════════════════════════════════════════════════════════
    # GROUP F — Behavioural Signals  (new — added in latest Android update)
    # ════════════════════════════════════════════════════════════════════════
    "totalAppsCount": {
        "type":   float,
        "unit":   "count",
        "source": "PackageManager.getInstalledPackages()",
        "notes":  "Total number of installed apps on the device.",
        "s1_weight": 0.8,
    },
    "upiTransactionsToday": {
        "type":   float,
        "unit":   "count",
        "source": "App launch breakdown filtered by 13 known UPI package IDs "
                  "(GPay, PhonePe, Paytm, ICICI, HDFC, SBI, etc.)",
        "notes":  "UPI/payment app launches today. Financial engagement proxy — "
                  "impulsivity marker in BPD/mania patterns.",
        "s1_weight": 1.1,
    },
    "appUninstallsToday": {
        "type":   float,
        "unit":   "count",
        "source": "Incremental diff of PackageManager.getInstalledPackages() count "
                  "stored in SharedPreferences between ticks",
        "notes":  "Apps removed today. Captures impulsive digital housekeeping behavior.",
        "s1_weight": 0.9,
    },
    "appInstallsToday": {
        "type":   float,
        "unit":   "count",
        "source": "PackageInfo.firstInstallTime compared against start of day",
        "notes":  "New apps installed today.",
        "s1_weight": 0.8,
    },

    # ════════════════════════════════════════════════════════════════════════
    # GROUP G — Calendar & Engagement  (new — added in latest Android update)
    # ════════════════════════════════════════════════════════════════════════
    "calendarEventsToday": {
        "type":   float,
        "unit":   "count",
        "source": "ContentResolver → CalendarContract.Events "
                  "(DTSTART between midnight and end of day)",
        "notes":  "Calendar entries scheduled for today. "
                  "Drop-off in planned social/work events can indicate withdrawal.",
        "s1_weight": 0.9,
    },
    "mediaCountToday": {
        "type":   float,
        "unit":   "count",
        "source": "MediaStore.Files — DATE_ADDED since midnight (external storage)",
        "notes":  "Photos, videos, and audio files added today. "
                  "Captures creative/social activity level.",
        "s1_weight": 0.7,
    },
    "downloadsToday": {
        "type":   float,
        "unit":   "count",
        "source": "Filesystem scan of Environment.DIRECTORY_DOWNLOADS — "
                  "files with lastModified() ≥ start of day",
        "notes":  "Files downloaded today. Bypasses MediaStore index lag "
                  "by checking the filesystem layer directly.",
        "s1_weight": 0.6,
    },
    "backgroundAudioHours": {
        "type":   float,
        "unit":   "hours",
        "source": "AudioManager.isMusicActive() debounced across 15-min ticks",
        "notes":  "Total time intentional audio (Spotify/YouTube/etc) was playing "
                  "in the background. High indicator of 'digital cocooning'.",
        "s1_weight": 1.1,
    },
    "dailySteps": {
        "type":   float,
        "unit":   "count",
        "source": "Sensor.TYPE_STEP_COUNTER (delta since midnight)",
        "notes":  "Total physical steps taken today. Core physical activity metric.",
        "s1_weight": 1.4,
    },
}


# ── Rich breakdowns (stored per snapshot but NOT part of the anomaly model) ──
# These are available in DataRepository and the UI but are not fed into System 1.
supplementary_breakdowns = {
    "appBreakdown": {
        "type":   "Map<str, int>",
        "format": "package_name → foreground_minutes",
        "source": "Usage event FOREGROUND/BACKGROUND pairs per package",
        "notes":  "Per-app screen time (minutes). Shown in Insights screen.",
    },
    "notificationBreakdown": {
        "type":   "Map<str, int>",
        "format": "package_name → notification_count",
        "source": "UsageEvents type=12 per package",
        "notes":  "Per-app notification count. Used in Insights screen.",
    },
    "appLaunchesBreakdown": {
        "type":   "Map<str, int>",
        "format": "package_name → launch_count",
        "source": "UsageEvents ACTIVITY_RESUMED per package (debounced)",
        "notes":  "Per-app launch count. Used in Insights screen.",
    },
}


# ── System 1 feature weights summary (for quick reference) ───────────────────
# Sorted descending by clinical priority weight.
FEATURE_WEIGHTS = {feat: meta["s1_weight"] for feat, meta in personality_vector.items()}
FEATURE_WEIGHTS_SORTED = dict(
    sorted(FEATURE_WEIGHTS.items(), key=lambda x: x[1], reverse=True)
)

if __name__ == "__main__":
    print(f"Total features in model:  {len(personality_vector)}")
    print(f"Supplementary breakdowns: {len(supplementary_breakdowns)}\n")

    groups = {}
    for feat, meta in personality_vector.items():
        source_line = meta["source"].split("(")[0].strip()
        group_key = source_line[:40]
        groups.setdefault(source_line[:5], []).append(feat)

    print("Features by System 1 weight (descending):")
    print(f"  {'Feature':<26}  {'Weight':>6}  Unit")
    print("  " + "-" * 55)
    for feat, w in FEATURE_WEIGHTS_SORTED.items():
        unit = personality_vector[feat]["unit"]
        print(f"  {feat:<26}  {w:>6.1f}  {unit}")
