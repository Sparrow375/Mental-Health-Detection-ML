package com.example.mhealth.logic.db

import androidx.room.Entity

/**
 * Stores computed daily DNA metrics — one row per day per user.
 * Populated at midnight before purging raw app_sessions / notification_events.
 *
 * Used for:
 *   - DNA baseline progress counting (independent of L1 daily_features)
 *   - Historical daily DNA comparison after baseline is established
 *   - Building DNA baseline from daily aggregates instead of raw sessions
 */
@Entity(
    tableName = "daily_dna_snapshot",
    primaryKeys = ["userId", "date"]
)
data class DailyDnaSnapshotEntity(
    val userId: String,
    val date: String,                       // YYYY-MM-DD

    // ── Phone DNA metrics ──────────────────────────────────────────────────
    val totalSessions: Int,
    val totalScreenTimeHours: Float,
    val firstPickupHour: Float?,            // null if no sessions
    val lastActivityHour: Float?,
    val activeWindowHours: Float?,
    val avgSessionMinutes: Float,
    val microSessionPct: Float,             // < 2 min
    val shortSessionPct: Float,             // 2–15 min
    val mediumSessionPct: Float,            // 15–30 min
    val deepSessionPct: Float,              // 30–60 min
    val marathonSessionPct: Float,          // > 60 min
    val selfOpenPct: Float,
    val notificationOpenPct: Float,
    val totalNotifications: Int,
    val notificationTapRate: Float,
    val notificationDismissRate: Float,
    val notificationIgnoreRate: Float,
    val uniqueAppsUsed: Int,
    val topAppPackage: String?,
    val nightChecks: Int,                       // unlocks during sleep window

    // ── Per-app DNA as JSON array ──────────────────────────────────────────
    // Array of objects: { appPackage, appLabel, totalScreenTimeMinutes,
    //   sessionCount, avgSessionMinutes, selfOpenRatio, notificationOpenRatio,
    //   primaryTimeRange, notificationCount, notificationTapCount, avgTapLatencyMinutes }
    val appDnaJson: String,

    val createdAt: Long
)
