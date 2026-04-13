package com.example.mhealth.logic.db

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Stores per-notification event data for Level 2 Digital DNA.
 * Each row = one notification event (arrival + user action).
 *
 * Used by:
 *   - PhoneDNA: notification_open_rate, dismiss_rate, ignore_rate
 *   - AppDNA: notification_response_latency per app
 *   - L2 Texture: notification_to_session_ratio, notification_response_latency_shift
 */
@Entity(
    tableName = "notification_events",
    indices = [
        Index(value = ["app_package", "arrival_timestamp"]),
        Index(value = ["date"])
    ]
)
data class NotificationEventEntity(
    @PrimaryKey val event_id: String,       // UUID
    val app_package: String,                // package that posted the notification
    val arrival_timestamp: Long,            // epoch_ms when notification arrived
    val action: String,                     // "TAP" | "DISMISS" | "IGNORE"
    val tap_latency_min: Float?,            // minutes between arrival and tap (null if not tapped)
    val date: String                        // "YYYY-MM-DD" derived for querying
)