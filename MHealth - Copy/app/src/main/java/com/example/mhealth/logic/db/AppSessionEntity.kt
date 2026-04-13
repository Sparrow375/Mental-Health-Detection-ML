package com.example.mhealth.logic.db

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Stores per-session app usage data for Level 2 Behavioral DNA.
 * Each row = one app session (foreground period).
 *
 * Trigger detection:
 *   NOTIFICATION = session opened within 10s of notification from same package
 *   SELF = all other cases
 */
@Entity(
    tableName = "app_sessions",
    indices = [
        Index(value = ["app_package", "open_timestamp"]),
        Index(value = ["date"])
    ]
)
data class AppSessionEntity(
    @PrimaryKey val session_id: String,  // UUID
    val app_package: String,
    val open_timestamp: Long,            // epoch_ms
    val close_timestamp: Long,           // epoch_ms
    val trigger: String,                 // "SELF" | "NOTIFICATION" | "SHORTCUT" | "WIDGET" | "EXTERNAL"
    val interaction_count: Int,          // user interactions during session
    val date: String                     // "YYYY-MM-DD" derived for querying
)