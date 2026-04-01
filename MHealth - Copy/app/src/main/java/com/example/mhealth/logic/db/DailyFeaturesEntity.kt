package com.example.mhealth.logic.db

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Stores one row of 22 behavioural features per user per day.
 * Maps directly to the 22 features in PersonalityVector.toMap().
 * Used as the daily input to the Python analysis engine.
 */
@Entity(tableName = "daily_features")
data class DailyFeaturesEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,

    // Identity
    val userId: String,
    val date: String,           // "YYYY-MM-DD"

    // Screen / App Usage
    val screenTimeHours: Float = 0f,
    val unlockCount: Float = 0f,
    val appLaunchCount: Float = 0f,
    val notificationsToday: Float = 0f,
    val socialAppRatio: Float = 0f,

    // Communication
    val callsPerDay: Float = 0f,
    val callDurationMinutes: Float = 0f,
    val uniqueContacts: Float = 0f,
    val conversationFrequency: Float = 0f,

    // Location & Movement
    val dailyDisplacementKm: Float = 0f,
    val locationEntropy: Float = 0f,
    val homeTimeRatio: Float = 0f,
    val placesVisited: Float = 0f,

    // Sleep Proxy
    val wakeTimeHour: Float = 0f,
    val sleepTimeHour: Float = 0f,
    val sleepDurationHours: Float = 0f,
    val darkDurationHours: Float = 0f,

    // System
    val chargeDurationHours: Float = 0f,
    val memoryUsagePercent: Float = 0f,
    val networkWifiMB: Float = 0f,
    val networkMobileMB: Float = 0f,

    // New expanded features (v2)
    val downloadsToday: Float = 0f,
    val storageUsedGB: Float = 0f,
    val appUninstallsToday: Float = 0f,
    val upiTransactionsToday: Float = 0f,
    val totalAppsCount: Float = 0f,
    val backgroundAudioHours: Float = 0f,
    val mediaCountToday: Float = 0f,
    val appInstallsToday: Float = 0f,

    // Missed Sensory Data (Captured for Cloud Sync / Backup)
    val dailySteps: Float = 0f,
    val appBreakdownJson: String = "{}",
    val notificationBreakdownJson: String = "{}",
    val appLaunchesBreakdownJson: String = "{}",

    // Sync flag (for future Firebase cloud sync)
    val syncedToCloud: Boolean = false,

    // Developer testing
    val isSimulated: Boolean = false
)
