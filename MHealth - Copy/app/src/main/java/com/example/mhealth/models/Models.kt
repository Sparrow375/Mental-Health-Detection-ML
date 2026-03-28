package com.example.mhealth.models

import java.util.Date

/**
 * PersonalityVector — every metric maps directly to what Android's Digital Wellbeing API
 * exposes, plus additional sensors available through the platform SDK.
 *
 * Digital Wellbeing source APIs:
 *   - UsageEvents (FOREGROUND/BACKGROUND pairs)  → screenTimeHours, appBreakdown, unlockCount, appLaunchCount
 *   - UsageEvents (KEYGUARD_HIDDEN/SHOWN)        → unlockCount, wakeTimeHour, sleepTimeHour
 *   - UsageEvents (NOTIFICATION_SEEN/INTERACTED) → notificationsToday
 *   - NetworkStatsManager                         → networkWifiMB, networkMobileMB
 *   - ActivityManager.MemoryInfo                  → memoryUsagePercent
 *   - StatFs(dataDir)                             → storageUsagePercent
 *   - BatteryManager broadcast                    → batteryLevel, chargeDurationHours
 *   - TYPE_STEP_COUNTER sensor                    → stepCount (daily delta)
 *   - ContentResolver(CallLog)                    → callsPerDay
 *   - ContentResolver(Telephony.Sms)              → textsPerDay
 *   - ContentResolver(Contacts)                   → uniqueContacts
 *   - ContentResolver(CalendarContract)           → calendarEventsToday
 *   - FusedLocationProviderClient (multi-point)   → dailyDisplacementKm, locationEntropy, homeTimeRatio
 *   - UsageEvents gap analysis                    → sleepDurationHours, darkDurationHours
 */
data class PersonalityVector(
    // ── SCREEN / APP USAGE (Digital Wellbeing primary) ──────────────────────
    val screenTimeHours: Float = 0f,       // total foreground time today (hrs)
    val unlockCount: Float = 0f,           // KEYGUARD_HIDDEN events today
    val appLaunchCount: Float = 0f,        // ACTIVITY_RESUMED events today
    val notificationsToday: Float = 0f,    // NOTIFICATION_SEEN events today
    val socialAppRatio: Float = 0f,        // social app time / total time

    // ── COMMUNICATION ────────────────────────────────────────────────────────
    val callsPerDay: Float = 0f,
    val callDurationMinutes: Float = 0f,
    val uniqueContacts: Float = 0f,
    val conversationFrequency: Float = 0f, // total calls

    // ── LOCATION & MOVEMENT ──────────────────────────────────────────────────
    val dailyDisplacementKm: Float = 0f,
    val locationEntropy: Float = 0f,
    val homeTimeRatio: Float = 0f,
    val placesVisited: Float = 0f,

    // ── SLEEP PROXY (from phone dark/usage gaps) ─────────────────────────────
    val wakeTimeHour: Float = 0f,          // hour of first phone use today
    val sleepTimeHour: Float = 0f,         // hour of last phone use yesterday
    val sleepDurationHours: Float = 0f,    // estimated sleep (usage gap)
    val darkDurationHours: Float = 0f,     // total time screen was off/idle

    // ── SYSTEM ───────────────────────────────────────────────────────────────
    val chargeDurationHours: Float = 0f,
    val memoryUsagePercent: Float = 0f,
    val networkWifiMB: Float = 0f,
    val networkMobileMB: Float = 0f,
    val mediaCountToday: Float = 0f,
    val appInstallsToday: Float = 0f,
    val calendarEventsToday: Float = 0f,

    // ── NEW EXPANDED FEATURES ─────────────────────────────────────────────────
    val downloadsToday: Float = 0f,        // files downloaded today
    val storageUsedGB: Float = 0f,         // internal storage currently used (GB)
    val appUninstallsToday: Float = 0f,    // apps removed today
    val upiTransactionsToday: Float = 0f,  // UPI/payment app launches today
    val nightInterruptions: Float = 0f,    // phone unlocks between 00:00–05:00

    // ── OPTIONAL ─────────────────────────────────────────────────────────────
    val moodScore: Int? = null,

    // ── BASELINE INTERNALS ───────────────────────────────────────────────────
    val variances: Map<String, Float> = emptyMap(),
    val appBreakdown: Map<String, Long> = emptyMap(), // package → foreground minutes
    val notificationBreakdown: Map<String, Int> = emptyMap(), // package → notification count
    val appLaunchesBreakdown: Map<String, Int> = emptyMap() // package → launch count
) {
    fun toMap(): Map<String, Float> = mapOf(
        "screenTimeHours"      to screenTimeHours,
        "unlockCount"          to unlockCount,
        "appLaunchCount"       to appLaunchCount,
        "notificationsToday"   to notificationsToday,
        "socialAppRatio"       to socialAppRatio,
        "callsPerDay"          to callsPerDay,
        "callDurationMinutes"  to callDurationMinutes,
        "uniqueContacts"       to uniqueContacts,
        "conversationFrequency" to conversationFrequency,
        "dailyDisplacementKm"  to dailyDisplacementKm,
        "locationEntropy"      to locationEntropy,
        "homeTimeRatio"        to homeTimeRatio,
        "placesVisited"        to placesVisited,
        "wakeTimeHour"         to wakeTimeHour,
        "sleepTimeHour"        to sleepTimeHour,
        "sleepDurationHours"   to sleepDurationHours,
        "darkDurationHours"    to darkDurationHours,
        "chargeDurationHours"  to chargeDurationHours,
        "memoryUsagePercent"   to memoryUsagePercent,
        "networkWifiMB"        to networkWifiMB,
        "networkMobileMB"      to networkMobileMB,
        "calendarEventsToday"  to calendarEventsToday,
        "downloadsToday"       to downloadsToday,
        "storageUsedGB"        to storageUsedGB,
        "appUninstallsToday"   to appUninstallsToday,
        "upiTransactionsToday" to upiTransactionsToday,
        "nightInterruptions"   to nightInterruptions
    )
}

data class DailyReport(
    val dayNumber: Int,
    val date: Date,
    val anomalyScore: Float,
    val alertLevel: String,
    val flaggedFeatures: List<String>,
    val patternType: String,
    val sustainedDeviationDays: Int,
    val evidenceAccumulated: Float,
    val topDeviations: Map<String, Float>,
    val notes: String
)

/** GPS fix captured every 15 min for displacement/entropy calculation */
data class LatLonPoint(val lat: Double, val lon: Double, val timeMs: Long)

/** User profile metadata captured during onboarding */
data class UserProfile(
    val email: String = "",
    val name: String = "",
    val gender: String = "",
    val dateOfBirth: String = "",
    val age: Int = 0,
    val profession: String = "",
    val country: String = ""
)
