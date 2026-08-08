package com.example.mhealth.models

import java.util.Date

/**
 * PersonalityVector — 30 behavioural features captured from Android platform APIs.
 *
 * Source APIs:
 *   - UsageEvents (FOREGROUND/BACKGROUND pairs)  → screenTimeHours, appBreakdown, unlockCount, appLaunchCount
 *   - UsageEvents (KEYGUARD_HIDDEN/SHOWN)        → unlockCount, wakeTimeHour, sleepTimeHour
 *   - UsageEvents (NOTIFICATION_SEEN/INTERACTED) → notificationsToday
 *   - BatteryManager broadcast                    → chargeDurationHours, chargeRegularity
 *   - TYPE_STEP_COUNTER sensor                    → dailyStepCount
 *   - TYPE_STEP_DETECTOR sensor + time bucketing  → activeMinutes
 *   - AccessibilityService (key events)           → keystrokeSpeed, backspaceRatio
 *   - AccessibilityService (scroll events)        → scrollVelocity
 *   - TYPE_LIGHT sensor                           → daylightExposureMinutes
 *   - ContentResolver(CallLog)                    → callsPerDay, callDurationMinutes, conversationFrequency
 *   - ContentResolver(Contacts)                   → uniqueContacts
 *   - ContentResolver(CalendarContract)           → calendarEventsToday
 *   - FusedLocationProviderClient (multi-point)   → dailyDisplacementKm, locationEntropy, homeTimeRatio
 *   - UsageEvents gap analysis                    → sleepDurationHours
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

    // ── SLEEP PROXY (from phone dark/usage gaps) ─────────────────────────────
    val wakeTimeHour: Float = 0f,          // hour of first phone use today
    val sleepTimeHour: Float = 0f,         // hour of last phone use yesterday
    val sleepDurationHours: Float = 0f,    // estimated sleep (usage gap)

    // ── PHYSICAL ACTIVITY ────────────────────────────────────────────────────
    val dailyStepCount: Float = 0f,
    val activeMinutes: Float = 0f,

    // ── INTERACTION DYNAMICS ─────────────────────────────────────────────────
    val keystrokeSpeed: Float = 0f,
    val backspaceRatio: Float = 0f,
    val scrollVelocity: Float = 0f,

    // ── CIRCADIAN & ENVIRONMENT ──────────────────────────────────────────────
    val daylightExposureMinutes: Float = 0f,
    val chargeRegularity: Float = 0f,

    // ── SYSTEM ───────────────────────────────────────────────────────────────
    val chargeDurationHours: Float = 0f,

    // ── BEHAVIOURAL SIGNALS ──────────────────────────────────────────────────
    val upiTransactionsToday: Float = 0f,
    val appUninstallsToday: Float = 0f,
    val appInstallsToday: Float = 0f,

    // ── ENGAGEMENT ───────────────────────────────────────────────────────────
    val mediaCountToday: Float = 0f,
    val downloadsToday: Float = 0f,
    val musicTimeMinutes: Float = 0f,

    // ── OPTIONAL ─────────────────────────────────────────────────────────────
    val moodScore: Int? = null,

    // ── BASELINE INTERNALS ───────────────────────────────────────────────────
    val variances: Map<String, Float> = emptyMap(),
    val appBreakdown: Map<String, Long> = emptyMap(), // package → foreground minutes
    val notificationBreakdown: Map<String, Int> = emptyMap(), // package → notification count
    val appLaunchesBreakdown: Map<String, Int> = emptyMap(), // package → launch count
    val bgAudioBreakdown: Map<String, Long> = emptyMap() // package → audio ms
) {
    fun toMap(): Map<String, Float> = mapOf(
        "screenTimeHours" to screenTimeHours,
        "unlockCount" to unlockCount,
        "appLaunchCount" to appLaunchCount,
        "notificationsToday" to notificationsToday,
        "socialAppRatio" to socialAppRatio,
        "callsPerDay" to callsPerDay,
        "callDurationMinutes" to callDurationMinutes,
        "uniqueContacts" to uniqueContacts,
        "conversationFrequency" to conversationFrequency,
        "dailyDisplacementKm" to dailyDisplacementKm,
        "locationEntropy" to locationEntropy,
        "homeTimeRatio" to homeTimeRatio,
        "wakeTimeHour" to wakeTimeHour,
        "sleepTimeHour" to sleepTimeHour,
        "sleepDurationHours" to sleepDurationHours,
        "dailyStepCount" to dailyStepCount,
        "activeMinutes" to activeMinutes,
        "keystrokeSpeed" to keystrokeSpeed,
        "backspaceRatio" to backspaceRatio,
        "scrollVelocity" to scrollVelocity,
        "daylightExposureMinutes" to daylightExposureMinutes,
        "chargeRegularity" to chargeRegularity,
        "chargeDurationHours" to chargeDurationHours,
        "upiTransactionsToday" to upiTransactionsToday,
        "appUninstallsToday" to appUninstallsToday,
        "appInstallsToday" to appInstallsToday,
        "mediaCountToday" to mediaCountToday,
        "downloadsToday" to downloadsToday,
        "musicTimeMinutes" to musicTimeMinutes
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
) {
    val anomalyDetected: Boolean get() = anomalyScore > 0.38f || sustainedDeviationDays >= 3 || alertLevel != "green"
}

/** GPS fix captured every 15 min for displacement/entropy calculation.
 *  [speed] is the Doppler-measured speed in m/s from the GPS chip (0f if unavailable).
 *  Used to classify walking vs vehicle segments in displacement calculation. */
data class LatLonPoint(
    val lat: Double,
    val lon: Double,
    val timeMs: Long,
    val accuracy: Float = 0f,
    val speed: Float = 0f        // m/s — from Location.speed; 0f if chip didn't report it
)

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
