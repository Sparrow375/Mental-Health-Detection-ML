package com.example.mhealth.logic

import com.example.mhealth.logic.db.BaselineEntity
import com.example.mhealth.logic.db.DailyFeaturesEntity
import org.json.JSONArray
import org.json.JSONObject

/**
 * JsonConverter — bridges the Room SQLite layer with the Python engine.
 *
 * Builds the exact JSON structure expected by engine.py's run_analysis():
 *
 * {
 *   "current": { <22 feature key-value floats> },
 *   "baseline": {
 *     "<feature_name>": { "mean": <float>, "std": <float> },
 *     ...
 *   },
 *   "history": [
 *     { <22 feature key-value floats> },   // oldest first
 *     ...                                  // up to 14 days
 *   ],
 *   "baseline_contaminated": <bool>
 * }
 */
object JsonConverter {

    /**
     * Builds the Python engine input JSON from persisted Room data.
     *
     * @param current      Today's feature row
     * @param baseline     List of BaselineEntity rows (one per feature)
     * @param history      Last N daily feature rows (oldest first, max 14)
     * @return             JSON string ready to pass to engine.run_analysis()
     */
    fun toEngineJson(
        current: DailyFeaturesEntity,
        baseline: List<BaselineEntity>,
        history: List<DailyFeaturesEntity>
    ): String {
        val root = JSONObject()

        // ── "current" block ───────────────────────────────────────────────────
        root.put("current", featureEntityToJson(current))

        // ── "baseline" block ──────────────────────────────────────────────────
        val baselineJson = JSONObject()
        val contaminated = baseline.any { it.isContaminated }
        for (b in baseline) {
            val featureObj = JSONObject().apply {
                put("mean", b.baselineValue)
                put("std", b.stdDeviation)
            }
            baselineJson.put(b.featureName, featureObj)
        }
        root.put("baseline", baselineJson)

        // ── "history" block (older days as list of feature maps) ───────────────
        val historyArray = JSONArray()
        for (h in history) {
            historyArray.put(featureEntityToJson(h))
        }
        root.put("history", historyArray)

        // ── contamination flag ────────────────────────────────────────────────
        root.put("baseline_contaminated", contaminated)

        return root.toString()
    }

    /** Converts a DailyFeaturesEntity to a flat JSON object of feature → value */
    private fun featureEntityToJson(e: DailyFeaturesEntity): JSONObject = JSONObject().apply {
        put("screen_time_hours", e.screenTimeHours)
        put("unlock_count", e.unlockCount)
        put("app_launch_count", e.appLaunchCount)
        put("notifications_today", e.notificationsToday)
        put("social_app_ratio", e.socialAppRatio)
        put("calls_per_day", e.callsPerDay)
        put("call_duration_minutes", e.callDurationMinutes)
        put("unique_contacts", e.uniqueContacts)
        put("conversation_frequency", e.conversationFrequency)
        put("daily_displacement_km", e.dailyDisplacementKm)
        put("location_entropy", e.locationEntropy)
        put("home_time_ratio", e.homeTimeRatio)
        put("places_visited", e.placesVisited)
        put("wake_time_hour", e.wakeTimeHour)
        put("sleep_time_hour", e.sleepTimeHour)
        put("sleep_duration_hours", e.sleepDurationHours)
        put("dark_duration_hours", e.darkDurationHours)
        put("charge_duration_hours", e.chargeDurationHours)
        put("memory_usage_percent", e.memoryUsagePercent)
        put("network_wifi_mb", e.networkWifiMB)
        put("network_mobile_mb", e.networkMobileMB)
        put("conversation_duration_hours", e.callDurationMinutes / 60f)
        put("downloads_today", e.downloadsToday)
        put("storage_used_gb", e.storageUsedGB)
        put("app_uninstalls_today", e.appUninstallsToday)
        put("upi_transactions_today", e.upiTransactionsToday)
        put("night_interruptions", e.nightInterruptions)
    }

    /**
     * Helper to convert PersonalityVector directly to a DailyFeaturesEntity
     * so MonitoringService can persist each day without extra logic.
     */
    fun fromPersonalityVector(
        userId: String,
        date: String,
        v: com.example.mhealth.models.PersonalityVector,
        isSimulated: Boolean = false
    ): DailyFeaturesEntity = DailyFeaturesEntity(
        userId = userId,
        date = date,
        screenTimeHours = v.screenTimeHours,
        unlockCount = v.unlockCount,
        appLaunchCount = v.appLaunchCount,
        notificationsToday = v.notificationsToday,
        socialAppRatio = v.socialAppRatio,
        callsPerDay = v.callsPerDay,
        callDurationMinutes = v.callDurationMinutes,
        uniqueContacts = v.uniqueContacts,
        conversationFrequency = v.conversationFrequency,
        dailyDisplacementKm = v.dailyDisplacementKm,
        locationEntropy = v.locationEntropy,
        homeTimeRatio = v.homeTimeRatio,
        placesVisited = v.placesVisited,
        wakeTimeHour = v.wakeTimeHour,
        sleepTimeHour = v.sleepTimeHour,
        sleepDurationHours = v.sleepDurationHours,
        darkDurationHours = v.darkDurationHours,
        chargeDurationHours = v.chargeDurationHours,
        memoryUsagePercent = v.memoryUsagePercent,
        networkWifiMB = v.networkWifiMB,
        networkMobileMB = v.networkMobileMB,
        downloadsToday = v.downloadsToday,
        storageUsedGB = v.storageUsedGB,
        appUninstallsToday = v.appUninstallsToday,
        upiTransactionsToday = v.upiTransactionsToday,
        nightInterruptions = v.nightInterruptions,
        isSimulated = isSimulated
    )

    fun toPersonalityVector(
        e: DailyFeaturesEntity
    ): com.example.mhealth.models.PersonalityVector = com.example.mhealth.models.PersonalityVector(
        screenTimeHours = e.screenTimeHours,
        unlockCount = e.unlockCount,
        appLaunchCount = e.appLaunchCount,
        notificationsToday = e.notificationsToday,
        socialAppRatio = e.socialAppRatio,
        callsPerDay = e.callsPerDay,
        callDurationMinutes = e.callDurationMinutes,
        uniqueContacts = e.uniqueContacts,
        conversationFrequency = e.conversationFrequency,
        dailyDisplacementKm = e.dailyDisplacementKm,
        locationEntropy = e.locationEntropy,
        homeTimeRatio = e.homeTimeRatio,
        placesVisited = e.placesVisited,
        wakeTimeHour = e.wakeTimeHour,
        sleepTimeHour = e.sleepTimeHour,
        sleepDurationHours = e.sleepDurationHours,
        darkDurationHours = e.darkDurationHours,
        chargeDurationHours = e.chargeDurationHours,
        memoryUsagePercent = e.memoryUsagePercent,
        networkWifiMB = e.networkWifiMB,
        networkMobileMB = e.networkMobileMB,
        downloadsToday = e.downloadsToday,
        storageUsedGB = e.storageUsedGB,
        appUninstallsToday = e.appUninstallsToday,
        upiTransactionsToday = e.upiTransactionsToday,
        nightInterruptions = e.nightInterruptions
    )
}
