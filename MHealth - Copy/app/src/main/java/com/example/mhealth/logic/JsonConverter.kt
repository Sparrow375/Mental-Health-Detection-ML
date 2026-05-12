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
        put("screenTimeHours", e.screenTimeHours)
        put("unlockCount", e.unlockCount)
        put("appLaunchCount", e.appLaunchCount)
        put("notificationsToday", e.notificationsToday)
        put("socialAppRatio", e.socialAppRatio)
        put("callsPerDay", e.callsPerDay)
        put("callDurationMinutes", e.callDurationMinutes)
        put("uniqueContacts", e.uniqueContacts)
        put("conversationFrequency", e.conversationFrequency)
        put("dailyDisplacementKm", e.dailyDisplacementKm)
        put("locationEntropy", e.locationEntropy)
        put("homeTimeRatio", e.homeTimeRatio)
        put("wakeTimeHour", e.wakeTimeHour)
        put("sleepTimeHour", e.sleepTimeHour)
        put("sleepDurationHours", e.sleepDurationHours)
        put("darkDurationHours", e.darkDurationHours)
        put("chargeDurationHours", e.chargeDurationHours)
        put("memoryUsagePercent", e.memoryUsagePercent)
        put("networkWifiMB", e.networkWifiMB)
        put("networkMobileMB", e.networkMobileMB)
        put("downloadsToday", e.downloadsToday)
        put("storageUsedGB", e.storageUsedGB)
        put("appUninstallsToday", e.appUninstallsToday)
        put("upiTransactionsToday", e.upiTransactionsToday)
        put("totalAppsCount", e.totalAppsCount)
        put("mediaCountToday", e.mediaCountToday)
        put("appInstallsToday", e.appInstallsToday)
        put("musicTimeMinutes", e.musicTimeMinutes)
        put("calendarEventsToday", e.calendarEventsToday)
        put("dailySteps", e.dailySteps)
        
        // Pass individual app usage dictionaries to python engine
        try {
            put("appBreakdown", JSONObject(e.appBreakdownJson))
            put("notificationBreakdown", JSONObject(e.notificationBreakdownJson))
            put("appLaunchesBreakdown", JSONObject(e.appLaunchesBreakdownJson))
            put("bgAudioBreakdown", JSONObject(e.bgAudioBreakdownJson))
        } catch (ex: Exception) {
            put("appBreakdown", JSONObject())
            put("notificationBreakdown", JSONObject())
            put("appLaunchesBreakdown", JSONObject())
            put("bgAudioBreakdown", JSONObject())
        }
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
        totalAppsCount = v.totalAppsCount,
        musicTimeMinutes = v.musicTimeMinutes,
        mediaCountToday = v.mediaCountToday,
        appInstallsToday = v.appInstallsToday,
        calendarEventsToday = v.calendarEventsToday,
        isSimulated = isSimulated,
        dailySteps = v.dailySteps,
        appBreakdownJson = mapToJson(v.appBreakdown as Map<String, Number>),
        notificationBreakdownJson = mapToJson(v.notificationBreakdown as Map<String, Number>),
        appLaunchesBreakdownJson = mapToJson(v.appLaunchesBreakdown as Map<String, Number>),
        bgAudioBreakdownJson = mapToJson(v.bgAudioBreakdown as Map<String, Number>)
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
        totalAppsCount = e.totalAppsCount,
        musicTimeMinutes = e.musicTimeMinutes,
        mediaCountToday = e.mediaCountToday,
        appInstallsToday = e.appInstallsToday,
        calendarEventsToday = e.calendarEventsToday,
        dailySteps = e.dailySteps,
        appBreakdown = parseMapLong(e.appBreakdownJson),
        notificationBreakdown = parseMapInt(e.notificationBreakdownJson),
        appLaunchesBreakdown = parseMapInt(e.appLaunchesBreakdownJson),
        bgAudioBreakdown = parseMapLong(e.bgAudioBreakdownJson)
    )

    private fun mapToJson(map: Map<String, Number>): String {
        // Optimize storage by keeping only top 100 most significant entries (by value descending)
        // This prevents runaway bloat while preserving 99% of behavioral relevance.
        val optimizedMap = map.entries
            .sortedByDescending { it.value.toDouble() }
            .take(100)
            .associate { it.toPair() }

        return JSONObject(optimizedMap as Map<*, *>).toString()
    }

    /**
     * Convert a list of AppSessionEntity to a JSON string for the Python engine.
     * Includes derived fields (hour, duration_minutes) that phone_dna_builder expects.
     */
    fun sessionsToJson(sessions: List<com.example.mhealth.logic.db.AppSessionEntity>): String {
        val cal = java.util.Calendar.getInstance()
        val arr = JSONArray()
        for (s in sessions) {
            cal.timeInMillis = s.open_timestamp
            val hour = cal.get(java.util.Calendar.HOUR_OF_DAY) + cal.get(java.util.Calendar.MINUTE) / 60f
            val durationMin = (s.close_timestamp - s.open_timestamp).coerceAtLeast(0) / 60_000f

            arr.put(JSONObject().apply {
                put("app_package", s.app_package)
                put("open_timestamp", s.open_timestamp)
                put("open_timestamp_ms", s.open_timestamp)
                put("close_timestamp", s.close_timestamp)
                put("hour", hour.toDouble())
                put("duration_minutes", durationMin.toDouble())
                put("trigger", s.trigger)
                put("interaction_count", s.interaction_count)
                put("date", s.date)
            })
        }
        return arr.toString()
    }

    private fun parseMapLong(jsonStr: String): Map<String, Long> {
        val map = mutableMapOf<String, Long>()
        try {
            val obj = JSONObject(jsonStr)
            for (key in obj.keys()) {
                map[key] = obj.getLong(key)
            }
        } catch (e: Exception) {}
        return map
    }

    private fun parseMapInt(jsonStr: String): Map<String, Int> {
        val map = mutableMapOf<String, Int>()
        try {
            val obj = JSONObject(jsonStr)
            for (key in obj.keys()) {
                map[key] = obj.getInt(key)
            }
        } catch (e: Exception) {}
        return map
    }
}
