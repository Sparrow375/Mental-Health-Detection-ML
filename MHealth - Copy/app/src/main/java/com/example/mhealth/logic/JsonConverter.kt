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
import android.content.Context

object JsonConverter {

    /**
     * Builds the Python engine input JSON from persisted Room data.
     *
     * @param context      Android Context to retrieve user preferences
     * @param current      Today's feature row
     * @param baseline     List of BaselineEntity rows (one per feature)
     * @param history      Last N daily feature rows (oldest first, max 14)
     * @return             JSON string ready to pass to engine.run_analysis()
     */
    fun toEngineJson(
        context: Context,
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

        // ── onboarding screener scores ────────────────────────────────────────
        root.put("phq9_score", DataRepository.phq9Score.value)
        root.put("gad7_score", DataRepository.gad7Score.value)
        root.put("recent_life_events_count", DataRepository.recentLifeEventsCount.value)

        // ── Detailed User Profile details from SharedPreferences ───────────────
        val prefs = context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE)
        root.put("age", prefs.getInt("user_age", 25))
        root.put("gender", prefs.getString("user_gender", "prefer_not_to_say"))
        root.put("living_situation", prefs.getString("user_living_situation", "with_family"))
        root.put("employment", prefs.getString("user_profession", "student"))

        root.put("typical_wake", prefs.getFloat("user_typical_wake", 7.0f).toDouble())
        root.put("typical_sleep", prefs.getFloat("user_typical_sleep", 23.0f).toDouble())
        root.put("commute_minutes", prefs.getInt("user_commute_minutes", 30))
        root.put("routine_consistency", prefs.getString("user_routine_consistency", "flexible"))

        root.put("lifestyle_screen", prefs.getInt("user_lifestyle_screen", 3))
        root.put("lifestyle_communication", prefs.getInt("user_lifestyle_communication", 3))
        root.put("lifestyle_movement", prefs.getInt("user_lifestyle_movement", 3))
        root.put("lifestyle_sleep", prefs.getInt("user_lifestyle_sleep", 3))
        root.put("lifestyle_behavioral", prefs.getInt("user_lifestyle_behavioral", 3))
        root.put("lifestyle_engagement", prefs.getInt("user_lifestyle_engagement", 3))

        root.put("is_student", prefs.getBoolean("user_is_student", false))
        root.put("has_chronic_condition", prefs.getBoolean("user_has_chronic_condition", false))
        root.put("in_therapy", prefs.getBoolean("user_in_therapy", false))
        root.put("physical_health_rating", prefs.getInt("user_physical_health_rating", 7))

        return root.toString()
    }

    private fun featureEntityToJson(e: DailyFeaturesEntity): JSONObject = JSONObject().apply {
        put("date", e.date)
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
        put("dailyStepCount", e.dailyStepCount)
        put("activeMinutes", e.activeMinutes)
        put("keystrokeSpeed", e.keystrokeSpeed)
        put("backspaceRatio", e.backspaceRatio)
        put("scrollVelocity", e.scrollVelocity)
        put("daylightExposureMinutes", e.daylightExposureMinutes)
        put("chargeRegularity", e.chargeRegularity)
        put("chargeDurationHours", e.chargeDurationHours)
        put("upiTransactionsToday", e.upiTransactionsToday)
        put("appUninstallsToday", e.appUninstallsToday)
        put("appInstallsToday", e.appInstallsToday)
        put("calendarEventsToday", e.calendarEventsToday)
        put("mediaCountToday", e.mediaCountToday)
        put("downloadsToday", e.downloadsToday)
        put("musicTimeMinutes", e.musicTimeMinutes)
        
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
        dailyStepCount = v.dailyStepCount,
        activeMinutes = v.activeMinutes,
        keystrokeSpeed = v.keystrokeSpeed,
        backspaceRatio = v.backspaceRatio,
        scrollVelocity = v.scrollVelocity,
        daylightExposureMinutes = v.daylightExposureMinutes,
        chargeRegularity = v.chargeRegularity,
        chargeDurationHours = v.chargeDurationHours,
        upiTransactionsToday = v.upiTransactionsToday,
        appUninstallsToday = v.appUninstallsToday,
        appInstallsToday = v.appInstallsToday,
        calendarEventsToday = v.calendarEventsToday,
        mediaCountToday = v.mediaCountToday,
        downloadsToday = v.downloadsToday,
        musicTimeMinutes = v.musicTimeMinutes,
        isSimulated = isSimulated,
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
        dailyStepCount = e.dailyStepCount,
        activeMinutes = e.activeMinutes,
        keystrokeSpeed = e.keystrokeSpeed,
        backspaceRatio = e.backspaceRatio,
        scrollVelocity = e.scrollVelocity,
        daylightExposureMinutes = e.daylightExposureMinutes,
        chargeRegularity = e.chargeRegularity,
        chargeDurationHours = e.chargeDurationHours,
        upiTransactionsToday = e.upiTransactionsToday,
        appUninstallsToday = e.appUninstallsToday,
        appInstallsToday = e.appInstallsToday,
        calendarEventsToday = e.calendarEventsToday,
        mediaCountToday = e.mediaCountToday,
        downloadsToday = e.downloadsToday,
        musicTimeMinutes = e.musicTimeMinutes,
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
            val dayOfWeekJava = cal.get(java.util.Calendar.DAY_OF_WEEK)
            val pythonDayOfWeek = (dayOfWeekJava + 5) % 7

            arr.put(JSONObject().apply {
                put("app_package", s.app_package)
                put("open_timestamp", s.open_timestamp)
                put("open_timestamp_ms", s.open_timestamp)
                put("close_timestamp", s.close_timestamp)
                put("hour", hour.toDouble())
                put("duration_minutes", durationMin.toDouble())
                put("day_of_week", pythonDayOfWeek)
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

    suspend fun buildBackupJson(context: Context, userId: String): String {
        val db = com.example.mhealth.logic.db.MHealthDatabase.getInstance(context)
        val dailyHistory = db.dailyFeaturesDao().getAllFeatures(userId)
        val baselineRows = db.baselineDao().getBaseline(userId)
        val analysisReports = db.analysisResultDao().getAll(userId)
        val profile = db.userProfileDao().getProfile(userId)
        
        val masterJson = org.json.JSONObject()
        
        masterJson.put("profile", org.json.JSONObject().apply {
            put("userId", userId)
            put("baselineReady", profile?.baselineReady ?: false)
            put("onboardingDate", profile?.onboardingDate ?: "")
            put("currentStatus", profile?.currentStatus ?: "Collecting")
        })

        val baselineArr = org.json.JSONArray()
        baselineRows.forEach { row ->
            baselineArr.put(org.json.JSONObject().apply {
                put("feature", row.featureName)
                put("mean", row.baselineValue)
                put("std", row.stdDeviation)
                put("start", row.baselineStart)
                put("end", row.baselineEnd)
                put("contaminated", row.isContaminated)
            })
        }
        masterJson.put("baseline", baselineArr)

        val scoreByDate: Map<String, Float> = analysisReports.associate { it.date to it.effectiveScore }

        val historyArr = org.json.JSONArray()
        dailyHistory.forEach { day ->
            val dayObj = org.json.JSONObject()
            dayObj.put("date", day.date)
            dayObj.put("isSimulated", day.isSimulated)
            dayObj.put("anomaly_score", scoreByDate[day.date] ?: -1.0)

            val features = org.json.JSONObject().apply {
                put("screenTimeHours", day.screenTimeHours)
                put("unlockCount", day.unlockCount)
                put("appLaunchCount", day.appLaunchCount)
                put("notifications", day.notificationsToday)
                put("socialRatio", day.socialAppRatio)
                put("callsPerDay", day.callsPerDay)
                put("callDurationMins", day.callDurationMinutes)
                put("uniqueContacts", day.uniqueContacts)
                put("conversationFrequency", day.conversationFrequency)
                put("displacementKm", day.dailyDisplacementKm)
                put("locationEntropy", day.locationEntropy)
                put("homeTimeRatio", day.homeTimeRatio)
                put("wakeTimeHour", day.wakeTimeHour)
                put("sleepTimeHour", day.sleepTimeHour)
                put("sleepDurationHours", day.sleepDurationHours)
                put("dailyStepCount", day.dailyStepCount)
                put("activeMinutes", day.activeMinutes)
                put("keystrokeSpeed", day.keystrokeSpeed)
                put("backspaceRatio", day.backspaceRatio)
                put("scrollVelocity", day.scrollVelocity)
                put("daylightExposureMinutes", day.daylightExposureMinutes)
                put("chargeRegularity", day.chargeRegularity)
                put("chargeDurationHours", day.chargeDurationHours)
                put("upiTransactions", day.upiTransactionsToday)
                put("appUninstalls", day.appUninstallsToday)
                put("appInstalls", day.appInstallsToday)
                put("calendarEvents", day.calendarEventsToday)
                put("mediaCount", day.mediaCountToday)
                put("downloads", day.downloadsToday)
                put("musicTimeMinutes", day.musicTimeMinutes)
            }
            dayObj.put("metrics", features)

            dayObj.put("detailed_logs", org.json.JSONObject().apply {
                put("app_breakdown", org.json.JSONObject(day.appBreakdownJson))
                put("notifications_breakdown", org.json.JSONObject(day.notificationBreakdownJson))
                put("app_launches_breakdown", org.json.JSONObject(day.appLaunchesBreakdownJson))
            })
            
            historyArr.put(dayObj)
        }
        masterJson.put("daily_history", historyArr)

        val prefs = context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE)
        val dailyCheckinHistory = prefs.getString("daily_checkin_history", "[]") ?: "[]"
        masterJson.put("daily_checkin_history", org.json.JSONArray(dailyCheckinHistory))

        val liveVector = DataRepository.latestVector.value
        if (liveVector != null) {
            val todayStr = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.getDefault()).format(java.util.Date())
            val todayObj = org.json.JSONObject()
            todayObj.put("date", todayStr)
            todayObj.put("is_live_snapshot", true)
            todayObj.put("isSimulated", false)
            todayObj.put("metrics", org.json.JSONObject().apply {
                put("screenTimeHours",    liveVector.screenTimeHours)
                put("unlockCount",         liveVector.unlockCount)
                put("appLaunchCount",      liveVector.appLaunchCount)
                put("notifications",       liveVector.notificationsToday)
                put("socialRatio",         liveVector.socialAppRatio)
                put("callsPerDay",         liveVector.callsPerDay)
                put("callDurationMins",    liveVector.callDurationMinutes)
                put("uniqueContacts",      liveVector.uniqueContacts)
                put("conversationFrequency", liveVector.conversationFrequency)
                put("displacementKm",      liveVector.dailyDisplacementKm)
                put("locationEntropy",     liveVector.locationEntropy)
                put("homeTimeRatio",       liveVector.homeTimeRatio)
                put("wakeTimeHour",        liveVector.wakeTimeHour)
                put("sleepTimeHour",       liveVector.sleepTimeHour)
                put("sleepDurationHours",  liveVector.sleepDurationHours)
                put("dailyStepCount",      liveVector.dailyStepCount)
                put("activeMinutes",       liveVector.activeMinutes)
                put("keystrokeSpeed",      liveVector.keystrokeSpeed)
                put("backspaceRatio",      liveVector.backspaceRatio)
                put("scrollVelocity",      liveVector.scrollVelocity)
                put("daylightExposureMinutes", liveVector.daylightExposureMinutes)
                put("chargeRegularity",    liveVector.chargeRegularity)
                put("chargeDurationHours", liveVector.chargeDurationHours)
                put("upiTransactions",     liveVector.upiTransactionsToday)
                put("appUninstalls",       liveVector.appUninstallsToday)
                put("appInstalls",         liveVector.appInstallsToday)
                put("calendarEvents",      liveVector.calendarEventsToday)
                put("mediaCount",          liveVector.mediaCountToday)
                put("downloads",           liveVector.downloadsToday)
                put("musicTimeMinutes",    liveVector.musicTimeMinutes)
            })
            todayObj.put("location_snapshots", DataRepository.locationSnapshots.value.joinToString(";") { "${it.lat},${it.lon},${it.timeMs}" })
            masterJson.put("live_today", todayObj)
        }

        return masterJson.toString()
    }
}
