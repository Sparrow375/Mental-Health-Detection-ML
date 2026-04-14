package com.example.mhealth.logic

import android.content.Context
import android.content.SharedPreferences
import com.example.mhealth.logic.db.AnalysisResultEntity
import com.example.mhealth.logic.db.DailyFeaturesEntity
import com.example.mhealth.logic.db.MHealthDatabase
import com.example.mhealth.models.DailyReport
import com.example.mhealth.models.LatLonPoint
import com.example.mhealth.models.PersonalityVector
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

object DataRepository {

    private var prefs: SharedPreferences? = null
    
    // Coroutine scope for stateIn() — survives for the lifetime of the process
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    // ─── Room-backed reactive flows (populated via initWithDb) ─────────────

    /** Emits the most recent AnalysisResultEntity whenever NightlyWorker saves a new result. */
    private val _latestAnalysisResult = MutableStateFlow<AnalysisResultEntity?>(null)
    val latestAnalysisResult: StateFlow<AnalysisResultEntity?> = _latestAnalysisResult

    /** Emits the live, non-persisted analysis for the current day's progress. */
    private val _provisionalAnalysis = MutableStateFlow<com.example.mhealth.models.DailyReport?>(null)
    val provisionalAnalysis: StateFlow<com.example.mhealth.models.DailyReport?> = _provisionalAnalysis

    /** Emits the last 30 analysis results (newest first) for the history sparkline. */
    private val _analysisHistory = MutableStateFlow<List<AnalysisResultEntity>>(emptyList())
    val analysisHistory: StateFlow<List<AnalysisResultEntity>> = _analysisHistory

    /** Emits the last 7 daily feature vectors for trend sparklines on the Monitor screen. */
    private val _weeklyFeatureHistory = MutableStateFlow<List<PersonalityVector>>(emptyList())
    val weeklyFeatureHistory: StateFlow<List<PersonalityVector>> = _weeklyFeatureHistory

    /** Emits the System 1 Profile JSON (DNA Baseline, Clusters, Texture Profiles) from person_dna table. */
    private val _s1ProfileJson = MutableStateFlow<String?>(null)
    val s1ProfileJson: StateFlow<String?> = _s1ProfileJson

    /**
     * Wire the Room-backed StateFlows after the DB is available (call from MonitoringService/Application).
     * Safe to call multiple times — subsequent calls are no-ops.
     */
    @Volatile private var dbInitialized = false
    fun initWithDb(context: Context, userId: String) {
        if (dbInitialized) return
        dbInitialized = true
        val db = MHealthDatabase.getInstance(context.applicationContext)
        // Launch two long-lived coroutines that push Room Flow emissions into MutableStateFlows
        scope.launch {
            db.analysisResultDao().getLatestFlow(userId).collect { entity ->
                _latestAnalysisResult.value = entity
            }
        }
        scope.launch {
            db.analysisResultDao().getLatestNFlow(userId, limit = 30).collect { list ->
                _analysisHistory.value = list
            }
        }
        // Load baseline progress and history
        scope.launch {
            val count = db.dailyFeaturesDao().count(userId)
            _baselineProgress.value = count + 1
            
            val entities = db.dailyFeaturesDao().getLatestN(userId, 7)
            _weeklyFeatureHistory.value = entities.map { it.toPersonalityVector() }.reversed()
        }
        // Load System 1 profile from person_dna table
        scope.launch {
            val dnaEntity = db.personDnaDao().getByUserId(userId)
            if (dnaEntity != null) {
                _s1ProfileJson.value = dnaEntity.dna_json
            }
        }
    }

    /** Update the System 1 profile (called from NightlyAnalysisWorker after profile build). */
    fun updateS1Profile(profileJson: String) {
        _s1ProfileJson.value = profileJson
    }

    private fun DailyFeaturesEntity.toPersonalityVector() = PersonalityVector(
        screenTimeHours = screenTimeHours,
        unlockCount = unlockCount,
        appLaunchCount = appLaunchCount,
        notificationsToday = notificationsToday,
        socialAppRatio = socialAppRatio,
        callsPerDay = callsPerDay,
        callDurationMinutes = callDurationMinutes,
        uniqueContacts = uniqueContacts,
        conversationFrequency = conversationFrequency,
        dailyDisplacementKm = dailyDisplacementKm,
        locationEntropy = locationEntropy,
        homeTimeRatio = homeTimeRatio,
        placesVisited = placesVisited,
        wakeTimeHour = wakeTimeHour,
        sleepTimeHour = sleepTimeHour,
        sleepDurationHours = sleepDurationHours,
        darkDurationHours = darkDurationHours,
        chargeDurationHours = chargeDurationHours,
        memoryUsagePercent = memoryUsagePercent,
        networkWifiMB = networkWifiMB,
        networkMobileMB = networkMobileMB,
        downloadsToday = downloadsToday,
        storageUsedGB = storageUsedGB,
        appUninstallsToday = appUninstallsToday,
        upiTransactionsToday = upiTransactionsToday,
        totalAppsCount = totalAppsCount,
        backgroundAudioHours = backgroundAudioHours,
        mediaCountToday = mediaCountToday,
        appInstallsToday = appInstallsToday,
        calendarEventsToday = calendarEventsToday,
        dailySteps = dailySteps,
        appBreakdown = emptyMap(), // This is usually null-filled from DB row, but we add our local one
        bgAudioBreakdown = _bgAudioBreakdown.value
    )



    // Latest computed personality vector (updated ~every 15 min)
    private val _latestVector = MutableStateFlow<PersonalityVector?>(null)
    val latestVector: StateFlow<PersonalityVector?> = _latestVector

    // Daily reports (post-baseline monitoring)
    private val _reports = MutableStateFlow<List<DailyReport>>(emptyList())
    val reports: StateFlow<List<DailyReport>> = _reports

    // The established baseline personality vector (P₀)
    private val _baseline = MutableStateFlow<PersonalityVector?>(null)
    val baseline: StateFlow<PersonalityVector?> = _baseline

    // Whether we are still in baseline-building phase
    private val _isBuildingBaseline = MutableStateFlow(true)
    val isBuildingBaseline: StateFlow<Boolean> = _isBuildingBaseline

    // Number of days collected toward baseline
    private val _baselineProgress = MutableStateFlow(0)
    val baselineProgress: StateFlow<Int> = _baselineProgress

    // Raw vectors collected so far during the baseline building phase
    private val _collectedBaselineVectors = MutableStateFlow<List<PersonalityVector>>(emptyList())
    val collectedBaselineVectors: StateFlow<List<PersonalityVector>> = _collectedBaselineVectors

    // Intraday hourly snapshots — for live sparkline on Monitor screen
    private val _hourlySnapshots = MutableStateFlow<List<PersonalityVector>>(emptyList())
    val hourlySnapshots: StateFlow<List<PersonalityVector>> = _hourlySnapshots

    // Multi-point GPS track for today — used for displacement & entropy
    private val _locationSnapshots = MutableStateFlow<List<LatLonPoint>>(emptyList())
    val locationSnapshots: StateFlow<List<LatLonPoint>> = _locationSnapshots

    // Current GPS state (STATIONARY/WALKING/VEHICLE) for adaptive tracking
    private val _gpsState = MutableStateFlow("Stationary")
    val gpsState: StateFlow<String> = _gpsState

    // Optional user mood check-in score (1-10)
    private val _moodScore = MutableStateFlow<Int?>(null)
    val moodScore: StateFlow<Int?> = _moodScore

    // Step count baseline captured at first collection of the day
    private val _stepBaseline = MutableStateFlow<Float?>(null)
    val stepBaseline: StateFlow<Float?> = _stepBaseline

    private val _accumulatedChargeHours = MutableStateFlow(0f)
    val accumulatedChargeHours: StateFlow<Float> = _accumulatedChargeHours

    // Accumulated background audio ms — updated per tick by AudioManager.isMusicActive()
    private val _accumulatedBgAudioMs = MutableStateFlow(0L)
    val accumulatedBgAudioMs: StateFlow<Long> = _accumulatedBgAudioMs

    // Per-app background audio breakdown: package -> ms
    private val _bgAudioBreakdown = MutableStateFlow<Map<String, Long>>(emptyMap())
    val bgAudioBreakdown: StateFlow<Map<String, Long>> = _bgAudioBreakdown

    // Shared notification arrival times for NLS → DataCollector trigger detection
    // Updated by MHealthNotificationListenerService, read by DataCollector.logSessionsFromEvents
    private val _recentNotificationTimes = mutableMapOf<String, Long>()
    fun setRecentNotificationTime(pkg: String, timestampMs: Long) {
        _recentNotificationTimes[pkg] = timestampMs
    }
    fun getRecentNotificationTime(pkg: String): Long = _recentNotificationTimes[pkg] ?: 0L
    fun clearRecentNotificationTime(pkg: String) { _recentNotificationTimes.remove(pkg) }

    // Saved home location (lat/lon) for homeTimeRatio calculation
    private val _homeLocation = MutableStateFlow<Pair<Double, Double>?>(null)
    val homeLocation: StateFlow<Pair<Double, Double>?> = _homeLocation

    // Track the last processed Calendar Day of Year stringently across app reboots
    private val _lastProcessedDay = MutableStateFlow(-1)
    val lastProcessedDay: StateFlow<Int> = _lastProcessedDay

    // DND On/Off Timestamps for Sleep Detection
    private val _dndOnMs = MutableStateFlow(-1L)
    val dndOnMs: StateFlow<Long> = _dndOnMs

    private val _dndOffMs = MutableStateFlow(-1L)
    val dndOffMs: StateFlow<Long> = _dndOffMs

    // User Profile & Onboarding
    private val _userProfile = MutableStateFlow<com.example.mhealth.models.UserProfile?>(null)
    val userProfile: StateFlow<com.example.mhealth.models.UserProfile?> = _userProfile

    private val _firstLoginComplete = MutableStateFlow(false)
    val firstLoginComplete: StateFlow<Boolean> = _firstLoginComplete

    // Dev Configuration
    private val _baselineDaysRequired = MutableStateFlow(28)
    val baselineDaysRequired: StateFlow<Int> = _baselineDaysRequired

    // DNA Baseline (Level 2) — separate from L1 baseline
    private val _dnaBaselineDaysRequired = MutableStateFlow(14)
    val dnaBaselineDaysRequired: StateFlow<Int> = _dnaBaselineDaysRequired

    private val _monitoringIntervalMinutes = MutableStateFlow(15L)
    val monitoringIntervalMinutes: StateFlow<Long> = _monitoringIntervalMinutes

    private val _forceNewDayTrigger = MutableStateFlow(0)
    val forceNewDayTrigger: StateFlow<Int> = _forceNewDayTrigger

    private val _resetTrigger = MutableStateFlow(0)
    val resetTrigger: StateFlow<Int> = _resetTrigger

    // --- Persistence & Init ---
    
    fun init(context: Context) {
        if (prefs != null) return
        prefs = context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE)

        // Dev Settings
        _baselineDaysRequired.value = prefs?.getInt("dev_baseline_days", 28) ?: 28
        _dnaBaselineDaysRequired.value = prefs?.getInt("dev_dna_baseline_days", 14) ?: 14
        _monitoringIntervalMinutes.value = prefs?.getLong("dev_monitoring_interval", 15L) ?: 15L
        
        // Restore Onboarding State
        _firstLoginComplete.value = prefs?.getBoolean("first_login_complete", false) ?: false
        val savedEmail = prefs?.getString("user_email", "") ?: ""
        if (savedEmail.isNotEmpty()) {
            _userProfile.value = com.example.mhealth.models.UserProfile(
                email = savedEmail,
                name = prefs?.getString("user_name", "") ?: "",
                gender = prefs?.getString("user_gender", "") ?: "",
                dateOfBirth = prefs?.getString("user_dob", "") ?: "",
                age = prefs?.getInt("user_age", 0) ?: 0,
                profession = prefs?.getString("user_profession", "") ?: "",
                country = prefs?.getString("user_country", "") ?: ""
            )
        }

        // Restore step baseline
        val savedStepBaseline = prefs?.getFloat("step_baseline_today", -1f)
        if (savedStepBaseline != null && savedStepBaseline >= 0f) {
            _stepBaseline.value = savedStepBaseline
        }

        // Restore locations
        val savedLocsStr = prefs?.getString("loc_snapshots_today", "") ?: ""
        if (savedLocsStr.isNotEmpty()) {
            try {
                val locs = savedLocsStr.split(";").filter { it.isNotBlank() }.map { 
                    val parts = it.split(",")
                    LatLonPoint(
                        parts[0].toDouble(), 
                        parts[1].toDouble(), 
                        parts[2].toLong(),
                        if (parts.size > 3) parts[3].toFloat() else 0f,
                        if (parts.size > 4) parts[4].toFloat() else 0f   // speed — backwards compat
                    )
                }
                _locationSnapshots.value = locs
            } catch (e: Exception) {}
        }

        // Restore accumulated charge hours
        _accumulatedChargeHours.value = prefs?.getFloat("charge_hours_today", 0f) ?: 0f

        // Restore accumulated background audio ms
        _accumulatedBgAudioMs.value = prefs?.getLong("bg_audio_ms_today", 0L) ?: 0L
        _bgAudioBreakdown.value = loadMapFromPrefs("bg_audio_breakdown_today")

        // Restore home location (stored as Float to use NaN as sentinel)
        val homeLat = prefs?.getFloat("home_location_lat", Float.NaN) ?: Float.NaN
        val homeLon = prefs?.getFloat("home_location_lon", Float.NaN) ?: Float.NaN
        if (!homeLat.isNaN() && !homeLon.isNaN()) {
            _homeLocation.value = Pair(homeLat.toDouble(), homeLon.toDouble())
        }

        // Restore last processed calendar day
        _lastProcessedDay.value = prefs?.getInt("last_processed_day", -1) ?: -1

        // Restore DND timestamps
        _dndOnMs.value = prefs?.getLong("dnd_on_ts", -1L) ?: -1L
        _dndOffMs.value = prefs?.getLong("dnd_off_ts", -1L) ?: -1L
    }

    // --- Mutators ---

    fun saveUserProfile(profile: com.example.mhealth.models.UserProfile) {
        _userProfile.value = profile
        prefs?.edit()?.apply {
            putString("user_email", profile.email)
            putString("user_name", profile.name)
            putString("user_gender", profile.gender)
            putString("user_dob", profile.dateOfBirth)
            putInt("user_age", profile.age)
            putString("user_profession", profile.profession)
            putString("user_country", profile.country)
            putBoolean("first_login_complete", true)
        }?.apply()
        _firstLoginComplete.value = true
    }

    fun setBaselineDaysRequired(days: Int) {
        _baselineDaysRequired.value = days
        prefs?.edit()?.putInt("dev_baseline_days", days)?.apply()
    }

    fun setDnaBaselineDaysRequired(days: Int) {
        _dnaBaselineDaysRequired.value = days
        prefs?.edit()?.putInt("dev_dna_baseline_days", days)?.apply()
    }

    fun setMonitoringIntervalMinutes(minutes: Long) {
        _monitoringIntervalMinutes.value = minutes
        prefs?.edit()?.putLong("dev_monitoring_interval", minutes)?.apply()
    }

    fun triggerNewDay() {
        _forceNewDayTrigger.value += 1
    }

    fun triggerReset() {
        _resetTrigger.value += 1
    }

    fun setIsBuildingBaseline(building: Boolean) {
        _isBuildingBaseline.value = building
    }

    fun updateLatestVector(vector: PersonalityVector) {
        _latestVector.value = vector
    }

    fun updateProvisionalAnalysis(result: com.example.mhealth.models.DailyReport?) {
        _provisionalAnalysis.value = result
    }

    fun addReport(report: DailyReport) {
        val current = _reports.value.toMutableList()
        current.add(report)
        _reports.value = current
    }

    fun updateReports(reports: List<DailyReport>) {
        _reports.value = reports
    }

    fun clearReports() {
        _reports.value = emptyList()
    }

    fun setBaseline(vector: PersonalityVector) {
        _baseline.value = vector
        _isBuildingBaseline.value = false
    }

    fun clearBaseline() {
        _baseline.value = null
    }

    fun updateBaselineProgress(days: Int) {
        _baselineProgress.value = days
    }

    fun updateCollectedBaselineVectors(vectors: List<PersonalityVector>) {
        _collectedBaselineVectors.value = vectors.toList()
    }

    fun addHourlySnapshot(vector: PersonalityVector) {
        val updated = (_hourlySnapshots.value + vector).takeLast(24)
        _hourlySnapshots.value = updated
    }

    fun addLocationSnapshot(point: LatLonPoint) {
        val updated = (_locationSnapshots.value + point).takeLast(288) // 24h @ 5-min continuous tracking
        _locationSnapshots.value = updated
        saveLocationsToPrefs(updated)
    }

    fun updateGpsState(state: String) {
        _gpsState.value = state
    }

    fun clearDailyLocationSnapshots() {
        _locationSnapshots.value = emptyList()
        saveLocationsToPrefs(emptyList())
    }

    private fun saveLocationsToPrefs(list: List<LatLonPoint>) {
        // Format: lat,lon,timeMs,accuracy,speed — speed added for vehicle filtering (backward compat)
        val str = list.joinToString(";") { "${it.lat},${it.lon},${it.timeMs},${it.accuracy},${it.speed}" }
        prefs?.edit()?.putString("loc_snapshots_today", str)?.apply()
    }

    fun setMoodScore(score: Int) {
        _moodScore.value = score.coerceIn(1, 10)
    }

    fun addChargeTime(hours: Float) {
        val newTotal = _accumulatedChargeHours.value + hours
        _accumulatedChargeHours.value = newTotal
        prefs?.edit()?.putFloat("charge_hours_today", newTotal)?.apply()
    }

    fun addBgAudioTime(packageName: String?, ms: Long) {
        val newTotal = _accumulatedBgAudioMs.value + ms
        _accumulatedBgAudioMs.value = newTotal
        prefs?.edit()?.putLong("bg_audio_ms_today", newTotal)?.apply()

        if (packageName != null) {
            val currentMap = _bgAudioBreakdown.value.toMutableMap()
            val existing = currentMap[packageName] ?: 0L
            currentMap[packageName] = existing + ms
            _bgAudioBreakdown.value = currentMap
            saveMapToPrefs("bg_audio_breakdown_today", currentMap)
        }
    }

    fun setHomeLocation(lat: Double, lon: Double) {
        _homeLocation.value = Pair(lat, lon)
        prefs?.edit()?.apply {
            putFloat("home_location_lat", lat.toFloat())
            putFloat("home_location_lon", lon.toFloat())
        }?.apply()
    }

    fun getHomeLatitude(): Double? = _homeLocation.value?.first
    fun getHomeLongitude(): Double? = _homeLocation.value?.second

    /**
     * Returns the last GPS fix from YESTERDAY (saved just before midnight wipe).
     * Used by DataCollector to anchor the overnight homeTimeRatio bridge:
     * if the patient was home at 23:59, the hours from midnight → first-snap-today count as home.
     */
    fun getLastLocationBeforeMidnight(): LatLonPoint? {
        val str = prefs?.getString("last_location_before_midnight", null) ?: return null
        return try {
            val parts = str.split(",")
            LatLonPoint(
                parts[0].toDouble(), parts[1].toDouble(), parts[2].toLong(),
                if (parts.size > 3) parts[3].toFloat() else 0f,
                if (parts.size > 4) parts[4].toFloat() else 0f
            )
        } catch (e: Exception) { null }
    }

    fun setStepBaseline(steps: Float) {
        if (_stepBaseline.value == null) {
            _stepBaseline.value = steps
            prefs?.edit()?.putFloat("step_baseline_today", steps)?.apply()
        }
    }

    fun setLastProcessedDay(day: Int) {
        _lastProcessedDay.value = day
        prefs?.edit()?.putInt("last_processed_day", day)?.apply()
    }

    fun setDndOnTimestamp(ts: Long) {
        _dndOnMs.value = ts
        prefs?.edit()?.putLong("dnd_on_ts", ts)?.apply()
    }

    fun setDndOffTimestamp(ts: Long) {
        _dndOffMs.value = ts
        prefs?.edit()?.putLong("dnd_off_ts", ts)?.apply()
    }

    fun resetDailyState() {
        // FIX: Before wiping today's location snapshots, persist the last known GPS fix.
        // This is used tomorrow as the overnight anchor for the homeTimeRatio midnight bridge.
        // Without this, the bridge has nothing to anchor on and misses all sleep hours.
        val lastSnap = _locationSnapshots.value.lastOrNull()
        if (lastSnap != null) {
            prefs?.edit()?.putString(
                "last_location_before_midnight",
                "${lastSnap.lat},${lastSnap.lon},${lastSnap.timeMs},${lastSnap.accuracy},${lastSnap.speed}"
            )?.apply()
        }

        _hourlySnapshots.value = emptyList()
        _locationSnapshots.value = emptyList()
        _stepBaseline.value = null
        _moodScore.value = null
        _accumulatedChargeHours.value = 0f
        _accumulatedBgAudioMs.value = 0L
        _bgAudioBreakdown.value = emptyMap()
        _dndOnMs.value = -1L
        _dndOffMs.value = -1L
        
        prefs?.edit()?.apply {
            remove("step_baseline_today")
            remove("loc_snapshots_today")
            remove("charge_hours_today")
            remove("bg_audio_ms_today")
            remove("bg_audio_breakdown_today")
            remove("dnd_on_ts")
            remove("dnd_off_ts")
            remove("prev_pkg_count")   // reset so appUninstalls recalculates fresh each day
        }?.apply()
    }

    fun restoreTodayState(locs: List<LatLonPoint>, chargeHrs: Float, bgAudio: Long, stepBase: Float) {
        _locationSnapshots.value = locs
        saveLocationsToPrefs(locs)
        
        _accumulatedChargeHours.value = chargeHrs
        prefs?.edit()?.putFloat("charge_hours_today", chargeHrs)?.apply()
        
        _accumulatedBgAudioMs.value = bgAudio
        prefs?.edit()?.putLong("bg_audio_ms_today", bgAudio)?.apply()
        
        if (stepBase >= 0f) {
            _stepBaseline.value = stepBase
            prefs?.edit()?.putFloat("step_baseline_today", stepBase)?.apply()
        }
    }

    fun clearAllState() {
        _baselineProgress.value = 0
        _collectedBaselineVectors.value = emptyList()
        _baseline.value = null
        _isBuildingBaseline.value = true
        resetDailyState()
    }

    private fun saveMapToPrefs(key: String, map: Map<String, Long>) {
        val json = org.json.JSONObject()
        map.forEach { (k, v) -> json.put(k, v) }
        prefs?.edit()?.putString(key, json.toString())?.apply()
    }

    private fun loadMapFromPrefs(key: String): Map<String, Long> {
        val jsonStr = prefs?.getString(key, "{}") ?: "{}"
        try {
            val json = org.json.JSONObject(jsonStr)
            val map = mutableMapOf<String, Long>()
            json.keys().forEach { k -> map[k] = json.getLong(k) }
            return map
        } catch (e: Exception) {
            return emptyMap()
        }
    }
}
