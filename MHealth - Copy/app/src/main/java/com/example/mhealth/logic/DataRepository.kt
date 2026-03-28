package com.example.mhealth.logic

import android.content.Context
import android.content.SharedPreferences
import com.example.mhealth.logic.db.AnalysisResultEntity
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

    /** Emits the last 30 analysis results (newest first) for the history sparkline. */
    private val _analysisHistory = MutableStateFlow<List<AnalysisResultEntity>>(emptyList())
    val analysisHistory: StateFlow<List<AnalysisResultEntity>> = _analysisHistory

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
    }



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

    // Optional user mood check-in score (1-10)
    private val _moodScore = MutableStateFlow<Int?>(null)
    val moodScore: StateFlow<Int?> = _moodScore

    // Step count baseline captured at first collection of the day
    private val _stepBaseline = MutableStateFlow<Float?>(null)
    val stepBaseline: StateFlow<Float?> = _stepBaseline

    private val _accumulatedChargeHours = MutableStateFlow(0f)
    val accumulatedChargeHours: StateFlow<Float> = _accumulatedChargeHours

    // Track the last processed Calendar Day of Year stringently across app reboots
    private val _lastProcessedDay = MutableStateFlow(-1)
    val lastProcessedDay: StateFlow<Int> = _lastProcessedDay

    // User Profile & Onboarding
    private val _userProfile = MutableStateFlow<com.example.mhealth.models.UserProfile?>(null)
    val userProfile: StateFlow<com.example.mhealth.models.UserProfile?> = _userProfile

    private val _firstLoginComplete = MutableStateFlow(false)
    val firstLoginComplete: StateFlow<Boolean> = _firstLoginComplete

    // Dev Configuration
    private val _baselineDaysRequired = MutableStateFlow(28)
    val baselineDaysRequired: StateFlow<Int> = _baselineDaysRequired

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
                    LatLonPoint(parts[0].toDouble(), parts[1].toDouble(), parts[2].toLong())
                }
                _locationSnapshots.value = locs
            } catch (e: Exception) {}
        }

        // Restore accumulated charge hours
        _accumulatedChargeHours.value = prefs?.getFloat("charge_hours_today", 0f) ?: 0f

        // Restore last processed calendar day
        _lastProcessedDay.value = prefs?.getInt("last_processed_day", -1) ?: -1
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

    fun addReport(report: DailyReport) {
        _reports.value = _reports.value + report
    }

    fun setBaseline(vector: PersonalityVector) {
        _baseline.value = vector
        _isBuildingBaseline.value = false
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
        val updated = (_locationSnapshots.value + point).takeLast(96) // max 24h @ 15min
        _locationSnapshots.value = updated
        saveLocationsToPrefs(updated)
    }

    fun clearDailyLocationSnapshots() {
        _locationSnapshots.value = emptyList()
        saveLocationsToPrefs(emptyList())
    }

    private fun saveLocationsToPrefs(list: List<LatLonPoint>) {
        val str = list.joinToString(";") { "${it.lat},${it.lon},${it.timeMs}" }
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

    fun resetDailyState() {
        _hourlySnapshots.value = emptyList()
        _locationSnapshots.value = emptyList()
        _stepBaseline.value = null
        _moodScore.value = null
        _accumulatedChargeHours.value = 0f
        
        prefs?.edit()?.apply {
            remove("step_baseline_today")
            remove("loc_snapshots_today")
            remove("charge_hours_today")
            remove("prev_pkg_count")   // reset so appUninstalls recalculates fresh each day
        }?.apply()
    }

    fun clearAllState() {
        _baselineProgress.value = 0
        _collectedBaselineVectors.value = emptyList()
        _baseline.value = null
        _isBuildingBaseline.value = true
        resetDailyState()
    }
}
