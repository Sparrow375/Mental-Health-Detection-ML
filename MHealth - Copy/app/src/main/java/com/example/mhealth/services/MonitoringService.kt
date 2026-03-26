package com.example.mhealth.services

import android.app.*
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.example.mhealth.MainActivity
import com.example.mhealth.logic.AnomalyDetector
import com.example.mhealth.logic.DataCollector
import com.example.mhealth.logic.DataRepository
import com.example.mhealth.logic.JsonConverter
import com.example.mhealth.logic.db.BaselineEntity
import com.example.mhealth.logic.db.MHealthDatabase
import com.example.mhealth.logic.db.UserProfileEntity
import com.example.mhealth.models.PersonalityVector
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.collectLatest
import java.text.SimpleDateFormat
import java.util.*

class MonitoringService : Service() {

    private lateinit var dataCollector: DataCollector
    private var detector: AnomalyDetector? = null

    private val serviceScope = CoroutineScope(Dispatchers.IO + Job())
    private var trackingJob: Job? = null
    private val dateFmt = SimpleDateFormat("yyyy-MM-dd", Locale.US)

    private val collectedDailyVectors = mutableListOf<PersonalityVector>()
    private var collectionTickCount = 0
    private var nightlyWorkerScheduled = false

    override fun onCreate() {
        super.onCreate()
        dataCollector = DataCollector(this)
        // Wire Room-backed StateFlows so AnalysisScreen/InsightsScreen update reactively
        val userId = DataRepository.userProfile.value?.email ?: "default_user"
        DataRepository.initWithDb(applicationContext, userId)
        
        restoreStateFromRoom()
        
        startForegroundNotification()
        scheduleMonitoring()
    }

    private fun restoreStateFromRoom() {
        val userId = DataRepository.userProfile.value?.email ?: "default_user"
        serviceScope.launch {
            try {
                val db = MHealthDatabase.getInstance(this@MonitoringService)
                val profile = db.userProfileDao().get(userId)
                
                if (profile?.baselineReady == true) {
                    val baselineEntities = db.baselineDao().getBaseline(userId)
                    if (baselineEntities.isNotEmpty()) {
                        val baselineFields = baselineEntities.associate { it.featureName to it.baselineValue }
                        val variances = baselineEntities.associate { it.featureName to it.stdDeviation }
                        val baseline = PersonalityVector(
                            screenTimeHours = baselineFields["screenTimeHours"] ?: 0f,
                            unlockCount = baselineFields["unlockCount"] ?: 0f,
                            appLaunchCount = baselineFields["appLaunchCount"] ?: 0f,
                            notificationsToday = baselineFields["notificationsToday"] ?: 0f,
                            socialAppRatio = baselineFields["socialAppRatio"] ?: 0f,
                            callsPerDay = baselineFields["callsPerDay"] ?: 0f,
                            callDurationMinutes = baselineFields["callDurationMinutes"] ?: 0f,
                            uniqueContacts = baselineFields["uniqueContacts"] ?: 0f,
                            conversationFrequency = baselineFields["conversationFrequency"] ?: 0f,
                            dailyDisplacementKm = baselineFields["dailyDisplacementKm"] ?: 0f,
                            locationEntropy = baselineFields["locationEntropy"] ?: 0f,
                            homeTimeRatio = baselineFields["homeTimeRatio"] ?: 0f,
                            placesVisited = baselineFields["placesVisited"] ?: 0f,
                            wakeTimeHour = baselineFields["wakeTimeHour"] ?: 0f,
                            sleepTimeHour = baselineFields["sleepTimeHour"] ?: 0f,
                            sleepDurationHours = baselineFields["sleepDurationHours"] ?: 0f,
                            darkDurationHours = baselineFields["darkDurationHours"] ?: 0f,
                            chargeDurationHours = baselineFields["chargeDurationHours"] ?: 0f,
                            memoryUsagePercent = baselineFields["memoryUsagePercent"] ?: 0f,
                            networkWifiMB = baselineFields["networkWifiMB"] ?: 0f,
                            networkMobileMB = baselineFields["networkMobileMB"] ?: 0f,
                            variances = variances as MutableMap<String, Float>
                        )
                        DataRepository.setBaseline(baseline)
                        detector = AnomalyDetector(baseline)
                    }
                }
                
                // Always load recent history for the Recent Trends UI, whether building or actively monitoring
                val pastFeatures = db.dailyFeaturesDao().getLatestN(userId, 60).reversed()
                val pastVectors = pastFeatures.map { JsonConverter.toPersonalityVector(it) }
                collectedDailyVectors.clear()
                collectedDailyVectors.addAll(pastVectors)
                DataRepository.updateBaselineProgress(collectedDailyVectors.size)
                DataRepository.updateCollectedBaselineVectors(collectedDailyVectors)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    private fun startForegroundNotification() {
        val channelId = "mhealth_monitoring"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                channelId, "MHealth Monitoring", NotificationManager.IMPORTANCE_LOW
            ).apply { description = "Passive mental health pattern monitoring" }
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
        val notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("MHealth Active")
            .setContentText("Passively monitoring device patterns…")
            .setSmallIcon(android.R.drawable.ic_menu_info_details)
            .setContentIntent(
                PendingIntent.getActivity(
                    this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE
                )
            )
            .setOngoing(true)
            .build()
        startForeground(1, notification)
    }

    private fun scheduleMonitoring() {
        // dynamic interval listener
        serviceScope.launch {
            DataRepository.monitoringIntervalMinutes.collectLatest { intervalMin ->
                trackingJob?.cancel()
                trackingJob = launch {
                    while (isActive) {
                        runTick()
                        delay(intervalMin * 60 * 1000L)
                    }
                }
            }
        }
        
        // dev force new-day trigger listener
        serviceScope.launch {
            DataRepository.forceNewDayTrigger.collect { triggers ->
                if (triggers > 0) {
                    val today = Calendar.getInstance().get(Calendar.DAY_OF_YEAR)
                    DataRepository.setLastProcessedDay(if (today <= 1) 365 else today - 1)
                    runTick()
                }
            }
        }

        // dev force reset trigger listener
        serviceScope.launch {
            DataRepository.resetTrigger.collect { triggers ->
                if (triggers > 0) {
                    collectedDailyVectors.clear()
                    DataRepository.clearAllState()
                    detector = null
                    
                    val userId = DataRepository.userProfile.value?.email ?: "default_user"
                    val db = MHealthDatabase.getInstance(this@MonitoringService)
                    try {
                        db.dailyFeaturesDao().clearAll(userId)
                        db.baselineDao().clearBaseline(userId)
                        db.analysisResultDao().clearAll(userId)
                        val profile = db.userProfileDao().get(userId)
                        if (profile != null) {
                            db.userProfileDao().upsert(profile.copy(baselineReady = false))
                        }
                    } catch (e: Exception) {
                        e.printStackTrace()
                    }
                }
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        serviceScope.cancel()
    }

    private fun runTick() {
        try {
            collectionTickCount++

            // 1) Capture a GPS point every tick
            dataCollector.captureLocationSnapshot()

            // 2) Collect a full snapshot using accumulated GPS track
            val locationSnaps = DataRepository.locationSnapshots.value
            val snapshot = dataCollector.collectSnapshot(locationSnaps)

            // Always update live data for home screen
            DataRepository.updateLatestVector(snapshot)
            DataRepository.addHourlySnapshot(snapshot)

            // 3) Accumulate charging time
            val battery = dataCollector.getBatteryInfo()
            if (battery.isCharging) {
                // assume tick duration matches current repository interval in hours
                val hoursPerTick = DataRepository.monitoringIntervalMinutes.value / 60f
                DataRepository.addChargeTime(hoursPerTick)
            }

            val today = Calendar.getInstance().get(Calendar.DAY_OF_YEAR)
            val savedDay = DataRepository.lastProcessedDay.value

            // On new day, reset daily accumulators
            if (today != savedDay && savedDay != -1) {
                DataRepository.resetDailyState()
            }

            if (DataRepository.isBuildingBaseline.value) {
                handleBaselineBuilding(snapshot, today, savedDay)
            } else {
                handleAnomalyDetection(snapshot, today, savedDay)
            }

            if (today != savedDay) {
                DataRepository.setLastProcessedDay(today)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun handleBaselineBuilding(snapshot: PersonalityVector, today: Int, savedDay: Int) {
        val hour = Calendar.getInstance().get(Calendar.HOUR_OF_DAY)
        val targetBaselineDays = DataRepository.baselineDaysRequired.value
        if (today != savedDay && savedDay != -1) {
            if (collectedDailyVectors.size < targetBaselineDays) {
                collectedDailyVectors.add(snapshot)
                DataRepository.updateBaselineProgress(collectedDailyVectors.size)
                DataRepository.updateCollectedBaselineVectors(collectedDailyVectors)

                // Persist end-of-day snapshot to Room
                persistDailySnapshot(snapshot, savedDay)
            }
        }

        if (collectedDailyVectors.size >= targetBaselineDays) {
            val baseline = buildBaseline(collectedDailyVectors)
            DataRepository.setBaseline(baseline)
            detector = AnomalyDetector(baseline)

            // Persist baseline to Room and schedule nightly worker
            if (!nightlyWorkerScheduled) {
                serviceScope.launch {
                    persistBaselineToRoom(baseline, targetBaselineDays)
                    scheduleNightlyWorker()
                    nightlyWorkerScheduled = true
                }
            }
        }
    }

    private fun handleAnomalyDetection(snapshot: PersonalityVector, today: Int, savedDay: Int) {
        val report = detector?.analyze(snapshot, DataRepository.reports.value.size + 1)
        report?.let {
            if (today != savedDay && savedDay != -1) {
                // Persist end-of-monitoring-day snapshot to Room
                persistDailySnapshot(snapshot, savedDay)

                DataRepository.addReport(it)
                if (it.alertLevel == "orange" || it.alertLevel == "red") {
                    sendAlertNotification(it.alertLevel, it.notes)
                }
            }
        }
    }

    // ── Room persistence helpers ──────────────────────────────────────────────

    private fun persistDailySnapshot(snapshot: PersonalityVector, dayOfYear: Int) {
        val userId = DataRepository.userProfile.value?.email ?: "default_user"
        val cal = Calendar.getInstance().apply { set(Calendar.DAY_OF_YEAR, dayOfYear) }
        val dateStr = dateFmt.format(cal.time)
        val entity = JsonConverter.fromPersonalityVector(userId, dateStr, snapshot)
        serviceScope.launch {
            try {
                MHealthDatabase.getInstance(this@MonitoringService)
                    .dailyFeaturesDao().insert(entity)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    private suspend fun persistBaselineToRoom(
        baseline: PersonalityVector,
        baselineDays: Int
    ) {
        val userId = DataRepository.userProfile.value?.email ?: "default_user"
        val db = MHealthDatabase.getInstance(this@MonitoringService)
        val today = dateFmt.format(Date())

        val entities = baseline.toMap().map { (feature, mean) ->
            BaselineEntity(
                userId         = userId,
                featureName    = feature,
                baselineValue  = mean,
                stdDeviation   = baseline.variances[feature] ?: 1f,
                baselineStart  = today,
                baselineEnd    = today
            )
        }
        db.baselineDao().insertAll(entities)
        db.userProfileDao().upsert(
            UserProfileEntity(
                userId        = userId,
                baselineReady = true,
                baselineDays  = baselineDays,
                currentStatus = "Monitoring"
            )
        )
    }

    private fun scheduleNightlyWorker() {
        val userId = DataRepository.userProfile.value?.email ?: "default_user"
        NightlyAnalysisWorker.schedule(this@MonitoringService, userId)
    }

    private fun buildBaseline(vectors: List<PersonalityVector>): PersonalityVector {
        if (vectors.isEmpty()) return PersonalityVector()
        val features = vectors.first().toMap().keys
        val averages = mutableMapOf<String, Float>()
        val variances = mutableMapOf<String, Float>()

        features.forEach { feature ->
            val vals = vectors.map { it.toMap()[feature] ?: 0f }
            val avg = vals.average().toFloat()
            averages[feature] = avg
            variances[feature] = calculateSD(vals, avg)
        }

        return PersonalityVector(
            screenTimeHours = averages["screenTimeHours"] ?: 0f,
            unlockCount = averages["unlockCount"] ?: 0f,
            socialAppRatio = averages["socialAppRatio"] ?: 0f,
            callsPerDay = averages["callsPerDay"] ?: 0f,
            callDurationMinutes = averages["callDurationMinutes"] ?: 0f,
            uniqueContacts = averages["uniqueContacts"] ?: 0f,
            dailyDisplacementKm = averages["dailyDisplacementKm"] ?: 0f,
            locationEntropy = averages["locationEntropy"] ?: 0f,
            homeTimeRatio = averages["homeTimeRatio"] ?: 0f,
            placesVisited = averages["placesVisited"] ?: 0f,
            wakeTimeHour = averages["wakeTimeHour"] ?: 0f,
            sleepTimeHour = averages["sleepTimeHour"] ?: 0f,
            sleepDurationHours = averages["sleepDurationHours"] ?: 0f,
            darkDurationHours = averages["darkDurationHours"] ?: 0f,
            chargeDurationHours = averages["chargeDurationHours"] ?: 0f,
            conversationFrequency = averages["conversationFrequency"] ?: 0f,
            memoryUsagePercent = averages["memoryUsagePercent"] ?: 0f,
            networkWifiMB = averages["networkWifiMB"] ?: 0f,
            networkMobileMB = averages["networkMobileMB"] ?: 0f,
            mediaCountToday = averages["mediaCountToday"] ?: 0f,
            appInstallsToday = averages["appInstallsToday"] ?: 0f,
            variances = variances
        )
    }

    /** Returns standard deviation (used as variance bound in AnomalyDetector) */
    private fun calculateSD(values: List<Float>, mean: Float): Float {
        if (values.size < 2) return 1.0f
        val sd = kotlin.math.sqrt(values.map { (it - mean) * (it - mean) }.average()).toFloat()
        return if (sd < 0.01f) 0.01f else sd
    }

    private fun sendAlertNotification(level: String, notes: String) {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.notify(
            2, NotificationCompat.Builder(this, "mhealth_monitoring")
                .setContentTitle("Pattern Change: ${level.uppercase()}")
                .setContentText(notes)
                .setSmallIcon(android.R.drawable.ic_dialog_alert)
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setAutoCancel(true)
                .build()
        )
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
