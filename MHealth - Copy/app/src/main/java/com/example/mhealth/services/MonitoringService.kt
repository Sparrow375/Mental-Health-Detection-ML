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
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.firestore.FirebaseFirestore
import kotlinx.coroutines.tasks.await
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
                    runTick(isSimulated = true)
                }
            }
        }

        // dev force reset trigger listener
        serviceScope.launch {
            DataRepository.resetTrigger.collect { triggers ->
                if (triggers > 0) {
                    val userId = DataRepository.userProfile.value?.email ?: "default_user"
                    val db = MHealthDatabase.getInstance(this@MonitoringService)
                    try {
                        db.dailyFeaturesDao().clearSimulated(userId)
                        
                        val targetBaselineDays = DataRepository.baselineDaysRequired.value
                        val remainingFeatures = db.dailyFeaturesDao().getLatestN(userId, 60)
                        if (remainingFeatures.size < targetBaselineDays) {
                            db.baselineDao().clearBaseline(userId)
                            db.analysisResultDao().clearAll(userId)
                            val profile = db.userProfileDao().get(userId)
                            if (profile != null) {
                                db.userProfileDao().upsert(profile.copy(baselineReady = false))
                            }
                            DataRepository.setIsBuildingBaseline(true)
                            detector = null
                        }
                        
                        restoreStateFromRoom()
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

    private fun runTick(isSimulated: Boolean = false) {
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
                handleBaselineBuilding(snapshot, today, savedDay, isSimulated)
            } else {
                handleAnomalyDetection(snapshot, today, savedDay, isSimulated)
            }

            if (today != savedDay) {
                DataRepository.setLastProcessedDay(today)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun handleBaselineBuilding(snapshot: PersonalityVector, today: Int, savedDay: Int, isSimulated: Boolean) {
        val hour = Calendar.getInstance().get(Calendar.HOUR_OF_DAY)
        val targetBaselineDays = DataRepository.baselineDaysRequired.value
        if (today != savedDay && savedDay != -1) {
            if (collectedDailyVectors.size < targetBaselineDays) {
                collectedDailyVectors.add(snapshot)
                DataRepository.updateBaselineProgress(collectedDailyVectors.size)
                DataRepository.updateCollectedBaselineVectors(collectedDailyVectors)

                // Persist end-of-day snapshot to Room
                persistDailySnapshot(snapshot, savedDay, isSimulated)
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

    private fun handleAnomalyDetection(snapshot: PersonalityVector, today: Int, savedDay: Int, isSimulated: Boolean) {
        val report = detector?.analyze(snapshot, DataRepository.reports.value.size + 1)
        report?.let {
            if (today != savedDay && savedDay != -1) {
                // Persist end-of-monitoring-day snapshot to Room
                persistDailySnapshot(snapshot, savedDay, isSimulated)

                DataRepository.addReport(it)
                if (it.alertLevel == "orange" || it.alertLevel == "red") {
                    sendAlertNotification(it.alertLevel, it.notes)
                }
            }
        }
    }

    // ── Room persistence helpers ──────────────────────────────────────────────

    private fun persistDailySnapshot(snapshot: PersonalityVector, dayOfYear: Int, isSimulated: Boolean) {
        val userId = DataRepository.userProfile.value?.email ?: "default_user"
        val cal = Calendar.getInstance().apply { set(Calendar.DAY_OF_YEAR, dayOfYear) }
        val dateStr = dateFmt.format(cal.time)
        val entity = JsonConverter.fromPersonalityVector(userId, dateStr, snapshot, isSimulated)
        serviceScope.launch {
            try {
                MHealthDatabase.getInstance(this@MonitoringService)
                    .dailyFeaturesDao().insert(entity)
                    
                // Automatically push un-synced data to Firebase database
                syncUnstagedDailyFeaturesToFirebase()
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

        // Upload firmly established baseline to Cloud Backup
        val uid = FirebaseAuth.getInstance().currentUser?.uid
        if (uid != null) {
            try {
                val firestore = FirebaseFirestore.getInstance()
                firestore.collection("users").document(uid).update("baseline_ready", true).await()
                
                val baselineRef = firestore.collection("users").document(uid).collection("baseline")
                baseline.toMap().forEach { (feature, mean) ->
                    val std = baseline.variances[feature] ?: 1f
                    val data = hashMapOf(
                        "featureName" to feature,
                        "baselineValue" to mean,
                        "stdDeviation" to std,
                        "baselineStart" to today,
                        "baselineEnd" to today
                    )
                    baselineRef.document(feature).set(data).await()
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
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

        val n = vectors.size
        // 10% Trimmed Mean: Removes extreme 10% high & 10% low outliers from the calibration period
        val trimCount = (n * 0.10).toInt().coerceAtLeast(0)

        features.forEach { feature ->
            val vals = vectors.map { it.toMap()[feature] ?: 0f }
            
            // Trim outliers for robust mean calculation
            val sortedVals = vals.sorted()
            val trimmedVals = if (n > 4 && trimCount > 0) {
                sortedVals.subList(trimCount, n - trimCount)
            } else {
                sortedVals
            }
            
            val robustAvg = trimmedVals.average().toFloat()
            averages[feature] = robustAvg
            
            // Variance is computed against the robust average using the full valid dataset
            variances[feature] = calculateSD(vals, robustAvg)
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

    private suspend fun syncUnstagedDailyFeaturesToFirebase() {
        val uid = FirebaseAuth.getInstance().currentUser?.uid ?: return
        val db = MHealthDatabase.getInstance(this@MonitoringService)
        val userId = DataRepository.userProfile.value?.email ?: return
        
        try {
            val unsynced = db.dailyFeaturesDao().getUnsynced(userId)
            if (unsynced.isEmpty()) return
            
            val firestore = FirebaseFirestore.getInstance()
            val collectionRef = firestore.collection("users").document(uid).collection("daily_features")
            
            for (entity in unsynced) {
                // Ensure simulated testing data does not contaminate Firebase
                if (entity.isSimulated) {
                    db.dailyFeaturesDao().markSynced(entity.id)
                    continue
                }
                
                val data = hashMapOf(
                    "screenTimeHours" to entity.screenTimeHours,
                    "unlockCount" to entity.unlockCount,
                    "appLaunchCount" to entity.appLaunchCount,
                    "notificationsToday" to entity.notificationsToday,
                    "socialAppRatio" to entity.socialAppRatio,
                    "callsPerDay" to entity.callsPerDay,
                    "callDurationMinutes" to entity.callDurationMinutes,
                    "uniqueContacts" to entity.uniqueContacts,
                    "conversationFrequency" to entity.conversationFrequency,
                    "dailyDisplacementKm" to entity.dailyDisplacementKm,
                    "locationEntropy" to entity.locationEntropy,
                    "homeTimeRatio" to entity.homeTimeRatio,
                    "placesVisited" to entity.placesVisited,
                    "wakeTimeHour" to entity.wakeTimeHour,
                    "sleepTimeHour" to entity.sleepTimeHour,
                    "sleepDurationHours" to entity.sleepDurationHours,
                    "darkDurationHours" to entity.darkDurationHours,
                    "chargeDurationHours" to entity.chargeDurationHours,
                    "memoryUsagePercent" to entity.memoryUsagePercent,
                    "networkWifiMB" to entity.networkWifiMB,
                    "networkMobileMB" to entity.networkMobileMB,
                    "downloadsToday" to entity.downloadsToday,
                    "storageUsedGB" to entity.storageUsedGB,
                    "appUninstallsToday" to entity.appUninstallsToday,
                    "upiTransactionsToday" to entity.upiTransactionsToday,
                    "nightInterruptions" to entity.nightInterruptions
                )
                
                collectionRef.document(entity.date).set(data).await()
                db.dailyFeaturesDao().markSynced(entity.id)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
