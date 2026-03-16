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
import com.example.mhealth.models.PersonalityVector
import java.util.*

import kotlinx.coroutines.*
import kotlinx.coroutines.flow.collectLatest

class MonitoringService : Service() {

    private lateinit var dataCollector: DataCollector
    private var detector: AnomalyDetector? = null
    
    private val serviceScope = CoroutineScope(Dispatchers.IO + Job())
    private var trackingJob: Job? = null

    private val collectedDailyVectors = mutableListOf<PersonalityVector>()
    private var collectionTickCount = 0 // increments occasionally

    override fun onCreate() {
        super.onCreate()
        dataCollector = DataCollector(this)
        startForegroundNotification()
        scheduleMonitoring()
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
                    DataRepository.setLastProcessedDay(-1)
                    runTick()
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
        // Only add one vector per *day* to baseline corpus (end-of-day snapshot logic)
        val hour = Calendar.getInstance().get(Calendar.HOUR_OF_DAY)
        val targetBaselineDays = DataRepository.baselineDaysRequired.value
        if (today != savedDay && savedDay != -1) {
            // New day — add yesterday's final snapshot to baseline corpus
            if (collectedDailyVectors.size < targetBaselineDays) {
                collectedDailyVectors.add(snapshot)
                DataRepository.updateBaselineProgress(collectedDailyVectors.size)
            }
        }

        if (collectedDailyVectors.size >= targetBaselineDays) {
            val baseline = buildBaseline(collectedDailyVectors)
            DataRepository.setBaseline(baseline)
            detector = AnomalyDetector(baseline)
        }
    }

    private fun handleAnomalyDetection(snapshot: PersonalityVector, today: Int, savedDay: Int) {
        val report = detector?.analyze(snapshot, DataRepository.reports.value.size + 1)
        report?.let {
            // Update or replace today's report (only add once per day at end, update live otherwise)
            if (today != savedDay && savedDay != -1) {
                DataRepository.addReport(it)
                if (it.alertLevel == "orange" || it.alertLevel == "red") {
                    sendAlertNotification(it.alertLevel, it.notes)
                }
            }
        }
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
