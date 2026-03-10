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

class MonitoringService : Service() {

    private lateinit var dataCollector: DataCollector
    private var detector: AnomalyDetector? = null
    private val timer = Timer()
    private val baselineDaysRequired = 28
    private val collectedDailyVectors = mutableListOf<PersonalityVector>()

    override fun onCreate() {
        super.onCreate()
        dataCollector = DataCollector(this)
        startForegroundService()
        scheduleMonitoring()
    }

    private fun startForegroundService() {
        val channelId = "mhealth_monitoring"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(channelId, "MHealth Monitoring", NotificationManager.IMPORTANCE_LOW)
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }

        val notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("MHealth Active")
            .setContentText("Collecting daily baseline data...")
            .setSmallIcon(android.R.drawable.ic_menu_info_details)
            .setContentIntent(PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE))
            .build()

        startForeground(1, notification)
    }

    private fun scheduleMonitoring() {
        // Collect data every 3 hours, but we focus on the 00:00 - Now window for the "daily" snapshot
        timer.schedule(object : TimerTask() {
            override fun run() {
                val currentData = dataCollector.collectDailyData()
                DataRepository.updateLatestVector(currentData)

                if (DataRepository.isBuildingBaseline.value) {
                    processBaselineBuilding(currentData)
                } else {
                    processAnomalyDetection(currentData)
                }
            }
        }, 0, 3 * 60 * 60 * 1000)
    }

    private fun processBaselineBuilding(currentData: PersonalityVector) {
        // In a real app, we'd store these in a DB and only add one per day.
        // For dev build, we'll simulate progress.
        val progress = DataRepository.baselineProgress.value
        
        // Check if it's a new day to increment progress (simplified for demo)
        // If we haven't finished building baseline:
        if (progress < baselineDaysRequired) {
            collectedDailyVectors.add(currentData)
            // Normally you'd update this once per day at 23:59
            // DataRepository.updateBaselineProgress(collectedDailyVectors.size)
        }

        if (progress >= baselineDaysRequired) {
            val baseline = calculateBaseline(collectedDailyVectors)
            detector = AnomalyDetector(baseline)
            DataRepository.setBaseline(baseline)
        }
    }

    private fun calculateBaseline(vectors: List<PersonalityVector>): PersonalityVector {
        if (vectors.isEmpty()) return PersonalityVector()
        
        // Average all features to create the baseline personality profile
        val avgScreenTime = vectors.map { it.screenTimeHours }.average().toFloat()
        val avgUnlocks = vectors.map { it.unlockCount }.average().toFloat()
        val avgSocial = vectors.map { it.socialAppRatio }.average().toFloat()
        val avgCalls = vectors.map { it.callsPerDay }.average().toFloat()
        val avgTexts = vectors.map { it.textsPerDay }.average().toFloat()
        val avgContacts = vectors.map { it.uniqueContacts }.average().toFloat()
        
        // Calculate variances for the detector
        val variances = mutableMapOf<String, Float>()
        variances["screenTimeHours"] = calculateVariance(vectors.map { it.screenTimeHours }, avgScreenTime)
        variances["unlockCount"] = calculateVariance(vectors.map { it.unlockCount }, avgUnlocks)
        // ... add more variances
        
        return PersonalityVector(
            screenTimeHours = avgScreenTime,
            unlockCount = avgUnlocks,
            socialAppRatio = avgSocial,
            callsPerDay = avgCalls,
            textsPerDay = avgTexts,
            uniqueContacts = avgContacts,
            variances = variances
        )
    }

    private fun calculateVariance(values: List<Float>, mean: Float): Float {
        if (values.size < 2) return 1.0f
        return values.map { (it - mean) * (it - mean) }.average().toFloat()
    }

    private fun processAnomalyDetection(currentData: PersonalityVector) {
        val report = detector?.analyze(currentData, 1) // Day count relative to baseline end
        report?.let {
            DataRepository.addReport(it)
            if (it.alertLevel == "orange" || it.alertLevel == "red") {
                sendAlertNotification(it.alertLevel, it.notes)
            }
        }
    }

    private fun sendAlertNotification(level: String, notes: String) {
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val notification = NotificationCompat.Builder(this, "mhealth_monitoring")
            .setContentTitle("Anomaly Detected: ${level.uppercase()}")
            .setContentText(notes)
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .build()
        notificationManager.notify(2, notification)
    }

    override fun onBind(intent: Intent?): IBinder? = null
    override fun onDestroy() { timer.cancel(); super.onDestroy() }
}
