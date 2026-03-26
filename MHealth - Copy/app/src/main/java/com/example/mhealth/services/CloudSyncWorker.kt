package com.example.mhealth.services

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.example.mhealth.logic.db.MHealthDatabase
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.firestore.FirebaseFirestore
import kotlinx.coroutines.tasks.await

class CloudSyncWorker(appContext: Context, workerParams: WorkerParameters) :
    CoroutineWorker(appContext, workerParams) {

    override suspend fun doWork(): Result {
        val auth = FirebaseAuth.getInstance()
        val user = auth.currentUser ?: return Result.failure()
        val uid = user.uid

        val firestore = FirebaseFirestore.getInstance()
        val db = MHealthDatabase.getInstance(applicationContext)

        try {
            // 1. Validate Active Device ID
            val prefs = applicationContext.getSharedPreferences("mhealth_prefs", Context.MODE_PRIVATE)
            val localDeviceId = prefs.getString("device_id", null) ?: return Result.failure()

            val profileDoc = firestore.collection("users").document(uid).get().await()
            val activeDeviceId = profileDoc.getString("active_device_id")

            if (localDeviceId != activeDeviceId) {
                // This phone is no longer the active device. Stop syncing to prevent corruption.
                return Result.failure()
            }

            // 2. Sync Daily Features
            val unsyncedFeatures = db.dailyFeaturesDao().getUnsynced(uid)
            val dailyDataRef = firestore.collection("users").document(uid).collection("daily_data")
            
            for (feature in unsyncedFeatures) {
                val dataMap = hashMapOf(
                    "date" to feature.date,
                    "screenTimeHours" to feature.screenTimeHours,
                    "unlockCount" to feature.unlockCount,
                    "appLaunchCount" to feature.appLaunchCount,
                    "notificationsToday" to feature.notificationsToday,
                    "socialAppRatio" to feature.socialAppRatio,
                    "callsPerDay" to feature.callsPerDay,
                    "callDurationMinutes" to feature.callDurationMinutes,
                    "uniqueContacts" to feature.uniqueContacts,
                    "conversationFrequency" to feature.conversationFrequency,
                    "dailyDisplacementKm" to feature.dailyDisplacementKm,
                    "locationEntropy" to feature.locationEntropy,
                    "homeTimeRatio" to feature.homeTimeRatio,
                    "placesVisited" to feature.placesVisited,
                    "wakeTimeHour" to feature.wakeTimeHour,
                    "sleepTimeHour" to feature.sleepTimeHour,
                    "sleepDurationHours" to feature.sleepDurationHours,
                    "darkDurationHours" to feature.darkDurationHours,
                    "chargeDurationHours" to feature.chargeDurationHours,
                    "memoryUsagePercent" to feature.memoryUsagePercent,
                    "networkWifiMB" to feature.networkWifiMB,
                    "networkMobileMB" to feature.networkMobileMB
                )
                
                dailyDataRef.document(feature.date).set(dataMap).await()
                db.dailyFeaturesDao().markSynced(feature.id)
            }

            // 3. Sync Analysis Results
            val unsyncedResults = db.analysisResultDao().getUnsynced(uid)
            val resultsRef = firestore.collection("users").document(uid).collection("results")

            for (result in unsyncedResults) {
                val resultMap = hashMapOf(
                    "anomaly_detected" to result.anomalyDetected,
                    "anomaly_message" to result.anomalyMessage,
                    "prototype_match" to result.prototypeMatch,
                    "match_message" to result.matchMessage,
                    "date" to result.date
                )
                
                resultsRef.document(result.date).set(resultMap).await()
                db.analysisResultDao().markSynced(result.id)
            }

            return Result.success()

        } catch (e: Exception) {
            e.printStackTrace()
            return Result.retry()
        }
    }
}
