package com.example.mhealth.services

import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.example.mhealth.logic.db.MHealthDatabase
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.firestore.FirebaseFirestore
import kotlinx.coroutines.tasks.await

class CloudSyncWorker(appContext: Context, workerParams: WorkerParameters) :
    CoroutineWorker(appContext, workerParams) {

    companion object {
        private const val TAG = "MHealth.CloudSync"
        private const val SYNC_WORK_NAME = "PeriodicCloudSync"

        fun schedulePeriodic(context: Context) {
            val constraints = androidx.work.Constraints.Builder()
                .setRequiredNetworkType(androidx.work.NetworkType.CONNECTED)
                .build()

            val periodicWork = androidx.work.PeriodicWorkRequestBuilder<CloudSyncWorker>(
                4, java.util.concurrent.TimeUnit.HOURS
            ).setConstraints(constraints).build()

            androidx.work.WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                SYNC_WORK_NAME,
                androidx.work.ExistingPeriodicWorkPolicy.UPDATE,
                periodicWork
            )
            Log.d(TAG, "Scheduled periodic CloudSyncWorker to run every 4 hours")

            // Also trigger an immediate sync right now so the user doesn't have to wait 4 hours
            val immediateWork = androidx.work.OneTimeWorkRequestBuilder<CloudSyncWorker>()
                .setConstraints(constraints)
                .build()
            androidx.work.WorkManager.getInstance(context).enqueueUniqueWork(
                SYNC_WORK_NAME + "_Immediate",
                androidx.work.ExistingWorkPolicy.REPLACE,
                immediateWork
            )
        }
    }

    override suspend fun doWork(): Result {
        val auth = FirebaseAuth.getInstance()
        val user = auth.currentUser ?: run {
            Log.e(TAG, "No authenticated user, skipping sync")
            return Result.failure()
        }
        val uid = user.uid

        val firestore = FirebaseFirestore.getInstance()
        val db = MHealthDatabase.getInstance(applicationContext)

        try {
            Log.d(TAG, "=== CloudSyncWorker started for user: ${user.email} (uid: $uid) ===")

            // 1. Validate Active Device ID
            val prefs = applicationContext.getSharedPreferences("mhealth_prefs", Context.MODE_PRIVATE)
            val localDeviceId = prefs.getString("device_id", null)
            if (localDeviceId == null) {
                Log.e(TAG, "No local device ID found, skipping sync")
                return Result.failure()
            }
            Log.d(TAG, "Local device ID: $localDeviceId")

            val profileDoc = firestore.collection("users").document(uid).get().await()
            val activeDeviceId = profileDoc.getString("active_device_id")
            Log.d(TAG, "Active device ID in Firestore: $activeDeviceId")

            if (localDeviceId != activeDeviceId) {
                Log.w(TAG, "Device ID mismatch! Updating active_device_id in Firestore to match local device instead of skipping sync.")
                firestore.collection("users").document(uid)
                    .set(mapOf("active_device_id" to localDeviceId), com.google.firebase.firestore.SetOptions.merge()).await()
            }
            Log.d(TAG, "Device ID validated successfully")

            // 2. Sync Daily Features
            // IMPORTANT: Room stores all data keyed by email, not Firebase UID.
            // Using `uid` here would return 0 rows and silently skip all syncing.
            val email = user.email ?: run {
                Log.e(TAG, "No email found for user, skipping sync")
                return Result.failure()
            }

            val unsyncedFeatures = db.dailyFeaturesDao().getUnsynced(email)
            Log.d(TAG, "Found ${unsyncedFeatures.size} unsynced daily features")

            // FIXED: Write to 'daily_features' (was incorrectly 'daily_data')
            val dailyFeaturesRef = firestore.collection("users").document(uid).collection("daily_features")

            var syncedFeaturesCount = 0
            for (feature in unsyncedFeatures) {
                if (feature.isSimulated) {
                    db.dailyFeaturesDao().markSynced(feature.id)
                    continue
                }
                try {
                    val dataMap = hashMapOf<String, Any>(
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
                        "networkMobileMB" to feature.networkMobileMB,
                        "downloadsToday" to feature.downloadsToday,
                        "storageUsedGB" to feature.storageUsedGB,
                        "appUninstallsToday" to feature.appUninstallsToday,
                        "upiTransactionsToday" to feature.upiTransactionsToday,
                        "totalAppsCount" to feature.totalAppsCount,
                        "dailySteps" to feature.dailySteps,
                        // FIXED: Added missing fields that MonitoringService writes
                        "backgroundAudioHours" to feature.backgroundAudioHours,
                        "mediaCountToday" to feature.mediaCountToday,
                        "appInstallsToday" to feature.appInstallsToday,
                        "calendarEventsToday" to feature.calendarEventsToday,
                        // FIXED: Store breakdowns as JSON strings (matching MonitoringService format)
                        // Previously parsed to Map with 'appBreakdown' key — now consistent
                        "appBreakdownJson" to feature.appBreakdownJson,
                        "notificationBreakdownJson" to feature.notificationBreakdownJson,
                        "appLaunchesBreakdownJson" to feature.appLaunchesBreakdownJson,
                        "bgAudioBreakdownJson" to feature.bgAudioBreakdownJson
                    )

                    dailyFeaturesRef.document(feature.date).set(dataMap).await()
                    db.dailyFeaturesDao().markSynced(feature.id)
                    syncedFeaturesCount++
                    Log.d(TAG, "Synced daily feature for date: ${feature.date}")
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to sync feature for ${feature.date}: ${e.message}", e)
                }
            }
            Log.d(TAG, "Successfully synced $syncedFeaturesCount/${unsyncedFeatures.size} daily features")

            // 3. Sync Analysis Results
            // efficiently force-sync all results to ensure Firestore has the anomaly_score
            db.analysisResultDao().resetSyncFlags(email)
            val unsyncedResults = db.analysisResultDao().getUnsynced(email)
            Log.d(TAG, "Found ${unsyncedResults.size} unsynced analysis results")

            val resultsRef = firestore.collection("users").document(uid).collection("results")

            var syncedResultsCount = 0
            for (result in unsyncedResults) {
                try {
                    val resultMap = hashMapOf<String, Any>(
                        "date" to result.date,
                        "anomaly_detected" to result.anomalyDetected,
                        "anomaly_score" to result.anomalyScore,
                        "anomaly_message" to result.anomalyMessage,
                        "alert_level" to result.alertLevel,
                        "sustained_days" to result.sustainedDays,
                        "prototype_match" to result.prototypeMatch,
                        "match_message" to result.matchMessage,
                        "prototype_confidence" to result.prototypeConfidence,
                        "gate_results" to result.gateResults
                    )

                    resultsRef.document(result.date).set(resultMap).await()
                    db.analysisResultDao().markSynced(result.id)
                    syncedResultsCount++
                    Log.d(TAG, "✓ Synced result: ${result.date} | score=${result.anomalyScore} | alert=${result.alertLevel} | prototype=${result.prototypeMatch} | confidence=${result.prototypeConfidence}")
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to sync result for ${result.date}: ${e.message}", e)
                }
            }
            Log.d(TAG, "Successfully synced $syncedResultsCount/${unsyncedResults.size} analysis results")

            // 4. Update total recorded days (baseline progress)
            // FIXED: Never decrease baseline_progress — protects against Room destructive
            // migration wiping local data and overwriting the Firestore count with 0.
            val localProgress = db.dailyFeaturesDao().count(email)
            val firestoreProgress = profileDoc.getLong("baseline_progress")?.toInt() ?: 0
            val progressToSet = maxOf(localProgress, firestoreProgress)
            Log.d(TAG, "baseline_progress: local=$localProgress, firestore=$firestoreProgress, writing=$progressToSet")
            firestore.collection("users").document(uid)
                .set(mapOf("baseline_progress" to progressToSet), com.google.firebase.firestore.SetOptions.merge()).await()

            Log.d(TAG, "=== CloudSyncWorker completed successfully ===")
            return Result.success()

        } catch (e: Exception) {
            Log.e(TAG, "CloudSyncWorker failed: ${e.message}", e)
            return Result.retry()
        }
    }
}
