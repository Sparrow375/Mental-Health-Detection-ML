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
                androidx.work.ExistingPeriodicWorkPolicy.KEEP,
                periodicWork
            )
            Log.d(TAG, "Scheduled periodic CloudSyncWorker to run every 4 hours")
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

            val dailyDataRef = firestore.collection("users").document(uid).collection("daily_data")

            fun jsonToMap(jsonStr: String): Map<String, Any> {
                val map = mutableMapOf<String, Any>()
                try {
                    val obj = org.json.JSONObject(jsonStr)
                    for (key in obj.keys()) {
                        map[key] = obj.get(key)
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to parse JSON: $jsonStr", e)
                }
                return map
            }

            var syncedFeaturesCount = 0
            for (feature in unsyncedFeatures) {
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

                        "appBreakdown" to jsonToMap(feature.appBreakdownJson),
                        "notificationBreakdown" to jsonToMap(feature.notificationBreakdownJson),
                        "appLaunchesBreakdown" to jsonToMap(feature.appLaunchesBreakdownJson)
                    )

                    dailyDataRef.document(feature.date).set(dataMap).await()
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
                    val resultMap = hashMapOf(
                        "anomaly_detected" to result.anomalyDetected,
                        "anomaly_message" to result.anomalyMessage,
                        "prototype_match" to result.prototypeMatch,
                        "match_message" to result.matchMessage,
                        "anomaly_score" to result.anomalyScore,
                        "alert_level" to result.alertLevel,
                        "date" to result.date
                    )

                    resultsRef.document(result.date).set(resultMap).await()
                    db.analysisResultDao().markSynced(result.id)
                    syncedResultsCount++
                    Log.d(TAG, "✓ Synced analysis result for date: ${result.date} | anomaly_score: ${result.anomalyScore} | alert: ${result.alertLevel}")
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to sync result for ${result.date}: ${e.message}", e)
                }
            }
            Log.d(TAG, "Successfully synced $syncedResultsCount/${unsyncedResults.size} analysis results")

            // 4. Update total recorded days (baseline progress)
            val baselineProgress = db.dailyFeaturesDao().count(email)
            Log.d(TAG, "Updating baseline_progress to: $baselineProgress")
            // Use set(merge=true) instead of update() — update() throws if the user doc
            // doesn't exist yet (e.g. just after first-time registration).
            firestore.collection("users").document(uid)
                .set(mapOf("baseline_progress" to baselineProgress), com.google.firebase.firestore.SetOptions.merge()).await()

            Log.d(TAG, "=== CloudSyncWorker completed successfully ===")
            return Result.success()

        } catch (e: Exception) {
            Log.e(TAG, "CloudSyncWorker failed: ${e.message}", e)
            return Result.retry()
        }
    }
}
