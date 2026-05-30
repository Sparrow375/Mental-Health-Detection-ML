package com.example.mhealth.services

import android.content.Context
import android.util.Log
import com.example.mhealth.logic.db.MHealthDatabase
import com.example.mhealth.models.PersonalityVector
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.firestore.FirebaseFirestore
import kotlinx.coroutines.tasks.await

object FirebaseSyncHelper {
    private const val TAG = "MHealth.FirebaseSync"

    suspend fun syncBaseline(context: Context, baseline: PersonalityVector, today: Int) {
        val uid = FirebaseAuth.getInstance().currentUser?.uid
        if (uid != null) {
            try {
                val firestore = FirebaseFirestore.getInstance()
                firestore.collection("users").document(uid)
                    .set(mapOf("baseline_ready" to true), com.google.firebase.firestore.SetOptions.merge()).await()
                
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
                Log.d(TAG, "Baseline synced to Firebase successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Error syncing baseline to Firebase", e)
            }
        }
    }

    suspend fun syncUnstagedDailyFeatures(context: Context, email: String) {
        val uid = FirebaseAuth.getInstance().currentUser?.uid ?: return
        val db = MHealthDatabase.getInstance(context)
        
        try {
            val unsynced = db.dailyFeaturesDao().getUnsynced(email)
            if (unsynced.isEmpty()) return
            
            val firestore = FirebaseFirestore.getInstance()
            val collectionRef = firestore.collection("users").document(uid).collection("daily_features")
            
            for (entity in unsynced) {
                if (entity.isSimulated) {
                    db.dailyFeaturesDao().markSynced(entity.id)
                    continue
                }
                
                val data = hashMapOf(
                    "date" to entity.date,
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
                    "wakeTimeHour" to entity.wakeTimeHour,
                    "sleepTimeHour" to entity.sleepTimeHour,
                    "sleepDurationHours" to entity.sleepDurationHours,
                    "dailyStepCount" to entity.dailyStepCount,
                    "activeMinutes" to entity.activeMinutes,
                    "keystrokeSpeed" to entity.keystrokeSpeed,
                    "backspaceRatio" to entity.backspaceRatio,
                    "scrollVelocity" to entity.scrollVelocity,
                    "daylightExposureMinutes" to entity.daylightExposureMinutes,
                    "chargeRegularity" to entity.chargeRegularity,
                    "chargeDurationHours" to entity.chargeDurationHours,
                    "upiTransactionsToday" to entity.upiTransactionsToday,
                    "appUninstallsToday" to entity.appUninstallsToday,
                    "appInstallsToday" to entity.appInstallsToday,
                    "calendarEventsToday" to entity.calendarEventsToday,
                    "mediaCountToday" to entity.mediaCountToday,
                    "downloadsToday" to entity.downloadsToday,
                    "musicTimeMinutes" to entity.musicTimeMinutes,
                    "appBreakdownJson" to entity.appBreakdownJson,
                    "notificationBreakdownJson" to entity.notificationBreakdownJson,
                    "appLaunchesBreakdownJson" to entity.appLaunchesBreakdownJson,
                    "bgAudioBreakdownJson" to entity.bgAudioBreakdownJson
                )
                
                collectionRef.document(entity.date).set(data).await()
                db.dailyFeaturesDao().markSynced(entity.id)
            }

            // Sync DNA profile
            try {
                val dnaEntity = db.personDnaDao().getByUserId(email)
                if (dnaEntity != null) {
                    firestore.collection("users").document(uid)
                        .collection("dna_profile").document("s1_profile")
                        .set(mapOf(
                            "dna_json" to dnaEntity.dna_json,
                            "updated_at" to System.currentTimeMillis()
                        )).await()
                }
            } catch (e: Exception) {
                Log.w(TAG, "DNA profile sync failed: ${e.message}")
            }

            // Sync notification events (last 7 days)
            try {
                val sevenDaysAgo = System.currentTimeMillis() - 7 * 24 * 3600_000L
                val recentNotifs = db.notificationEventDao().getAll()
                    .filter { it.arrival_timestamp >= sevenDaysAgo }
                if (recentNotifs.isNotEmpty()) {
                    val notifBatch = firestore.collection("users").document(uid).collection("notification_events")
                    for (ne in recentNotifs.takeLast(200)) {
                        notifBatch.document(ne.event_id).set(hashMapOf(
                            "app_package" to ne.app_package,
                            "arrival_timestamp" to ne.arrival_timestamp,
                            "action" to ne.action,
                            "tap_latency_min" to (ne.tap_latency_min ?: -1f),
                            "date" to ne.date
                        )).await()
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Notification events sync failed: ${e.message}")
            }

            // Sync app sessions (last 7 days)
            try {
                val sevenDaysAgoMs = System.currentTimeMillis() - 7 * 24 * 3600_000L
                val recentSessions = db.appSessionDao().getAll()
                    .filter { it.open_timestamp >= sevenDaysAgoMs }
                if (recentSessions.isNotEmpty()) {
                    val sessionBatch = firestore.collection("users").document(uid).collection("app_sessions")
                    for (s in recentSessions.takeLast(200)) {
                        sessionBatch.document(s.session_id).set(hashMapOf(
                            "app_package" to s.app_package,
                            "open_timestamp" to s.open_timestamp,
                            "close_timestamp" to s.close_timestamp,
                            "trigger" to s.trigger,
                            "interaction_count" to s.interaction_count,
                            "date" to s.date
                        )).await()
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "App sessions sync failed: ${e.message}")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to sync features to Firebase: ${e.message}", e)
        }
    }
}
