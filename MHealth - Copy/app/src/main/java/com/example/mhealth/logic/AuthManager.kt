package com.example.mhealth.logic

import android.content.Context
import com.example.mhealth.logic.db.AnalysisResultEntity
import com.example.mhealth.logic.db.BaselineEntity
import com.example.mhealth.logic.db.DailyFeaturesEntity
import com.example.mhealth.logic.db.MHealthDatabase
import com.example.mhealth.logic.db.UserProfileEntity
import com.example.mhealth.models.UserProfile
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.firestore.FirebaseFirestore
import kotlinx.coroutines.tasks.await
import java.util.UUID

class AuthManager(private val context: Context) {
    private val auth = FirebaseAuth.getInstance()
    private val firestore = FirebaseFirestore.getInstance()

    // Create new account strictly. Fail if email exists.
    suspend fun createUser(email: String, name: String = ""): Result<Boolean> {
        return try {
            // Temporary hardcoded password for testing phase
            val password = "user1234"
            auth.createUserWithEmailAndPassword(email, password).await()
            setupFirestoreProfile(name)
            Result.success(true)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    // Cloud recovery for users who reinstalled the app, or login from a new device.
    // After successful sign-in this device becomes the authoritative sync device by
    // writing its ID to Firestore. The previous device's CloudSyncWorker will then
    // detect the mismatch and stop syncing — preventing data corruption.
    suspend fun signInExistingUser(email: String): Result<UserProfile> {
        return try {
            val password = "user1234"
            auth.signInWithEmailAndPassword(email, password).await()
            val user = auth.currentUser ?: return Result.failure(Exception("No user found"))
            val uid = user.uid

            // ✅ FIX: Claim this device as the active sync device.
            // This is the key step that allows the same email to log in on a new
            // device. The old device's CloudSyncWorker will stop when it detects
            // its local device_id no longer matches active_device_id in Firestore.
            val localDeviceId = getLocalDeviceId()
            firestore.collection("users").document(uid)
                .set(
                    mapOf("active_device_id" to localDeviceId),
                    com.google.firebase.firestore.SetOptions.merge()
                ).await()

            val doc = firestore.collection("users").document(uid).get().await()

            val profile = UserProfile(
                email       = email,
                name        = doc.getString("name") ?: "Recovered User",
                gender      = doc.getString("gender") ?: "",
                dateOfBirth = doc.getString("dateOfBirth") ?: "",
                age         = (doc.getLong("age") ?: 0L).toInt(),
                profession  = doc.getString("profession") ?: "",
                country     = doc.getString("country") ?: ""
            )

            downloadDataToRoom(uid)
            Result.success(profile)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** Upload the full questionnaire-collected profile to the Firestore user document. */
    suspend fun updateFirestoreFullProfile(profile: UserProfile) {
        val uid = auth.currentUser?.uid ?: return
        val update = mapOf(
            "name"        to profile.name,
            "gender"      to profile.gender,
            "dateOfBirth" to profile.dateOfBirth,
            "age"         to profile.age,
            "profession"  to profile.profession,
            "country"     to profile.country
        )
        try {
            firestore.collection("users").document(uid).update(update as Map<String, Any>).await()
        } catch (e: Exception) {
            // Document may not exist yet — use set with merge
            firestore.collection("users").document(uid).set(update, com.google.firebase.firestore.SetOptions.merge()).await()
        }
    }

    private suspend fun setupFirestoreProfile(name: String) {
        val user = auth.currentUser ?: return
        val uid = user.uid
        val docRef = firestore.collection("users").document(uid)
        val localDeviceId = getLocalDeviceId()

        val docSnapshot = docRef.get().await()
        if (docSnapshot.exists()) {
            // Account already exists in Firestore (e.g. user registered on another device).
            // Claim this device as the active sync device and download existing data.
            docRef.set(
                mapOf("active_device_id" to localDeviceId),
                com.google.firebase.firestore.SetOptions.merge()
            ).await()
            downloadDataToRoom(uid)
        } else {
            // Brand-new account — create the Firestore document.
            val patientId = "Patient_${UUID.randomUUID().toString().take(6)}"
            val profileData = hashMapOf(
                "email"            to user.email,
                "name"             to name.ifBlank { "Patient" },
                "patient_id"       to patientId,
                "active_device_id" to localDeviceId,
                "status"           to "Normal",
                "onboarding_date"  to System.currentTimeMillis(),
                "baseline_progress" to 0
            )
            docRef.set(profileData).await()

            val db = MHealthDatabase.getInstance(context)
            db.userProfileDao().upsert(
                UserProfileEntity(
                    userId = user.email ?: uid,
                    currentStatus = "Collecting"
                )
            )
        }
    }

    private fun getLocalDeviceId(): String {
        val prefs = context.getSharedPreferences("mhealth_prefs", Context.MODE_PRIVATE)
        var deviceId = prefs.getString("device_id", null)
        if (deviceId == null) {
            deviceId = UUID.randomUUID().toString()
            prefs.edit().putString("device_id", deviceId).apply()
        }
        return deviceId
    }

    private suspend fun downloadDataToRoom(uid: String) {
        val db = MHealthDatabase.getInstance(context)
        val emailId = auth.currentUser?.email ?: uid

        // 1. Fetch Profile metadata
        val profileDoc = firestore.collection("users").document(uid).get().await()
        val status            = profileDoc.getString("status") ?: "Monitoring"
        val onboardingDateMs  = profileDoc.getLong("onboarding_date") ?: System.currentTimeMillis()
        val isReady           = profileDoc.getBoolean("baseline_ready") ?: false
        val baselineProgress  = profileDoc.getLong("baseline_progress")?.toInt() ?: 0

        db.userProfileDao().upsert(
            UserProfileEntity(
                userId        = emailId,
                onboardingDate = onboardingDateMs.toString(),
                baselineReady = isReady,
                currentStatus = status
            )
        )

        // 2. Fetch Baselines
        val baselinesSnapshot = firestore.collection("users").document(uid).collection("baseline").get().await()
        val baselineEntities = mutableListOf<BaselineEntity>()
        for (doc in baselinesSnapshot.documents) {
            val featureName = doc.getString("featureName") ?: continue
            val mean = doc.getDouble("baselineValue")?.toFloat() ?: 0f
            val std  = doc.getDouble("stdDeviation")?.toFloat()  ?: 0f
            baselineEntities.add(
                BaselineEntity(
                    userId        = emailId,
                    featureName   = featureName,
                    baselineValue = mean,
                    stdDeviation  = std,
                    baselineStart = doc.getString("baselineStart") ?: "",
                    baselineEnd   = doc.getString("baselineEnd")   ?: ""
                )
            )
        }
        if (baselineEntities.isNotEmpty()) db.baselineDao().insertAll(baselineEntities)

        // 3. Restore ALL daily features from the cloud
        // Note: CloudSyncWorker writes to 'daily_features'. We fetch from there.
        val featuresSnapshot = firestore.collection("users").document(uid).collection("daily_features").get().await()
        val featureEntities = mutableListOf<DailyFeaturesEntity>()
        for (doc in featuresSnapshot.documents) {
            try {
                featureEntities.add(
                    DailyFeaturesEntity(
                        userId = emailId,
                        date = doc.id,
                        screenTimeHours = doc.getDouble("screenTimeHours")?.toFloat() ?: 0f,
                        unlockCount = doc.getDouble("unlockCount")?.toFloat() ?: 0f,
                        appLaunchCount = doc.getDouble("appLaunchCount")?.toFloat() ?: 0f,
                        notificationsToday = doc.getDouble("notificationsToday")?.toFloat() ?: 0f,
                        socialAppRatio = doc.getDouble("socialAppRatio")?.toFloat() ?: 0f,
                        callsPerDay = doc.getDouble("callsPerDay")?.toFloat() ?: 0f,
                        callDurationMinutes = doc.getDouble("callDurationMinutes")?.toFloat() ?: 0f,
                        uniqueContacts = doc.getDouble("uniqueContacts")?.toFloat() ?: 0f,
                        conversationFrequency = doc.getDouble("conversationFrequency")?.toFloat() ?: 0f,
                        dailyDisplacementKm = doc.getDouble("dailyDisplacementKm")?.toFloat() ?: 0f,
                        locationEntropy = doc.getDouble("locationEntropy")?.toFloat() ?: 0f,
                        homeTimeRatio = doc.getDouble("homeTimeRatio")?.toFloat() ?: 0f,
                        placesVisited = 0f,  // Deprecated — no longer tracked
                        wakeTimeHour = doc.getDouble("wakeTimeHour")?.toFloat() ?: 0f,
                        sleepTimeHour = doc.getDouble("sleepTimeHour")?.toFloat() ?: 0f,
                        sleepDurationHours = doc.getDouble("sleepDurationHours")?.toFloat() ?: 0f,
                        darkDurationHours = doc.getDouble("darkDurationHours")?.toFloat() ?: 0f,
                        chargeDurationHours = doc.getDouble("chargeDurationHours")?.toFloat() ?: 0f,
                        memoryUsagePercent = doc.getDouble("memoryUsagePercent")?.toFloat() ?: 0f,
                        networkWifiMB = doc.getDouble("networkWifiMB")?.toFloat() ?: 0f,
                        networkMobileMB = doc.getDouble("networkMobileMB")?.toFloat() ?: 0f,
                        downloadsToday = doc.getDouble("downloadsToday")?.toFloat() ?: 0f,
                        storageUsedGB = doc.getDouble("storageUsedGB")?.toFloat() ?: 0f,
                        appUninstallsToday = doc.getDouble("appUninstallsToday")?.toFloat() ?: 0f,
                        upiTransactionsToday = doc.getDouble("upiTransactionsToday")?.toFloat() ?: 0f,
                        totalAppsCount = doc.getDouble("totalAppsCount")?.toFloat() ?: 0f,
                        backgroundAudioHours = doc.getDouble("backgroundAudioHours")?.toFloat() ?: 0f,
                        mediaCountToday = doc.getDouble("mediaCountToday")?.toFloat() ?: 0f,
                        appInstallsToday = doc.getDouble("appInstallsToday")?.toFloat() ?: 0f,
                        calendarEventsToday = doc.getDouble("calendarEventsToday")?.toFloat() ?: 0f,
                        dailySteps = doc.getDouble("dailySteps")?.toFloat() ?: 0f,
                        // JSON BREAKDOWNS (Essential for Digital DNA and L2 Analysis)
                        appBreakdownJson = doc.getString("appBreakdownJson") ?: "{}",
                        notificationBreakdownJson = doc.getString("notificationBreakdownJson") ?: "{}",
                        appLaunchesBreakdownJson = doc.getString("appLaunchesBreakdownJson") ?: "{}",
                        bgAudioBreakdownJson = doc.getString("bgAudioBreakdownJson") ?: "{}",
                        syncedToCloud = true,
                        isSimulated = false
                    )
                )
            } catch (e: Exception) {
                // Skip faulty docs
            }
        }
        if (featureEntities.isNotEmpty()) {
            db.dailyFeaturesDao().insertAll(featureEntities)
        }

        // 4. Restore ALL analysis results from the cloud
        val resultsSnapshot = firestore.collection("users").document(uid).collection("results").get().await()
        for (doc in resultsSnapshot.documents) {
            val date = doc.getString("date") ?: doc.id
            val resultEntity = AnalysisResultEntity(
                userId           = emailId,
                date             = date,
                anomalyDetected  = doc.getBoolean("anomaly_detected") ?: false,
                anomalyScore     = doc.getDouble("anomaly_score")?.toFloat() ?: 0f,
                anomalyMessage   = doc.getString("anomaly_message")   ?: "",
                alertLevel       = doc.getString("alert_level")       ?: "green",
                sustainedDays    = (doc.getLong("sustained_days") ?: 0L).toInt(),
                prototypeMatch   = doc.getString("prototype_match")   ?: "Normal",
                matchMessage     = doc.getString("match_message")     ?: "",
                prototypeConfidence = ((doc.getDouble("prototype_confidence") ?: 0.0) / 10.0).coerceIn(0.0, 1.0).toFloat(),
                gateResults      = doc.getString("gate_results")      ?: "{}",
                syncedToCloud    = true,
                // L2 Digital DNA fields
                l2Modifier       = doc.getDouble("l2_modifier")?.toFloat() ?: 1.0f,
                coherence        = doc.getDouble("coherence")?.toFloat()   ?: 0f,
                rhythmDissolution = doc.getDouble("rhythm_dissolution")?.toFloat() ?: 0f,
                sessionIncoherence = doc.getDouble("session_incoherence")?.toFloat() ?: 0f,
                effectiveScore   = doc.getDouble("effective_score")?.toFloat() ?: 0f,
                evidenceAccumulated = doc.getDouble("evidence_accumulated")?.toFloat() ?: 0f,
                patternType      = doc.getString("pattern_type") ?: "stable",
                flaggedFeatures  = doc.getString("flagged_features") ?: "[]"
            )
            db.analysisResultDao().insert(resultEntity)
        }
    }
}
