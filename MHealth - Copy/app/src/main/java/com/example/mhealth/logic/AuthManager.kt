package com.example.mhealth.logic

import android.content.Context
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.FirebaseAuthInvalidUserException
import com.google.firebase.firestore.FirebaseFirestore
import com.example.mhealth.logic.db.MHealthDatabase
import com.example.mhealth.logic.db.BaselineEntity
import com.example.mhealth.logic.db.UserProfileEntity
import kotlinx.coroutines.tasks.await
import java.util.UUID

class AuthManager(private val context: Context) {
    private val auth = FirebaseAuth.getInstance()
    private val firestore = FirebaseFirestore.getInstance()

    // Login or Create with Email + Default Password
    suspend fun signInOrCreateUser(email: String, name: String = ""): Result<Boolean> {
        return try {
            // TODO: Temporary hardcoded password for testing phase. Must be replaced with OTP/Email link before production.
            val password = "user1234"
            try {
                // Try to create the user first
                auth.createUserWithEmailAndPassword(email, password).await()
            } catch (e: Exception) {
                // If it fails (e.g., account already exists), fall back to sign in
                auth.signInWithEmailAndPassword(email, password).await()
            }
            // After successful auth, fetch or setup Firestore profile
            setupFirestoreProfile(name)
            Result.success(true)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    private suspend fun setupFirestoreProfile(name: String) {
        val user = auth.currentUser ?: return
        val uid = user.uid
        val docRef = firestore.collection("users").document(uid)
        
        val docSnapshot = docRef.get().await()
        if (docSnapshot.exists()) {
            val existingDeviceId = docSnapshot.getString("active_device_id")
            val localDeviceId = getLocalDeviceId()
            
            // Scenario 2: active device id exists and differs
            if (existingDeviceId != null && existingDeviceId != localDeviceId) {
                // For now, override it
            }
            
            // Update active_device_id
            docRef.update("active_device_id", localDeviceId).await()
            
            // Download existing data to Room
            downloadDataToRoom(uid)
            
        } else {
            // Create new profile
            val localDeviceId = getLocalDeviceId()
            val patientId = "Patient_${UUID.randomUUID().toString().take(6)}"
            val profileData = hashMapOf(
                "email" to user.email,
                "name" to name.ifBlank { "Patient" },
                "patient_id" to patientId,
                "active_device_id" to localDeviceId,
                "status" to "Normal",
                "onboarding_date" to System.currentTimeMillis()
            )
            docRef.set(profileData).await()
            
            val db = MHealthDatabase.getInstance(context)
            db.userProfileDao().upsert(
                UserProfileEntity(
                    userId = uid,
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
        
        // 1. Fetch Profile
        val profileDoc = firestore.collection("users").document(uid).get().await()
        val status = profileDoc.getString("status") ?: "Monitoring"
        val onboardingDateMs = profileDoc.getLong("onboarding_date") ?: System.currentTimeMillis()
        val isReady = profileDoc.getBoolean("baseline_ready") ?: false
        
        db.userProfileDao().upsert(
            UserProfileEntity(
                userId = uid,
                onboardingDate = onboardingDateMs.toString(),
                baselineReady = isReady,
                currentStatus = status
            )
        )
        
        // 2. Fetch Baselines
        val baselinesSnapshot = firestore.collection("users").document(uid).collection("baseline").get().await()
        val entities = mutableListOf<BaselineEntity>()
        for (doc in baselinesSnapshot.documents) {
            val featureName = doc.getString("featureName") ?: continue
            val mean = doc.getDouble("baselineValue")?.toFloat() ?: 0f
            val std = doc.getDouble("stdDeviation")?.toFloat() ?: 0f
            
            entities.add(
                BaselineEntity(
                    userId = uid,
                    featureName = featureName,
                    baselineValue = mean,
                    stdDeviation = std,
                    baselineStart = doc.getString("baselineStart") ?: "",
                    baselineEnd = doc.getString("baselineEnd") ?: ""
                )
            )
        }
        if (entities.isNotEmpty()) {
            db.baselineDao().insertAll(entities)
        }
    }
}
