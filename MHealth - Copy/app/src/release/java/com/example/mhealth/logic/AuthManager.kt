package com.example.mhealth.logic

import android.content.Context
import com.example.mhealth.logic.db.MHealthDatabase
import com.example.mhealth.logic.db.UserProfileEntity
import com.example.mhealth.models.UserProfile

class AuthManager(private val context: Context) {

    // Local-only account creation. Strict offline database insert.
    suspend fun createUser(email: String, name: String = ""): Result<Boolean> {
        return try {
            val db = MHealthDatabase.getInstance(context)
            
            // Check if profile already exists locally
            val existing = db.userProfileDao().getProfile(email)
            if (existing != null) {
                return Result.failure(Exception("An account with this email already exists on this device."))
            }

            // Create local-only user profile
            val localProfile = UserProfileEntity(
                userId = email,
                onboardingDate = System.currentTimeMillis().toString(),
                baselineReady = false,
                currentStatus = "Collecting"
            )
            db.userProfileDao().upsert(localProfile)
            Result.success(true)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    // Local-only sign-in for users with data already stored in local Room DB.
    suspend fun signInExistingUser(email: String): Result<UserProfile> {
        return try {
            val db = MHealthDatabase.getInstance(context)
            val entity = db.userProfileDao().getProfile(email)
                ?: return Result.failure(Exception("No account found for this email on this device."))

            // Fetch name or return local patient fallback
            val profileName = if (entity.currentStatus == "Collecting") "Lumen Patient" else "User"

            val profile = UserProfile(
                email       = email,
                name        = profileName,
                gender      = "",
                dateOfBirth = "",
                age         = 0,
                profession  = "",
                country     = ""
            )
            Result.success(profile)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /** No-op in offline Release build. Profile updates are saved locally. */
    suspend fun updateFirestoreFullProfile(profile: UserProfile) {
        // No-op offline stub.
    }
}
