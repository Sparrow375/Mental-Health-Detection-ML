package com.swasthiti.logic.db

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Stores onboarding and monitoring lifecycle state per user.
 * currentStatus values: "Collecting" | "BaselineBuilding" | "Monitoring" | "Flagged"
 */
@Entity(tableName = "user_profile_db")
data class UserProfileEntity(
    @PrimaryKey val userId: String,
    val onboardingDate: String = "",          // "YYYY-MM-DD"
    val baselineReady: Boolean = false,
    val baselineDays: Int = 28,
    val currentStatus: String = "Collecting", // "Collecting"|"Monitoring"|"Flagged"
    val baselineContaminated: Boolean = false,
    val updatedAt: Long = System.currentTimeMillis()
)

