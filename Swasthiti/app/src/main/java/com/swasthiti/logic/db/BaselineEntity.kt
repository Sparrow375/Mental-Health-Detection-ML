package com.swasthiti.logic.db

import androidx.room.Entity

/**
 * Stores per-feature baseline statistics once the 28-day onboarding period completes.
 * Each row = one feature's mean and standard deviation for a given user.
 *
 * The is_contaminated flag is set by the 3-Gate screener (System 2 / baseline_screener.py).
 * If contaminated, the Python engine falls back to population-synthetic baseline values.
 */
@Entity(
    tableName = "baseline",
    primaryKeys = ["userId", "featureName"]
)
data class BaselineEntity(
    val userId: String,
    val featureName: String,
    val baselineValue: Float,      // mean over 28 days
    val stdDeviation: Float,       // std over 28 days
    val baselineStart: String,     // "YYYY-MM-DD"
    val baselineEnd: String,       // "YYYY-MM-DD"
    val isContaminated: Boolean = false,
    val updatedAt: Long = System.currentTimeMillis()
)

