package com.example.mhealth.logic.db

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "badge_achievements")
data class BadgeEntity(
    @PrimaryKey val badgeId: String,
    val badgeName: String,
    val description: String,
    val category: String,
    val isUnlocked: Boolean = false,
    val earnedAt: String? = null,
    val currentStreak: Int = 0,
    val targetStreak: Int = 7
)
