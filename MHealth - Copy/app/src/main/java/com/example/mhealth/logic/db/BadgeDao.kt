package com.example.mhealth.logic.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface BadgeDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertOrUpdate(badge: BadgeEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(badges: List<BadgeEntity>)

    @Query("SELECT * FROM badge_achievements ORDER BY isUnlocked DESC, badgeName ASC")
    fun getAllBadgesFlow(): Flow<List<BadgeEntity>>

    @Query("SELECT * FROM badge_achievements ORDER BY isUnlocked DESC, badgeName ASC")
    suspend fun getAllBadges(): List<BadgeEntity>

    @Query("SELECT * FROM badge_achievements WHERE badgeId = :badgeId LIMIT 1")
    suspend fun getBadgeById(badgeId: String): BadgeEntity?

    @Query("UPDATE badge_achievements SET isUnlocked = 1, earnedAt = :earnedAt WHERE badgeId = :badgeId")
    suspend fun unlockBadge(badgeId: String, earnedAt: String): Int
}
