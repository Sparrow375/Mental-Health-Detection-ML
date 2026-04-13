package com.swasthiti.logic.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Upsert

@Dao
interface UserProfileDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(entity: UserProfileEntity)

    @Query("SELECT * FROM user_profile_db WHERE userId = :userId LIMIT 1")
    suspend fun get(userId: String): UserProfileEntity?

    @Query("UPDATE user_profile_db SET baselineReady = :ready WHERE userId = :userId")
    suspend fun setBaselineReady(userId: String, ready: Boolean): Int

    @Query("UPDATE user_profile_db SET currentStatus = :status WHERE userId = :userId")
    suspend fun updateStatus(userId: String, status: String): Int
}

