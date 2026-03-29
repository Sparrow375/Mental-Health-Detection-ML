package com.example.mhealth.logic.db

import androidx.room.*

@Dao
interface DailyFeaturesDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(entity: DailyFeaturesEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(entities: List<DailyFeaturesEntity>)

    @Query("SELECT * FROM daily_features WHERE userId = :userId AND date = :date LIMIT 1")
    suspend fun getByDate(userId: String, date: String): DailyFeaturesEntity?

    @Query("SELECT * FROM daily_features WHERE userId = :userId AND syncedToCloud = 0")
    suspend fun getUnsynced(userId: String): List<DailyFeaturesEntity>

    @Query("UPDATE daily_features SET syncedToCloud = 1 WHERE id = :id")
    suspend fun markSynced(id: Long): Int

    @Query("SELECT * FROM daily_features WHERE userId = :userId ORDER BY date DESC LIMIT :limit")
    suspend fun getLatestN(userId: String, limit: Int): List<DailyFeaturesEntity>

    @Query("SELECT * FROM daily_features WHERE userId = :userId ORDER BY date ASC")
    suspend fun getAllFeatures(userId: String): List<DailyFeaturesEntity>

    @Query("DELETE FROM daily_features WHERE userId = :userId")
    suspend fun clearAll(userId: String): Int

    @Query("DELETE FROM daily_features WHERE userId = :userId AND isSimulated = 1")
    suspend fun clearSimulated(userId: String): Int
}
