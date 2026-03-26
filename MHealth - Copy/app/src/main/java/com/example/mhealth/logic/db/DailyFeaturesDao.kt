package com.example.mhealth.logic.db

import androidx.room.*

@Dao
interface DailyFeaturesDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(entity: DailyFeaturesEntity)

    @Query("SELECT * FROM daily_features WHERE userId = :userId AND date = :date LIMIT 1")
    suspend fun getByDate(userId: String, date: String): DailyFeaturesEntity?

    /** Returns all rows not yet uploaded to Firebase */
    @Query("SELECT * FROM daily_features WHERE userId = :userId AND syncedToCloud = 0")
    suspend fun getUnsynced(userId: String): List<DailyFeaturesEntity>

    @Query("UPDATE daily_features SET syncedToCloud = 1 WHERE id = :id")
    suspend fun markSynced(id: Long)

    /** Returns the last N rows ordered by date descending (for history window passed to Python) */
    @Query("SELECT * FROM daily_features WHERE userId = :userId ORDER BY date DESC LIMIT :limit")
    suspend fun getLatestN(userId: String, limit: Int): List<DailyFeaturesEntity>

    @Query("DELETE FROM daily_features WHERE userId = :userId")
    suspend fun clearAll(userId: String)
}
