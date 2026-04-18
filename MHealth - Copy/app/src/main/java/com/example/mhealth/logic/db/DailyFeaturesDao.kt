package com.example.mhealth.logic.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import androidx.room.Upsert

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

    @Query("SELECT COUNT(*) FROM daily_features WHERE userId = :userId")
    suspend fun count(userId: String): Int

    @Query("DELETE FROM daily_features WHERE userId = :userId")
    suspend fun clearAll(userId: String): Int

    @Query("DELETE FROM daily_features WHERE userId = :userId AND isSimulated = 1")
    suspend fun clearSimulated(userId: String): Int

    /** Upsert today's in-progress snapshot by userId+date. Preserves the existing row's id and sync state. */
    @androidx.room.Transaction
    suspend fun upsertByDate(entity: DailyFeaturesEntity) {
        val existing = getByDate(entity.userId, entity.date)
        if (existing != null) {
            // Update the existing row — keep its id and sync flag
            insert(entity.copy(id = existing.id, syncedToCloud = existing.syncedToCloud))
        } else {
            insert(entity)
        }
    }
}
