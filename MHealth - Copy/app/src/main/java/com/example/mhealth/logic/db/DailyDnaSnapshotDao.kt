package com.example.mhealth.logic.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query

@Dao
interface DailyDnaSnapshotDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(entity: DailyDnaSnapshotEntity)

    @Query("SELECT * FROM daily_dna_snapshot WHERE userId = :userId AND date = :date LIMIT 1")
    suspend fun getByDate(userId: String, date: String): DailyDnaSnapshotEntity?

    @Query("SELECT COUNT(DISTINCT date) FROM daily_dna_snapshot WHERE userId = :userId")
    suspend fun countDistinctDays(userId: String): Int

    @Query("SELECT * FROM daily_dna_snapshot WHERE userId = :userId ORDER BY date ASC")
    suspend fun getAll(userId: String): List<DailyDnaSnapshotEntity>

    @Query("SELECT * FROM daily_dna_snapshot WHERE userId = :userId ORDER BY date DESC LIMIT :limit")
    suspend fun getLatestN(userId: String, limit: Int): List<DailyDnaSnapshotEntity>

    @Query("DELETE FROM daily_dna_snapshot WHERE userId = :userId AND date = :date")
    suspend fun deleteByDate(userId: String, date: String): Int

    @Query("DELETE FROM daily_dna_snapshot WHERE userId = :userId")
    suspend fun clearAll(userId: String): Int
}
