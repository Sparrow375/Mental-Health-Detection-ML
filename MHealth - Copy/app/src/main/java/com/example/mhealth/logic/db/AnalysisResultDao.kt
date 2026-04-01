package com.example.mhealth.logic.db

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Dao
interface AnalysisResultDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(entity: AnalysisResultEntity)

    @Query("SELECT * FROM analysis_results WHERE userId = :userId ORDER BY date DESC LIMIT 1")
    suspend fun getLatest(userId: String): AnalysisResultEntity?

    /** Reactive version — emits a new value each time NightlyAnalysisWorker inserts a row. */
    @Query("SELECT * FROM analysis_results WHERE userId = :userId ORDER BY date DESC LIMIT 1")
    fun getLatestFlow(userId: String): Flow<AnalysisResultEntity?>

    @Query("SELECT * FROM analysis_results WHERE userId = :userId ORDER BY date DESC LIMIT :limit")
    suspend fun getLatestN(userId: String, limit: Int): List<AnalysisResultEntity>

    /** Reactive version for history list — emits on every insert/update. */
    @Query("SELECT * FROM analysis_results WHERE userId = :userId ORDER BY date DESC LIMIT :limit")
    fun getLatestNFlow(userId: String, limit: Int = 30): Flow<List<AnalysisResultEntity>>

    @Query("SELECT * FROM analysis_results WHERE userId = :userId AND syncedToCloud = 0")
    suspend fun getUnsynced(userId: String): List<AnalysisResultEntity>

    @Query("UPDATE analysis_results SET syncedToCloud = 1 WHERE id = :id")
    suspend fun markSynced(id: Long): Int

    @Query("SELECT * FROM analysis_results WHERE userId = :userId ORDER BY date ASC")
    suspend fun getAll(userId: String): List<AnalysisResultEntity>

    @Query("DELETE FROM analysis_results WHERE userId = :userId")
    suspend fun clearAll(userId: String): Int
}
