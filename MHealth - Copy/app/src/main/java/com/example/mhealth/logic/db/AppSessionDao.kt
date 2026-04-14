package com.example.mhealth.logic.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query

/**
 * DAO for app_sessions table — Level 2 Behavioral DNA session data.
 */
@Dao
interface AppSessionDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(session: AppSessionEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(sessions: List<AppSessionEntity>)

    /** Get all sessions for a given date (YYYY-MM-DD). */
    @Query("SELECT * FROM app_sessions WHERE date = :date ORDER BY open_timestamp ASC")
    suspend fun getByDate(date: String): List<AppSessionEntity>

    /** Get sessions for a date range (last N days), ordered chronologically. */
    @Query("SELECT * FROM app_sessions WHERE date >= :startDate AND date <= :endDate ORDER BY open_timestamp ASC")
    suspend fun getByDateRange(startDate: String, endDate: String): List<AppSessionEntity>

    /** Get last 28 days of sessions for DNA building. */
    @Query("SELECT * FROM app_sessions WHERE open_timestamp >= :sinceEpochMs ORDER BY open_timestamp ASC")
    suspend fun getSessionsSince(sinceEpochMs: Long): List<AppSessionEntity>

    /** Get today's sessions for a specific app package. */
    @Query("SELECT * FROM app_sessions WHERE app_package = :pkg AND date = :date ORDER BY open_timestamp ASC")
    suspend fun getByPackageAndDate(pkg: String, date: String): List<AppSessionEntity>

    /** Count total sessions stored (for diagnostics). */
    @Query("SELECT COUNT(*) FROM app_sessions")
    suspend fun count(): Int

    /** Delete sessions older than the given epoch_ms (cleanup). */
    @Query("DELETE FROM app_sessions WHERE close_timestamp < :beforeEpochMs")
    suspend fun deleteOlderThan(beforeEpochMs: Long): Int

    /** Get distinct app packages in the last N days. */
    @Query("SELECT DISTINCT app_package FROM app_sessions WHERE open_timestamp >= :sinceEpochMs")
    suspend fun getDistinctPackagesSince(sinceEpochMs: Long): List<String>

    /** Get all sessions (for Firebase sync). */
    @Query("SELECT * FROM app_sessions ORDER BY open_timestamp ASC")
    suspend fun getAll(): List<AppSessionEntity>
}
