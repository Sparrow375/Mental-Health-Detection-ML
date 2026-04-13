package com.example.mhealth.logic.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query

/**
 * DAO for notification_events table — Level 2 Digital DNA notification data.
 */
@Dao
interface NotificationEventDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(event: NotificationEventEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(events: List<NotificationEventEntity>)

    /** Get all notification events for a given date (YYYY-MM-DD). */
    @Query("SELECT * FROM notification_events WHERE date = :date ORDER BY arrival_timestamp ASC")
    suspend fun getByDate(date: String): List<NotificationEventEntity>

    /** Get notification events for a date range. */
    @Query("SELECT * FROM notification_events WHERE date >= :startDate AND date <= :endDate ORDER BY arrival_timestamp ASC")
    suspend fun getByDateRange(startDate: String, endDate: String): List<NotificationEventEntity>

    /** Get all notification events since a given epoch_ms (for DNA building). */
    @Query("SELECT * FROM notification_events WHERE arrival_timestamp >= :sinceEpochMs ORDER BY arrival_timestamp ASC")
    suspend fun getEventsSince(sinceEpochMs: Long): List<NotificationEventEntity>

    /** Get today's notification events for a specific app package. */
    @Query("SELECT * FROM notification_events WHERE app_package = :pkg AND date = :date ORDER BY arrival_timestamp ASC")
    suspend fun getByPackageAndDate(pkg: String, date: String): List<NotificationEventEntity>

    /** Count total notification events stored. */
    @Query("SELECT COUNT(*) FROM notification_events")
    suspend fun count(): Int

    /** Delete notification events older than the given epoch_ms (cleanup). */
    @Query("DELETE FROM notification_events WHERE arrival_timestamp < :beforeEpochMs")
    suspend fun deleteOlderThan(beforeEpochMs: Long): Int

    /** Get distinct app packages in notification events since a given time. */
    @Query("SELECT DISTINCT app_package FROM notification_events WHERE arrival_timestamp >= :sinceEpochMs")
    suspend fun getDistinctPackagesSince(sinceEpochMs: Long): List<String>
}