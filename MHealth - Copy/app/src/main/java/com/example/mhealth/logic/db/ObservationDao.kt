package com.example.mhealth.logic.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Dao
interface ObservationDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(entity: ObservationEntity)

    @Query("SELECT * FROM observations WHERE userId = :userId ORDER BY date DESC LIMIT 1")
    suspend fun getLatest(userId: String): ObservationEntity?

    @Query("SELECT * FROM observations WHERE userId = :userId ORDER BY date DESC LIMIT 1")
    fun getLatestFlow(userId: String): Flow<ObservationEntity?>

    @Query("SELECT * FROM observations WHERE userId = :userId ORDER BY date DESC")
    fun getAllFlow(userId: String): Flow<List<ObservationEntity>>

    @Query("SELECT * FROM observations WHERE userId = :userId AND date = :date LIMIT 1")
    suspend fun getByDate(userId: String, date: String): ObservationEntity?

    @Query("SELECT * FROM observations WHERE userId = :userId AND date = :date LIMIT 1")
    fun getByDateFlow(userId: String, date: String): Flow<ObservationEntity?>

    @Update
    suspend fun update(entity: ObservationEntity)

    @Query("UPDATE observations SET feedbackState = :state, feedbackCategory = :category, feedbackNotes = :notes WHERE userId = :userId AND date = :date")
    suspend fun updateFeedback(userId: String, date: String, state: String, category: String, notes: String): Int

    @Query("DELETE FROM observations WHERE userId = :userId")
    suspend fun clearAll(userId: String): Int
}
