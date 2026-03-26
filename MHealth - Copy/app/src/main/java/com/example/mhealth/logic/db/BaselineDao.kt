package com.example.mhealth.logic.db

import androidx.room.*

@Dao
interface BaselineDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(entities: List<BaselineEntity>)

    @Query("SELECT * FROM baseline WHERE userId = :userId")
    suspend fun getBaseline(userId: String): List<BaselineEntity>

    @Query("UPDATE baseline SET isContaminated = :contaminated WHERE userId = :userId")
    suspend fun setContaminated(userId: String, contaminated: Boolean)

    @Query("DELETE FROM baseline WHERE userId = :userId")
    suspend fun clearBaseline(userId: String)
}
