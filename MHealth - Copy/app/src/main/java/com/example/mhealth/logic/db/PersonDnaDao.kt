package com.example.mhealth.logic.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query

/**
 * DAO for person_dna table — Level 2 Behavioral DNA persistence.
 */
@Dao
interface PersonDnaDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(dna: PersonDnaEntity)

    /** Get the DNA for a given user. Returns null if not yet built. */
    @Query("SELECT * FROM person_dna WHERE person_id = :userId LIMIT 1")
    suspend fun getByUserId(userId: String): PersonDnaEntity?

    /** Update existing DNA JSON. */
    @Query("UPDATE person_dna SET dna_json = :dnaJson, last_updated = :lastUpdated WHERE person_id = :userId")
    suspend fun updateDna(userId: String, dnaJson: String, lastUpdated: Long)

    /** Delete DNA for a user (e.g., on reset). */
    @Query("DELETE FROM person_dna WHERE person_id = :userId")
    suspend fun deleteByUserId(userId: String)
}
