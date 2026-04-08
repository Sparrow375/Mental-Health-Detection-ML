package com.example.mhealth.logic.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import androidx.room.Upsert

@Dao
interface UserCredentialsDao {

    /**
     * Register a new user. Returns the new rowId, or -1 if the email already exists
     * (OnConflictStrategy.IGNORE ensures no exception is thrown on duplicate email).
     */
    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun register(entity: UserCredentialsEntity): Long

    /** Look up credentials by email. Returns null if not found. */
    @Query("SELECT * FROM user_credentials WHERE email = :email LIMIT 1")
    suspend fun findByEmail(email: String): UserCredentialsEntity?

    @Query("DELETE FROM user_credentials WHERE email = :email")
    suspend fun deleteByEmail(email: String)

    /** Returns true if at least one account exists — used to detect first launch. */
    @Query("SELECT COUNT(*) FROM user_credentials")
    suspend fun count(): Int
}
