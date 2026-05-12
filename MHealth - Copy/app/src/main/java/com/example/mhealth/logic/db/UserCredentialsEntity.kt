package com.example.mhealth.logic.db

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Stores local user credentials created during first-time registration.
 * Email is the primary key and is unique — prevents duplicate accounts.
 * Password is stored as a SHA-256 hex digest.
 */
@Entity(
    tableName = "user_credentials",
    indices = [Index(value = ["email"], unique = true)]
)
data class UserCredentialsEntity(
    @PrimaryKey val email: String,
    val name: String,
    val passwordHash: String,        // SHA-256 of the raw password
    val createdAt: Long = System.currentTimeMillis()
)
