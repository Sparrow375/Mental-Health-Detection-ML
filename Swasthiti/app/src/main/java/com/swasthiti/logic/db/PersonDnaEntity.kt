package com.swasthiti.logic.db

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Stores serialized PersonDNA for Level 2 Behavioral DNA system.
 * One row per user — updated after each nightly analysis.
 * The dna_json field contains the full PersonDNA serialized as JSON.
 */
@Entity(
    tableName = "person_dna",
    indices = [Index(value = ["person_id"], unique = true)]
)
data class PersonDnaEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val person_id: String,             // user ID (email)
    val dna_json: String,              // Full PersonDNA serialized as JSON
    val created_at: Long,              // epoch_ms when first created
    val last_updated: Long             // epoch_ms when last updated
)
