package com.example.mhealth.logic.db

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "observations")
data class ObservationEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val userId: String,
    val date: String,                      // "YYYY-MM-DD"
    val category: String,                  // "Sleep", "Activity", "Digital", "Mobility", "General"
    val title: String,
    val narrative: String,                 // The surfaced story
    val feedbackState: String = "unresolved", // unresolved | confirmed | corrected | noted
    val feedbackCategory: String = "",    // schedule_shift | travel | illness | stress | none
    val feedbackNotes: String = "",       // Custom user detail notes
    val baselineConfidence: Float = 1.0f,  // 0f to 1f confidence score
    val isQuietDay: Boolean = false,       // whether it is a quiet day/reassuring note
    val flaggedFeatures: String = "[]",    // JSON list of features involved
    val createdAt: Long = System.currentTimeMillis()
)
