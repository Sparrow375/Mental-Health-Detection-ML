package com.example.mhealth.logic.db

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * One row per analysis run (daily).
 * Contains the output of both System 1 (anomaly detection) and System 2 (prototype matching).
 *
 * Fields:
 *   anomalyDetected   – true if System 1 raised a sustained anomaly
 *   anomalyMessage    – human-readable sentence from the engine
 *   sustainedDays     – consecutive days above anomaly threshold
 *   prototypeMatch    – closest disorder prototype ("Normal", "Depression", "BPD", ...)
 *   matchMessage      – confidence + explanation sentence
 *   prototypeConfidence – 0.0–1.0 match confidence from System 2
 *   gateResults       – JSON string of the 3-gate screener results
 *   alertLevel        – "green" | "yellow" | "orange" | "red"
 */
@Entity(tableName = "analysis_results")
data class AnalysisResultEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val userId: String,
    val date: String,                      // "YYYY-MM-DD"
    val anomalyDetected: Boolean = false,
    val anomalyMessage: String = "",
    val anomalyScore: Float = 0f,          // 0.0–1.0 composite anomaly score from System 1
    val sustainedDays: Int = 0,
    val alertLevel: String = "green",
    val prototypeMatch: String = "Normal",
    val matchMessage: String = "",
    val prototypeConfidence: Float = 0f,
    val gateResults: String = "{}",        // JSON blob of gate pass/fail
    val syncedToCloud: Boolean = false,
    val createdAt: Long = System.currentTimeMillis()
)
