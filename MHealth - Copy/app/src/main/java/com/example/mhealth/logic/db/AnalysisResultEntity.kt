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
 *
 *   L2 Digital DNA fields:
 *   l2Modifier          – L2 modifier [0.15–2.0]: suppresses or amplifies L1 score
 *   coherence           – context coherence [0–1]: how well today matches known archetype
 *   rhythmDissolution   – rhythm dissolution [0–1]: how much usage rhythm has scattered
 *   sessionIncoherence  – session incoherence [0–1]: abandon rate + duration collapse
 *   effectiveScore      – L1 × L2 modifier: the score that feeds evidence accumulator
 *   evidenceAccumulated – running accumulated evidence with compounding
 *   patternType         – "stable" | "rapid_cycling" | "acute_spike" | "gradual_drift" | "mixed"
 *   flaggedFeatures     – JSON list of flagged feature strings (e.g. 'sleepDuration (2.41 SD)')
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
    val createdAt: Long = System.currentTimeMillis(),

    // L2 Digital DNA fields
    val l2Modifier: Float = 1.0f,              // [0.15–2.0] suppresses or amplifies L1
    val coherence: Float = 0f,                 // [0–1] context match quality
    val rhythmDissolution: Float = 0f,         // [0–1] usage rhythm scatter
    val sessionIncoherence: Float = 0f,        // [0–1] session quality degradation
    val effectiveScore: Float = 0f,            // L1 × L2 modifier
    val evidenceAccumulated: Float = 0f,       // running evidence with compounding
    val patternType: String = "stable",        // stable | rapid_cycling | acute_spike | gradual_drift | mixed
    val flaggedFeatures: String = "[]"         // JSON list of flagged feature strings
)
