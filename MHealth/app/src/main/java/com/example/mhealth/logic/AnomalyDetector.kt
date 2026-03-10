package com.example.mhealth.logic

import com.example.mhealth.models.DailyReport
import com.example.mhealth.models.PersonalityVector
import java.util.*
import kotlin.math.*

class AnomalyDetector(private val baseline: PersonalityVector) {
    private val historyWindow = 7
    private val featureHistory = mutableMapOf<String, MutableList<Float>>()
    
    // Sustained deviation tracking
    private var sustainedDeviationDays = 0
    private var evidenceAccumulated = 0f
    private val anomalyScoreHistory = mutableListOf<Float>()
    
    private val ANOMALY_SCORE_THRESHOLD = 0.35f
    private val SUSTAINED_THRESHOLD_DAYS = 4
    private val EVIDENCE_THRESHOLD = 2.0f

    init {
        val features = listOf(
            "screenTimeHours", "unlockCount", "socialAppRatio", "callsPerDay",
            "textsPerDay", "uniqueContacts", "responseTimeMinutes", "dailyDisplacementKm",
            "locationEntropy", "homeTimeRatio", "placesVisited", "wakeTimeHour",
            "sleepTimeHour", "sleepDurationHours", "darkDurationHours", "chargeDurationHours",
            "conversationFrequency"
        )
        features.forEach { featureHistory[it] = mutableListOf() }
    }

    fun analyze(currentData: PersonalityVector, dayNumber: Int): DailyReport {
        val currentMap = currentData.toMap()
        val deviations = calculateDeviations(currentMap)
        val velocities = calculateVelocities(currentMap)
        val anomalyScore = calculateAnomalyScore(deviations, velocities)

        updateSustainedTracking(anomalyScore)

        val alertLevel = determineAlertLevel(anomalyScore, deviations)
        val patternType = detectPatternType()
        val flaggedFeatures = identifyFlaggedFeatures(deviations)
        val topDeviations = getTopDeviations(deviations)
        val notes = generateNotes(anomalyScore, alertLevel, patternType)

        return DailyReport(
            dayNumber = dayNumber,
            date = Date(),
            anomalyScore = anomalyScore,
            alertLevel = alertLevel,
            flaggedFeatures = flaggedFeatures,
            patternType = patternType,
            sustainedDeviationDays = sustainedDeviationDays,
            evidenceAccumulated = evidenceAccumulated,
            topDeviations = topDeviations,
            notes = notes
        )
    }

    private fun calculateDeviations(currentData: Map<String, Float>): Map<String, Float> {
        val deviations = mutableMapOf<String, Float>()
        val baselineMap = baseline.toMap()
        val variances = baseline.variances

        currentData.forEach { (feature, value) ->
            val baselineVal = baselineMap[feature] ?: 0f
            val variance = variances[feature] ?: 1f // Avoid division by zero
            deviations[feature] = (value - baselineVal) / if (variance != 0f) variance else 1f
        }
        return deviations
    }

    private fun calculateVelocities(currentData: Map<String, Float>): Map<String, Float> {
        val velocities = mutableMapOf<String, Float>()
        val alpha = 0.4f

        currentData.forEach { (feature, value) ->
            val history = featureHistory[feature]!!
            history.add(value)
            if (history.size > historyWindow) history.removeAt(0)

            if (history.size < 2) {
                velocities[feature] = 0f
            } else {
                var ewma = history[0]
                val ewmaValues = mutableListOf<Float>()
                history.forEach { val_ ->
                    ewma = alpha * val_ + (1 - alpha) * ewma
                    ewmaValues.add(ewma)
                }
                val slope = (ewmaValues.last() - ewmaValues.first()) / ewmaValues.size
                val baselineVal = baseline.toMap()[feature] ?: 1f
                velocities[feature] = if (baselineVal != 0f) slope / baselineVal else 0f
            }
        }
        return velocities
    }

    private fun calculateAnomalyScore(deviations: Map<String, Float>, velocities: Map<String, Float>): Float {
        val magnitudeScore = deviations.values.map { abs(it) }.average().toFloat()
        val normalizedMagnitude = min(magnitudeScore / 3.0f, 1.0f)

        val velocityScore = velocities.values.map { abs(it) }.average().toFloat()
        val normalizedVelocity = min(velocityScore * 10f, 1.0f)

        return 0.7f * normalizedMagnitude + 0.3f * normalizedVelocity
    }

    private fun updateSustainedTracking(anomalyScore: Float) {
        anomalyScoreHistory.add(anomalyScore)
        if (anomalyScoreHistory.size > 14) anomalyScoreHistory.removeAt(0)

        if (anomalyScore > ANOMALY_SCORE_THRESHOLD) {
            sustainedDeviationDays++
            evidenceAccumulated += anomalyScore * (1 + sustainedDeviationDays * 0.1f)
        } else {
            sustainedDeviationDays = max(0, sustainedDeviationDays - 1)
            evidenceAccumulated *= 0.92f
        }
    }

    private fun determineAlertLevel(anomalyScore: Float, deviations: Map<String, Float>): String {
        val criticalFeatures = listOf("sleepDurationHours", "screenTimeHours", "dailyDisplacementKm")
        val criticalDeviation = criticalFeatures.map { abs(deviations[it] ?: 0f) }.maxOrNull() ?: 0f

        val hasSustainedDeviation = sustainedDeviationDays >= SUSTAINED_THRESHOLD_DAYS || evidenceAccumulated >= EVIDENCE_THRESHOLD

        if (!hasSustainedDeviation) return "green"

        return when {
            anomalyScore < 0.35f && criticalDeviation < 2.0f -> "green"
            anomalyScore < 0.50f && criticalDeviation < 2.5f -> "yellow"
            anomalyScore < 0.65f || criticalDeviation < 3.0f -> "orange"
            else -> "red"
        }
    }

    private fun detectPatternType(): String {
        if (anomalyScoreHistory.size < 7) return "insufficient_data"
        val recent = anomalyScoreHistory.takeLast(7)
        val mean = recent.average().toFloat()
        val std = sqrt(recent.map { (it - mean).pow(2) }.average()).toFloat()

        return when {
            mean < 0.3f -> "stable"
            std > 0.15f -> "rapid_cycling"
            mean > 0.5f -> "acute_spike"
            else -> "mixed_pattern"
        }
    }

    private fun identifyFlaggedFeatures(deviations: Map<String, Float>): List<String> {
        return deviations.filter { abs(it.value) > 1.5f }.map { "${it.key} (${"%.2f".format(it.value)} SD)" }
    }

    private fun getTopDeviations(deviations: Map<String, Float>): Map<String, Float> {
        return deviations.entries.sortedByDescending { abs(it.value) }.take(5).associate { it.toPair() }
    }

    private fun generateNotes(anomalyScore: Float, alertLevel: String, patternType: String): String {
        val notes = mutableListOf<String>()
        if (sustainedDeviationDays >= SUSTAINED_THRESHOLD_DAYS) notes.add("Sustained deviation (${sustainedDeviationDays} days)")
        if (evidenceAccumulated >= EVIDENCE_THRESHOLD) notes.add("Evidence: ${"%.2f".format(evidenceAccumulated)}")
        if (alertLevel != "green") notes.add("Alert: ${alertLevel.uppercase()}")
        return if (notes.isEmpty()) "Normal operation" else notes.joinToString(" | ")
    }
}
