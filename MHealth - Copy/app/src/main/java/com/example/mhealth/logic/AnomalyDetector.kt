package com.example.mhealth.logic

import com.example.mhealth.models.DailyReport
import com.example.mhealth.models.PersonalityVector
import java.util.Date
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/** Mirror of the same helper in MonitoringService — keeps sleep/wake on a midnight-safe scale. */
private fun normalizeTimeToNoon(rawHour: Float): Float = (rawHour - 12f + 24f) % 24f
private val FEATURE_META = mapOf(
    "screenTimeHours"      to 1.4f, "unlockCount"          to 1.2f, "appLaunchCount"       to 0.9f,
    "notificationsToday"   to 0.8f, "socialAppRatio"       to 1.3f, "callsPerDay"          to 1.3f,
    "callDurationMinutes"  to 1.2f, "uniqueContacts"       to 1.1f, "conversationFrequency" to 0.9f,
    "dailyDisplacementKm"  to 1.5f, "locationEntropy"      to 1.3f, "homeTimeRatio"        to 1.2f,
    "wakeTimeHour"         to 1.4f, "sleepTimeHour"        to 1.3f,
    "sleepDurationHours"   to 1.6f, "darkDurationHours"    to 1.0f, "chargeDurationHours"  to 0.8f,
    "memoryUsagePercent"   to 0.5f, "networkWifiMB"        to 0.6f, "networkMobileMB"      to 0.6f,
    "storageUsedGB"        to 0.4f, "totalAppsCount"       to 0.8f, "upiTransactionsToday" to 1.1f,
    "appUninstallsToday"   to 0.9f, "appInstallsToday"     to 0.8f, "calendarEventsToday"  to 0.9f,
    "mediaCountToday"      to 0.7f, "downloadsToday"       to 0.6f, "backgroundAudioHours" to 0.9f,
    "dailySteps"           to 1.0f
)

class AnomalyDetector(
    private val baseline: PersonalityVector,
    historicalAnomalyScores: List<Float> = emptyList()
) {
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
        // These MUST exactly match the keys returned by PersonalityVector.toMap()
        val features = listOf(
            "screenTimeHours", "unlockCount", "appLaunchCount", "notificationsToday",
            "socialAppRatio", "callsPerDay", "callDurationMinutes", "uniqueContacts",
            "conversationFrequency", "dailyDisplacementKm", "locationEntropy",
            "homeTimeRatio", "wakeTimeHour", "sleepTimeHour",
            "sleepDurationHours", "darkDurationHours", "chargeDurationHours",
            "memoryUsagePercent", "networkWifiMB", "networkMobileMB", "calendarEventsToday",
            "downloadsToday", "storageUsedGB", "appUninstallsToday",
            "upiTransactionsToday", "totalAppsCount", "mediaCountToday",
            "appInstallsToday", "dailySteps", "backgroundAudioHours"
        )
        features.forEach { featureHistory[it] = mutableListOf() }

        // Initialize anomaly score history from Room data (past analysis results)
        anomalyScoreHistory.addAll(historicalAnomalyScores.takeLast(14))
    }

    fun analyze(currentData: PersonalityVector, dayNumber: Int, isProvisional: Boolean = false): DailyReport {
        val currentMap = currentData.toMap()
        val deviations = calculateDeviations(currentMap)
        val velocities = calculateVelocities(currentMap, isProvisional)
        val anomalyScore = calculateAnomalyScore(deviations, velocities)

        updateSustainedTracking(anomalyScore, isProvisional)

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
            val variance = variances[feature] ?: 1f
            val weight = FEATURE_META[feature] ?: 1.0f

            val safeVariance = if (feature.endsWith("Hour")) {
                max(variance, 0.5f)
            } else {
                max(variance, 1.0f)
            }

            // Apply Noon-Offset so 11:55 PM vs 12:05 AM is a 10-min diff, not a 23h diff (only for absolute clock hours)
            val currentNorm  = if (feature.endsWith("Hour")) normalizeTimeToNoon(value)       else value
            val baselineNorm = if (feature.endsWith("Hour")) normalizeTimeToNoon(baselineVal) else baselineVal

            // Z-Score * Weight
            deviations[feature] = ((currentNorm - baselineNorm) / safeVariance) * weight
        }
        return deviations
    }

    private fun calculateVelocities(currentData: Map<String, Float>, isProvisional: Boolean): Map<String, Float> {
        val velocities = mutableMapOf<String, Float>()
        val alpha = 0.4f

        currentData.forEach { (feature, value) ->
            val normValue = if (feature.endsWith("Hour")) normalizeTimeToNoon(value) else value

            val history = featureHistory[feature] ?: mutableListOf<Float>().also { featureHistory[feature] = it }
            
            // Only mutate historical queues if this is the final daily analysis
            val tempHistory = if (isProvisional) {
                val temp = history.toMutableList()
                temp.add(normValue)
                if (temp.size > historyWindow) temp.removeAt(0)
                temp
            } else {
                history.add(normValue)
                if (history.size > historyWindow) history.removeAt(0)
                history
            }

            if (tempHistory.size < 2) {
                velocities[feature] = 0f
            } else {
                var ewma = tempHistory[0]
                val ewmaValues = mutableListOf<Float>()
                tempHistory.forEach { val_ ->
                    ewma = alpha * val_ + (1 - alpha) * ewma
                    ewmaValues.add(ewma)
                }
                val slope = (ewmaValues.last() - ewmaValues.first()) / ewmaValues.size
                val baselineRaw = baseline.toMap()[feature] ?: 1f
                val baselineNorm = if (feature.endsWith("Hour")) normalizeTimeToNoon(baselineRaw) else baselineRaw
                velocities[feature] = if (baselineNorm != 0f) slope / baselineNorm else 0f
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

    private fun updateSustainedTracking(anomalyScore: Float, isProvisional: Boolean) {
        if (isProvisional) return // Never mutate sustained evidence on 15min provisional ticks

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
        val criticalFeatures = listOf(
            "sleepDurationHours", "screenTimeHours", "dailyDisplacementKm",
            "socialAppRatio", "totalAppsCount", "upiTransactionsToday",
            "wakeTimeHour", "callsPerDay"
        )
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
