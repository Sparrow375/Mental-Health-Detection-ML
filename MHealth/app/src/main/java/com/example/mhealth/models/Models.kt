package com.example.mhealth.models

import java.util.Date

data class PersonalityVector(
    val screenTimeHours: Float = 0f,
    val unlockCount: Float = 0f,
    val socialAppRatio: Float = 0f,
    val callsPerDay: Float = 0f,
    val textsPerDay: Float = 0f,
    val uniqueContacts: Float = 0f,
    val responseTimeMinutes: Float = 0f,
    val dailyDisplacementKm: Float = 0f,
    val locationEntropy: Float = 0f,
    val homeTimeRatio: Float = 0f,
    val placesVisited: Float = 0f,
    val wakeTimeHour: Float = 0f,
    val sleepTimeHour: Float = 0f,
    val sleepDurationHours: Float = 0f,
    val darkDurationHours: Float = 0f,
    val chargeDurationHours: Float = 0f,
    val conversationFrequency: Float = 0f,
    val variances: Map<String, Float> = emptyMap(),
    val appBreakdown: Map<String, Long> = emptyMap() // Package to Minutes
) {
    fun toMap(): Map<String, Float> {
        return mapOf(
            "screenTimeHours" to screenTimeHours,
            "unlockCount" to unlockCount,
            "socialAppRatio" to socialAppRatio,
            "callsPerDay" to callsPerDay,
            "textsPerDay" to textsPerDay,
            "uniqueContacts" to uniqueContacts,
            "dailyDisplacementKm" to dailyDisplacementKm,
            "locationEntropy" to locationEntropy,
            "homeTimeRatio" to homeTimeRatio,
            "placesVisited" to placesVisited,
            "wakeTimeHour" to wakeTimeHour,
            "sleepTimeHour" to sleepTimeHour,
            "sleepDurationHours" to sleepDurationHours,
            "darkDurationHours" to darkDurationHours,
            "chargeDurationHours" to chargeDurationHours,
            "conversationFrequency" to conversationFrequency
        )
    }
}

data class DailyReport(
    val dayNumber: Int,
    val date: Date,
    val anomalyScore: Float,
    val alertLevel: String,
    val flaggedFeatures: List<String>,
    val patternType: String,
    val sustainedDeviationDays: Int,
    val evidenceAccumulated: Float,
    val topDeviations: Map<String, Float>,
    val notes: String
)
