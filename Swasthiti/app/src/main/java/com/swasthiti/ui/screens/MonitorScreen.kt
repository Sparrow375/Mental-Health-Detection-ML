package com.swasthiti.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Sensors
import androidx.compose.material.icons.filled.Shield
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.swasthiti.logic.DataRepository
import com.swasthiti.ui.components.*
import com.swasthiti.ui.charts.ArcProgressRing
import com.swasthiti.ui.charts.AnomalyScoreGauge
import com.swasthiti.ui.charts.SparklineChart
import com.swasthiti.ui.theme.*
import androidx.compose.ui.graphics.Brush
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material.icons.filled.CheckCircle
import com.swasthiti.FeatureTableCard
import com.swasthiti.PerAppBreakdownCard
import com.swasthiti.BgAudioBreakdownCard

@Composable
fun MonitorScreen() {
    val progress by DataRepository.baselineProgress.collectAsState()
    val isBuilding by DataRepository.isBuildingBaseline.collectAsState()
    val vector by DataRepository.latestVector.collectAsState()
    val baseline by DataRepository.baseline.collectAsState()
    val hourly by DataRepository.hourlySnapshots.collectAsState()
    val reports by DataRepository.reports.collectAsState()
    val baselineDaysReq by DataRepository.baselineDaysRequired.collectAsState()
    val baselineVectors by DataRepository.collectedBaselineVectors.collectAsState()
    val analysisResult by DataRepository.latestAnalysisResult.collectAsState()
    val analysisHistory by DataRepository.analysisHistory.collectAsState()
    val s1ProfileJson by DataRepository.s1ProfileJson.collectAsState()

    LazyColumn(Modifier.fillMaxSize()) {
        item {
            Box(
                Modifier.fillMaxWidth()
                    .background(Brush.horizontalGradient(listOf(SoftCyan, ChartPurple)))
                    .padding(20.dp)
            ) {
                Column {
                    Text("Baseline & Monitoring", fontSize = 24.sp, fontWeight = FontWeight.Bold, color = Color.White)
                    Text("Layers 2 & 3 — ${if (isBuilding) "Building Personal Normal" else "Continuous Tracking"}", fontSize = 13.sp, color = Color.White.copy(0.85f))
                }
            }
        }

        // Baseline progress arc
        item {
            InfoCard("Baseline Progress (P₀)", headerColor = SoftCyan) {
                if (isBuilding) {
                    val target = baselineDaysReq.toFloat()
                    val frac = (progress / target).coerceIn(0f, 1f)
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        ArcProgressRing(progress.toFloat(), target, SoftCyan, "Days", "/ ${target.toInt()}", size = 90.dp)
                        Spacer(Modifier.width(16.dp))
                        Column {
                            Text("Learning Your Unique Patterns", fontWeight = FontWeight.SemiBold, color = TextPrimary)
                            Text("Day $progress of ${target.toInt()} in mathematically establishing your scientific P₀ baseline. Collecting multidimensional behavioral data continuously for accuracy.", fontSize = 12.sp, color = TextSecondary, lineHeight = 16.sp)
                            Spacer(Modifier.height(6.dp))
                            LinearProgressIndicator(
                                progress = { frac },
                                color = SoftCyan,
                                trackColor = SoftCyan.copy(0.15f),
                                modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(4.dp))
                            )
                        }
                    }
                } else {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        val statusText = if (analysisResult != null) {
                            "Baseline Locked - ${analysisResult?.alertLevel?.uppercase() ?: "UNKNOWN"} Status"
                        } else {
                            "Scientific Baseline Established"
                        }
                        val statusColor = analysisResult?.let { alertColorForLevel(it.alertLevel) } ?: AlertGreen
                        val isHighRisk = analysisResult?.alertLevel?.lowercase() in listOf("orange", "red")
                        val icon = if (isHighRisk) Icons.Default.Warning else Icons.Default.CheckCircle
                        
                        Icon(icon, null, tint = statusColor, modifier = Modifier.size(40.dp))
                        Spacer(Modifier.width(12.dp))
                        Column {
                            Text(statusText, fontWeight = FontWeight.SemiBold, color = statusColor)
                            
                            val descriptionText = if (analysisResult != null) {
                                "Your current behavioral vector is being compared against your ${baselineDaysReq}-day P₀ baseline. " + 
                                when(analysisResult?.alertLevel?.lowercase()) {
                                    "green" -> "Data indicates high alignment with your normal routines."
                                    "yellow" -> "Slight deviations from your baseline detected. Tracking for potential shifts."
                                    "orange" -> "Moderate departure from baseline established. Behavioral patterns show significant variance."
                                    "red" -> "Critical deviation from your established P₀. Immediate attention recommended."
                                    else -> "Continuous monitoring active."
                                }
                            } else {
                                "${baselineDaysReq}-day P₀ vector is locked. Real-time multidimensional tracking is now active."
                            }
                            Text(descriptionText, fontSize = 12.sp, color = TextSecondary, lineHeight = 16.sp)
                        }
                    }
                }
                
                if (baselineVectors.isNotEmpty()) {
                    Spacer(Modifier.height(20.dp))
                    Text(if (isBuilding) "Multi-Sensor Formation Trend" else "Composite Behavioral Index", fontSize = 13.sp, color = TextPrimary, fontWeight = FontWeight.Medium)
                    Spacer(Modifier.height(12.dp))
                    
                    val composite = baselineVectors.takeLast(baselineDaysReq).map { v ->
                        val screen = (v.screenTimeHours / 12f).coerceIn(0f, 1f) * 40f
                        val move = (v.dailyDisplacementKm / 20f).coerceIn(0f, 1f) * 30f
                        val comms = (v.callsPerDay / 10f).coerceIn(0f, 1f) * 30f
                        screen + move + comms
                    }
                    
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.Bottom) {
                        Text("Activity Index (Last $baselineDaysReq Days)", fontSize = 11.sp, color = TextSecondary)
                        if (composite.isNotEmpty()) {
                            Text("%.0f".format(composite.last()), fontSize = 14.sp, fontWeight = FontWeight.Bold, color = SoftCyan)
                        }
                    }
                    Spacer(Modifier.height(4.dp))
                    SparklineChart(composite, SoftCyan, Modifier.fillMaxWidth().height(80.dp), showDots = true)
                }
            }
        }

        // Intraday sparklines
        item {
            InfoCard("Today's Intraday Trends", headerColor = ChartPurple) {
                if (hourly.size < 2) {
                    Text("Collecting hourly snapshots…", color = TextSecondary, fontSize = 12.sp)
                } else {
                    val screenTimes = hourly.map { it.screenTimeHours }
                    val places = hourly.map { it.placesVisited }
                    val distances = hourly.map { it.dailyDisplacementKm }
                    SparklineLabel("Screen Time (hrs)", screenTimes, OceanBlue)
                    Spacer(Modifier.height(12.dp))
                    SparklineLabel("Distance (km)", distances, ChartRed)
                    Spacer(Modifier.height(12.dp))
                    SparklineLabel("Places Visited", places, ChartPurple)
                }
            }
        }

        // Current vs Baseline comparison (only available post-baseline)
        if (!isBuilding && baseline != null && vector != null) {
            item {
                InfoCard("Current vs Baseline", headerColor = OceanBlue) {
                    val v = checkNotNull(vector); val b = checkNotNull(baseline)
                    val rows = listOf(
                        Triple("Screen Time", v.screenTimeHours, b.screenTimeHours),
                        Triple("Places Visited", v.placesVisited, b.placesVisited),
                        Triple("Calls/Day", v.callsPerDay, b.callsPerDay),
                        Triple("Social Ratio %", v.socialAppRatio * 100, b.socialAppRatio * 100),
                        Triple("Sleep Hours", v.sleepDurationHours, b.sleepDurationHours),
                        Triple("Displacement (km)", v.dailyDisplacementKm, b.dailyDisplacementKm)
                    )
                    rows.forEach { (label, cur, base) ->
                        ComparisonRow(label, cur, base)
                    }
                }
            }
        }

        // Full baseline feature table (all features, mean ± σ vs current)
        if (!isBuilding && baseline != null && vector != null) {
            item {
                baseline?.let { b -> vector?.let { v -> FeatureTableCard(baseline = b, current = v) } }
            }
        }

        // Per-App Breakdown section
        if (!isBuilding && vector != null) {
            item {
                vector?.let { v -> 
                    PerAppBreakdownCard(vector = v)
                    BgAudioBreakdownCard(vector = v)
                }
            }
        }

        // ── System 1 DNA Profile Section ────────────────────────────────────
        item {
            DnaProfileSection(profileJson = s1ProfileJson)
        }

        // ── L2 Digital DNA Section ──────────────────────────────────────────
        item {
            ScreenHeader(
                "Digital DNA",
                "Level 2 behavioral anomaly analysis",
                Icons.Default.Shield
            )
        }

        // 5. Alert Status + Effective Score Gauge
        item {
            val result = analysisResult
            val alertLevel = result?.alertLevel ?: "green"
            val effectiveScore = result?.effectiveScore ?: 0f
            val alertColor = alertColorForLevel(alertLevel)

            InfoCard("Alert Status & Score", headerColor = alertColor) {
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceEvenly,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    // Alert level badge
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Box(
                            modifier = Modifier
                                .size(56.dp)
                                .clip(CircleShape)
                                .background(alertColor.copy(alpha = 0.15f)),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                alertIconForLevel(alertLevel),
                                fontSize = 24.sp
                            )
                        }
                        Spacer(Modifier.height(4.dp))
                        Text(
                            alertLevel.uppercase(),
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold,
                            color = alertColor
                        )
                        Text(
                            "Alert Level",
                            fontSize = 10.sp,
                            color = TextSecondary
                        )
                    }

                    // Effective Score gauge
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        AnomalyScoreGauge(
                            score = effectiveScore,
                            modifier = Modifier.size(120.dp, 70.dp)
                        )
                        Text(
                            "%.2f".format(effectiveScore),
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold,
                            color = TextPrimary
                        )
                        Text("Effective Score", fontSize = 10.sp, color = TextSecondary)
                    }
                }
            }
        }

        // 6. L2 Behavioral Texture Metrics
        item {
            val result = analysisResult
            InfoCard("Behavioral Texture (L2)", headerColor = SwasthitiAccentPurple) {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    // L2 Modifier
                    L2MetricRow(
                        label = "L2 Modifier",
                        value = result?.l2Modifier ?: 1.0f,
                        maxValue = 2.0f,
                        color = l2ModifierColor(result?.l2Modifier ?: 1.0f),
                        description = modifierDescription(result?.l2Modifier ?: 1.0f)
                    )

                    HorizontalDivider(color = Color(0xFFE2E8F0), thickness = 1.dp)

                    // Context Coherence
                    L2MetricRow(
                        label = "Context Coherence",
                        value = result?.coherence ?: 0f,
                        maxValue = 1.0f,
                        color = SwasthitiTeal,
                        description = if ((result?.coherence ?: 0f) > 0.5f) "Matches known pattern" else "Unfamiliar pattern"
                    )

                    HorizontalDivider(color = Color(0xFFE2E8F0), thickness = 1.dp)

                    // Rhythm Dissolution
                    L2MetricRow(
                        label = "Rhythm Dissolution",
                        value = result?.rhythmDissolution ?: 0f,
                        maxValue = 1.0f,
                        color = if ((result?.rhythmDissolution ?: 0f) > 0.5f) AlertOrange else SwasthitiIndigo,
                        description = if ((result?.rhythmDissolution ?: 0f) > 0.5f) "Usage rhythm scattered" else "Rhythm intact"
                    )

                    HorizontalDivider(color = Color(0xFFE2E8F0), thickness = 1.dp)

                    // Session Incoherence
                    L2MetricRow(
                        label = "Session Incoherence",
                        value = result?.sessionIncoherence ?: 0f,
                        maxValue = 1.0f,
                        color = if ((result?.sessionIncoherence ?: 0f) > 0.5f) AlertRed else SwasthitiChartIndigo,
                        description = if ((result?.sessionIncoherence ?: 0f) > 0.5f) "Sessions degrading" else "Sessions healthy"
                    )
                }
            }
        }

        // 7. Evidence Accumulation & Pattern
        item {
            val result = analysisResult
            val evidenceHistory = analysisHistory.map { it.evidenceAccumulated }

            InfoCard("Evidence Engine", headerColor = SwasthitiChartSlate) {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    // Evidence stats row
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceEvenly
                    ) {
                        EvidenceStatPill(
                            "Evidence",
                            "%.2f".format(result?.evidenceAccumulated ?: 0f),
                            SwasthitiIndigo
                        )
                        EvidenceStatPill(
                            "Sustained",
                            "${result?.sustainedDays ?: 0}d",
                            if ((result?.sustainedDays ?: 0) >= 5) AlertOrange else SwasthitiTeal
                        )
                        EvidenceStatPill(
                            "Pattern",
                            patternLabel(result?.patternType ?: "stable"),
                            patternColor(result?.patternType ?: "stable")
                        )
                    }

                    // Evidence sparkline (last 14 days)
                    if (evidenceHistory.size >= 2) {
                        SparklineLabel(
                            "Evidence Trend (${evidenceHistory.size}d)",
                            evidenceHistory,
                            SwasthitiChartIndigo
                        )
                    }

                    // Flagged Features
                    val flagged = parseFlaggedFeatures(result?.flaggedFeatures ?: "[]")
                    if (flagged.isNotEmpty()) {
                        HorizontalDivider(color = Color(0xFFE2E8F0), thickness = 1.dp)
                        Text(
                            "Flagged Features",
                            fontSize = 12.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = TextSecondary
                        )
                        Spacer(Modifier.height(4.dp))
                        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                            flagged.take(5).forEach { feature ->
                                FlaggedFeatureChip(feature)
                            }
                        }
                    }
                }
            }
        }

        item { Spacer(Modifier.height(16.dp)) }
    }
}

// ── Helper composables for L2 visualization ──────────────────────────────────

@Composable
private fun L2MetricRow(
    label: String,
    value: Float,
    maxValue: Float,
    color: Color,
    description: String
) {
    Column {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(label, fontSize = 12.sp, color = TextSecondary, fontWeight = FontWeight.Medium)
            Text(
                "%.2f".format(value),
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold,
                color = color
            )
        }
        Spacer(Modifier.height(4.dp))
        // Progress bar
        Box(
            Modifier
                .fillMaxWidth()
                .height(6.dp)
                .clip(RoundedCornerShape(3.dp))
                .background(color.copy(alpha = 0.15f))
        ) {
            Box(
                Modifier
                    .fillMaxHeight()
                    .fillMaxWidth((value / maxValue.coerceAtLeast(0.01f)).coerceIn(0f, 1f))
                    .clip(RoundedCornerShape(3.dp))
                    .background(color)
            )
        }
        Spacer(Modifier.height(2.dp))
        Text(description, fontSize = 10.sp, color = TextMuted)
    }
}

@Composable
private fun EvidenceStatPill(
    label: String,
    value: String,
    color: Color
) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(10.dp))
                .background(color.copy(alpha = 0.1f))
                .padding(horizontal = 14.dp, vertical = 6.dp)
        ) {
            Text(
                value,
                fontSize = 14.sp,
                fontWeight = FontWeight.ExtraBold,
                color = color
            )
        }
        Spacer(Modifier.height(3.dp))
        Text(label, fontSize = 10.sp, color = TextSecondary, fontWeight = FontWeight.Medium)
    }
}

@Composable
private fun FlaggedFeatureChip(feature: String) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(8.dp))
            .background(AlertOrange.copy(alpha = 0.08f))
            .padding(horizontal = 10.dp, vertical = 5.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(6.dp)
                    .clip(CircleShape)
                    .background(AlertOrange)
            )
            Spacer(Modifier.width(6.dp))
            Text(
                feature,
                fontSize = 11.sp,
                color = AlertOrange,
                fontWeight = FontWeight.Medium
            )
        }
    }
}

// ── Utility functions ────────────────────────────────────────────────────────

private fun alertColorForLevel(level: String): Color = when (level.lowercase()) {
    "green" -> AlertGreen
    "yellow" -> AlertYellow
    "orange" -> AlertOrange
    "red" -> AlertRed
    else -> AlertGreen
}

private fun alertIconForLevel(level: String): String = when (level.lowercase()) {
    "green" -> "✅"
    "yellow" -> "⚠️"
    "orange" -> "🟠"
    "red" -> "🔴"
    else -> "✅"
}

private fun l2ModifierColor(modifier: Float): Color = when {
    modifier < 0.5f -> AlertGreen       // Strong suppression — known context
    modifier < 0.9f -> SwasthitiTeal      // Moderate suppression
    modifier < 1.1f -> SwasthitiIndigo    // Neutral
    modifier < 1.5f -> AlertOrange      // Moderate amplification
    else -> AlertRed                     // Strong amplification — clinical signal
}

private fun modifierDescription(modifier: Float): String = when {
    modifier < 0.5f -> "Strongly suppressed — known context"
    modifier < 0.9f -> "Partially suppressed — mostly matches baseline"
    modifier < 1.1f -> "Neutral — mixed signals"
    modifier < 1.5f -> "Amplified — unfamiliar pattern detected"
    else -> "Strongly amplified — clinical signal"
}

private fun patternLabel(pattern: String): String = when (pattern) {
    "rapid_cycling" -> "Cycling"
    "acute_spike" -> "Acute"
    "gradual_drift" -> "Drift"
    "mixed_pattern" -> "Mixed"
    "stable" -> "Stable"
    else -> pattern.replaceFirstChar { it.uppercase() }
}

private fun patternColor(pattern: String): Color = when (pattern) {
    "stable" -> AlertGreen
    "gradual_drift" -> SwasthitiTeal
    "mixed_pattern" -> AlertYellow
    "rapid_cycling" -> AlertOrange
    "acute_spike" -> AlertRed
    else -> TextSecondary
}

private fun parseFlaggedFeatures(json: String): List<String> {
    return try {
        // Simple parse: remove [ ] and quotes, split by comma
        val cleaned = json.trim().removeSurrounding("[", "]").removeSurrounding("\"", "\"")
        if (cleaned.isBlank()) emptyList()
        else cleaned.split("\",\"").map { it.removeSurrounding("\"").trim() }.filter { it.isNotBlank() }
    } catch (e: Exception) {
        emptyList()
    }
}
