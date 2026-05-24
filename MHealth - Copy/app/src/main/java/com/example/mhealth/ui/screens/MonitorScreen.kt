package com.example.mhealth.ui.screens
 
import kotlin.math.abs

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Circle
import androidx.compose.material.icons.filled.Hub
import androidx.compose.material.icons.filled.Sensors
import androidx.compose.material.icons.filled.Shield
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.mhealth.logic.DataRepository
import com.example.mhealth.alertColor
import com.example.mhealth.ui.components.*
import com.example.mhealth.ui.charts.ArcProgressRing
import com.example.mhealth.ui.charts.AnomalyScoreGauge
import com.example.mhealth.ui.charts.SparklineChart
import com.example.mhealth.ui.theme.*
import androidx.compose.ui.graphics.Brush
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material.icons.filled.CheckCircle
import com.example.mhealth.FeatureTableCard
import com.example.mhealth.PerAppBreakdownCard
import com.example.mhealth.BgAudioBreakdownCard
import org.json.JSONObject

@Composable
fun MonitorScreen() {
    val progress by DataRepository.baselineProgress.collectAsState()
    val isBuilding by DataRepository.isBuildingBaseline.collectAsState()
    val vector by DataRepository.latestVector.collectAsState()
    val baseline by DataRepository.baseline.collectAsState()
    val hourly by DataRepository.hourlySnapshots.collectAsState()
    val baselineDaysReq by DataRepository.baselineDaysRequired.collectAsState()
    val baselineVectors by DataRepository.collectedBaselineVectors.collectAsState()
    val latestResult by DataRepository.latestAnalysisResult.collectAsState()
    val s1ProfileJson by DataRepository.s1ProfileJson.collectAsState()

    // Parse anchor clusters from profile JSON
    val profileObj = remember(s1ProfileJson) {
        if (s1ProfileJson.isNullOrBlank() || s1ProfileJson == "{}") null
        else try { JSONObject(s1ProfileJson!!) } catch (_: Exception) { null }
    }

    LazyColumn(Modifier.fillMaxSize()) {
        item { HeaderSection(isBuilding) }

        item {
            BaselineProgressCard(
                progress = progress,
                target = baselineDaysReq,
                isBuilding = isBuilding,
                latestResult = latestResult,
                baselineVectors = baselineVectors
            )
        }

        if (!isBuilding && latestResult != null) {
            item {
                AnomalyScoreFlowCard(latestResult = latestResult!!)
            }
        }

        if (!isBuilding && profileObj != null) {
            item { AnchorClustersCard(profileObj!!) }
        }

        item { IntradayTrendsCard(hourly) }

        if (!isBuilding && baseline != null && vector != null) {
            item {
                ComparisonCard(vector = vector!!, baseline = baseline!!)
            }
            item {
                FeatureTableCard(baseline = baseline!!, current = vector!!)
            }
        }

        if (!isBuilding && vector != null) {
            item {
                PerAppBreakdownCard(vector = vector!!)
                BgAudioBreakdownCard(vector = vector!!)
            }
        }

        item { Spacer(Modifier.height(16.dp)) }
    }
}

@Composable
private fun HeaderSection(isBuilding: Boolean) {
    Box(
        Modifier.fillMaxWidth()
            .background(Brush.horizontalGradient(listOf(OceanBlue, AccentBlue)))
            .padding(20.dp)
    ) {
        Column {
            Text("Baseline & Monitoring", fontSize = 24.sp, fontWeight = FontWeight.Bold, color = Color.White)
            Text("Layers 2 & 3 — ${if (isBuilding) "Building Personal Normal" else "Continuous Tracking"}", fontSize = 13.sp, color = Color.White.copy(0.85f))
        }
    }
}

@Composable
private fun BaselineProgressCard(
    progress: Int,
    target: Int,
    isBuilding: Boolean,
    latestResult: com.example.mhealth.logic.db.AnalysisResultEntity?,
    baselineVectors: List<com.example.mhealth.models.PersonalityVector>
) {
    val isMonitoring = !isBuilding
    val displayProgress = remember(progress, target, isMonitoring) {
        if (isMonitoring) progress.toFloat() else progress.toFloat().coerceAtMost(target.toFloat())
    }
    val displayMax = remember(progress, target, isMonitoring) {
        if (isMonitoring) progress.toFloat().coerceAtLeast(1f) else target.toFloat()
    }

    InfoCard(
        if (isMonitoring) "Monitoring Active (P₀)" else "Baseline Progress (P₀)",
        headerColor = if (isMonitoring) AlertGreen else SoftCyan
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            ArcProgressRing(
                value = displayProgress,
                maxValue = displayMax,
                color = if (isMonitoring) AlertGreen else SoftCyan,
                label = if (isMonitoring) "Monitored" else "Days",
                unit = if (isMonitoring) "Total" else "/ $target",
                size = 90.dp
            )
            Spacer(Modifier.width(16.dp))
            Column {
                if (isMonitoring) {
                    val statusColor = latestResult?.let { alertColor(it.alertLevel) } ?: AlertGreen
                    Text("Continuous Tracking Active", fontWeight = FontWeight.Bold, color = statusColor)
                    Text("Tracking your days over P₀. Comparison against your $target-day established baseline is live.", fontSize = 12.sp, color = TextSecondary, lineHeight = 16.sp)
                } else {
                    val frac = (progress / target.toFloat().coerceAtLeast(1f)).coerceIn(0f, 1f)
                    Text("Learning Your Unique Patterns", fontWeight = FontWeight.Bold, color = TextPrimary)
                    Text("Day $progress of $target in establishing your P₀ baseline.", fontSize = 12.sp, color = TextSecondary, lineHeight = 16.sp)
                    Spacer(Modifier.height(8.dp))
                    LinearProgressIndicator(
                        progress = { frac },
                        color = SoftCyan,
                        trackColor = SoftCyan.copy(0.15f),
                        modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(4.dp))
                    )
                }
            }
        }

        if (isMonitoring && latestResult != null) {
            Spacer(Modifier.height(16.dp))
            HorizontalDivider(color = Color.Gray.copy(0.1f))
            Spacer(Modifier.height(12.dp))
            val statusColor = alertColor(latestResult.alertLevel)
            Text(
                when(latestResult.alertLevel.lowercase()) {
                    "green" -> "Data indicates high alignment with your normal routines."
                    "yellow" -> "Slight deviations from your baseline detected."
                    "orange" -> "Significant departure from baseline established."
                    "red" -> "Critical deviation from your established P₀."
                    else -> "Continuous monitoring active."
                },
                fontSize = 12.sp, color = statusColor, fontWeight = FontWeight.Medium
            )
        }

        if (baselineVectors.isNotEmpty()) {
            Spacer(Modifier.height(20.dp))
            Text(if (isBuilding) "Multi-Sensor Formation Trend" else "Composite Behavioral Index", fontSize = 13.sp, color = TextPrimary, fontWeight = FontWeight.Medium)
            Spacer(Modifier.height(12.dp))

            val composite = remember(baselineVectors, target) {
                baselineVectors.takeLast(target).map { v ->
                    (v.screenTimeHours / 12f).coerceIn(0f, 1f) * 40f +
                    (v.dailyDisplacementKm / 20f).coerceIn(0f, 1f) * 30f +
                    (v.callsPerDay / 10f).coerceIn(0f, 1f) * 30f
                }
            }

            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.Bottom) {
                Text("Activity Index (Last $target Days)", fontSize = 11.sp, color = TextSecondary)
                if (composite.isNotEmpty()) {
                    Text("%.0f".format(composite.last()), fontSize = 14.sp, fontWeight = FontWeight.Bold, color = SoftCyan)
                }
            }
            Spacer(Modifier.height(4.dp))
            SparklineChart(composite, SoftCyan, Modifier.fillMaxWidth().height(80.dp), showDots = true)
        }
    }
}

@Composable
private fun IntradayTrendsCard(hourly: List<com.example.mhealth.models.PersonalityVector>) {
    InfoCard("Today's Intraday Trends", headerColor = ChartPurple) {
        if (hourly.size < 2) {
            Text("Collecting hourly snapshots…", color = TextSecondary, fontSize = 12.sp)
        } else {
            val screenTimes = remember(hourly) { hourly.map { it.screenTimeHours } }
            val distances = remember(hourly) { hourly.map { it.dailyDisplacementKm } }

            SparklineLabel("Screen Time (hrs)", screenTimes, OceanBlue)
            Spacer(Modifier.height(12.dp))
            SparklineLabel("Distance (km)", distances, ChartRed)
        }
    }
}

@Composable
private fun ComparisonCard(
    vector: com.example.mhealth.models.PersonalityVector,
    baseline: com.example.mhealth.models.PersonalityVector
) {
    InfoCard("Current vs Baseline", headerColor = OceanBlue) {
        val rows = remember(vector, baseline) {
            listOf(
                Triple("Screen Time", vector.screenTimeHours, baseline.screenTimeHours),
                Triple("Calls/Day", vector.callsPerDay, baseline.callsPerDay),
                Triple("Social Ratio %", vector.socialAppRatio * 100, baseline.socialAppRatio * 100),
                Triple("Sleep Hours", vector.sleepDurationHours, baseline.sleepDurationHours),
                Triple("Displacement (km)", vector.dailyDisplacementKm, baseline.dailyDisplacementKm)
            )
        }
        rows.forEach { (label, cur, base) ->
            Column {
                ComparisonRow(label, cur, base)
                val diff = cur - base
                val pct = if (base != 0f) ((diff / base) * 100).toInt() else 0
                val deltaColor = when {
                    abs(pct) <= 10 -> AlertGreen
                    abs(pct) <= 30 -> AlertYellow
                    else -> AlertRed
                }
                if (pct != 0) {
                    Text(
                        "${if (pct > 0) "+" else ""}${pct}% from baseline",
                        fontSize = 10.sp, color = deltaColor, fontWeight = FontWeight.Medium,
                        modifier = Modifier.padding(start = 4.dp, bottom = 4.dp)
                    )
                }
            }
        }
    }
}

// ── L1 Anchor Clusters Card (Behavioral Archetypes) ──────────────────────────

@Composable
private fun AnchorClustersCard(profile: JSONObject) {
    val clustersArr = profile.optJSONArray("anchor_clusters") ?: return
    if (clustersArr.length() == 0) return

    val clusterMethod = clustersArr.optJSONObject(0)?.optString("method", "clinical_pca_meanshift")
        ?.replace("_", " ") ?: "PCA + Mean-Shift"

    InfoCard(
        "Behavioral Archetypes (L1 Clusters)",
        headerColor = AccentPurple
    ) {
        Text(
            "${clustersArr.length()} cluster(s) · $clusterMethod",
            color = TextSecondary, fontSize = 11.sp
        )
        Spacer(Modifier.height(8.dp))

        for (i in 0 until clustersArr.length()) {
            val cluster = clustersArr.optJSONObject(i) ?: continue
            val clusterId = cluster.optInt("cluster_id", 0)
            val memberCount = cluster.optInt("member_count", 0)
            val radius = cluster.optDouble("radius", 0.0)
            val centroidFeatures = cluster.optJSONObject("centroid_features") ?: continue
            val memberDates = cluster.optJSONArray("member_dates")

            val clusterColor = when (i % 4) {
                0 -> AccentPurple; 1 -> AccentBlue; 2 -> AccentGreen; else -> AccentOrange
            }

            Card(
                modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                colors = CardDefaults.cardColors(containerColor = BgLight),
                shape = RoundedCornerShape(8.dp)
            ) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.Circle, null, tint = clusterColor, modifier = Modifier.size(12.dp))
                        Spacer(Modifier.width(6.dp))
                        Text("Archetype $clusterId", color = clusterColor, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                        Spacer(Modifier.weight(1f))
                        Text("$memberCount days", color = TextSecondary, fontSize = 11.sp)
                        Spacer(Modifier.width(8.dp))
                        Text("r=${String.format("%.2f", radius)}", color = TextSecondary, fontSize = 11.sp)
                    }
                    Spacer(Modifier.height(6.dp))

                    // Centroid feature bars (top 6 features)
                    val maxVal = centroidFeatures.keys().asSequence().mapNotNull {
                        kotlin.runCatching { Math.abs(centroidFeatures.optDouble(it)) }.getOrNull()
                    }.maxOrNull() ?: 1.0

                    centroidFeatures.keys().asSequence().take(6).forEach { feat ->
                        val value = centroidFeatures.optDouble(feat, 0.0)
                        val fraction = (Math.abs(value) / maxVal).toFloat().coerceIn(0f, 1f)
                        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(vertical = 1.dp)) {
                            Text(feat, color = TextSecondary, fontSize = 9.sp, modifier = Modifier.width(120.dp),
                                maxLines = 1, overflow = TextOverflow.Ellipsis)
                            Box(
                                modifier = Modifier.weight(1f).height(4.dp).background(BorderLight, RoundedCornerShape(2.dp))
                            ) {
                                Box(
                                    modifier = Modifier.fillMaxWidth(fraction).fillMaxHeight()
                                        .background(clusterColor, RoundedCornerShape(2.dp))
                                )
                            }
                            Text(String.format("%.1f", value), color = TextSecondary, fontSize = 9.sp,
                                modifier = Modifier.width(40.dp))
                        }
                    }

                    // Member dates
                    if (memberDates != null && memberDates.length() > 0) {
                        Spacer(Modifier.height(4.dp))
                        Text("Dates: ${(0 until minOf(memberDates.length(), 5)).joinToString(", ") { memberDates.getString(it) }}${if (memberDates.length() > 5) " …" else ""}",
                            color = TextSecondary, fontSize = 9.sp)
                    }
                }
            }
        }
    }
}

@Composable
private fun AnomalyScoreFlowCard(
    latestResult: com.example.mhealth.logic.db.AnalysisResultEntity
) {
    InfoCard(
        title = "Daily Anomaly Diagnostics Flow",
        headerColor = AccentBlue
    ) {
        Column(Modifier.fillMaxWidth()) {
            Text(
                "Each night, the dual-layer diagnostic pipeline processes 30 Surface features and millions of session data points. Here is today's step-by-step mathematical flow.",
                fontSize = 12.sp,
                color = TextSecondary,
                lineHeight = 16.sp
            )
            Spacer(Modifier.height(16.dp))

            // STEP 1: Layer 1
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier
                        .size(24.dp)
                        .background(AccentBlue.copy(0.15f), CircleShape),
                    contentAlignment = Alignment.Center
                ) {
                    Text("1", color = AccentBlue, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                }
                Spacer(Modifier.width(8.dp))
                Text(
                    "Layer 1: Raw Surface Anomaly Score",
                    fontWeight = FontWeight.Bold,
                    fontSize = 13.sp,
                    color = TextPrimary
                )
                Spacer(Modifier.weight(1f))
                Text(
                    String.format("%.3f", latestResult.anomalyScore),
                    fontWeight = FontWeight.ExtraBold,
                    fontSize = 15.sp,
                    color = AccentBlue
                )
            }
            Spacer(Modifier.height(4.dp))
            Text(
                "Aggregated Mahalanobis Z-score magnitude and EWMA velocity of all 30 L1 telemetry features.",
                fontSize = 10.sp,
                color = TextSecondary,
                lineHeight = 14.sp
            )
            Spacer(Modifier.height(8.dp))
            LinearProgressIndicator(
                progress = { latestResult.anomalyScore.coerceIn(0f, 1f) },
                color = AccentBlue,
                trackColor = BorderLight,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(6.dp)
                    .clip(RoundedCornerShape(3.dp))
            )

            // CONNECTOR ×
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 10.dp),
                contentAlignment = Alignment.Center
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        Modifier
                            .width(1.dp)
                            .height(16.dp)
                            .background(BorderLight)
                    )
                    Spacer(Modifier.width(12.dp))
                    Text("×", fontSize = 20.sp, fontWeight = FontWeight.Bold, color = TextSecondary)
                    Spacer(Modifier.width(12.dp))
                    Box(
                        Modifier
                            .width(1.dp)
                            .height(16.dp)
                            .background(BorderLight)
                    )
                }
            }

            // STEP 2: Layer 2
            val modifierColor = when {
                latestResult.l2Modifier < 0.9f -> AlertGreen
                latestResult.l2Modifier > 1.1f -> AlertOrange
                else -> TextSecondary
            }
            val modifierLabel = when {
                latestResult.l2Modifier < 0.9f -> "Suppression (Matches Known Routine)"
                latestResult.l2Modifier > 1.1f -> "Amplification (Degraded/Disorganized)"
                else -> "Neutral (No Modifier Influence)"
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier
                        .size(24.dp)
                        .background(modifierColor.copy(0.15f), CircleShape),
                    contentAlignment = Alignment.Center
                ) {
                    Text("2", color = modifierColor, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                }
                Spacer(Modifier.width(8.dp))
                Text(
                    "Layer 2: Digital DNA Modifier",
                    fontWeight = FontWeight.Bold,
                    fontSize = 13.sp,
                    color = TextPrimary
                )
                Spacer(Modifier.weight(1f))
                Text(
                    String.format("× %.3f", latestResult.l2Modifier),
                    fontWeight = FontWeight.ExtraBold,
                    fontSize = 15.sp,
                    color = modifierColor
                )
            }
            Spacer(Modifier.height(4.dp))
            Text(
                "Effect: $modifierLabel",
                fontSize = 11.sp,
                color = modifierColor,
                fontWeight = FontWeight.SemiBold
            )
            Spacer(Modifier.height(8.dp))

            Card(
                colors = CardDefaults.cardColors(containerColor = BorderLight.copy(0.2f)),
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    // Context Coherence
                    Column {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(
                                "Context Coherence",
                                fontSize = 12.sp,
                                color = TextPrimary,
                                fontWeight = FontWeight.SemiBold
                            )
                            Spacer(Modifier.weight(1f))
                            Text(
                                String.format("%.2f", latestResult.coherence),
                                fontSize = 12.sp,
                                color = AlertGreen,
                                fontWeight = FontWeight.Bold
                            )
                        }
                        Text(
                            "Measures alignment with discovered DBSCAN baseline archetypes (reduces anomaly).",
                            fontSize = 10.sp,
                            color = TextSecondary,
                            lineHeight = 13.sp
                        )
                        Spacer(Modifier.height(4.dp))
                        LinearProgressIndicator(
                            progress = { latestResult.coherence.coerceIn(0f, 1f) },
                            color = AlertGreen,
                            trackColor = BorderLight,
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(4.dp)
                                .clip(RoundedCornerShape(2.dp))
                        )
                    }

                    // Rhythm Dissolution
                    Column {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(
                                "Rhythm Dissolution",
                                fontSize = 12.sp,
                                color = TextPrimary,
                                fontWeight = FontWeight.SemiBold
                            )
                            Spacer(Modifier.weight(1f))
                            Text(
                                String.format("%.2f", latestResult.rhythmDissolution),
                                fontSize = 12.sp,
                                color = AlertOrange,
                                fontWeight = FontWeight.Bold
                            )
                        }
                        Text(
                            "KL divergence of today's hourly app usage from typical day-of-week DNA (amplifies anomaly).",
                            fontSize = 10.sp,
                            color = TextSecondary,
                            lineHeight = 13.sp
                        )
                        Spacer(Modifier.height(4.dp))
                        LinearProgressIndicator(
                            progress = { latestResult.rhythmDissolution.coerceIn(0f, 1f) },
                            color = AlertOrange,
                            trackColor = BorderLight,
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(4.dp)
                                .clip(RoundedCornerShape(2.dp))
                        )
                    }

                    // Session Incoherence
                    Column {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(
                                "Session Incoherence",
                                fontSize = 12.sp,
                                color = TextPrimary,
                                fontWeight = FontWeight.SemiBold
                            )
                            Spacer(Modifier.weight(1f))
                            Text(
                                String.format("%.2f", latestResult.sessionIncoherence),
                                fontSize = 12.sp,
                                color = ChartRed,
                                fontWeight = FontWeight.Bold
                            )
                        }
                        Text(
                            "Abandon rate spike + duration collapse on high-depth apps (amplifies anomaly).",
                            fontSize = 10.sp,
                            color = TextSecondary,
                            lineHeight = 13.sp
                        )
                        Spacer(Modifier.height(4.dp))
                        LinearProgressIndicator(
                            progress = { latestResult.sessionIncoherence.coerceIn(0f, 1f) },
                            color = ChartRed,
                            trackColor = BorderLight,
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(4.dp)
                                .clip(RoundedCornerShape(2.dp))
                        )
                    }
                }
            }

            // CONNECTOR =
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 10.dp),
                contentAlignment = Alignment.Center
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        Modifier
                            .width(1.dp)
                            .height(16.dp)
                            .background(BorderLight)
                    )
                    Spacer(Modifier.width(12.dp))
                    Text("=", fontSize = 20.sp, fontWeight = FontWeight.Bold, color = TextSecondary)
                    Spacer(Modifier.width(12.dp))
                    Box(
                        Modifier
                            .width(1.dp)
                            .height(16.dp)
                            .background(BorderLight)
                    )
                }
            }

            // STEP 3: Effective Score
            val effectiveScoreColor = alertColor(latestResult.alertLevel)
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier
                        .size(28.dp)
                        .background(effectiveScoreColor.copy(0.15f), CircleShape),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        Icons.Default.Shield,
                        contentDescription = null,
                        tint = effectiveScoreColor,
                        modifier = Modifier.size(16.dp)
                    )
                }
                Spacer(Modifier.width(8.dp))
                Text(
                    "Effective Fused Score",
                    fontWeight = FontWeight.Bold,
                    fontSize = 14.sp,
                    color = TextPrimary
                )
                Spacer(Modifier.weight(1f))
                Text(
                    String.format("%.3f", latestResult.effectiveScore),
                    fontWeight = FontWeight.ExtraBold,
                    fontSize = 18.sp,
                    color = effectiveScoreColor
                )
            }
            Spacer(Modifier.height(4.dp))
            Text(
                "Trigger threshold: 0.380 (Score below 0.38 is healthy)",
                fontSize = 10.sp,
                color = TextSecondary
            )
            Spacer(Modifier.height(8.dp))
            Box(
                Modifier
                    .fillMaxWidth()
                    .height(8.dp)
                    .background(BorderLight, RoundedCornerShape(4.dp))
            ) {
                Box(
                    Modifier
                        .fillMaxWidth(latestResult.effectiveScore.coerceIn(0f, 1f))
                        .fillMaxHeight()
                        .background(effectiveScoreColor, RoundedCornerShape(4.dp))
                )
                // Threshold mark at 38%
                Box(
                    Modifier
                        .fillMaxWidth(0.38f)
                        .fillMaxHeight()
                        .width(2.dp)
                        .background(TextPrimary)
                )
            }
            Spacer(Modifier.height(4.dp))
            Row(Modifier.fillMaxWidth()) {
                Text("0.00", fontSize = 9.sp, color = TextSecondary)
                Spacer(Modifier.weight(0.38f))
                Text(
                    "Gate (0.38)",
                    fontSize = 9.sp,
                    color = TextPrimary,
                    fontWeight = FontWeight.Bold
                )
                Spacer(Modifier.weight(0.62f))
                Text(
                    "1.00",
                    fontSize = 9.sp,
                    color = TextSecondary,
                    modifier = Modifier.align(Alignment.End)
                )
            }

            // STEP 4: Evidence
            Spacer(Modifier.height(16.dp))
            HorizontalDivider(color = Color.Gray.copy(0.1f))
            Spacer(Modifier.height(12.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    if (latestResult.effectiveScore > 0.38f) Icons.Default.Warning else Icons.Default.CheckCircle,
                    contentDescription = null,
                    tint = if (latestResult.effectiveScore > 0.38f) AlertOrange else AlertGreen,
                    modifier = Modifier.size(18.dp)
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    if (latestResult.effectiveScore > 0.38f) "Compounding Sustained Evidence" else "Normal (Decaying Evidence)",
                    fontWeight = FontWeight.Bold,
                    fontSize = 12.sp,
                    color = TextPrimary
                )
                Spacer(Modifier.weight(1f))
                Text(
                    String.format("%.3f", latestResult.evidenceAccumulated),
                    fontWeight = FontWeight.Bold,
                    fontSize = 13.sp,
                    color = if (latestResult.effectiveScore > 0.38f) AlertOrange else AlertGreen
                )
            }
            Spacer(Modifier.height(4.dp))
            Text(
                if (latestResult.effectiveScore > 0.38f)
                    "Score > 0.38 triggers evidence compounding (+15% scale per sustained day) to evaluate diagnostic alert levels."
                else
                    "Score <= 0.38 signals healthy routine alignment. Cumulative anomaly evidence decays by 8% today.",
                fontSize = 10.sp,
                color = TextSecondary,
                lineHeight = 14.sp
            )
        }
    }
}
