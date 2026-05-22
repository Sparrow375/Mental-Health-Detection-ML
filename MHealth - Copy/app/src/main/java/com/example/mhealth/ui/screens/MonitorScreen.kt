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

        // L1 Anchor Clusters — behavioral archetypes (System 1)
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
                label = "Days",
                unit = "/ $target",
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
