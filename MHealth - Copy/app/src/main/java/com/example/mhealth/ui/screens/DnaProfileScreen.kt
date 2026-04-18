package com.example.mhealth.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import org.json.JSONObject
import org.json.JSONArray
import androidx.compose.runtime.getValue
import com.example.mhealth.logic.DataRepository
import com.example.mhealth.ui.theme.*
import com.example.mhealth.ui.components.ScreenHeader

import androidx.compose.ui.platform.LocalContext
import kotlinx.coroutines.launch
import kotlinx.coroutines.Dispatchers
import com.example.mhealth.logic.PythonEngine
import com.example.mhealth.logic.db.MHealthDatabase
import android.util.Log
import com.example.mhealth.ui.components.InfoCard
import com.example.mhealth.ui.components.MetricPill
import com.example.mhealth.ui.components.CollapsibleCard
import com.example.mhealth.ui.components.MiniStat
import com.example.mhealth.ui.components.PhoneMetric
import com.example.mhealth.ui.components.TextureMetric
import com.example.mhealth.models.PersonalityVector



// ── DNA Screen Constants ───────────────────────────────────────────────────
private const val TAG = "DnaProfileScreen"

// MainActivity chart colors mapped to DNA profile Light Theme 
private val MhealthIndigo = Color(0xFF4F46E5)
private val ChartPurple = AccentPurple
private val OceanBlue = AccentBlue
private val SoftCyan = AccentCyan
private val ChartGreen = AccentGreen
private val ChartRed = AccentRed
private val AlertOrange = AccentOrange
private val AlertYellow = Color(0xFFEAB308)
private val ChartBlue = AccentBlue

private val groupColors = mapOf(
    "screen_app" to AccentBlue,
    "communication" to AccentGreen,
    "location" to AccentOrange,
    "sleep" to AccentPurple,
    "system" to TextSecondary,
    "behavioral" to AccentCyan,
    "engagement" to AccentRed,
)

// ── Main composable ──────────────────────────────────────────────────────────

@Composable
fun DnaScreen() {
    var selectedTabIndex by remember { mutableStateOf(0) }
    val profileJson by DataRepository.s1ProfileJson.collectAsState()
    val baselineDays by DataRepository.dnaBaselineDaysRequired.collectAsState()
    val currentProgress by DataRepository.dnaBaselineProgress.collectAsState()
    val latestVector by DataRepository.latestVector.collectAsState()
    
    Column(Modifier.fillMaxSize().background(BgLight)) {
        ScreenHeader(
            title = "Behavioral DNA",
            subtitle = "Pattern Fingerprinting & Daily Metrics",
            icon = Icons.Default.Favorite
        )

        androidx.compose.material3.TabRow(
            selectedTabIndex = selectedTabIndex,
            containerColor = BgLight,
            contentColor = AccentBlue
        ) {
            androidx.compose.material3.Tab(selected = selectedTabIndex == 0, onClick = { selectedTabIndex = 0 }) {
                Text("Baseline Profile", modifier = Modifier.padding(16.dp), color = if (selectedTabIndex == 0) AccentBlue else TextSecondary)
            }
            androidx.compose.material3.Tab(selected = selectedTabIndex == 1, onClick = { selectedTabIndex = 1 }) {
                Text("Today's DNA", modifier = Modifier.padding(16.dp), color = if (selectedTabIndex == 1) AccentBlue else TextSecondary)
            }
        }
        
        if (selectedTabIndex == 0) {
            DnaProfileSection(profileJson, baselineDays, currentProgress)
        } else {
            TodayDnaMetricsSection(latestVector)
        }
    }
}

@Composable
fun DnaProfileSection(profileJson: String?, baselineDays: Int = 28, currentProgress: Int = 0) {
    val profile = remember(profileJson) {
        if (profileJson.isNullOrBlank() || profileJson == "{}") null
        else try { JSONObject(profileJson) } catch (_: Exception) { null }
    }

    // Determine if we have a "complete" enough profile to show charts
    // If not, show the building/empty state with the Finalize button
    val hasValidDna = profile != null && 
            (profile.has("personality_vector") || profile.has("anchor_clusters"))

    if (!hasValidDna) {
        DnaProfileEmptyState(baselineDays, currentProgress)
        return
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        // Header
        DnaProfileHeader(profile)

        // Personality Vector (29-feature baseline)
        PersonalityVectorCard(profile)

        // Feature Importance Grid
        FeatureImportanceCard(profile)

        // Anchor Clusters
        AnchorClustersCard(profile)

        // App DNA Profiles
        AppDnaProfilesCard(profile)

        // Phone DNA
        PhoneDnaCard(profile)

        // Texture Profiles
        TextureProfilesCard(profile)

        Spacer(modifier = Modifier.height(24.dp))
    }
}

// ── Empty state ──────────────────────────────────────────────────────────────

@Composable
private fun DnaProfileEmptyState(baselineDays: Int = 28, currentProgress: Int = 0) {
    val context = LocalContext.current
    val isAnalysing by DataRepository.isDnaAnalysing.collectAsState()

    Column(
        modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Card(
            modifier = Modifier.fillMaxWidth().padding(16.dp),
            colors = CardDefaults.cardColors(containerColor = CardLight),
            shape = RoundedCornerShape(12.dp),
            border = CardDefaults.outlinedCardBorder(true)
        ) {
            Column(
                modifier = Modifier.padding(24.dp).fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
            Icon(
                Icons.Default.Fingerprint,
                contentDescription = null,
                modifier = Modifier.size(48.dp),
                tint = AccentBlue.copy(alpha = 0.6f)
            )
            Spacer(Modifier.height(16.dp))
            Text(
                "Building DNA Profile",
                color = TextPrimary,
                fontWeight = FontWeight.Bold,
                fontSize = 18.sp
            )
            Spacer(Modifier.height(8.dp))
            Text(
                "Collecting your unique behavioral patterns. Progress: Day $currentProgress/$baselineDays",
                color = TextSecondary,
                fontSize = 14.sp,
                textAlign = androidx.compose.ui.text.style.TextAlign.Center
            )
            
            Spacer(Modifier.height(24.dp))
            
            LinearProgressIndicator(
                progress = (currentProgress.toFloat() / baselineDays.coerceAtLeast(1)).coerceIn(0f, 1f),
                modifier = Modifier.fillMaxWidth().height(8.dp),
                color = AccentBlue,
                trackColor = BorderLight,
                strokeCap = androidx.compose.ui.graphics.StrokeCap.Round
            )
            
            Spacer(Modifier.height(8.dp))
            Text(
                "Day $currentProgress of $baselineDays collected",
                color = TextSecondary,
                fontSize = 12.sp
            )

            // ── Dev & Finalize Controls ─────────────────────────────────────
            
            Spacer(Modifier.height(24.dp))
            
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                // 1. Finalize DNA Button (Enabled if we have at least 1 day of data)
                Button(
                    onClick = { 
                        android.widget.Toast.makeText(context, "Initiating DNA Baseline Construction...", android.widget.Toast.LENGTH_SHORT).show()
                        DataRepository.triggerDnaFinalize() 
                    },
                    enabled = !isAnalysing && currentProgress > 0,
                    modifier = Modifier.fillMaxWidth().height(48.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (currentProgress >= baselineDays) AccentGreen else AccentBlue
                    ),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    if (isAnalysing) {
                        CircularProgressIndicator(modifier = Modifier.size(20.dp), color = Color.White, strokeWidth = 2.dp)
                        Spacer(Modifier.width(8.dp))
                        Text("Building Profile...")
                    } else {
                        val canFinalize = currentProgress >= 3
                        Icon(if (canFinalize) Icons.Default.CheckCircle else Icons.Default.AutoFixHigh, contentDescription = null, modifier = Modifier.size(20.dp))
                        Spacer(Modifier.width(8.dp))
                        Text(
                            when {
                                currentProgress >= baselineDays -> "Finalize DNA Baseline"
                                canFinalize -> "Finalize DNA (3-Day Min Met)"
                                else -> "Building DNA (Day $currentProgress/28)"
                            }
                        )
                    }
                }

                // 2. Force New Day Button (Dev Only)
                OutlinedButton(
                    onClick = { DataRepository.triggerForceNewDay() },
                    modifier = Modifier.fillMaxWidth().height(48.dp),
                    border = androidx.compose.foundation.BorderStroke(1.dp, BorderLight),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Icon(Icons.Default.FastForward, contentDescription = null, tint = AccentOrange)
                    Spacer(Modifier.width(8.dp))
                    Text("Debug: Force Day Transition", color = AccentOrange)
                }

                if (currentProgress < baselineDays) {
                    Text(
                        "Analysis usually runs at midnight. Use 'Finalize' to build now.",
                        color = TextSecondary,
                        fontSize = 11.sp,
                        textAlign = androidx.compose.ui.text.style.TextAlign.Center
                    )
                } else {
                    Text(
                        "Threshold reached! Recommended to finalize now.",
                        color = AccentGreen,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

        }
    }
}
}

// ── Header ───────────────────────────────────────────────────────────────────

@Composable
private fun DnaProfileHeader(profile: JSONObject) {
    val daysOfData = profile.optInt("days_of_data", 0)
    val builtAt = profile.optString("built_at", "N/A")
    val pv = profile.optJSONObject("personality_vector")
    val confidence = pv?.optString("confidence", "LOW") ?: "LOW"
    val nClusters = profile.optJSONArray("anchor_clusters")?.length() ?: 0
    val nApps = profile.optJSONObject("app_dna_profiles")?.length() ?: 0

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = CardLight),
        shape = RoundedCornerShape(12.dp),
        border = CardDefaults.outlinedCardBorder(true)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Fingerprint, null, tint = AccentBlue, modifier = Modifier.size(28.dp))
                Spacer(Modifier.width(8.dp))
                Text("System 1 — Behavioral DNA Profile", color = TextPrimary, fontWeight = FontWeight.Bold, fontSize = 18.sp)
            }
            Spacer(Modifier.height(12.dp))

            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                StatChip("Days", "$daysOfData", AccentBlue)
                StatChip("Confidence", confidence, when(confidence) { "HIGH" -> AccentGreen; "MEDIUM" -> AccentOrange; else -> AccentRed })
                StatChip("Clusters", "$nClusters", AccentPurple)
                StatChip("Apps", "$nApps", AccentCyan)
            }

            if (builtAt != "N/A") {
                Spacer(Modifier.height(8.dp))
                Text("Built: $builtAt", color = TextSecondary, fontSize = 11.sp)
            }
        }
    }
}

@Composable
private fun StatChip(label: String, value: String, color: Color) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(value, color = color, fontWeight = FontWeight.Bold, fontSize = 16.sp)
        Text(label, color = TextSecondary, fontSize = 11.sp)
    }
}

// ── Personality Vector Card ──────────────────────────────────────────────────

@Composable
private fun PersonalityVectorCard(profile: JSONObject) {
    val pv = profile.optJSONObject("personality_vector") ?: return
    val means = pv.optJSONObject("means") ?: return
    val variances = pv.optJSONObject("variances") ?: return

    var expanded by remember { mutableStateOf(true) }

    CollapsibleCard(
        title = "Personality Vector (DNA Baseline)",
        subtitle = "${means.length()} features · ${pv.optString("confidence", "LOW")} confidence",
        icon = Icons.Default.Psychology,
        expanded = expanded,
        onToggle = { expanded = !expanded }
    ) {
        // Show top features sorted by weight
        val importance = profile.optJSONObject("feature_importance")
        val sortedFeatures = mutableListOf<Pair<String, JSONObject>>()
        if (importance != null) {
            val keys = importance.keys()
            while (keys.hasNext()) {
                val k = keys.next()
                val info = importance.optJSONObject(k)
                if (info != null) sortedFeatures.add(k to info)
            }
            sortedFeatures.sortByDescending { it.second.optDouble("weight", 1.0) }
        }

        if (sortedFeatures.isNotEmpty()) {
            val topForGraph = sortedFeatures.take(6)
            DnaRadarChart(
                labels = topForGraph.map { 
                    val raw = it.first
                    if (raw.length > 15) raw.replace("feature_", "").replace("app_", "").take(12) + "…" else raw 
                },
                values = topForGraph.map { it.second.optDouble("weight", 1.0).toFloat() },
                colors = topForGraph.map { groupColors[it.second.optString("group", "")] ?: AccentBlue },
                modifier = Modifier.fillMaxWidth().height(220.dp).padding(vertical = 16.dp)
            )
        }

        sortedFeatures.forEach { (feat, info) ->
            val mean = means.optDouble(feat, 0.0)
            val std = variances.optDouble(feat, 0.0)
            val weight = info.optDouble("weight", 1.0)
            val group = info.optString("group", "")
            val groupColor = groupColors[group] ?: TextSecondary

            Row(
                modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Group color indicator
                Box(
                    modifier = Modifier.width(3.dp).height(20.dp)
                        .background(groupColor, RoundedCornerShape(1.dp))
                )
                Spacer(Modifier.width(8.dp))

                Column(modifier = Modifier.weight(1f)) {
                    Text(feat, color = TextPrimary, fontSize = 12.sp, fontWeight = FontWeight.Medium)
                    Text("μ=${String.format("%.2f", mean)}  σ=${String.format("%.2f", std)}",
                        color = TextSecondary, fontSize = 10.sp)
                }

                // Weight badge
                Text(
                    String.format("%.1f", weight),
                    color = groupColor,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier
                        .background(groupColor.copy(alpha = 0.15f), RoundedCornerShape(4.dp))
                        .padding(horizontal = 6.dp, vertical = 2.dp)
                )
            }
        }
    }
}

// ── Feature Importance Radar-like Card ────────────────────────────────────────

@Composable
private fun FeatureImportanceCard(profile: JSONObject) {
    val groupSummaries = profile.optJSONObject("group_summaries") ?: return

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = CardLight),
        shape = RoundedCornerShape(12.dp),
        border = CardDefaults.outlinedCardBorder(true)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Radar, null, tint = AccentPurple, modifier = Modifier.size(20.dp))
                Spacer(Modifier.width(8.dp))
                Text("Feature Group Importance", color = TextPrimary, fontWeight = FontWeight.Bold, fontSize = 14.sp)
            }
            Spacer(Modifier.height(12.dp))

            val groups = mutableListOf<Pair<String, JSONObject>>()
            val keys = groupSummaries.keys()
            while (keys.hasNext()) {
                val k = keys.next()
                val v = groupSummaries.optJSONObject(k)
                if (v != null) groups.add(k to v)
            }
            groups.sortByDescending { it.second.optDouble("total_weight", 0.0) }

            val maxWeight = groups.maxOfOrNull { it.second.optDouble("total_weight", 0.0) } ?: 1.0

            if (groups.isNotEmpty()) {
                DnaRadarChart(
                    labels = groups.map { it.first.replace("_", " ") },
                    values = groups.map { it.second.optDouble("total_weight", 0.0).toFloat() },
                    colors = groups.map { groupColors[it.first] ?: TextSecondary },
                    modifier = Modifier.fillMaxWidth().height(200.dp).padding(vertical = 8.dp)
                )
            }

            groups.forEach { (groupName, info) ->
                val totalWeight = info.optDouble("total_weight", 0.0)
                val avgWeight = info.optDouble("avg_weight", 0.0)
                val featureCount = info.optJSONArray("features")?.length() ?: 0
                val color = groupColors[groupName] ?: TextSecondary
                val fraction = (totalWeight / maxWeight).toFloat().coerceIn(0f, 1f)

                Column(modifier = Modifier.padding(vertical = 4.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            groupName.replace("_", " ").replaceFirstChar { it.uppercase() },
                            color = color, fontSize = 12.sp, fontWeight = FontWeight.Medium
                        )
                        Text(
                            "${String.format("%.1f", totalWeight)} total · $featureCount features",
                            color = TextSecondary, fontSize = 10.sp
                        )
                    }
                    Spacer(Modifier.height(4.dp))
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(8.dp)
                            .background(BorderLight, RoundedCornerShape(4.dp))
                    ) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth(fraction)
                                .fillMaxHeight()
                                .background(color, RoundedCornerShape(4.dp))
                        )
                    }
                }
            }
        }
    }
}

// ── Anchor Clusters Card ─────────────────────────────────────────────────────

@Composable
private fun AnchorClustersCard(profile: JSONObject) {
    val clustersArr = profile.optJSONArray("anchor_clusters") ?: return
    if (clustersArr.length() == 0) return

    var expanded by remember { mutableStateOf(true) }

    CollapsibleCard(
        title = "Anchor Clusters (L1 Behavioral Archetypes)",
        subtitle = "${clustersArr.length()} cluster(s) discovered via DBSCAN",
        icon = Icons.Default.Hub,
        expanded = expanded,
        onToggle = { expanded = !expanded }
    ) {
        for (i in 0 until clustersArr.length()) {
            val cluster = clustersArr.optJSONObject(i) ?: continue
            val clusterId = cluster.optInt("cluster_id", 0)
            val memberCount = cluster.optInt("member_count", 0)
            val radius = cluster.optDouble("radius", 0.0)
            val centroidFeatures = cluster.optJSONObject("centroid_features") ?: continue
            val memberDates = cluster.optJSONArray("member_dates")

            Card(
                modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                colors = CardDefaults.cardColors(containerColor = BgLight),
                shape = RoundedCornerShape(8.dp)
            ) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.Circle, null, tint = AccentPurple, modifier = Modifier.size(12.dp))
                        Spacer(Modifier.width(6.dp))
                        Text("Cluster $clusterId", color = AccentPurple, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                        Spacer(Modifier.weight(1f))
                        Text("$memberCount days", color = TextSecondary, fontSize = 11.sp)
                        Spacer(Modifier.width(8.dp))
                        Text("r=${String.format("%.2f", radius)}", color = TextSecondary, fontSize = 11.sp)
                    }
                    Spacer(Modifier.height(6.dp))

                    // Centroid feature bars
                    val maxVal = centroidFeatures.keys().asSequence().mapNotNull {
                        kotlin.runCatching { Math.abs(centroidFeatures.optDouble(it)) }.getOrNull()
                    }.maxOrNull() ?: 1.0

                    centroidFeatures.keys().asSequence().take(6).forEach { feat ->
                        val value = centroidFeatures.optDouble(feat, 0.0)
                        val fraction = (Math.abs(value) / maxVal).toFloat().coerceIn(0f, 1f)
                        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(vertical = 1.dp)) {
                            Text(feat, color = TextSecondary, fontSize = 9.sp, modifier = Modifier.width(120.dp))
                            Box(
                                modifier = Modifier.weight(1f).height(4.dp).background(BorderLight, RoundedCornerShape(2.dp))
                            ) {
                                Box(
                                    modifier = Modifier.fillMaxWidth(fraction).fillMaxHeight()
                                        .background(AccentBlue, RoundedCornerShape(2.dp))
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

// ── App DNA Profiles Card ────────────────────────────────────────────────────

@Composable
private fun AppDnaProfilesCard(profile: JSONObject) {
    val appDnas = profile.optJSONObject("app_dna_profiles") ?: return
    if (appDnas.length() == 0) return

    var expanded by remember { mutableStateOf(true) }

    CollapsibleCard(
        title = "App DNA Profiles (Per-App Behavioral Fingerprints)",
        subtitle = "${appDnas.length()} apps profiled",
        icon = Icons.Default.Apps,
        expanded = expanded,
        onToggle = { expanded = !expanded }
    ) {
        // Sort apps by session count
        val sortedApps = appDnas.keys().asSequence().toList().sortedByDescending { pkg ->
            val appProfile = appDnas.optJSONObject(pkg)
            appProfile?.optDouble("sessions_per_active_day", 0.0)
                ?: appProfile?.optDouble("avg_session_minutes", 0.0)
                ?: 0.0
        }

        sortedApps.take(10).forEach { pkg ->
            val appProfile = appDnas.optJSONObject(pkg) ?: return@forEach
            val shortName = pkg.substringAfterLast(".")
            val avgSession = appProfile.optDouble("avg_session_minutes", 0.0)
            val abandonRate = appProfile.optDouble("abandon_rate", 0.0)
            val selfOpen = appProfile.optDouble("self_open_ratio", 0.0)
            val sessionsPerDay = appProfile.optDouble("sessions_per_active_day", 0.0)

            Card(
                modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
                colors = CardDefaults.cardColors(containerColor = BgLight),
                shape = RoundedCornerShape(6.dp)
            ) {
                Row(
                    modifier = Modifier.padding(8.dp).fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(Icons.Default.PhoneAndroid, null, tint = AccentCyan, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(6.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(shortName, color = TextPrimary, fontSize = 12.sp, fontWeight = FontWeight.Medium,
                            maxLines = 1, overflow = TextOverflow.Ellipsis)
                        Text(pkg, color = TextSecondary, fontSize = 9.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        MiniStat("Avg", "${String.format("%.1f", avgSession)}m", AccentBlue)
                        MiniStat("Sessions", "${String.format("%.0f", sessionsPerDay)}/d", AccentGreen)
                        MiniStat("Self", "${String.format("%.0f", selfOpen * 100)}%", AccentOrange)
                    }
                }
            }
        }
    }
}


// ── Phone DNA Card ───────────────────────────────────────────────────────────

@Composable
private fun PhoneDnaCard(profile: JSONObject) {
    val phoneDna = profile.optJSONObject("phone_dna") ?: return
    if (phoneDna.length() == 0) return

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = CardLight),
        shape = RoundedCornerShape(12.dp),
        border = CardDefaults.outlinedCardBorder(true)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Smartphone, null, tint = AccentGreen, modifier = Modifier.size(20.dp))
                Spacer(Modifier.width(8.dp))
                Text("Phone DNA (Device-Level)", color = TextPrimary, fontWeight = FontWeight.Bold, fontSize = 14.sp)
            }
            Spacer(Modifier.height(12.dp))

            // Session duration distribution chart
            val sessionDist = phoneDna.optJSONArray("session_duration_distribution")
            if (sessionDist != null && sessionDist.length() == 5) {
                Text("Session Duration Distribution", color = TextSecondary, fontSize = 11.sp, fontWeight = FontWeight.Medium)
                Spacer(Modifier.height(4.dp))

                val labels = listOf("<1m", "1-5m", "5-15m", "15-60m", ">60m")
                val values = (0 until 5).map { sessionDist.getDouble(it).toFloat() }
                val maxVal = values.maxOrNull()?.coerceAtLeast(0.01f) ?: 0.01f

                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                    labels.forEachIndexed { idx, label ->
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Box(
                                modifier = Modifier.width(36.dp).height(60.dp),
                                contentAlignment = Alignment.BottomCenter
                            ) {
                                Box(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .fillMaxHeight((values[idx] / maxVal).coerceIn(0.05f, 1f))
                                        .background(
                                            when(idx) {
                                                0 -> AccentCyan; 1 -> AccentBlue; 2 -> AccentGreen
                                                3 -> AccentOrange; else -> AccentRed
                                            },
                                            RoundedCornerShape(topStart = 3.dp, topEnd = 3.dp)
                                        )
                                )
                            }
                            Spacer(Modifier.height(2.dp))
                            Text(label, color = TextSecondary, fontSize = 8.sp)
                            Text("${String.format("%.0f", values[idx] * 100)}%", color = TextPrimary, fontSize = 9.sp, fontWeight = FontWeight.Bold)
                        }
                    }
                }
                Spacer(Modifier.height(12.dp))
            }

            // Key metrics grid
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                val firstPickup = phoneDna.optDouble("first_pickup_hour_mean", 0.0)
                val rhythmReg = phoneDna.optDouble("daily_rhythm_regularity", 0.0)
                val deepRatio = phoneDna.optDouble("deep_session_ratio", 0.0)
                val microRatio = phoneDna.optDouble("micro_session_ratio", 0.0)

                PhoneMetric("First Pickup", String.format("%.1f", firstPickup) + "h", AccentBlue)
                PhoneMetric("Rhythm Reg.", String.format("%.0f", rhythmReg * 100) + "%", AccentGreen)
                PhoneMetric("Deep Sessions", String.format("%.0f", deepRatio * 100) + "%", AccentOrange)
                PhoneMetric("Micro Sessions", String.format("%.0f", microRatio * 100) + "%", AccentCyan)
            }
        }
    }
}


// ── Texture Profiles Card ────────────────────────────────────────────────────

@Composable
private fun TextureProfilesCard(profile: JSONObject) {
    val textureArr = profile.optJSONArray("texture_profiles") ?: return
    if (textureArr.length() == 0) return

    var expanded by remember { mutableStateOf(true) }

    CollapsibleCard(
        title = "Contextual Texture Profiles (L2)",
        subtitle = "${textureArr.length()} archetype(s)",
        icon = Icons.Default.Texture,
        expanded = expanded,
        onToggle = { expanded = !expanded }
    ) {
        for (i in 0 until textureArr.length()) {
            val tp = textureArr.optJSONObject(i) ?: continue
            val summary = tp.optJSONObject("texture_summary") ?: continue
            val memberDays = tp.optInt("member_days", 0)
            val avgSessions = summary.optDouble("avg_sessions_per_day", 0.0)
            val avgAbandon = summary.optDouble("avg_abandon_rate", 0.0)
            val avgSelfOpen = summary.optDouble("avg_self_open_ratio", 0.0)
            val avgDeep = summary.optDouble("avg_deep_session_ratio", 0.0)
            val avgMicro = summary.optDouble("avg_micro_session_ratio", 0.0)
            val avgSwitching = summary.optDouble("avg_app_switching_rate", 0.0)
            val avgSessionMin = summary.optDouble("avg_session_minutes", 0.0)

            Text("Archetype ${tp.optInt("archetype_id", 0)} — $memberDays days",
                color = AccentPurple, fontWeight = FontWeight.Bold, fontSize = 13.sp)
            Spacer(Modifier.height(8.dp))

            // Metrics grid
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                TextureMetric("Sessions/Day", String.format("%.1f", avgSessions), AccentBlue)
                TextureMetric("Abandon Rate", String.format("%.0f", avgAbandon * 100) + "%", AccentRed)
                TextureMetric("Self-Open", String.format("%.0f", avgSelfOpen * 100) + "%", AccentGreen)
                TextureMetric("Avg Duration", String.format("%.1f", avgSessionMin) + "m", AccentOrange)
            }
            Spacer(Modifier.height(6.dp))
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                TextureMetric("Deep Ratio", String.format("%.0f", avgDeep * 100) + "%", AccentPurple)
                TextureMetric("Micro Ratio", String.format("%.0f", avgMicro * 100) + "%", AccentCyan)
                TextureMetric("Switching", String.format("%.0f", avgSwitching * 100) + "%", TextSecondary)
            }

            // Daily breakdown
            val dailyBreakdown = summary.optJSONArray("daily_breakdown")
            if (dailyBreakdown != null && dailyBreakdown.length() > 0) {
                Spacer(Modifier.height(8.dp))
                Text("Daily Breakdown (last ${dailyBreakdown.length()} days):",
                    color = TextSecondary, fontSize = 11.sp, fontWeight = FontWeight.Medium)
                Spacer(Modifier.height(4.dp))

                // Mini sparkline of sessions per day
                val sessionCounts = (0 until dailyBreakdown.length()).map {
                    dailyBreakdown.optJSONObject(it)?.optInt("total_sessions", 0) ?: 0
                }
                val maxSessions = sessionCounts.maxOrNull()?.coerceAtLeast(1) ?: 1

                Row(horizontalArrangement = Arrangement.spacedBy(2.dp)) {
                    sessionCounts.forEach { count ->
                        val frac = (count.toFloat() / maxSessions).coerceIn(0.05f, 1f)
                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .height(24.dp * frac)
                                .background(AccentBlue.copy(alpha = 0.7f), RoundedCornerShape(1.dp))
                        )
                    }
                }
            }
        }
    }
}


// ── Reusable Collapsible Card ────────────────────────────────────────────────


@Composable
fun TodayDnaMetricsSection(v: PersonalityVector?) {
    if (v == null) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text("Collecting today's DNA data...", color = TextSecondary)
        }
        return
    }
    
    androidx.compose.foundation.lazy.LazyColumn(Modifier.fillMaxSize().padding(bottom = 16.dp)) {
            // ── Phone DNA — Today's Device Fingerprint ──────────────────────────
            item {
                val context2 = LocalContext.current
                val dnaComputer = remember { com.example.mhealth.logic.AppDnaComputer(context2) }
                val phoneDna by produceState<com.example.mhealth.logic.AppDnaComputer.TodayPhoneDna?>(
                    initialValue = null,
                    key1 = v
                ) {
                    kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
                        value = try { dnaComputer.computeTodayPhoneDna() } catch (_: Exception) { null }
                    }
                }

                InfoCard("Phone DNA — Today", headerColor = MhealthIndigo) {
                    if (phoneDna == null) {
                        Box(Modifier.fillMaxWidth().height(80.dp), contentAlignment = Alignment.Center) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                CircularProgressIndicator(color = MhealthIndigo, modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                                Spacer(Modifier.width(8.dp))
                                Text("Computing DNA…", fontSize = 12.sp, color = TextSecondary)
                            }
                        }
                    } else {
                        val pd = phoneDna!!
                        Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
                            // Active Window
                            Text("📱 Activity Window", fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = TextSecondary)
                            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                                MetricPill("First Pickup", pd.firstPickupHour?.let { "%.0f:%02d".format(it, ((it % 1) * 60).toInt()) } ?: "—", OceanBlue)
                                MetricPill("Active Window", pd.activeWindowHours?.let { "%.1fh".format(it) } ?: "—", SoftCyan)
                                MetricPill("Unique Apps", "${pd.uniqueAppsUsed}", ChartPurple)
                            }

                            HorizontalDivider(color = Color(0xFFE2E8F0), thickness = 0.5.dp)

                            // Session Distribution
                            Text("⏱ Session Distribution", fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = TextSecondary)
                            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                                MetricPill("< 2m", "%.0f%%".format(pd.microSessionPct), ChartRed.copy(0.8f))
                                MetricPill("2–15m", "%.0f%%".format(pd.shortSessionPct), AlertOrange)
                                MetricPill("15–30m", "%.0f%%".format(pd.mediumSessionPct), AlertYellow)
                                MetricPill("30–60m", "%.0f%%".format(pd.deepSessionPct), ChartGreen)
                                MetricPill("60m+", "%.0f%%".format(pd.marathonSessionPct), ChartPurple)
                            }

                            HorizontalDivider(color = Color(0xFFE2E8F0), thickness = 0.5.dp)

                            // Trigger DNA
                            Text("🎯 Trigger DNA", fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = TextSecondary)
                            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                                MetricPill("Self-Open", "%.0f%%".format(pd.selfOpenPct), OceanBlue)
                                MetricPill("Notif-Open", "%.0f%%".format(pd.notificationOpenPct), AlertOrange)
                                MetricPill("Total Sess.", "${pd.totalSessions}", ChartBlue)
                            }

                            HorizontalDivider(color = Color(0xFFE2E8F0), thickness = 0.5.dp)

                            // Notification Reflexes
                            Text("🔔 Notification Reflexes", fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = TextSecondary)
                            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                                MetricPill("Tap Rate", "%.0f%%".format(pd.notificationTapRate * 100), ChartGreen)
                                MetricPill("Dismiss", "%.0f%%".format(pd.notificationDismissRate * 100), ChartRed)
                                MetricPill("Ignore", "%.0f%%".format(pd.notificationIgnoreRate * 100), AlertOrange)
                            }
                            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                                MetricPill("Arrivals", "${pd.totalNotifications}", MhealthIndigo)
                                MetricPill("Screen", "%.1fh".format(pd.totalScreenTimeHours), OceanBlue)
                                if (pd.topAppPackage != null) {
                                    val topLabel = try {
                                        context2.packageManager.getApplicationLabel(
                                            context2.packageManager.getApplicationInfo(pd.topAppPackage, 0)
                                        ).toString()
                                    } catch (_: Exception) { pd.topAppPackage.substringAfterLast(".") }
                                    MetricPill("Top App", topLabel, ChartPurple)
                                }
                            }

                            HorizontalDivider(color = Color(0xFFE2E8F0), thickness = 0.5.dp)

                            // Night Checks — phone usage during sleep window
                            Text("\uD83C\uDF19 Night Checks", fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = TextSecondary)
                            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                                MetricPill("Sleep Unlocks", "${pd.nightChecks}", if (pd.nightChecks > 3) ChartRed else ChartGreen)
                                MetricPill("Unique Apps", "${pd.uniqueAppsUsed}", MhealthIndigo)
                            }
                        }
                    }
                }
            }

            // ── App DNA — Per-App Behavioral Fingerprints ─────────────────────────
            item {
                val context3 = LocalContext.current
                val dnaComputer2 = remember { com.example.mhealth.logic.AppDnaComputer(context3) }
                val appDnaList by produceState<List<com.example.mhealth.logic.AppDnaComputer.TodayAppDna>>(
                    initialValue = emptyList(),
                    key1 = v
                ) {
                    kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
                        value = try { dnaComputer2.computeTodayAppDnaList() } catch (_: Exception) { emptyList() }
                    }
                }

                var expandedApp by remember { mutableStateOf<String?>(null) }

                InfoCard("App DNA — Per App", headerColor = ChartPurple) {
                    if (appDnaList.isEmpty()) {
                        Box(Modifier.fillMaxWidth().height(60.dp), contentAlignment = Alignment.Center) {
                            Text("No app sessions recorded today", fontSize = 12.sp, color = TextSecondary)
                        }
                    } else {
                        // Header row
                        Row(Modifier.fillMaxWidth().padding(bottom = 6.dp)) {
                            Text("App", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = TextSecondary, modifier = Modifier.weight(2.5f))
                            Text("Time", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = TextSecondary, modifier = Modifier.weight(1.2f))
                            Text("Sessions", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = TextSecondary, modifier = Modifier.weight(1.3f))
                            Spacer(Modifier.weight(0.5f))
                        }
                        HorizontalDivider(color = TextSecondary.copy(alpha = 0.15f), thickness = 0.5.dp)

                        appDnaList.forEach { app ->
                            val isExpanded = expandedApp == app.appPackage
                            val hrs = app.totalScreenTimeMinutes / 60
                            val mins = app.totalScreenTimeMinutes % 60
                            val timeStr = if (hrs > 0) "${hrs}h ${mins}m" else "${mins}m"

                            // App row (clickable)
                            Row(
                                Modifier.fillMaxWidth()
                                    .clickable {
                                        expandedApp = if (isExpanded) null else app.appPackage
                                    }
                                    .padding(vertical = 8.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    app.appLabel, fontSize = 12.sp, color = TextPrimary,
                                    modifier = Modifier.weight(2.5f),
                                    maxLines = 1, overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis
                                )
                                Text(timeStr, fontSize = 11.sp, color = TextSecondary, modifier = Modifier.weight(1.2f))
                                Text("${app.sessionCount}", fontSize = 11.sp, color = TextSecondary, modifier = Modifier.weight(1.3f))
                                Icon(
                                    if (isExpanded) Icons.Default.ArrowUpward else Icons.Default.ArrowDownward,
                                    contentDescription = if (isExpanded) "Collapse" else "Expand",
                                    modifier = Modifier.size(14.dp).weight(0.5f),
                                    tint = TextSecondary
                                )
                            }

                            // Expanded detail view — App DNA metrics
                            AnimatedVisibility(
                                visible = isExpanded,
                                enter = expandVertically() + fadeIn(),
                                exit = shrinkVertically() + fadeOut()
                            ) {
                                Card(
                                    modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp),
                                    colors = CardDefaults.cardColors(containerColor = Color(0xFFF8FAFC)),
                                    shape = RoundedCornerShape(12.dp)
                                ) {
                                    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                                        Text("🧬 ${app.appLabel} DNA", fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = ChartPurple)

                                        // Time window
                                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                                            MetricPill("Time Range", app.primaryTimeRange, OceanBlue)
                                            MetricPill("Avg Session", "%.1fm".format(app.avgSessionMinutes), SoftCyan)
                                        }
                                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                                            MetricPill("Min Session", "%.1fm".format(app.minSessionMinutes), ChartGreen)
                                            MetricPill("Max Session", "%.1fm".format(app.maxSessionMinutes), ChartRed)
                                        }

                                        HorizontalDivider(color = Color(0xFFE2E8F0), thickness = 0.5.dp)

                                        // Trigger DNA
                                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                                            MetricPill("Self-Open", "%.0f%%".format(app.selfOpenRatio * 100), OceanBlue)
                                            MetricPill("Notif-Open", "%.0f%%".format(app.notificationOpenRatio * 100), AlertOrange)
                                        }

                                        // Notifications for this app
                                        if (app.notificationCount > 0) {
                                            HorizontalDivider(color = Color(0xFFE2E8F0), thickness = 0.5.dp)
                                            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                                                MetricPill("🔔 Notifs", "${app.notificationCount}", MhealthIndigo)
                                                MetricPill("Tapped", "${app.notificationTapCount}", ChartGreen)
                                                MetricPill("Tap Latency", app.avgTapLatencyMinutes?.let { "%.1fm".format(it) } ?: "—", ChartBlue)
                                            }
                                        }
                                    }
                                }
                            }

                            HorizontalDivider(color = TextSecondary.copy(alpha = 0.08f), thickness = 0.5.dp)
                        }
                    }
                }
            }


    }
}

// ── Graph Components ─────────────────────────────────────────────────────────

@Composable
fun DnaRadarChart(
    labels: List<String>,
    values: List<Float>,
    colors: List<Color>,
    modifier: Modifier = Modifier
) {
    if (labels.isEmpty() || values.isEmpty() || labels.size != values.size) return
    val maxVal = values.maxOrNull()?.coerceAtLeast(0.01f) ?: 1f

    Canvas(modifier = modifier) {
        val radius = size.minDimension / 2f * 0.65f
        val center = Offset(size.width / 2f, size.height / 2f)
        val anglePerNode = 2 * Math.PI / labels.size

        // Background Web
        for (i in 1..4) {
            val r = radius * (i / 4f)
            val path = androidx.compose.ui.graphics.Path()
            for (j in labels.indices) {
                val angle = j * anglePerNode - Math.PI / 2
                val x = center.x + r * Math.cos(angle).toFloat()
                val y = center.y + r * Math.sin(angle).toFloat()
                if (j == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            path.close()
            drawPath(path, color = BorderLight, style = Stroke(1.dp.toPx()))
        }

        // Connect Center to Outer Nodes
        for (j in labels.indices) {
            val angle = j * anglePerNode - Math.PI / 2
            val x = center.x + radius * Math.cos(angle).toFloat()
            val y = center.y + radius * Math.sin(angle).toFloat()
            drawLine(
                color = BorderLight,
                start = center,
                end = Offset(x, y),
                strokeWidth = 1.dp.toPx()
            )
        }

        // Draw Data Polygon
        val dataPath = androidx.compose.ui.graphics.Path()
        val dataPoints = mutableListOf<Offset>()
        for (j in labels.indices) {
            val valRatio = (values[j] / maxVal).coerceIn(0f, 1f)
            val angle = j * anglePerNode - Math.PI / 2
            val x = center.x + radius * valRatio * Math.cos(angle).toFloat()
            val y = center.y + radius * valRatio * Math.sin(angle).toFloat()
            val point = Offset(x, y)
            dataPoints.add(point)
            if (j == 0) dataPath.moveTo(x, y) else dataPath.lineTo(x, y)
        }
        dataPath.close()

        drawPath(
            path = dataPath,
            color = AccentBlue.copy(alpha = 0.2f),
            style = androidx.compose.ui.graphics.drawscope.Fill
        )
        drawPath(
            path = dataPath,
            color = AccentBlue,
            style = Stroke(2.dp.toPx(), pathEffect = PathEffect.cornerPathEffect(4.dp.toPx()))
        )

        // Draw Nodes
        dataPoints.forEachIndexed { index, point ->
            drawCircle(
                color = colors.getOrElse(index) { AccentBlue },
                radius = 4.dp.toPx(),
                center = point
            )
        }

        // Draw Labels
        val textPaint = android.graphics.Paint().apply {
            textSize = 10.sp.toPx()
            this.color = android.graphics.Color.DKGRAY
            textAlign = android.graphics.Paint.Align.CENTER
            isAntiAlias = true
        }

        for (j in labels.indices) {
            val angle = j * anglePerNode - Math.PI / 2
            val isBottom = Math.sin(angle) > 0.5
            val isTop = Math.sin(angle) < -0.5
            val labelRadius = radius * 1.35f
            
            val x = center.x + labelRadius * Math.cos(angle).toFloat()
            val y = center.y + labelRadius * Math.sin(angle).toFloat() + if (isBottom) textPaint.textSize else if (isTop) -textPaint.textSize / 2 else textPaint.textSize / 3

            var text = labels[j].replace("_", " ").split(" ").joinToString(" ") { 
                it.replaceFirstChar { char -> char.uppercase() } 
            }
            if (text.length > 15) text = text.take(13) + "…"
            drawContext.canvas.nativeCanvas.drawText(text, x, y, textPaint)
        }
    }
}
