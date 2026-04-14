package com.example.mhealth.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
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
import com.example.mhealth.ui.components.ScreenHeader

// ── Color palette ────────────────────────────────────────────────────────────
private val BgDark = Color(0xFF0D1117)
private val CardDark = Color(0xFF161B22)
private val BorderDark = Color(0xFF30363D)
private val TextPrimary = Color(0xFFE6EDF3)
private val TextSecondary = Color(0xFF8B949E)
private val AccentBlue = Color(0xFF58A6FF)
private val AccentGreen = Color(0xFF3FB950)
private val AccentOrange = Color(0xFFD29922)
private val AccentRed = Color(0xFFF85149)
private val AccentPurple = Color(0xFFBC8CFF)
private val AccentCyan = Color(0xFF39D2C0)

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
    val profileJson by DataRepository.s1ProfileJson.collectAsState()
    
    Column(Modifier.fillMaxSize().background(BgDark)) {
        ScreenHeader(
            title = "Behavioral DNA",
            subtitle = "System 1 — Baseline Pattern Fingerprinting",
            icon = Icons.Default.Favorite
        )
        DnaProfileSection(profileJson)
    }
}

@Composable
fun DnaProfileSection(profileJson: String?) {
    if (profileJson.isNullOrBlank() || profileJson == "{}") {
        DnaProfileEmptyState()
        return
    }

    val profile = remember(profileJson) {
        try { JSONObject(profileJson) } catch (_: Exception) { null }
    }

    if (profile == null) {
        DnaProfileEmptyState()
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
private fun DnaProfileEmptyState() {
    Card(
        modifier = Modifier.fillMaxWidth().padding(16.dp),
        colors = CardDefaults.cardColors(containerColor = CardDark),
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
                tint = TextSecondary
            )
            Spacer(Modifier.height(12.dp))
            Text(
                "DNA Baseline Not Yet Built",
                color = TextPrimary,
                fontWeight = FontWeight.Bold,
                fontSize = 16.sp
            )
            Spacer(Modifier.height(4.dp))
            Text(
                "Complete the 28-day baseline period to generate your behavioral DNA profile.",
                color = TextSecondary,
                fontSize = 13.sp
            )
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
        colors = CardDefaults.cardColors(containerColor = CardDark),
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

    var expanded by remember { mutableStateOf(false) }

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
        colors = CardDefaults.cardColors(containerColor = CardDark),
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
                            .background(BorderDark, RoundedCornerShape(4.dp))
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
                colors = CardDefaults.cardColors(containerColor = BgDark),
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
                                modifier = Modifier.weight(1f).height(4.dp).background(BorderDark, RoundedCornerShape(2.dp))
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

    var expanded by remember { mutableStateOf(false) }

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
                colors = CardDefaults.cardColors(containerColor = BgDark),
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

@Composable
private fun MiniStat(label: String, value: String, color: Color) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(value, color = color, fontSize = 10.sp, fontWeight = FontWeight.Bold)
        Text(label, color = TextSecondary, fontSize = 8.sp)
    }
}

// ── Phone DNA Card ───────────────────────────────────────────────────────────

@Composable
private fun PhoneDnaCard(profile: JSONObject) {
    val phoneDna = profile.optJSONObject("phone_dna") ?: return
    if (phoneDna.length() == 0) return

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = CardDark),
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

@Composable
private fun PhoneMetric(label: String, value: String, color: Color) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(value, color = color, fontWeight = FontWeight.Bold, fontSize = 14.sp)
        Text(label, color = TextSecondary, fontSize = 9.sp, textAlign = androidx.compose.ui.text.style.TextAlign.Center)
    }
}

// ── Texture Profiles Card ────────────────────────────────────────────────────

@Composable
private fun TextureProfilesCard(profile: JSONObject) {
    val textureArr = profile.optJSONArray("texture_profiles") ?: return
    if (textureArr.length() == 0) return

    var expanded by remember { mutableStateOf(false) }

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

@Composable
private fun TextureMetric(label: String, value: String, color: Color) {
    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.padding(2.dp)) {
        Text(value, color = color, fontWeight = FontWeight.Bold, fontSize = 12.sp)
        Text(label, color = TextSecondary, fontSize = 8.sp, textAlign = androidx.compose.ui.text.style.TextAlign.Center)
    }
}

// ── Reusable Collapsible Card ────────────────────────────────────────────────

@Composable
private fun CollapsibleCard(
    title: String,
    subtitle: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    expanded: Boolean,
    onToggle: () -> Unit,
    content: @Composable () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = CardDark),
        shape = RoundedCornerShape(12.dp),
        border = CardDefaults.outlinedCardBorder(true)
    ) {
        Column {
            // Header row (always visible)
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { onToggle() }
                    .padding(16.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(icon, null, tint = AccentBlue, modifier = Modifier.size(20.dp))
                Spacer(Modifier.width(8.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(title, color = TextPrimary, fontWeight = FontWeight.Bold, fontSize = 14.sp)
                    Text(subtitle, color = TextSecondary, fontSize = 11.sp)
                }
                Icon(
                    if (expanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                    null, tint = TextSecondary, modifier = Modifier.size(20.dp)
                )
            }

            // Expandable content
            AnimatedVisibility(
                visible = expanded,
                enter = expandVertically() + fadeIn(),
                exit = fadeOut()
            ) {
                Column(modifier = Modifier.padding(start = 16.dp, end = 16.dp, bottom = 16.dp)) {
                    HorizontalDivider(color = BorderDark)
                    Spacer(Modifier.height(8.dp))
                    content()
                }
            }
        }
    }
}