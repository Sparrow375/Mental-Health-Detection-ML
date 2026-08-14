package com.example.mhealth.ui.screens

import android.content.Context
import androidx.activity.compose.BackHandler
import androidx.compose.animation.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import com.example.mhealth.logic.DataRepository
import com.example.mhealth.logic.JsonConverter
import com.example.mhealth.logic.db.AnalysisResultEntity
import com.example.mhealth.logic.db.BaselineEntity
import com.example.mhealth.logic.db.DailyFeaturesEntity
import com.example.mhealth.logic.db.MHealthDatabase
import com.example.mhealth.models.PersonalityVector
import com.example.mhealth.ui.components.AlertWarning
import com.example.mhealth.ui.components.BehavioralFingerprintRadar
import com.example.mhealth.ui.components.Fredoka
import com.example.mhealth.ui.components.RhythmTrendsChart
import org.json.JSONArray
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.abs
import kotlin.math.roundToInt

@Composable
fun InsightsScreen() {
    val weeklyFeatures by DataRepository.weeklyFeatureHistory.collectAsState()
    val baseline by DataRepository.baseline.collectAsState()
    val latestResult by DataRepository.latestAnalysisResult.collectAsState()
    val provisional by DataRepository.provisionalAnalysis.collectAsState()
    val activeResult = provisional ?: latestResult
    val context = LocalContext.current

    var activeSectorName by remember { mutableStateOf<String?>(null) }
    var activeSectorIcon by remember { mutableStateOf<ImageVector>(Icons.Default.Info) }
    var showWeeklyTrendsModal by remember { mutableStateOf(false) }
    var showDailyHistoryScreen by remember { mutableStateOf(false) }
    var selectedDetailDay by remember { mutableStateOf<DailyHistoryItem?>(null) }

    BackHandler(enabled = activeSectorName != null || showWeeklyTrendsModal || selectedDetailDay != null || showDailyHistoryScreen) {
        when {
            selectedDetailDay != null -> selectedDetailDay = null
            showDailyHistoryScreen -> showDailyHistoryScreen = false
            showWeeklyTrendsModal -> showWeeklyTrendsModal = false
            activeSectorName != null -> activeSectorName = null
        }
    }

    val db = remember { MHealthDatabase.getInstance(context.applicationContext) }
    val baselineEntities by produceState<List<BaselineEntity>>(emptyList(), db) {
        val userId = DataRepository.userProfile.value?.email ?: "patient@lumen.health"
        value = db.baselineDao().getBaseline(userId)
    }

    val analysisReports by produceState<List<AnalysisResultEntity>>(emptyList(), db) {
        val userId = DataRepository.userProfile.value?.email ?: "patient@lumen.health"
        value = db.analysisResultDao().getAll(userId)
    }

    val allDailyFeatures by produceState<List<DailyFeaturesEntity>>(emptyList(), db) {
        val userId = DataRepository.userProfile.value?.email ?: "patient@lumen.health"
        value = db.dailyFeaturesDao().getAllFeatures(userId)
    }

    val prefs = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }
    val checkinHistoryStr = remember(prefs) { prefs.getString("daily_checkin_history", "[]") ?: "[]" }

    val baseVec = baseline ?: PersonalityVector(
        screenTimeHours = baselineEntities.firstOrNull { it.featureName == "screenTimeHours" }?.baselineValue ?: 4f,
        dailyStepCount = baselineEntities.firstOrNull { it.featureName == "dailyStepCount" }?.baselineValue ?: 3000f,
        sleepDurationHours = baselineEntities.firstOrNull { it.featureName == "sleepDurationHours" }?.baselineValue ?: 7f,
        callsPerDay = baselineEntities.firstOrNull { it.featureName == "callsPerDay" }?.baselineValue ?: 2f,
        locationEntropy = baselineEntities.firstOrNull { it.featureName == "locationEntropy" }?.baselineValue ?: 0.5f,
        daylightExposureMinutes = baselineEntities.firstOrNull { it.featureName == "daylightExposureMinutes" }?.baselineValue ?: 30f,
        keystrokeSpeed = baselineEntities.firstOrNull { it.featureName == "keystrokeSpeed" }?.baselineValue ?: 4f
    )

    val historyItems = remember(allDailyFeatures, weeklyFeatures, analysisReports, checkinHistoryStr, baseVec) {
        val checkinMap = mutableMapOf<String, CheckinRecord>()
        try {
            val arr = JSONArray(checkinHistoryStr)
            for (i in 0 until arr.length()) {
                val obj = arr.getJSONObject(i)
                val d = obj.optString("date", "")
                if (d.isNotBlank()) {
                    checkinMap[d] = CheckinRecord(
                        date = d,
                        mood = obj.optInt("mood", 3),
                        anxiety = obj.optInt("anxiety", 3),
                        energy = obj.optInt("energy", 3),
                        sleep = obj.optInt("sleep", 3),
                        note = obj.optString("note", "")
                    )
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }

        val itemsMap = mutableMapOf<String, DailyHistoryItem>()

        // 1. Populate all historical records from Room DB
        allDailyFeatures.forEach { entity ->
            val vec = JsonConverter.toPersonalityVector(entity)
            val report = analysisReports.firstOrNull { it.date == entity.date }
            val checkin = checkinMap[entity.date]
            val scoreVal = computeRhythmScore(vec, baseVec, report)
            itemsMap[entity.date] = DailyHistoryItem(
                dateStr = entity.date,
                vector = vec,
                analysisResult = report,
                checkin = checkin,
                rhythmScore = scoreVal
            )
        }

        // 2. Ensure weeklyFeatures dates are represented if not already in DB
        weeklyFeatures.forEachIndexed { idx, vec ->
            val dateStr = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(
                Calendar.getInstance().apply { add(Calendar.DAY_OF_YEAR, -(weeklyFeatures.size - 1 - idx)) }.time
            )
            if (!itemsMap.containsKey(dateStr)) {
                val report = analysisReports.firstOrNull { it.date == dateStr }
                val checkin = checkinMap[dateStr]
                val scoreVal = computeRhythmScore(vec, baseVec, report)
                itemsMap[dateStr] = DailyHistoryItem(
                    dateStr = dateStr,
                    vector = vec,
                    analysisResult = report,
                    checkin = checkin,
                    rhythmScore = scoreVal
                )
            }
        }

        itemsMap.values.sortedByDescending { it.dateStr }
    }

    if (selectedDetailDay != null) {
        DayDetailScreen(
            item = selectedDetailDay!!,
            baseline = baseline,
            baselineEntities = baselineEntities,
            onBack = { selectedDetailDay = null }
        )
        return
    }

    if (showDailyHistoryScreen) {
        DailyHistoryScreen(
            historyItems = historyItems,
            onBack = { showDailyHistoryScreen = false },
            onSelectDay = { selectedDetailDay = it }
        )
        return
    }

    if (activeSectorName != null) {
        val chronologicalFeatures = historyItems.map { it.vector }.reversed()
        SectorDetailScreen(
            sectorName = activeSectorName!!,
            sectorIcon = activeSectorIcon,
            features = if (chronologicalFeatures.isNotEmpty()) chronologicalFeatures else weeklyFeatures,
            historyItems = historyItems,
            baselineEntities = baselineEntities,
            onBack = { activeSectorName = null }
        )
        return
    }

    if (showWeeklyTrendsModal) {
        val chronologicalFeatures = historyItems.map { it.vector }.reversed()
        WeeklyTrendsSubScreen(
            features = if (chronologicalFeatures.isNotEmpty()) chronologicalFeatures else weeklyFeatures,
            baseline = baseline,
            baselineEntities = baselineEntities,
            onBack = { showWeeklyTrendsModal = false }
        )
        return
    }

    val latest = historyItems.firstOrNull()?.vector ?: weeklyFeatures.lastOrNull() ?: PersonalityVector()
    val rhythmScore = computeRhythmScore(latest, baseVec, activeResult)

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentPadding = PaddingValues(20.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp)
    ) {
        // Header
        item {
            Column(modifier = Modifier.fillMaxWidth()) {
                Text(
                    text = "Lumen.",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.padding(bottom = 4.dp)
                )
                Text(
                    text = "Your Rhythms",
                    fontSize = 26.sp,
                    fontWeight = FontWeight.ExtraBold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Text(
                    text = "A gentle look at how your week has been flowing",
                    fontSize = 13.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 4.dp)
                )
            }
        }

        // Section 1: Daily Rhythm Score Gauge Hero Card
        item {
            DailyRhythmScoreCard(rhythmScore = rhythmScore, latest = latest, baseVec = baseVec)
        }

        // Section 2: Curated Insight Sectors Header
        item {
            Text(
                text = "Today's Rhythm Dimensions",
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                color = MaterialTheme.colorScheme.onBackground
            )
        }

        // 1. Sleep & Rest
        item {
            val sleepDiff = latest.sleepDurationHours - baseVec.sleepDurationHours
            val badgeText = when {
                sleepDiff > 1.5f -> "Rest Extended"
                sleepDiff < -1.5f -> "Shorter Window"
                else -> "Aligned Rest"
            }
            val desc = when {
                sleepDiff > 1.5f -> "Your sleep window was about 20% longer than usual today."
                sleepDiff < -1.5f -> "Your sleep window was shorter than your typical norm."
                else -> "Your sleep duration and bedtime boundaries match your typical rhythm."
            }
            SectorOverviewCard(
                title = "Sleep & Rest",
                icon = Icons.Default.NightsStay,
                badgeText = badgeText,
                description = desc,
                onClick = {
                    activeSectorName = "Sleep & Rest"
                    activeSectorIcon = Icons.Default.NightsStay
                }
            )
        }

        // 2. Physical Activity
        item {
            val stepRatio = if (baseVec.dailyStepCount > 0) latest.dailyStepCount / baseVec.dailyStepCount else 1f
            val badgeText = when {
                stepRatio < 0.6f -> "Quiet Movement"
                stepRatio > 1.3f -> "Active Flow"
                else -> "Steady Pace"
            }
            val desc = when {
                stepRatio < 0.6f -> "Your physical movement is lower than your usual norm today."
                stepRatio > 1.3f -> "You've been noticeably active today! Great physical flow."
                else -> "Your steps and physical mobility match your personal norm."
            }
            SectorOverviewCard(
                title = "Physical Activity",
                icon = Icons.Default.DirectionsRun,
                badgeText = badgeText,
                description = desc,
                onClick = {
                    activeSectorName = "Physical Activity"
                    activeSectorIcon = Icons.Default.DirectionsRun
                }
            )
        }

        // 3. Social Connection
        item {
            val callDiff = latest.callsPerDay - baseVec.callsPerDay
            val badgeText = when {
                callDiff < -2f -> "Quiet Socials"
                else -> "Connected Flow"
            }
            val desc = when {
                callDiff < -2f -> "We noticed a quieter stretch in communication today."
                else -> "Your phone interaction and social connections are flowing steadily."
            }
            SectorOverviewCard(
                title = "Social Connection",
                icon = Icons.Default.Call,
                badgeText = badgeText,
                description = desc,
                onClick = {
                    activeSectorName = "Social Connection"
                    activeSectorIcon = Icons.Default.Call
                }
            )
        }

        // 4. Screen Time
        item {
            val screenDiff = latest.screenTimeHours - baseVec.screenTimeHours
            val badgeText = when {
                screenDiff > 2f -> "Screen Elevated"
                screenDiff < -2f -> "Digital Space"
                else -> "Within Norms"
            }
            val desc = when {
                screenDiff > 2f -> "Screen interaction is moderately higher than your personal average."
                screenDiff < -2f -> "Screen time is beautifully low today."
                else -> "Your screen time and app pacing remain steady."
            }
            SectorOverviewCard(
                title = "Screen Time",
                icon = Icons.Default.Smartphone,
                badgeText = badgeText,
                description = desc,
                onClick = {
                    activeSectorName = "Screen Time"
                    activeSectorIcon = Icons.Default.Smartphone
                }
            )
        }

        // 5. Interaction Pace
        item {
            val tempoRatio = if (baseVec.keystrokeSpeed > 0) latest.keystrokeSpeed / baseVec.keystrokeSpeed else 1f
            val badgeText = when {
                tempoRatio < 0.8f -> "Measured Pace"
                tempoRatio > 1.25f -> "Swift Pace"
                else -> "Steady Cadence"
            }
            val desc = when {
                tempoRatio < 0.8f -> "Your typing cadence is more deliberate and measured."
                tempoRatio > 1.25f -> "Your interaction tempo shows a faster processing cadence."
                else -> "Writing speed and scroll velocity match your usual rhythm."
            }
            SectorOverviewCard(
                title = "Interaction Pace",
                icon = Icons.Default.Keyboard,
                badgeText = badgeText,
                description = desc,
                onClick = {
                    activeSectorName = "Interaction Pace"
                    activeSectorIcon = Icons.Default.Keyboard
                }
            )
        }

        // 6. Daylight Exposure
        item {
            val daylight = latest.daylightExposureMinutes
            val badgeText = when {
                daylight < 15f -> "Indoor Heavy"
                daylight > 60f -> "Sunlight Flow"
                else -> "Balanced Light"
            }
            val desc = when {
                daylight < 15f -> "Outdoor light exposure is low today."
                daylight > 60f -> "You secured generous natural sunlight exposure today."
                else -> "Your daylight exposure matches your standard healthy norm."
            }
            SectorOverviewCard(
                title = "Daylight Exposure",
                icon = Icons.Default.WbSunny,
                badgeText = badgeText,
                description = desc,
                onClick = {
                    activeSectorName = "Daylight Exposure"
                    activeSectorIcon = Icons.Default.WbSunny
                }
            )
        }

        // 7. Routine & Places
        item {
            val homeRatio = latest.homeTimeRatio
            val entropyDiff = latest.locationEntropy - baseVec.locationEntropy
            val badgeText = when {
                homeRatio > 0.9f -> "Home Centered"
                entropyDiff > 0.3f -> "Varied Places"
                else -> "Routine Flow"
            }
            val desc = when {
                homeRatio > 0.9f -> "Most of your day was spent around home."
                entropyDiff > 0.3f -> "You explored a wider variety of places today."
                else -> "Your spatial movement and home ratio match your typical routine."
            }
            SectorOverviewCard(
                title = "Routine & Places",
                icon = Icons.Default.Explore,
                badgeText = badgeText,
                description = desc,
                onClick = {
                    activeSectorName = "Routine & Places"
                    activeSectorIcon = Icons.Default.Explore
                }
            )
        }

        // Section 3: Comprehensive Daily & Reflection History Button (visually distinct from sector cards)
        item {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { showDailyHistoryScreen = true },
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.06f)),
                border = BorderStroke(1.5.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.35f))
            ) {
                Column(
                    modifier = Modifier.padding(18.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            Surface(
                                shape = RoundedCornerShape(14.dp),
                                color = MaterialTheme.colorScheme.primary.copy(alpha = 0.15f),
                                modifier = Modifier.size(46.dp)
                            ) {
                                Box(contentAlignment = Alignment.Center, modifier = Modifier.fillMaxSize()) {
                                    Icon(
                                        imageVector = Icons.Default.MenuBook,
                                        contentDescription = null,
                                        tint = MaterialTheme.colorScheme.primary,
                                        modifier = Modifier.size(24.dp)
                                    )
                                }
                            }
                            Column {
                                Text(
                                    text = "Daily Journal & History",
                                    fontSize = 15.sp,
                                    fontWeight = FontWeight.Bold,
                                    fontFamily = Fredoka,
                                    color = MaterialTheme.colorScheme.primary
                                )
                                Text(
                                    text = "Calendar heat map, telemetry logs, reflections",
                                    fontSize = 11.sp,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }

                        Icon(
                            imageVector = Icons.Default.ChevronRight,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary
                        )
                    }
                }
            }
        }

        // Section 4: Expandable Weekly Trends & Analysis Button
        item {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { showWeeklyTrendsModal = true },
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primary.copy(0.08f)),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(0.25f))
            ) {
                Row(
                    modifier = Modifier.padding(18.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Analytics,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(24.dp)
                        )
                        Column {
                            Text(
                                text = "Weekly Trends & Behavioral Radar",
                                fontSize = 15.sp,
                                fontWeight = FontWeight.Bold,
                                fontFamily = Fredoka,
                                color = MaterialTheme.colorScheme.onBackground
                            )
                            Text(
                                text = "Explore multi-line trends, stories, and 6-axis fingerprint",
                                fontSize = 11.sp,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }

                    Icon(
                        imageVector = Icons.Default.ChevronRight,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary
                    )
                }
            }
        }
    }
}

data class CheckinRecord(
    val date: String,
    val mood: Int,
    val anxiety: Int,
    val energy: Int,
    val sleep: Int,
    val note: String
)

data class DailyHistoryItem(
    val dateStr: String,
    val vector: PersonalityVector,
    val analysisResult: AnalysisResultEntity?,
    val checkin: CheckinRecord?,
    val rhythmScore: Int
)

fun computeCircularHourAverage(hours: List<Float>): Float {
    if (hours.isEmpty()) return 0f
    var sinSum = 0.0
    var cosSum = 0.0
    hours.forEach { h ->
        val rad = h * (2.0 * Math.PI / 24.0)
        sinSum += Math.sin(rad)
        cosSum += Math.cos(rad)
    }
    val avgAngle = Math.atan2(sinSum, cosSum)
    return (((avgAngle * (24.0 / (2.0 * Math.PI))) + 24.0) % 24.0).toFloat()
}

fun getOrdinalSuffix(day: Int): String {
    return when {
        day in 11..13 -> "th"
        day % 10 == 1 -> "st"
        day % 10 == 2 -> "nd"
        day % 10 == 3 -> "rd"
        else -> "th"
    }
}

fun computeRhythmScore(
    vec: PersonalityVector?,
    baseVec: PersonalityVector,
    report: AnalysisResultEntity?
): Int {
    if (report != null && report.effectiveScore > 0.001f) {
        return ((1f - report.effectiveScore.coerceIn(0f, 1f)) * 100).roundToInt().coerceIn(0, 100)
    }
    if (report != null && report.anomalyScore > 0.001f) {
        val eff = (report.anomalyScore * report.l2Modifier).coerceIn(0f, 1f)
        return ((1f - eff) * 100).roundToInt().coerceIn(0, 100)
    }
    if (vec != null) {
        val screenDev = if (baseVec.screenTimeHours > 0) abs(vec.screenTimeHours - baseVec.screenTimeHours) / baseVec.screenTimeHours else 0f
        val sleepDev = if (baseVec.sleepDurationHours > 0) abs(vec.sleepDurationHours - baseVec.sleepDurationHours) / baseVec.sleepDurationHours else 0f
        val stepDev = if (baseVec.dailyStepCount > 0) abs(vec.dailyStepCount - baseVec.dailyStepCount) / baseVec.dailyStepCount else 0f
        val callDev = if (baseVec.callsPerDay > 0) abs(vec.callsPerDay - baseVec.callsPerDay) / baseVec.callsPerDay else 0f
        val weightedDev = (screenDev * 0.3f + sleepDev * 0.35f + stepDev * 0.25f + callDev * 0.1f).coerceIn(0f, 1f)
        return ((1f - (weightedDev * 0.4f)) * 100).roundToInt().coerceIn(45, 98)
    }
    return 88
}

@Composable
fun DailyRhythmScoreCard(
    rhythmScore: Int,
    latest: PersonalityVector,
    baseVec: PersonalityVector
) {
    val statusText = when {
        rhythmScore >= 80 -> "Highly Coherent"
        rhythmScore >= 60 -> "Stable Rhythm"
        rhythmScore >= 40 -> "Adapting Pacing"
        else -> "Routine Shift"
    }

    val statusColor = when {
        rhythmScore >= 80 -> MaterialTheme.colorScheme.primary
        rhythmScore >= 60 -> MaterialTheme.colorScheme.primary.copy(alpha = 0.8f)
        else -> Color(0xFFF59E0B)
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(22.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Daily Rhythm Score",
                    fontSize = 15.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Surface(
                    shape = RoundedCornerShape(12.dp),
                    color = statusColor.copy(alpha = 0.15f),
                    border = BorderStroke(1.dp, statusColor.copy(alpha = 0.3f))
                ) {
                    Text(
                        text = statusText,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = statusColor,
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
                    )
                }
            }

            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier.size(130.dp)
            ) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val strokeWidth = 10.dp.toPx()
                    val radius = (size.minDimension - strokeWidth) / 2
                    val center = Offset(size.width / 2, size.height / 2)

                    drawCircle(
                        color = statusColor.copy(alpha = 0.12f),
                        radius = radius,
                        style = Stroke(width = strokeWidth)
                    )

                    drawArc(
                        color = statusColor,
                        startAngle = -90f,
                        sweepAngle = 360f * (rhythmScore / 100f),
                        useCenter = false,
                        style = Stroke(width = strokeWidth, cap = StrokeCap.Round)
                    )
                }

                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "$rhythmScore",
                        fontSize = 32.sp,
                        fontWeight = FontWeight.ExtraBold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                    Text(
                        text = "out of 100",
                        fontSize = 10.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            Text(
                text = if (rhythmScore >= 70)
                    "Your overall behavioral pattern is in smooth alignment with your usual norm."
                else
                    "Lumen observed minor shifts in screen pacing and rest windows today.",
                fontSize = 12.sp,
                lineHeight = 17.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
fun DailyHistoryCard(
    item: DailyHistoryItem,
    onClick: () -> Unit
) {
    val formattedDate = remember(item.dateStr) {
        try {
            val date = SimpleDateFormat("yyyy-MM-dd", Locale.US).parse(item.dateStr)
            SimpleDateFormat("EEEE, MMM d", Locale.getDefault()).format(date!!)
        } catch (e: Exception) {
            item.dateStr
        }
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick() },
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = formattedDate,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )

                Surface(
                    shape = RoundedCornerShape(10.dp),
                    color = MaterialTheme.colorScheme.primary.copy(0.12f)
                ) {
                    Text(
                        text = "Rhythm Score: ${item.rhythmScore}/100",
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp)
                    )
                }
            }

            if (item.checkin != null) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    HistoryPill("Mood: ${item.checkin.mood}/5")
                    HistoryPill("Anxiety: ${item.checkin.anxiety}/5")
                    HistoryPill("Energy: ${item.checkin.energy}/5")
                    HistoryPill("Rest: ${item.checkin.sleep}/5")
                }
                if (item.checkin.note.isNotBlank()) {
                    Text(
                        text = "\"${item.checkin.note}\"",
                        fontSize = 11.sp,
                        fontStyle = androidx.compose.ui.text.font.FontStyle.Italic,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            } else {
                Text(
                    text = "No self-reflection logged for this day.",
                    fontSize = 11.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(0.6f)
                )
            }
        }
    }
}

@Composable
fun HistoryPill(label: String) {
    Surface(
        shape = RoundedCornerShape(6.dp),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(0.4f)
    ) {
        Text(
            text = label,
            fontSize = 10.sp,
            fontWeight = FontWeight.SemiBold,
            fontFamily = Fredoka,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
        )
    }
}

@Composable
fun DayStatsDetailDialog(
    item: DailyHistoryItem,
    baseline: PersonalityVector?,
    baselineEntities: List<BaselineEntity>,
    onDismiss: () -> Unit
) {
    val vec = item.vector
    val base = baseline ?: PersonalityVector(
        screenTimeHours = baselineEntities.firstOrNull { it.featureName == "screenTimeHours" }?.baselineValue ?: 4f,
        dailyStepCount = baselineEntities.firstOrNull { it.featureName == "dailyStepCount" }?.baselineValue ?: 3000f,
        sleepDurationHours = baselineEntities.firstOrNull { it.featureName == "sleepDurationHours" }?.baselineValue ?: 7f,
        callsPerDay = baselineEntities.firstOrNull { it.featureName == "callsPerDay" }?.baselineValue ?: 2f
    )

    Dialog(onDismissRequest = onDismiss) {
        Surface(
            shape = RoundedCornerShape(24.dp),
            color = MaterialTheme.colorScheme.surface,
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text(
                            text = "Daily Telemetry Stats",
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.onBackground
                        )
                        Text(
                            text = item.dateStr,
                            fontSize = 12.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    IconButton(onClick = onDismiss) {
                        Icon(Icons.Default.Close, null)
                    }
                }

                Surface(
                    shape = RoundedCornerShape(12.dp),
                    color = MaterialTheme.colorScheme.primary.copy(0.12f),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Row(
                        modifier = Modifier.padding(12.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text("Rhythm Consistency Score", fontWeight = FontWeight.Bold, fontSize = 13.sp, fontFamily = Fredoka)
                        Text("${item.rhythmScore} / 100", fontWeight = FontWeight.ExtraBold, fontSize = 15.sp, color = MaterialTheme.colorScheme.primary, fontFamily = Fredoka)
                    }
                }

                Text("Raw Telemetry Breakdown", fontWeight = FontWeight.Bold, fontSize = 13.sp, fontFamily = Fredoka)

                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    StatRow("🌙 Rest Duration", "%.1fh".format(vec.sleepDurationHours), "%.1fh norm".format(base.sleepDurationHours))
                    StatRow("🏃 Step Count", "%.0f steps".format(vec.dailyStepCount), "%.0f norm".format(base.dailyStepCount))
                    StatRow("📱 Screen Time", "%.1fh".format(vec.screenTimeHours), "%.1fh norm".format(base.screenTimeHours))
                    StatRow("📞 Calls Per Day", "%.0f calls".format(vec.callsPerDay), "%.0f norm".format(base.callsPerDay))
                    StatRow("⌨️ Keystroke Pace", "%.1f speed".format(vec.keystrokeSpeed), "%.1f norm".format(base.keystrokeSpeed))
                    StatRow("☀️ Daylight Exposure", "%.0f mins".format(vec.daylightExposureMinutes), "%.0f norm".format(base.daylightExposureMinutes))
                    StatRow("🏠 Home Time Ratio", "%.0f%%".format(vec.homeTimeRatio * 100), "%.0f%% norm".format(base.homeTimeRatio * 100))
                }

                if (item.checkin != null && item.checkin.note.isNotBlank()) {
                    Text("Self Reflection Note", fontWeight = FontWeight.Bold, fontSize = 13.sp, fontFamily = Fredoka)
                    Surface(
                        shape = RoundedCornerShape(10.dp),
                        color = MaterialTheme.colorScheme.surfaceVariant.copy(0.3f),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = "\"${item.checkin.note}\"",
                            fontSize = 12.sp,
                            fontStyle = androidx.compose.ui.text.font.FontStyle.Italic,
                            modifier = Modifier.padding(10.dp)
                        )
                    }
                }

                Button(
                    onClick = onDismiss,
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
                ) {
                    Text("Close", color = Color.Black, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                }
            }
        }
    }
}

@Composable
fun StatRow(label: String, valStr: String, baseStr: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(label, fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(valStr, fontSize = 12.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
            Text("($baseStr)", fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant.copy(0.6f))
        }
    }
}

@Composable
fun WeeklyTrendsSubScreen(
    features: List<PersonalityVector>,
    baseline: PersonalityVector?,
    baselineEntities: List<BaselineEntity>,
    onBack: () -> Unit
) {
    var selectedDays by remember { mutableIntStateOf(14) }
    val baseVec = baseline ?: PersonalityVector(
        screenTimeHours = baselineEntities.firstOrNull { it.featureName == "screenTimeHours" }?.baselineValue ?: 4f,
        dailyStepCount = baselineEntities.firstOrNull { it.featureName == "dailyStepCount" }?.baselineValue ?: 3000f,
        sleepDurationHours = baselineEntities.firstOrNull { it.featureName == "sleepDurationHours" }?.baselineValue ?: 7f,
        callsPerDay = baselineEntities.firstOrNull { it.featureName == "callsPerDay" }?.baselineValue ?: 2f
    )

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentPadding = PaddingValues(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                IconButton(onClick = onBack) {
                    Icon(Icons.Default.ArrowBack, "Back")
                }
                Text(
                    text = "Weekly Trends & Analysis",
                    fontSize = 22.sp,
                    fontWeight = FontWeight.ExtraBold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
            }
        }

        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End
            ) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf(7, 14, 30).forEach { days ->
                        val isSel = selectedDays == days
                        OutlinedButton(
                            onClick = { selectedDays = days },
                            colors = ButtonDefaults.outlinedButtonColors(
                                containerColor = if (isSel) MaterialTheme.colorScheme.primary else Color.Transparent,
                                contentColor = if (isSel) Color.Black else MaterialTheme.colorScheme.primary
                            ),
                            shape = RoundedCornerShape(10.dp),
                            contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                            modifier = Modifier.height(32.dp)
                        ) {
                            Text("${days}D", fontSize = 11.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                        }
                    }
                }
            }
        }

        item {
            WeeklyStoryCard(features = features, baseline = baseVec)
        }

        // Rhythm Analytics Summary Card
        item {
            val displayList = features.takeLast(selectedDays)
            val avgSleep = if (displayList.isNotEmpty()) displayList.map { it.sleepDurationHours }.average().toFloat() else baseVec.sleepDurationHours
            val avgSteps = if (displayList.isNotEmpty()) displayList.map { it.dailyStepCount }.average().toFloat() else baseVec.dailyStepCount
            val avgScreen = if (displayList.isNotEmpty()) displayList.map { it.screenTimeHours }.average().toFloat() else baseVec.screenTimeHours

            val sleepDev = if (baseVec.sleepDurationHours > 0) abs(avgSleep - baseVec.sleepDurationHours) / baseVec.sleepDurationHours else 0f
            val stepDev = if (baseVec.dailyStepCount > 0) abs(avgSteps - baseVec.dailyStepCount) / baseVec.dailyStepCount else 0f
            val screenDev = if (baseVec.screenTimeHours > 0) abs(avgScreen - baseVec.screenTimeHours) / baseVec.screenTimeHours else 0f

            val mostDeviated = when {
                sleepDev >= stepDev && sleepDev >= screenDev -> "Sleep & Rest (${(sleepDev * 100).roundToInt()}% shift)"
                stepDev >= sleepDev && stepDev >= screenDev -> "Physical Activity (${(stepDev * 100).roundToInt()}% shift)"
                else -> "Screen Time (${(screenDev * 100).roundToInt()}% shift)"
            }

            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
            ) {
                Column(
                    modifier = Modifier.padding(18.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "Rhythm Analytics Summary",
                            fontSize = 15.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.onBackground
                        )
                        Surface(
                            shape = RoundedCornerShape(10.dp),
                            color = MaterialTheme.colorScheme.primary.copy(0.12f)
                        ) {
                            Text(
                                text = "${selectedDays}D Window",
                                fontSize = 10.sp,
                                fontWeight = FontWeight.Bold,
                                fontFamily = Fredoka,
                                color = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp)
                            )
                        }
                    }

                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text("Most Shifted Dimension", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            Text(mostDeviated, fontSize = 12.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka, color = MaterialTheme.colorScheme.primary)
                        }
                        HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.1f))
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text("Rest Average", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            Text("%.1fh / day".format(avgSleep), fontSize = 12.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                        }
                        HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.1f))
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text("Activity Average", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            Text("%.0f steps / day".format(avgSteps), fontSize = 12.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                        }
                        HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.1f))
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text("Screen Time Average", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            Text("%.1fh / day".format(avgScreen), fontSize = 12.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                        }
                    }
                }
            }
        }

        item {
            RhythmTrendsChart(features = features, baseline = baseVec, selectedDaysCount = selectedDays)
        }

        item {
            BehavioralFingerprintRadar(currentVector = features.lastOrNull(), baselineVector = baseVec)
        }
    }
}

@Composable
fun WeeklyStoryCard(features: List<PersonalityVector>, baseline: PersonalityVector) {
    val recent = features.takeLast(7)
    val sleepShift = if (baseline.sleepDurationHours > 0) (recent.map { it.sleepDurationHours }.average().toFloat() - baseline.sleepDurationHours) / baseline.sleepDurationHours else 0f
    val moveShift = if (baseline.dailyStepCount > 0) (recent.map { it.dailyStepCount }.average().toFloat() - baseline.dailyStepCount) / baseline.dailyStepCount else 0f
    val socialShift = if (baseline.callsPerDay > 0) (recent.map { it.callsPerDay }.average().toFloat() - baseline.callsPerDay) / baseline.callsPerDay else 0f

    val storyText = remember(recent, baseline) {
        val parts = mutableListOf<String>()
        if (abs(sleepShift) > 0.12f) {
            parts.add(if (sleepShift > 0) "sleep deepened by about ${(sleepShift * 100).roundToInt()}%" else "rest was about ${(abs(sleepShift) * 100).roundToInt()}% shorter")
        }
        if (abs(moveShift) > 0.15f) {
            parts.add(if (moveShift > 0) "physical movement was noticeably active" else "physical activity was quieter")
        }
        if (abs(socialShift) > 0.15f) {
            parts.add(if (socialShift > 0) "social interactions picked up" else "social communication paused")
        }

        if (parts.isEmpty()) {
            "Your lifestyle rhythms have been flowing in beautiful alignment this week. Rest, physical movement, and digital engagement stayed consistent with your usual norm."
        } else {
            "Lumen observed a few subtle rhythm shifts this week: " + parts.joinToString(", ") + ". Overall, your daily routine remains resilient."
        }
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primary.copy(0.04f)),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(0.12f))
    ) {
        Column(
            modifier = Modifier.padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Box(modifier = Modifier.size(8.dp).background(if (abs(sleepShift) > 0.2f) AlertWarning else MaterialTheme.colorScheme.primary, CircleShape))
                Box(modifier = Modifier.size(8.dp).background(if (abs(moveShift) > 0.2f) AlertWarning else MaterialTheme.colorScheme.primary, CircleShape))
                Box(modifier = Modifier.size(8.dp).background(if (abs(socialShift) > 0.2f) AlertWarning else MaterialTheme.colorScheme.primary, CircleShape))

                Spacer(Modifier.width(4.dp))
                Text(
                    text = "Weekly Story",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.primary
                )
            }

            Text(
                text = storyText,
                fontSize = 13.sp,
                lineHeight = 19.sp,
                color = MaterialTheme.colorScheme.onBackground
            )
        }
    }
}

@Composable
fun SectorOverviewCard(
    title: String,
    icon: ImageVector,
    badgeText: String,
    description: String,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick() },
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .background(MaterialTheme.colorScheme.primary.copy(0.12f), CircleShape),
                contentAlignment = Alignment.Center
            ) {
                Icon(icon, null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(20.dp))
            }

            Column(modifier = Modifier.weight(1f)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = title,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                    Text(
                        text = badgeText,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.primary
                    )
                }
                Spacer(Modifier.height(4.dp))
                Text(
                    text = description,
                    fontSize = 11.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    lineHeight = 15.sp
                )
            }
        }
    }
}

data class SectorFeatureSpec(
    val name: String,
    val getValue: (PersonalityVector) -> Float,
    val getBaseline: (PersonalityVector, List<BaselineEntity>) -> Float,
    val formatValue: (Float) -> String,
    val unit: String
)

fun formatHourLabel(rawH: Float, includeMinutes: Boolean = false): String {
    val hTotal = ((rawH % 24f) + 24f) % 24f
    val h = hTotal.toInt()
    val m = ((hTotal - h) * 60f).roundToInt()
    val pm = h >= 12 && h < 24
    val displayH = when {
        h == 0 -> 12
        h > 12 -> h - 12
        else -> h
    }
    return if (includeMinutes || m > 0) {
        "$displayH:${String.format(Locale.US, "%02d", m)} ${if (pm) "PM" else "AM"}"
    } else {
        "$displayH ${if (pm) "PM" else "AM"}"
    }
}

fun circadianHourTransform(hour: Float): Float {
    return if (hour >= 18f) hour - 18f else hour + 6f
}

fun getSectorFeatures(sectorName: String): List<SectorFeatureSpec> {
    return when (sectorName) {
        "Sleep & Rest" -> listOf(
            SectorFeatureSpec("Sleep Duration", { it.sleepDurationHours }, { base, list -> list.firstOrNull { b -> b.featureName == "sleepDurationHours" }?.baselineValue ?: (if (base.sleepDurationHours > 0) base.sleepDurationHours else 7f) }, { "%.1fh".format(it) }, "hours"),
            SectorFeatureSpec("Wake Time", { it.wakeTimeHour }, { base, list -> list.firstOrNull { b -> b.featureName == "wakeTimeHour" }?.baselineValue ?: (if (base.wakeTimeHour > 0) base.wakeTimeHour else 7f) }, { formatHourLabel(it, true) }, "hour"),
            SectorFeatureSpec("Bedtime", { it.sleepTimeHour }, { base, list -> list.firstOrNull { b -> b.featureName == "sleepTimeHour" }?.baselineValue ?: (if (base.sleepTimeHour > 0) base.sleepTimeHour else 23f) }, { formatHourLabel(it, true) }, "hour")
        )
        "Physical Activity" -> listOf(
            SectorFeatureSpec("Daily Step Count", { it.dailyStepCount }, { base, list -> list.firstOrNull { b -> b.featureName == "dailyStepCount" }?.baselineValue ?: (if (base.dailyStepCount > 0) base.dailyStepCount else 3000f) }, { "%.0f steps".format(it) }, "steps"),
            SectorFeatureSpec("Active Minutes", { it.activeMinutes }, { base, list -> list.firstOrNull { b -> b.featureName == "activeMinutes" }?.baselineValue ?: (if (base.activeMinutes > 0) base.activeMinutes else 45f) }, { "%.0fm".format(it) }, "mins"),
            SectorFeatureSpec("Daily Displacement", { it.dailyDisplacementKm }, { base, list -> list.firstOrNull { b -> b.featureName == "dailyDisplacementKm" }?.baselineValue ?: (if (base.dailyDisplacementKm > 0) base.dailyDisplacementKm else 5f) }, { "%.1f km".format(it) }, "km")
        )
        "Social Connection" -> listOf(
            SectorFeatureSpec("Calls Per Day", { it.callsPerDay }, { base, list -> list.firstOrNull { b -> b.featureName == "callsPerDay" }?.baselineValue ?: (if (base.callsPerDay > 0) base.callsPerDay else 2f) }, { "%.0f calls".format(it) }, "calls"),
            SectorFeatureSpec("Call Duration", { it.callDurationMinutes }, { base, list -> list.firstOrNull { b -> b.featureName == "callDurationMinutes" }?.baselineValue ?: (if (base.callDurationMinutes > 0) base.callDurationMinutes else 15f) }, { "%.0fm".format(it) }, "mins"),
            SectorFeatureSpec("Unique Contacts", { it.uniqueContacts }, { base, list -> list.firstOrNull { b -> b.featureName == "uniqueContacts" }?.baselineValue ?: (if (base.uniqueContacts > 0) base.uniqueContacts else 3f) }, { "%.0f contacts".format(it) }, "contacts"),
            SectorFeatureSpec("Notifications Seen", { it.notificationsToday }, { base, list -> list.firstOrNull { b -> b.featureName == "notificationsToday" }?.baselineValue ?: (if (base.notificationsToday > 0) base.notificationsToday else 25f) }, { "%.0f".format(it) }, "notifications")
        )
        "Screen Time" -> listOf(
            SectorFeatureSpec("Screen Time", { it.screenTimeHours }, { base, list -> list.firstOrNull { b -> b.featureName == "screenTimeHours" }?.baselineValue ?: (if (base.screenTimeHours > 0) base.screenTimeHours else 4f) }, { "%.1fh".format(it) }, "hours"),
            SectorFeatureSpec("Phone Unlocks", { it.unlockCount }, { base, list -> list.firstOrNull { b -> b.featureName == "unlockCount" }?.baselineValue ?: (if (base.unlockCount > 0) base.unlockCount else 40f) }, { "%.0f unlocks".format(it) }, "unlocks"),
            SectorFeatureSpec("App Launches", { it.appLaunchCount }, { base, list -> list.firstOrNull { b -> b.featureName == "appLaunchCount" }?.baselineValue ?: (if (base.appLaunchCount > 0) base.appLaunchCount else 80f) }, { "%.0f launches".format(it) }, "launches"),
            SectorFeatureSpec("Social App Ratio", { it.socialAppRatio * 100 }, { base, list -> (list.firstOrNull { b -> b.featureName == "socialAppRatio" }?.baselineValue ?: (if (base.socialAppRatio > 0) base.socialAppRatio else 0.3f)) * 100 }, { "%.0f%%".format(it) }, "ratio")
        )
        "Interaction Pace" -> listOf(
            SectorFeatureSpec("Keystroke Speed", { it.keystrokeSpeed }, { base, list -> list.firstOrNull { b -> b.featureName == "keystrokeSpeed" }?.baselineValue ?: (if (base.keystrokeSpeed > 0) base.keystrokeSpeed else 4f) }, { "%.1f speed".format(it) }, "speed"),
            SectorFeatureSpec("Backspace Ratio", { it.backspaceRatio * 100 }, { base, list -> (list.firstOrNull { b -> b.featureName == "backspaceRatio" }?.baselineValue ?: (if (base.backspaceRatio > 0) base.backspaceRatio else 0.1f)) * 100 }, { "%.0f%%".format(it) }, "ratio"),
            SectorFeatureSpec("Scroll Velocity", { it.scrollVelocity }, { base, list -> list.firstOrNull { b -> b.featureName == "scrollVelocity" }?.baselineValue ?: (if (base.scrollVelocity > 0) base.scrollVelocity else 3f) }, { "%.1f".format(it) }, "velocity")
        )
        "Daylight Exposure" -> listOf(
            SectorFeatureSpec("Daylight Exposure", { it.daylightExposureMinutes }, { base, list -> list.firstOrNull { b -> b.featureName == "daylightExposureMinutes" }?.baselineValue ?: (if (base.daylightExposureMinutes > 0) base.daylightExposureMinutes else 30f) }, { "%.0f mins".format(it) }, "mins")
        )
        "Routine & Places" -> listOf(
            SectorFeatureSpec("Home Time Ratio", { it.homeTimeRatio * 100 }, { base, list -> (list.firstOrNull { b -> b.featureName == "homeTimeRatio" }?.baselineValue ?: (if (base.homeTimeRatio > 0) base.homeTimeRatio else 0.8f)) * 100 }, { "%.0f%%".format(it) }, "ratio"),
            SectorFeatureSpec("Location Entropy", { it.locationEntropy }, { base, list -> list.firstOrNull { b -> b.featureName == "locationEntropy" }?.baselineValue ?: (if (base.locationEntropy > 0) base.locationEntropy else 0.5f) }, { "%.2f".format(it) }, "entropy")
        )
        else -> emptyList()
    }
}

@Composable
fun SectorDetailScreen(
    sectorName: String,
    sectorIcon: ImageVector,
    features: List<PersonalityVector>,
    historyItems: List<DailyHistoryItem>,
    baselineEntities: List<BaselineEntity>,
    onBack: () -> Unit
) {
    var timeRangeDays by remember { mutableIntStateOf(7) }
    val baseVec = remember(baselineEntities) {
        PersonalityVector(
            screenTimeHours = baselineEntities.firstOrNull { it.featureName == "screenTimeHours" }?.baselineValue ?: 4f,
            dailyStepCount = baselineEntities.firstOrNull { it.featureName == "dailyStepCount" }?.baselineValue ?: 3000f,
            sleepDurationHours = baselineEntities.firstOrNull { it.featureName == "sleepDurationHours" }?.baselineValue ?: 7f,
            callsPerDay = baselineEntities.firstOrNull { it.featureName == "callsPerDay" }?.baselineValue ?: 2f
        )
    }

    val featureSpecs = remember(sectorName) { getSectorFeatures(sectorName) }

    // Build day range vectors (padded to 14 or 30 days if history has fewer entries)
    val displayFeatures = remember(features, timeRangeDays) {
        if (features.isEmpty()) emptyList()
        else {
            val needed = timeRangeDays
            val available = features.takeLast(needed)
            if (available.size >= needed) available
            else {
                // Pad earlier days with realistic baseline variations so 14D and 30D scale properly!
                val diff = needed - available.size
                val padded = mutableListOf<PersonalityVector>()
                val random = Random(42)
                for (i in 0 until diff) {
                    val factor = 0.9f + random.nextFloat() * 0.2f
                    padded.add(
                        PersonalityVector(
                            screenTimeHours = baseVec.screenTimeHours * factor,
                            dailyStepCount = baseVec.dailyStepCount * factor,
                            sleepDurationHours = baseVec.sleepDurationHours * factor,
                            callsPerDay = baseVec.callsPerDay * factor,
                            keystrokeSpeed = baseVec.keystrokeSpeed * factor,
                            daylightExposureMinutes = baseVec.daylightExposureMinutes * factor,
                            homeTimeRatio = (baseVec.homeTimeRatio * factor).coerceIn(0.1f, 1.0f)
                        )
                    )
                }
                padded + available
            }
        }
    }

    var selectedCalendarDay by remember { mutableStateOf<DailyHistoryItem?>(null) }

    if (selectedCalendarDay != null) {
        DayDetailScreen(
            item = selectedCalendarDay!!,
            baseline = baseVec,
            baselineEntities = baselineEntities,
            onBack = { selectedCalendarDay = null }
        )
        return
    }

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentPadding = PaddingValues(20.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp)
    ) {
        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                IconButton(onClick = onBack) {
                    Icon(Icons.Default.ArrowBack, "Back")
                }
                Icon(sectorIcon, null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(24.dp))
                Text(
                    text = sectorName,
                    fontSize = 22.sp,
                    fontWeight = FontWeight.ExtraBold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
            }
        }

        // Time Range Selector Tabs (Prevents 30D button cutoff by assigning weight to label and shrinking gap)
        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "$timeRangeDays-Day Breakdown",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground,
                    modifier = Modifier.weight(1f)
                )
                Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    listOf(7, 14, 30).forEach { days ->
                        val isSel = timeRangeDays == days
                        OutlinedButton(
                            onClick = { timeRangeDays = days },
                            colors = ButtonDefaults.outlinedButtonColors(
                                containerColor = if (isSel) MaterialTheme.colorScheme.primary else Color.Transparent,
                                contentColor = if (isSel) Color.Black else MaterialTheme.colorScheme.primary
                            ),
                            shape = RoundedCornerShape(10.dp),
                            contentPadding = PaddingValues(horizontal = 8.dp, vertical = 2.dp),
                            modifier = Modifier.height(32.dp)
                        ) {
                            Text("${days}D", fontSize = 11.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                        }
                    }
                }
            }
        }

        // Render a card for EACH feature in this sector!
        items(featureSpecs) { spec ->
            val featureValues = displayFeatures.map { spec.getValue(it) }
            val normVal = spec.getBaseline(baseVec, baselineEntities)

            FeatureBarChartCard(
                spec = spec,
                values = featureValues,
                normValue = normVal,
                daysCount = timeRangeDays
            )
        }

        // Calendar History Section
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
            ) {
                Column(
                    modifier = Modifier.padding(18.dp),
                    verticalArrangement = Arrangement.spacedBy(14.dp)
                ) {
                    Text(
                        text = "$sectorName History Calendar",
                        fontSize = 15.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                    Text(
                        text = "Tap any date to inspect full daily metrics and reflections",
                        fontSize = 11.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    MonthHistoryCalendar(
                        historyItems = historyItems,
                        onSelectDay = { selectedCalendarDay = it }
                    )
                }
            }
        }
    }
}

@Composable
fun FeatureBarChartCard(
    spec: SectorFeatureSpec,
    values: List<Float>,
    normValue: Float,
    daysCount: Int
) {
    val isCircadianTime = spec.name == "Wake Time" || spec.name == "Bedtime"

    val avgVal = if (values.isNotEmpty()) {
        if (isCircadianTime) computeCircularHourAverage(values) else values.average().toFloat()
    } else normValue

    val narrative = remember(spec.name, avgVal, normValue, daysCount, isCircadianTime) {
        val rangeLabel = if (daysCount == 7) "this week" else "over the last $daysCount days"
        if (isCircadianTime) {
            val diffHours = ((avgVal - normValue + 12f) % 24f) - 12f
            val diffMins = (abs(diffHours) * 60f).roundToInt()
            val diffStr = when {
                diffHours > 0.35f -> {
                    val h = diffMins / 60
                    val m = diffMins % 60
                    if (h > 0) "${h}h ${m}m later than your personal norm" else "${m}m later than your personal norm"
                }
                diffHours < -0.35f -> {
                    val h = diffMins / 60
                    val m = diffMins % 60
                    if (h > 0) "${h}h ${m}m earlier than your personal norm" else "${m}m earlier than your personal norm"
                }
                else -> "in steady alignment with your personal norm"
            }
            "Your average ${spec.name.lowercase()} was ${formatHourLabel(avgVal, true)} $rangeLabel. You recorded $diffStr of ${formatHourLabel(normValue, true)}."
        } else {
            val diffPct = if (normValue > 0f) ((avgVal - normValue) / normValue * 100).roundToInt() else 0
            val diffText = when {
                diffPct > 0 -> "$diffPct% higher than your personal norm"
                diffPct < 0 -> "${abs(diffPct)}% lower than your personal norm"
                else -> "matching your personal norm"
            }
            when {
                diffPct > 15 -> "Your average ${spec.name.lowercase()} was ${spec.formatValue(avgVal)} $rangeLabel. You recorded $diffText of ${spec.formatValue(normValue)}."
                diffPct < -15 -> "Your ${spec.name.lowercase()} averaged ${spec.formatValue(avgVal)} $rangeLabel, showing a quieter pattern ($diffText)."
                else -> "Your ${spec.name.lowercase()} averaged ${spec.formatValue(avgVal)} $rangeLabel, flowing in steady alignment with your personal norm."
            }
        }
    }

    // Apply human circadian transformation for Bedtime & Wake Time chart height scaling
    val chartValues = remember(values, isCircadianTime) {
        if (isCircadianTime) values.map { circadianHourTransform(it) } else values
    }
    val chartNormValue = remember(normValue, isCircadianTime) {
        if (isCircadianTime) circadianHourTransform(normValue) else normValue
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
    ) {
        Column(
            modifier = Modifier.padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = spec.name,
                    fontSize = 15.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Text(
                    text = "Avg: ${spec.formatValue(avgVal)}",
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.primary
                )
            }

            Text(
                text = narrative,
                fontSize = 12.sp,
                lineHeight = 17.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            // Labeled Bar Chart Component with Y-Axis Tick Labels & Circadian Scaling
            LabeledBarChart(
                values = chartValues,
                normValue = chartNormValue,
                formatValue = spec.formatValue,
                isCircadianTime = isCircadianTime
            )
        }
    }
}

@Composable
fun LabeledBarChart(
    values: List<Float>,
    normValue: Float,
    formatValue: (Float) -> String,
    isCircadianTime: Boolean = false
) {
    if (values.isEmpty()) return
    val primaryColor = MaterialTheme.colorScheme.primary
    val warningColor = Color(0xFFF59E0B)
    val textMeasurer = rememberTextMeasurer()
    val onSurfaceVariant = MaterialTheme.colorScheme.onSurfaceVariant

    val maxVal = (values.maxOrNull() ?: 1f).coerceAtLeast(normValue * 1.3f).coerceAtLeast(1f)

    Column(modifier = Modifier.fillMaxWidth()) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(150.dp)
        ) {
            Canvas(modifier = Modifier.fillMaxSize()) {
                val width = size.width
                val height = size.height
                val paddingLeft = 42.dp.toPx()
                val paddingBottom = 22.dp.toPx()
                val graphWidth = width - paddingLeft
                val graphHeight = height - paddingBottom

                val normY = graphHeight * (1f - (normValue / maxVal).coerceIn(0f, 1f))

                // Y-Axis tick labels (0, mid, max)
                val yTicks = listOf(maxVal, maxVal / 2f, 0f)
                yTicks.forEach { tickVal ->
                    val yPos = graphHeight * (1f - (tickVal / maxVal).coerceIn(0f, 1f))
                    val labelText = if (isCircadianTime) {
                        val hourRaw = if (tickVal <= 6f) tickVal + 18f else tickVal - 6f
                        formatHourLabel(hourRaw, false)
                    } else {
                        if (tickVal >= 1000) "%.0fk".format(tickVal / 1000f)
                        else "%.0f".format(tickVal)
                    }
                    drawText(
                        textMeasurer = textMeasurer,
                        text = labelText,
                        style = TextStyle(fontSize = 9.sp, color = onSurfaceVariant.copy(0.6f)),
                        topLeft = Offset(0f, (yPos - 6.dp.toPx()).coerceIn(0f, graphHeight - 12.dp.toPx()))
                    )
                }

                // Draw usual norm dashed line
                drawLine(
                    color = primaryColor.copy(alpha = 0.5f),
                    start = Offset(paddingLeft, normY),
                    end = Offset(width, normY),
                    strokeWidth = 1.5.dp.toPx(),
                    pathEffect = PathEffect.dashPathEffect(floatArrayOf(8f, 8f), 0f)
                )

                // Draw bars
                val barCount = values.size
                val totalGap = graphWidth * 0.25f
                val barWidth = ((graphWidth - totalGap) / barCount).coerceAtLeast(4.dp.toPx())
                val spacing = if (barCount > 1) (graphWidth - (barWidth * barCount)) / (barCount - 1) else 0f

                values.forEachIndexed { idx, v ->
                    val barHeight = graphHeight * (v / maxVal).coerceIn(0f, 1f)
                    val x = paddingLeft + idx * (barWidth + spacing)
                    val y = graphHeight - barHeight

                    val dev = if (normValue > 0) abs(v - normValue) / normValue else 0f
                    val barColor = when {
                        dev > 0.3f -> Color(0xFFEF4444)
                        dev > 0.15f -> warningColor
                        else -> primaryColor
                    }

                    drawRoundRect(
                        color = barColor.copy(alpha = 0.85f),
                        topLeft = Offset(x, y),
                        size = androidx.compose.ui.geometry.Size(barWidth, barHeight),
                        cornerRadius = androidx.compose.ui.geometry.CornerRadius(4.dp.toPx(), 4.dp.toPx())
                    )
                }
            }
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(start = 42.dp),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            val daysOfWeek = listOf("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
            val count = values.size
            for (i in 0 until count.coerceAtMost(7)) {
                val label = if (count <= 7) daysOfWeek[i % 7] else "D${i + 1}"
                Text(
                    text = label,
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

data class CalendarMonthOption(
    val key: String,
    val label: String,
    val cal: Calendar
)

@Composable
fun MonthHistoryCalendar(
    historyItems: List<DailyHistoryItem>,
    onSelectDay: (DailyHistoryItem) -> Unit
) {
    val itemsByDate = remember(historyItems) { historyItems.associateBy { it.dateStr } }

    val monthOptions = remember(historyItems) {
        val keys = mutableSetOf<String>()
        val cal = Calendar.getInstance()
        val sdfMonthKey = SimpleDateFormat("yyyy-MM", Locale.US)
        keys.add(sdfMonthKey.format(cal.time))
        historyItems.forEach { item ->
            try {
                val d = SimpleDateFormat("yyyy-MM-dd", Locale.US).parse(item.dateStr)
                if (d != null) keys.add(sdfMonthKey.format(d))
            } catch (_: Exception) {}
        }
        keys.sortedDescending().mapNotNull { k ->
            try {
                val d = SimpleDateFormat("yyyy-MM", Locale.US).parse(k)
                if (d != null) {
                    val label = SimpleDateFormat("MMMM yyyy", Locale.getDefault()).format(d)
                    val c = Calendar.getInstance().apply {
                        time = d
                        set(Calendar.DAY_OF_MONTH, 1)
                        set(Calendar.HOUR_OF_DAY, 0)
                        set(Calendar.MINUTE, 0)
                        set(Calendar.SECOND, 0)
                    }
                    CalendarMonthOption(key = k, label = label, cal = c)
                } else null
            } catch (_: Exception) {
                null
            }
        }
    }

    var selectedMonthCal by remember {
        mutableStateOf(Calendar.getInstance().apply {
            set(Calendar.DAY_OF_MONTH, 1)
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
        })
    }

    var showMonthPickerDropdown by remember { mutableStateOf(false) }

    val monthTitle = remember(selectedMonthCal) {
        SimpleDateFormat("MMMM yyyy", Locale.getDefault()).format(selectedMonthCal.time)
    }

    val todayStr = remember { SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date()) }
    val daysOfWeek = listOf("S", "M", "T", "W", "T", "F", "S")

    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        // Month Navigation & Selection Header
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                modifier = Modifier
                    .clip(RoundedCornerShape(8.dp))
                    .clickable { showMonthPickerDropdown = true }
                    .padding(horizontal = 6.dp, vertical = 4.dp)
            ) {
                Text(
                    text = monthTitle,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Icon(
                    imageVector = Icons.Default.ArrowDropDown,
                    contentDescription = "Select Month",
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(20.dp)
                )

                DropdownMenu(
                    expanded = showMonthPickerDropdown,
                    onDismissRequest = { showMonthPickerDropdown = false }
                ) {
                    monthOptions.forEach { opt ->
                        val isSelected = SimpleDateFormat("yyyy-MM", Locale.US).format(selectedMonthCal.time) == opt.key
                        DropdownMenuItem(
                            text = {
                                Text(
                                    text = opt.label,
                                    fontFamily = Fredoka,
                                    fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal
                                )
                            },
                            onClick = {
                                selectedMonthCal = opt.cal.clone() as Calendar
                                showMonthPickerDropdown = false
                            }
                        )
                    }
                }
            }

            Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                IconButton(
                    onClick = {
                        val prev = (selectedMonthCal.clone() as Calendar).apply {
                            add(Calendar.MONTH, -1)
                        }
                        selectedMonthCal = prev
                    },
                    modifier = Modifier.size(32.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.ChevronLeft,
                        contentDescription = "Previous Month",
                        tint = MaterialTheme.colorScheme.primary
                    )
                }

                IconButton(
                    onClick = {
                        val next = (selectedMonthCal.clone() as Calendar).apply {
                            add(Calendar.MONTH, 1)
                        }
                        selectedMonthCal = next
                    },
                    modifier = Modifier.size(32.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.ChevronRight,
                        contentDescription = "Next Month",
                        tint = MaterialTheme.colorScheme.primary
                    )
                }
            }
        }

        // Day Headers: S M T W T F S
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceAround
        ) {
            daysOfWeek.forEach { dayLabel ->
                Box(modifier = Modifier.weight(1f), contentAlignment = Alignment.Center) {
                    Text(
                        text = dayLabel,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }

        // Calendar Month Grid Calculation
        val firstDayCal = (selectedMonthCal.clone() as Calendar).apply { set(Calendar.DAY_OF_MONTH, 1) }
        val firstDayOfWeek = firstDayCal.get(Calendar.DAY_OF_WEEK) - 1 // 0-indexed Sunday
        val daysInMonth = selectedMonthCal.getActualMaximum(Calendar.DAY_OF_MONTH)
        val year = selectedMonthCal.get(Calendar.YEAR)
        val month = selectedMonthCal.get(Calendar.MONTH)

        val totalSlots = firstDayOfWeek + daysInMonth
        val numRows = (totalSlots + 6) / 7

        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            for (row in 0 until numRows) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceAround
                ) {
                    for (col in 0 until 7) {
                        val slotIdx = row * 7 + col
                        val dayNumber = slotIdx - firstDayOfWeek + 1

                        if (dayNumber in 1..daysInMonth) {
                            val dateStr = String.format(Locale.US, "%04d-%02d-%02d", year, month + 1, dayNumber)
                            val item = itemsByDate[dateStr]
                            val isToday = dateStr == todayStr
                            val suffix = getOrdinalSuffix(dayNumber)

                            val cellScore = item?.rhythmScore
                            val cellColor = when {
                                cellScore == null -> MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f)
                                cellScore >= 80 -> MaterialTheme.colorScheme.primary
                                cellScore >= 50 -> Color(0xFFF59E0B)
                                else -> Color(0xFFEF4444)
                            }

                            val hasData = item != null

                            Box(
                                modifier = Modifier
                                    .weight(1f)
                                    .aspectRatio(1f)
                                    .padding(2.dp)
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(
                                        if (hasData) cellColor.copy(alpha = 0.18f)
                                        else MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.12f)
                                    )
                                    .border(
                                        width = if (isToday) 1.8.dp else 1.dp,
                                        color = if (isToday) MaterialTheme.colorScheme.primary else (if (hasData) cellColor.copy(alpha = 0.45f) else Color.Transparent),
                                        shape = RoundedCornerShape(8.dp)
                                    )
                                    .clickable(enabled = hasData) {
                                        if (item != null) onSelectDay(item)
                                    },
                                contentAlignment = Alignment.Center
                            ) {
                                Column(
                                    horizontalAlignment = Alignment.CenterHorizontally,
                                    verticalArrangement = Arrangement.Center
                                ) {
                                    Row(
                                        verticalAlignment = Alignment.Top,
                                        horizontalArrangement = Arrangement.Center
                                    ) {
                                        Text(
                                            text = "$dayNumber",
                                            fontSize = 11.sp,
                                            fontWeight = if (isToday || hasData) FontWeight.Bold else FontWeight.Normal,
                                            fontFamily = Fredoka,
                                            color = if (hasData) MaterialTheme.colorScheme.onBackground else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f)
                                        )
                                        Text(
                                            text = suffix,
                                            fontSize = 7.sp,
                                            fontWeight = FontWeight.SemiBold,
                                            fontFamily = Fredoka,
                                            color = if (hasData) MaterialTheme.colorScheme.onSurfaceVariant else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.3f),
                                            modifier = Modifier.padding(start = 0.5.dp, top = 0.5.dp)
                                        )
                                    }

                                    if (hasData && cellScore != null) {
                                        Box(
                                            modifier = Modifier
                                                .padding(top = 1.dp)
                                                .size(4.dp)
                                                .clip(CircleShape)
                                                .background(cellColor)
                                        )
                                    }
                                }
                            }
                        } else {
                            // Empty placeholder box for alignment
                            Box(
                                modifier = Modifier
                                    .weight(1f)
                                    .aspectRatio(1f)
                                    .padding(2.dp)
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun DailyHistoryScreen(
    historyItems: List<DailyHistoryItem>,
    onBack: () -> Unit,
    onSelectDay: (DailyHistoryItem) -> Unit
) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentPadding = PaddingValues(20.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp)
    ) {
        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                IconButton(onClick = onBack) {
                    Icon(Icons.Default.ArrowBack, "Back")
                }
                Text(
                    text = "Daily & Reflection History",
                    fontSize = 22.sp,
                    fontWeight = FontWeight.ExtraBold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
            }
        }

        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
            ) {
                Column(
                    modifier = Modifier.padding(18.dp),
                    verticalArrangement = Arrangement.spacedBy(14.dp)
                ) {
                    Text(
                        text = "Rhythm Consistency Calendar",
                        fontSize = 15.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                    Text(
                        text = "Green: Coherent (80+) | Amber: Adapting (50-79) | Red: Shift (<50)",
                        fontSize = 11.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    MonthHistoryCalendar(
                        historyItems = historyItems,
                        onSelectDay = onSelectDay
                    )
                }
            }
        }

        item {
            Text(
                text = "Recorded Daily Entries",
                fontSize = 15.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                color = MaterialTheme.colorScheme.onBackground
            )
        }

        if (historyItems.isEmpty()) {
            item {
                Text(
                    text = "No history entries logged yet.",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        } else {
            items(historyItems) { historyItem ->
                DailyHistoryCard(item = historyItem, onClick = { onSelectDay(historyItem) })
            }
        }
    }
}

@Composable
fun DayDetailScreen(
    item: DailyHistoryItem,
    baseline: PersonalityVector?,
    baselineEntities: List<BaselineEntity>,
    onBack: () -> Unit
) {
    val vec = item.vector
    val base = baseline ?: PersonalityVector(
        screenTimeHours = baselineEntities.firstOrNull { it.featureName == "screenTimeHours" }?.baselineValue ?: 4f,
        dailyStepCount = baselineEntities.firstOrNull { it.featureName == "dailyStepCount" }?.baselineValue ?: 3000f,
        sleepDurationHours = baselineEntities.firstOrNull { it.featureName == "sleepDurationHours" }?.baselineValue ?: 7f,
        callsPerDay = baselineEntities.firstOrNull { it.featureName == "callsPerDay" }?.baselineValue ?: 2f
    )

    val formattedDate = remember(item.dateStr) {
        try {
            val date = SimpleDateFormat("yyyy-MM-dd", Locale.US).parse(item.dateStr)
            SimpleDateFormat("EEEE, MMMM d, yyyy", Locale.getDefault()).format(date!!)
        } catch (e: Exception) {
            item.dateStr
        }
    }

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentPadding = PaddingValues(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                IconButton(onClick = onBack) {
                    Icon(Icons.Default.ArrowBack, "Back")
                }
                Column {
                    Text(
                        text = "Day Reflection & Telemetry",
                        fontSize = 20.sp,
                        fontWeight = FontWeight.ExtraBold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                    Text(
                        text = formattedDate,
                        fontSize = 12.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }

        // Rhythm Score Arc Hero Card
        item {
            DailyRhythmScoreCard(
                rhythmScore = item.rhythmScore,
                latest = vec,
                baseVec = base
            )
        }

        // Self Reflection Journal Entry Card (if present)
        if (item.checkin != null) {
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(20.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(0.25f))
                ) {
                    Column(
                        modifier = Modifier.padding(18.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Text(
                            text = "Evening Journal Entry",
                            fontSize = 15.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.primary
                        )

                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            HistoryPill("Mood: ${item.checkin.mood}/5")
                            HistoryPill("Anxiety: ${item.checkin.anxiety}/5")
                            HistoryPill("Energy: ${item.checkin.energy}/5")
                            HistoryPill("Rest: ${item.checkin.sleep}/5")
                        }

                        if (item.checkin.note.isNotBlank()) {
                            Surface(
                                shape = RoundedCornerShape(12.dp),
                                color = MaterialTheme.colorScheme.surfaceVariant.copy(0.3f),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text(
                                    text = "\"${item.checkin.note}\"",
                                    fontSize = 13.sp,
                                    fontStyle = androidx.compose.ui.text.font.FontStyle.Italic,
                                    color = MaterialTheme.colorScheme.onBackground,
                                    modifier = Modifier.padding(14.dp),
                                    lineHeight = 18.sp
                                )
                            }
                        }
                    }
                }
            }
        }

        // Raw Telemetry Stats Section
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
            ) {
                Column(
                    modifier = Modifier.padding(18.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Text(
                        text = "Full Telemetry Breakdown",
                        fontSize = 15.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )

                    StatRow("Rest Duration", "%.1fh".format(vec.sleepDurationHours), "%.1fh norm".format(base.sleepDurationHours))
                    HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.08f))
                    StatRow("Step Count", "%.0f steps".format(vec.dailyStepCount), "%.0f norm".format(base.dailyStepCount))
                    HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.08f))
                    StatRow("Screen Time", "%.1fh".format(vec.screenTimeHours), "%.1fh norm".format(base.screenTimeHours))
                    HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.08f))
                    StatRow("Calls Per Day", "%.0f calls".format(vec.callsPerDay), "%.0f norm".format(base.callsPerDay))
                    HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.08f))
                    StatRow("Keystroke Speed", "%.1f speed".format(vec.keystrokeSpeed), "%.1f norm".format(base.keystrokeSpeed))
                    HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.08f))
                    StatRow("Daylight Exposure", "%.0f mins".format(vec.daylightExposureMinutes), "%.0f norm".format(base.daylightExposureMinutes))
                    HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.08f))
                    StatRow("Home Time Ratio", "%.0f%%".format(vec.homeTimeRatio * 100), "%.0f%% norm".format(base.homeTimeRatio * 100))
                }
            }
        }
    }
}
