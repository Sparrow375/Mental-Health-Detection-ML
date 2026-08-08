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
import com.example.mhealth.logic.db.AnalysisResultEntity
import com.example.mhealth.logic.db.BaselineEntity
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
    var selectedDetailDay by remember { mutableStateOf<DailyHistoryItem?>(null) }

    BackHandler(enabled = activeSectorName != null || showWeeklyTrendsModal || selectedDetailDay != null) {
        when {
            selectedDetailDay != null -> selectedDetailDay = null
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

    val prefs = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }
    val checkinHistoryStr = remember(prefs) { prefs.getString("daily_checkin_history", "[]") ?: "[]" }

    val historyItems = remember(weeklyFeatures, analysisReports, checkinHistoryStr) {
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

        weeklyFeatures.mapIndexed { idx, vec ->
            val dateStr = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()).format(
                Calendar.getInstance().apply { add(Calendar.DAY_OF_YEAR, -(weeklyFeatures.size - 1 - idx)) }.time
            )
            val report = analysisReports.firstOrNull { it.date == dateStr }
            val checkin = checkinMap[dateStr]
            val scoreVal = if (report != null) ((1f - report.effectiveScore.coerceIn(0f, 1f)) * 100).roundToInt() else 85
            
            DailyHistoryItem(
                dateStr = dateStr,
                vector = vec,
                analysisResult = report,
                checkin = checkin,
                rhythmScore = scoreVal
            )
        }.reversed()
    }

    if (activeSectorName != null) {
        SectorDetailScreen(
            sectorName = activeSectorName!!,
            sectorIcon = activeSectorIcon,
            features = weeklyFeatures,
            baselineEntities = baselineEntities,
            onBack = { activeSectorName = null }
        )
        return
    }

    if (showWeeklyTrendsModal) {
        WeeklyTrendsSubScreen(
            features = weeklyFeatures,
            baseline = baseline,
            baselineEntities = baselineEntities,
            onBack = { showWeeklyTrendsModal = false }
        )
        return
    }

    if (selectedDetailDay != null) {
        DayStatsDetailDialog(
            item = selectedDetailDay!!,
            baseline = baseline,
            baselineEntities = baselineEntities,
            onDismiss = { selectedDetailDay = null }
        )
    }

    val latest = weeklyFeatures.lastOrNull() ?: PersonalityVector()
    val baseVec = baseline ?: PersonalityVector(
        screenTimeHours = baselineEntities.firstOrNull { it.featureName == "screenTimeHours" }?.baselineValue ?: 4f,
        dailyStepCount = baselineEntities.firstOrNull { it.featureName == "dailyStepCount" }?.baselineValue ?: 3000f,
        sleepDurationHours = baselineEntities.firstOrNull { it.featureName == "sleepDurationHours" }?.baselineValue ?: 7f,
        callsPerDay = baselineEntities.firstOrNull { it.featureName == "callsPerDay" }?.baselineValue ?: 2f,
        locationEntropy = baselineEntities.firstOrNull { it.featureName == "locationEntropy" }?.baselineValue ?: 0.5f,
        daylightExposureMinutes = baselineEntities.firstOrNull { it.featureName == "daylightExposureMinutes" }?.baselineValue ?: 30f,
        keystrokeSpeed = baselineEntities.firstOrNull { it.featureName == "keystrokeSpeed" }?.baselineValue ?: 4f
    )

    val currentEffScore = activeResult?.effectiveScore ?: 0.15f
    val rhythmScore = ((1.0f - currentEffScore.coerceIn(0f, 1f)) * 100).roundToInt()

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

        // Section 3: Comprehensive Daily & Reflection History
        item {
            Column(modifier = Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text(
                    text = "Daily & Reflection History",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Text(
                    text = "Tap any day card to view that day's complete statistics and reflections",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }

        if (historyItems.isEmpty()) {
            item {
                Text(
                    text = "No history recorded yet. Monitored days will appear here.",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        } else {
            items(historyItems) { historyItem ->
                DailyHistoryCard(item = historyItem, onClick = { selectedDetailDay = historyItem })
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

@Composable
fun SectorDetailScreen(
    sectorName: String,
    sectorIcon: ImageVector,
    features: List<PersonalityVector>,
    baselineEntities: List<BaselineEntity>,
    onBack: () -> Unit
) {
    var timeRangeDays by remember { mutableIntStateOf(7) }
    val displayFeatures = remember(features, timeRangeDays) { features.takeLast(timeRangeDays) }

    val series = remember(displayFeatures, sectorName) {
        displayFeatures.map { day ->
            when (sectorName) {
                "Sleep & Rest" -> day.sleepDurationHours
                "Physical Activity" -> day.dailyStepCount
                "Social Connection" -> day.callsPerDay
                "Screen Time" -> day.screenTimeHours
                "Interaction Pace" -> day.keystrokeSpeed
                "Daylight Exposure" -> day.daylightExposureMinutes
                "Routine & Places" -> day.homeTimeRatio
                else -> 0f
            }
        }
    }

    val avgVal = if (series.isNotEmpty()) series.average().toFloat() else 0f
    val baseVal = remember(baselineEntities, sectorName) {
        val key = when (sectorName) {
            "Sleep & Rest" -> "sleepDurationHours"
            "Physical Activity" -> "dailyStepCount"
            "Social Connection" -> "callsPerDay"
            "Screen Time" -> "screenTimeHours"
            "Interaction Pace" -> "keystrokeSpeed"
            "Daylight Exposure" -> "daylightExposureMinutes"
            "Routine & Places" -> "homeTimeRatio"
            else -> ""
        }
        baselineEntities.firstOrNull { it.featureName == key }?.baselineValue ?: avgVal.coerceAtLeast(1f)
    }

    val pctDiff = if (baseVal > 0) ((avgVal - baseVal) / baseVal * 100).roundToInt() else 0

    val unitLabel = when (sectorName) {
        "Sleep & Rest" -> "hours"
        "Physical Activity" -> "steps"
        "Social Connection" -> "calls"
        "Screen Time" -> "hours"
        "Interaction Pace" -> "speed"
        "Daylight Exposure" -> "mins"
        "Routine & Places" -> "home ratio"
        else -> ""
    }

    val avgFormatted = when (sectorName) {
        "Sleep & Rest", "Screen Time" -> "%.1fh".format(avgVal)
        "Physical Activity" -> "%.0f steps".format(avgVal)
        "Social Connection" -> "%.0f calls".format(avgVal)
        "Interaction Pace" -> "%.1f pace".format(avgVal)
        "Daylight Exposure" -> "%.0f mins".format(avgVal)
        "Routine & Places" -> "%.0f%%".format(avgVal * 100)
        else -> "%.1f".format(avgVal)
    }

    val baseFormatted = when (sectorName) {
        "Sleep & Rest", "Screen Time" -> "%.1fh".format(baseVal)
        "Physical Activity" -> "%.0f steps".format(baseVal)
        "Social Connection" -> "%.0f calls".format(baseVal)
        "Interaction Pace" -> "%.1f pace".format(baseVal)
        "Daylight Exposure" -> "%.0f mins".format(baseVal)
        "Routine & Places" -> "%.0f%%".format(baseVal * 100)
        else -> "%.1f".format(baseVal)
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

        // Time Range Tabs
        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End
            ) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf(7, 14, 30).forEach { days ->
                        val isSel = timeRangeDays == days
                        OutlinedButton(
                            onClick = { timeRangeDays = days },
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

        // Narrative Summary Card (No Progress Bar!)
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
                        text = "$timeRangeDays-Day Summary",
                        fontSize = 15.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )

                    val diffText = if (pctDiff >= 0) "$pctDiff% above" else "${abs(pctDiff)}% below"
                    Text(
                        text = "Your average: $avgFormatted ($diffText your usual norm of $baseFormatted)",
                        fontSize = 13.sp,
                        lineHeight = 18.sp,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary,
                        fontFamily = Fredoka
                    )

                    Spacer(Modifier.height(4.dp))

                    Text(
                        text = "Daily Trend Values",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )

                    // Line chart with raw values
                    RawValuesLineChart(series = series, baseVal = baseVal)
                }
            }
        }

        // Outlier Callout Section
        val outlier = displayFeatures.maxByOrNull { abs(series.getOrElse(displayFeatures.indexOf(it)) { 0f } - baseVal) }
        if (outlier != null) {
            val outlierIdx = displayFeatures.indexOf(outlier)
            val cal = Calendar.getInstance().apply { add(Calendar.DAY_OF_YEAR, -(displayFeatures.size - 1 - outlierIdx)) }
            val dayName = SimpleDateFormat("EEEE", Locale.getDefault()).format(cal.time)
            val outlierVal = series.getOrElse(outlierIdx) { 0f }
            val formatVal = when (sectorName) {
                "Sleep & Rest" -> "%.1fh rest".format(outlierVal)
                "Physical Activity" -> "%.0f steps".format(outlierVal)
                "Social Connection" -> "%.0f calls".format(outlierVal)
                "Screen Time" -> "%.1fh screen time".format(outlierVal)
                "Interaction Pace" -> "%.1f pace".format(outlierVal)
                "Daylight Exposure" -> "%.0f mins daylight".format(outlierVal)
                "Routine & Places" -> "%.0f%% home time".format(outlierVal * 100)
                else -> "%.1f".format(outlierVal)
            }

            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(0.2f)),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
                ) {
                    Row(
                        modifier = Modifier.padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Icon(Icons.Default.Tune, contentDescription = null, tint = AlertWarning)
                        Column {
                            Text(
                                text = "Highest Shift: $dayName",
                                fontSize = 13.sp,
                                fontWeight = FontWeight.Bold,
                                fontFamily = Fredoka,
                                color = MaterialTheme.colorScheme.onBackground
                            )
                            Text(
                                text = "$dayName recorded $formatVal, showing the largest relative shift compared to your usual norm.",
                                fontSize = 11.sp,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }
            }
        }

        // Daily Raw Values List Section
        item {
            Text(
                text = "Daily Entries",
                fontSize = 15.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                color = MaterialTheme.colorScheme.onBackground
            )
        }

        items(series.size) { idx ->
            val valItem = series[idx]
            val cal = Calendar.getInstance().apply { add(Calendar.DAY_OF_YEAR, -(series.size - 1 - idx)) }
            val dateLabel = SimpleDateFormat("EEEE, MMM d", Locale.getDefault()).format(cal.time)
            val valStr = when (sectorName) {
                "Sleep & Rest", "Screen Time" -> "%.1fh".format(valItem)
                "Physical Activity" -> "%.0f steps".format(valItem)
                "Social Connection" -> "%.0f calls".format(valItem)
                "Interaction Pace" -> "%.1f speed".format(valItem)
                "Daylight Exposure" -> "%.0f mins".format(valItem)
                "Routine & Places" -> "%.0f%%".format(valItem * 100)
                else -> "%.1f".format(valItem)
            }

            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.08f))
            ) {
                Row(
                    modifier = Modifier.padding(14.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(dateLabel, fontSize = 13.sp, fontWeight = FontWeight.SemiBold, fontFamily = Fredoka)
                    Text(valStr, fontSize = 13.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary, fontFamily = Fredoka)
                }
            }
        }
    }
}

@Composable
fun RawValuesLineChart(series: List<Float>, baseVal: Float) {
    if (series.isEmpty()) return
    val textMeasurer = rememberTextMeasurer()
    val primaryColor = MaterialTheme.colorScheme.primary
    val onSurfaceVariant = MaterialTheme.colorScheme.onSurfaceVariant

    val minV = (series.minOrNull() ?: 0f).coerceAtMost(baseVal * 0.5f)
    val maxV = (series.maxOrNull() ?: 1f).coerceAtLeast(baseVal * 1.5f)

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(160.dp)
    ) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val width = size.width
            val height = size.height
            val paddingLeft = 36.dp.toPx()
            val paddingBottom = 20.dp.toPx()
            val graphWidth = width - paddingLeft
            val graphHeight = height - paddingBottom

            val range = (maxV - minV).coerceAtLeast(1f)

            fun getY(v: Float): Float {
                return graphHeight * (1.0f - (v - minV) / range)
            }

            // Draw baseline dashed horizontal reference line
            val baseY = getY(baseVal)
            drawLine(
                color = primaryColor.copy(alpha = 0.5f),
                start = Offset(paddingLeft, baseY),
                end = Offset(width, baseY),
                strokeWidth = 1.5.dp.toPx(),
                pathEffect = PathEffect.dashPathEffect(floatArrayOf(10f, 10f), 0f)
            )

            // Series line
            val count = series.size
            val stepX = if (count > 1) graphWidth / (count - 1) else graphWidth
            val path = Path()

            series.forEachIndexed { i, v ->
                val x = paddingLeft + i * stepX
                val y = getY(v)
                if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
                drawCircle(color = primaryColor, radius = 3.5.dp.toPx(), center = Offset(x, y))
            }

            drawPath(path = path, color = primaryColor, style = Stroke(width = 2.5.dp.toPx(), cap = StrokeCap.Round))
        }
    }
}
