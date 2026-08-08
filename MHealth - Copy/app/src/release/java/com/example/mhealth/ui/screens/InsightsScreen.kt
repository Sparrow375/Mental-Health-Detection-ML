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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.mhealth.logic.DataRepository
import com.example.mhealth.logic.db.BaselineEntity
import com.example.mhealth.logic.db.MHealthDatabase
import com.example.mhealth.models.PersonalityVector
import com.example.mhealth.ui.components.AlertWarning
import com.example.mhealth.ui.components.BehavioralFingerprintRadar
import com.example.mhealth.ui.components.MiniSparkline
import com.example.mhealth.ui.components.RhythmTrendsChart
import com.example.mhealth.ui.components.StaggeredFadeIn
import com.example.mhealth.ui.components.Fredoka
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.abs
import kotlin.math.roundToInt

@Composable
fun InsightsScreen() {
    val weeklyFeatures by DataRepository.weeklyFeatureHistory.collectAsState()
    val baseline by DataRepository.baseline.collectAsState()
    val isDnaReady by DataRepository.isDnaBaselineReady.collectAsState()
    val context = LocalContext.current
    
    var activeSectorName by remember { mutableStateOf<String?>(null) }
    var activeSectorIcon by remember { mutableStateOf<ImageVector>(Icons.Default.Info) }

    BackHandler(enabled = activeSectorName != null) {
        activeSectorName = null
    }

    val db = remember { MHealthDatabase.getInstance(context.applicationContext) }
    val baselineEntities by produceState<List<BaselineEntity>>(emptyList(), db) {
        val userId = DataRepository.userProfile.value?.email ?: "patient@lumen.health"
        value = db.baselineDao().getBaseline(userId)
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

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentPadding = PaddingValues(24.dp),
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
                    modifier = Modifier.padding(bottom = 6.dp)
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

        // Section 1: Curated Insight Sectors Header
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
                sleepDiff < -1.5f -> "Your sleep window was shorter than your typical baseline."
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
                stepRatio < 0.6f -> "Your physical movement is lower than your usual baseline today."
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

        // 7. Routine & Places (NEW - replaces Charging)
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

        // Section: Long-Term Weekly Patterns Header
        item {
            Spacer(Modifier.height(8.dp))
            Text(
                text = "Weekly Trends & Analysis",
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                color = MaterialTheme.colorScheme.onBackground
            )
        }

        // Weekly Story Narrative Card
        item {
            WeeklyStoryCard(features = weeklyFeatures, baseline = baseVec)
        }

        // Multi-Dimensional Rhythm Trends Chart
        item {
            RhythmTrendsChart(features = weeklyFeatures, baseline = baseVec)
        }

        // 6-Axis Behavioral Fingerprint Radar
        item {
            BehavioralFingerprintRadar(currentVector = latest, baselineVector = baseVec)
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
            "Your lifestyle rhythms have been flowing in beautiful alignment this week. Rest, physical movement, and digital engagement stayed consistent with your baseline."
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
                // Mini visual status indicator dots
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
    val recent = features.takeLast(7)
    val series = remember(recent, sectorName) {
        recent.map { day ->
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

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentPadding = PaddingValues(24.dp),
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

        // Narrative Summary Card
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
                        text = "Weekly Narrative",
                        fontSize = 15.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )

                    val summaryStr = when {
                        pctDiff > 10 -> "Your average for $sectorName was about $pctDiff% higher than your baseline this week."
                        pctDiff < -10 -> "Your average for $sectorName was about ${abs(pctDiff)}% below your baseline this week."
                        else -> "Your $sectorName stayed in close alignment with your personal baseline throughout the week."
                    }
                    Text(
                        text = summaryStr,
                        fontSize = 13.sp,
                        lineHeight = 18.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    Spacer(Modifier.height(4.dp))

                    // Relative Baseline Comparison Bar
                    Text(
                        text = "Weekly Average vs Personal Baseline",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.primary
                    )
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        val barFraction = (1.0f + (pctDiff / 100f)).coerceIn(0.2f, 1.8f) / 2.0f
                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .height(8.dp)
                                .background(MaterialTheme.colorScheme.surfaceVariant, CircleShape)
                        ) {
                            Box(
                                modifier = Modifier
                                    .fillMaxHeight()
                                    .fillMaxWidth(barFraction)
                                    .background(MaterialTheme.colorScheme.primary, CircleShape)
                            )
                        }
                        Text(
                            text = if (pctDiff >= 0) "+$pctDiff%" else "$pctDiff%",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }

                    Spacer(Modifier.height(4.dp))

                    Text(
                        text = "7-Day Trend Sparkline",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.primary
                    )
                    MiniSparkline(values = series)
                }
            }
        }

        // Outlier Callout Section
        item {
            Text(
                text = "Notable Days & Tips",
                fontSize = 15.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                color = MaterialTheme.colorScheme.onBackground
            )
        }

        val outlier = recent.maxByOrNull { abs(series.getOrElse(recent.indexOf(it)) { 0f } - baseVal) }
        if (outlier != null) {
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
                                text = "Highest Shift Day",
                                fontSize = 13.sp,
                                fontWeight = FontWeight.Bold,
                                fontFamily = Fredoka,
                                color = MaterialTheme.colorScheme.onBackground
                            )
                            Text(
                                text = "Showed the largest relative deviation compared to your standard norm.",
                                fontSize = 11.sp,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }
            }
        }

        item {
            val tipStr = when (sectorName) {
                "Sleep & Rest" -> "Keeping consistent sleep and wake hours builds strong circadian rhythms."
                "Physical Activity" -> "A quick 10-minute afternoon walk helps release tension and steady mood."
                "Social Connection" -> "Reaching out to a trusted contact or friend brings emotional balance."
                "Screen Time" -> "Taking brief 15-minute digital sunset breaks before bed aids restful sleep."
                "Interaction Pace" -> "A steady typing and scroll cadence reflects balanced energy levels."
                "Daylight Exposure" -> "Getting morning natural light helps anchor your internal body clock."
                "Routine & Places" -> "Visiting new spatial environments naturally expands cognitive vitality."
                else -> "Maintaining daily routine consistency supports overall lifestyle balance."
            }
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.08f)),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.2f))
            ) {
                Row(
                    modifier = Modifier.padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Icon(Icons.Default.Lightbulb, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                    Column {
                        Text(
                            text = "Gentle Insight",
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.primary
                        )
                        Text(
                            text = tipStr,
                            fontSize = 11.sp,
                            color = MaterialTheme.colorScheme.onBackground,
                            lineHeight = 16.sp
                        )
                    }
                }
            }
        }
    }
}
