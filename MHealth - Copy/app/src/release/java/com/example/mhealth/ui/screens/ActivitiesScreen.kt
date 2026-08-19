package com.example.mhealth.ui.screens

import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.widget.Toast
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.util.Log
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.example.mhealth.logic.DataRepository
import com.example.mhealth.logic.db.BadgeEntity
import com.example.mhealth.logic.db.BaselineEntity
import com.example.mhealth.logic.db.MHealthDatabase
import com.example.mhealth.models.PersonalityVector
import com.example.mhealth.ui.components.AlertWarning
import com.example.mhealth.ui.components.AlertRose
import com.example.mhealth.ui.components.StaggeredFadeIn
import com.example.mhealth.ui.components.ToggleRow
import com.example.mhealth.ui.components.Fredoka
import com.example.mhealth.ui.components.rememberNavBarPadding
import kotlinx.coroutines.delay
import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale
import java.util.UUID
import kotlin.math.roundToInt

import com.example.mhealth.WindDownOverlay
import com.example.mhealth.DigitalDetoxTimerOverlay

@Composable
fun ActivitiesScreen() {
    val context = LocalContext.current
    val prefs = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }
    val badges by DataRepository.badges.collectAsState()

    var showBadgeGallery by remember { mutableStateOf(false) }
    var showManageHabits by remember { mutableStateOf(false) }
    var showWindDownOverlay by remember { mutableStateOf(false) }
    var showDetoxOverlay by remember { mutableStateOf(false) }
    var showDetoxSetupDialog by remember { mutableStateOf(false) }
    var detoxMinutes by remember { mutableIntStateOf(30) }
    var showQuestsSubScreen by remember { mutableStateOf(false) }

    androidx.activity.compose.BackHandler(enabled = showQuestsSubScreen) {
        showQuestsSubScreen = false
    }

    if (showQuestsSubScreen) {
        QuestsScreen(
            prefs = prefs,
            badges = badges,
            onBack = { showQuestsSubScreen = false }
        )
        return
    }

    if (showBadgeGallery) {
        BadgeGalleryDialog(badges = badges, onDismiss = { showBadgeGallery = false })
    }
    if (showManageHabits) {
        ManageHabitsDialog(prefs = prefs, onDismiss = { showManageHabits = false })
    }
    if (showWindDownOverlay) {
        WindDownOverlay(sleepTarget = 8.0f, onDismiss = { showWindDownOverlay = false })
    }
    if (showDetoxSetupDialog) {
        DetoxDurationSetupDialog(
            initialMinutes = detoxMinutes,
            onConfirm = { mins ->
                detoxMinutes = mins
                showDetoxSetupDialog = false
                showDetoxOverlay = true
            },
            onDismiss = { showDetoxSetupDialog = false }
        )
    }
    if (showDetoxOverlay) {
        DigitalDetoxTimerOverlay(durationMinutes = detoxMinutes, onDismiss = { showDetoxOverlay = false })
    }

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentPadding = PaddingValues(24.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp)
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
                    text = "Your Activities",
                    fontSize = 26.sp,
                    fontWeight = FontWeight.ExtraBold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Text(
                    text = "Habit quests, wind-down companion, and focus tools",
                    fontSize = 13.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 4.dp)
                )
            }
        }

        // Habit Quests Section
        item {
            HabitQuestsCard(
                prefs = prefs,
                badges = badges,
                onManageClick = { showManageHabits = true },
                onGalleryClick = { showBadgeGallery = true },
                onOpenQuests = { showQuestsSubScreen = true }
            )
        }

        // Mindful Breathing Reset (1-min / 3-min lotus breathing exercise)
        item {
            MindfulBreathingCard()
        }

        // Wind-Down Companion
        item {
            WindDownCard(onStartClick = { showWindDownOverlay = true })
        }

        // Digital Detox
        item {
            DigitalDetoxCard(onStartClick = { showDetoxSetupDialog = true })
        }
    }
}

data class CustomQuest(
    val id: String,
    val title: String,
    val description: String,
    val category: String, // Movement, Screen, Sleep, Mindfulness, Hydration, Reading, General
    val targetQuantity: Int,
    val unit: String,
    val currentProgress: Int = 0,
    val streak: Int = 0,
    val lastCompletedDate: String = "",
    val isAutoTracked: Boolean = false
)

fun loadCustomQuests(prefs: SharedPreferences): List<CustomQuest> {
    val jsonStr = prefs.getString("custom_habits_json_v2", null)
    if (jsonStr != null) {
        try {
            val arr = JSONArray(jsonStr)
            val list = mutableListOf<CustomQuest>()
            for (i in 0 until arr.length()) {
                val obj = arr.getJSONObject(i)
                list.add(
                    CustomQuest(
                        id = obj.optString("id", UUID.randomUUID().toString()),
                        title = obj.optString("title", "Custom Quest"),
                        description = obj.optString("description", ""),
                        category = obj.optString("category", "General"),
                        targetQuantity = obj.optInt("targetQuantity", 1),
                        unit = obj.optString("unit", "times"),
                        currentProgress = obj.optInt("currentProgress", 0),
                        streak = obj.optInt("streak", 0),
                        lastCompletedDate = obj.optString("lastCompletedDate", ""),
                        isAutoTracked = obj.optBoolean("isAutoTracked", false)
                    )
                )
            }
            return list
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
    // Fallback migration from legacy custom_habits_set
    val legacySet = prefs.getStringSet("custom_habits_set", emptySet()) ?: emptySet()
    val migrated = legacySet.mapIndexed { idx, item ->
        val parts = item.split("|")
        CustomQuest(
            id = "legacy_$idx",
            title = parts.getOrNull(0) ?: "Custom Habit",
            description = parts.getOrNull(1) ?: "",
            category = "General",
            targetQuantity = 1,
            unit = "times",
            streak = parts.getOrNull(2)?.toIntOrNull() ?: 0
        )
    }
    if (migrated.isNotEmpty()) {
        saveCustomQuests(prefs, migrated)
    }
    return migrated
}

fun saveCustomQuests(prefs: SharedPreferences, list: List<CustomQuest>) {
    val arr = JSONArray()
    for (q in list) {
        val obj = JSONObject().apply {
            put("id", q.id)
            put("title", q.title)
            put("description", q.description)
            put("category", q.category)
            put("targetQuantity", q.targetQuantity)
            put("unit", q.unit)
            put("currentProgress", q.currentProgress)
            put("streak", q.streak)
            put("lastCompletedDate", q.lastCompletedDate)
            put("isAutoTracked", q.isAutoTracked)
        }
        arr.put(obj)
    }
    prefs.edit().putString("custom_habits_json_v2", arr.toString()).apply()
}

@Composable
fun HabitQuestsCard(
    prefs: SharedPreferences,
    badges: List<BadgeEntity>,
    onManageClick: () -> Unit,
    onGalleryClick: () -> Unit,
    onOpenQuests: () -> Unit = {}
) {
    val context = LocalContext.current
    val unlockedCount = remember(badges) { badges.count { it.isUnlocked } }

    val sunsetEnabled = remember { prefs.getBoolean("habit_digital_sunset_enabled", false) }
    val circadianEnabled = remember { prefs.getBoolean("habit_circadian_anchor_enabled", false) }
    val movementEnabled = remember { prefs.getBoolean("habit_movement_boost_enabled", false) }
    val screenLimitEnabled = remember { prefs.getBoolean("habit_screen_limit_enabled", false) }
    val focusEnabled = remember { prefs.getBoolean("habit_focus_mode_enabled", false) }
    val mindfulEnabled = remember { prefs.getBoolean("habit_mindful_pause_enabled", false) }
    val daylightEnabled = remember { prefs.getBoolean("habit_daylight_boost_enabled", false) }

    val sunsetStreak = remember { prefs.getInt("habit_digital_sunset_streak", prefs.getInt("streak_digital_sunset", 0)) }
    val circadianStreak = remember { prefs.getInt("habit_circadian_anchor_streak", prefs.getInt("streak_circadian_anchor", 0)) }
    val movementStreak = remember { prefs.getInt("habit_movement_boost_streak", prefs.getInt("streak_movement_boost", 0)) }
    val screenLimitStreak = remember { prefs.getInt("habit_screen_limit_streak", 0) }
    val focusStreak = remember { prefs.getInt("habit_focus_mode_streak", prefs.getInt("streak_focus_mode", 0)) }
    val mindfulStreak = remember { prefs.getInt("habit_mindful_pause_streak", 0) }
    val daylightStreak = remember { prefs.getInt("habit_daylight_boost_streak", 0) }

    val movementTarget = remember { prefs.getInt("habit_movement_boost_target", 6000) }
    val screenLimitTarget = remember { prefs.getFloat("habit_screen_limit_target", 4.0f) }
    val sunsetTarget = remember { prefs.getInt("habit_digital_sunset_target", 30) }
    val mindfulTarget = remember { prefs.getInt("habit_mindful_pause_target", 1) }
    val daylightTarget = remember { prefs.getInt("habit_daylight_boost_target", 30) }

    var customQuests by remember { mutableStateOf(loadCustomQuests(prefs)) }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onOpenQuests() },
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
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.EmojiEvents,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(22.dp)
                    )
                    Text(
                        text = "Habit Quests",
                        fontSize = 17.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                }

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    TextButton(onClick = onGalleryClick) {
                        Text(
                            text = "Badges ($unlockedCount/${badges.size})",
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                    IconButton(onClick = onManageClick) {
                        Icon(Icons.Default.Tune, null, tint = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            }

            val activeCount = listOf(
                sunsetEnabled, circadianEnabled, movementEnabled, screenLimitEnabled,
                focusEnabled, mindfulEnabled, daylightEnabled
            ).count { it } + customQuests.size

            if (activeCount == 0) {
                Text(
                    text = "No active habit quests configured. Tap the settings icon to configure personalized goals for steps, screen time, bedtime, and custom habits.",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    lineHeight = 17.sp
                )
            } else {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    if (movementEnabled) {
                        QuestRow(
                            title = "Movement Boost",
                            subtitle = "Goal: ${String.format(java.util.Locale.US, "%,d", movementTarget)} steps/day",
                            streak = movementStreak,
                            icon = Icons.Default.DirectionsRun
                        )
                    }
                    if (screenLimitEnabled) {
                        QuestRow(
                            title = "Screen Time Limit",
                            subtitle = "Cap: ${String.format(java.util.Locale.US, "%.1f", screenLimitTarget)}h/day",
                            streak = screenLimitStreak,
                            icon = Icons.Default.PhoneAndroid
                        )
                    }
                    if (sunsetEnabled) {
                        QuestRow(
                            title = "Digital Sunset",
                            subtitle = "Screen-free ${sunsetTarget}m before sleep",
                            streak = sunsetStreak,
                            icon = Icons.Default.NightsStay
                        )
                    }
                    if (circadianEnabled) {
                        val bedHour = prefs.getFloat("habit_circadian_anchor_target", 23.0f)
                        val bedStr = if (bedHour >= 24f || bedHour < 1f) "12:00 AM" else if (bedHour == 23.5f) "11:30 PM" else if (bedHour == 23.0f) "11:00 PM" else if (bedHour == 22.5f) "10:30 PM" else "10:00 PM"
                        QuestRow(
                            title = "Circadian Anchor",
                            subtitle = "Bedtime boundary: $bedStr",
                            streak = circadianStreak,
                            icon = Icons.Default.Schedule
                        )
                    }
                    if (focusEnabled) {
                        val focusTarget = prefs.getFloat("habit_focus_mode_target", 0.20f)
                        QuestRow(
                            title = "Focus Mode",
                            subtitle = "Social apps below ${(focusTarget * 100).toInt()}% ratio",
                            streak = focusStreak,
                            icon = Icons.Default.CenterFocusStrong
                        )
                    }
                    if (mindfulEnabled) {
                        QuestRow(
                            title = "Mindful Pause",
                            subtitle = "Target: $mindfulTarget breathing session/day",
                            streak = mindfulStreak,
                            icon = Icons.Default.SelfImprovement
                        )
                    }
                    if (daylightEnabled) {
                        QuestRow(
                            title = "Daylight Boost",
                            subtitle = "Target: ${daylightTarget}m daylight exposure",
                            streak = daylightStreak,
                            icon = Icons.Default.WbSunny
                        )
                    }

                    // Custom Quests
                    customQuests.forEach { quest ->
                        val catIcon = when (quest.category) {
                            "Movement" -> Icons.Default.DirectionsRun
                            "Screen" -> Icons.Default.PhoneAndroid
                            "Sleep" -> Icons.Default.Bedtime
                            "Mindfulness" -> Icons.Default.SelfImprovement
                            "Hydration" -> Icons.Default.LocalDrink
                            "Reading" -> Icons.Default.MenuBook
                            else -> Icons.Default.Star
                        }
                        QuestRow(
                            title = quest.title,
                            subtitle = "${quest.description.ifEmpty { "Daily goal" }} (${quest.targetQuantity} ${quest.unit})",
                            streak = quest.streak,
                            icon = catIcon,
                            progressFraction = if (quest.targetQuantity > 0) (quest.currentProgress.toFloat() / quest.targetQuantity).coerceIn(0f, 1f) else null,
                            progressText = "${quest.currentProgress} / ${quest.targetQuantity} ${quest.unit}",
                            actionButton = if (!quest.isAutoTracked) {
                                {
                                    IconButton(
                                        onClick = {
                                            val updated = customQuests.map {
                                                if (it.id == quest.id) {
                                                    val newProg = it.currentProgress + 1
                                                    val completed = newProg >= it.targetQuantity
                                                    val newStreak = if (completed && it.currentProgress < it.targetQuantity) it.streak + 1 else it.streak
                                                    it.copy(currentProgress = newProg, streak = newStreak)
                                                } else it
                                            }
                                            saveCustomQuests(prefs, updated)
                                            customQuests = updated
                                            Toast.makeText(context, "Progress logged for ${quest.title}!", Toast.LENGTH_SHORT).show()
                                        },
                                        modifier = Modifier.size(32.dp)
                                    ) {
                                        Icon(Icons.Default.AddCircle, null, tint = MaterialTheme.colorScheme.primary)
                                    }
                                }
                            } else null
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun QuestRow(
    title: String,
    subtitle: String,
    streak: Int,
    icon: ImageVector,
    progressFraction: Float? = null,
    progressText: String? = null,
    actionButton: (@Composable () -> Unit)? = null
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(0.35f)),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.08f))
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                    modifier = Modifier.weight(1f)
                ) {
                    Box(
                        modifier = Modifier
                            .size(36.dp)
                            .background(MaterialTheme.colorScheme.primary.copy(0.12f), CircleShape),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(icon, null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(18.dp))
                    }
                    Column {
                        Text(
                            text = title,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.onBackground
                        )
                        Text(
                            text = subtitle,
                            fontSize = 11.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }

                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    if (actionButton != null) {
                        actionButton()
                    }

                    Surface(
                        shape = RoundedCornerShape(16.dp),
                        color = MaterialTheme.colorScheme.primary.copy(0.15f)
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(3.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.LocalFireDepartment,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.size(13.dp)
                            )
                            Text(
                                text = "$streak d",
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Bold,
                                fontFamily = Fredoka,
                                color = MaterialTheme.colorScheme.primary
                            )
                        }
                    }
                }
            }

            if (progressFraction != null) {
                Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
                    LinearProgressIndicator(
                        progress = { progressFraction },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(4.dp)
                            .clip(CircleShape),
                        color = MaterialTheme.colorScheme.primary,
                        trackColor = MaterialTheme.colorScheme.outline.copy(0.15f)
                    )
                    if (progressText != null) {
                        Text(
                            text = progressText,
                            fontSize = 10.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            textAlign = TextAlign.End,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun BadgeGalleryDialog(
    badges: List<BadgeEntity>,
    onDismiss: () -> Unit
) {
    Dialog(onDismissRequest = onDismiss) {
        Surface(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            shape = RoundedCornerShape(24.dp),
            color = MaterialTheme.colorScheme.surface
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Achievement Badges",
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                    IconButton(onClick = onDismiss, modifier = Modifier.size(28.dp)) {
                        Icon(Icons.Default.Close, null, tint = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }

                LazyVerticalGrid(
                    columns = GridCells.Fixed(2),
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                    modifier = Modifier.height(300.dp)
                ) {
                    items(badges) { badge ->
                        BadgeTile(badge = badge)
                    }
                }

                Button(
                    onClick = onDismiss,
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
                ) {
                    Text("Close Gallery", color = Color.Black, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                }
            }
        }
    }
}

@Composable
fun BadgeTile(badge: BadgeEntity) {
    val isUnlocked = badge.isUnlocked
    val bgColor = if (isUnlocked) MaterialTheme.colorScheme.primary.copy(0.12f) else MaterialTheme.colorScheme.surfaceVariant.copy(0.3f)
    val iconColor = if (isUnlocked) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant.copy(0.4f)

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = bgColor),
        border = BorderStroke(1.dp, if (isUnlocked) MaterialTheme.colorScheme.primary.copy(0.3f) else Color.Transparent)
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Icon(
                imageVector = if (isUnlocked) Icons.Default.WorkspacePremium else Icons.Default.Lock,
                contentDescription = null,
                tint = iconColor,
                modifier = Modifier.size(28.dp)
            )
            Text(
                text = badge.badgeName,
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                textAlign = TextAlign.Center,
                color = MaterialTheme.colorScheme.onBackground
            )
            Text(
                text = badge.description,
                fontSize = 10.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
                lineHeight = 13.sp
            )
        }
    }
}

@Composable
fun WindDownCard(onStartClick: () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
    ) {
        Row(
            modifier = Modifier.padding(18.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(14.dp),
                modifier = Modifier.weight(1f)
            ) {
                Box(
                    modifier = Modifier
                        .size(42.dp)
                        .background(Color(0xFF818CF8).copy(0.15f), CircleShape),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(Icons.Default.Bedtime, null, tint = Color(0xFF818CF8), modifier = Modifier.size(22.dp))
                }
                Column {
                    Text(
                        text = "Wind-Down Companion",
                        fontSize = 15.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                    Text(
                        text = "Prepare your mind for sleep with dimming",
                        fontSize = 11.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            Button(
                onClick = onStartClick,
                shape = RoundedCornerShape(10.dp),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF818CF8))
            ) {
                Text("Start", color = Color.White, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
            }
        }
    }
}

@Composable
fun DigitalDetoxCard(onStartClick: () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
    ) {
        Row(
            modifier = Modifier.padding(18.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(14.dp),
                modifier = Modifier.weight(1f)
            ) {
                Box(
                    modifier = Modifier
                        .size(42.dp)
                        .background(MaterialTheme.colorScheme.primary.copy(0.15f), CircleShape),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(Icons.Default.Timer, null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(22.dp))
                }
                Column {
                    Text(
                        text = "Digital Detox Timer",
                        fontSize = 15.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                    Text(
                        text = "Lock screen focus time for offline space",
                        fontSize = 11.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            Button(
                onClick = onStartClick,
                shape = RoundedCornerShape(10.dp),
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
            ) {
                Text("Focus", color = Color.Black, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
            }
        }
    }
}

@Composable
fun DetoxDurationSetupDialog(
    initialMinutes: Int,
    onConfirm: (Int) -> Unit,
    onDismiss: () -> Unit
) {
    var selectedMinutes by remember { mutableIntStateOf(initialMinutes) }
    var customText by remember { mutableStateOf(initialMinutes.toString()) }

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
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Text(
                    text = "Digital Detox Duration",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Text(
                    text = "Select or enter your desired detox duration in minutes:",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    listOf(15, 30, 45, 60).forEach { mins ->
                        val isSel = selectedMinutes == mins
                        OutlinedButton(
                            onClick = {
                                selectedMinutes = mins
                                customText = mins.toString()
                            },
                            colors = ButtonDefaults.outlinedButtonColors(
                                containerColor = if (isSel) MaterialTheme.colorScheme.primary else Color.Transparent,
                                contentColor = if (isSel) Color.Black else MaterialTheme.colorScheme.primary
                            ),
                            shape = RoundedCornerShape(10.dp),
                            modifier = Modifier
                                .weight(1f)
                                .height(36.dp)
                        ) {
                            Text("${mins}m", fontSize = 11.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                        }
                    }
                }

                OutlinedTextField(
                    value = customText,
                    onValueChange = { input ->
                        val filtered = input.filter { it.isDigit() }
                        customText = filtered
                        val parsed = filtered.toIntOrNull()
                        if (parsed != null && parsed in 1..480) {
                            selectedMinutes = parsed
                        }
                    },
                    label = { Text("Custom Minutes", fontSize = 12.sp) },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp)
                )

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.End,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    TextButton(onClick = onDismiss) {
                        Text("Cancel", fontFamily = Fredoka)
                    }
                    Spacer(Modifier.width(8.dp))
                    Button(
                        onClick = { onConfirm(selectedMinutes.coerceIn(1, 480)) },
                        shape = RoundedCornerShape(10.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
                    ) {
                        Text("Start Detox", color = Color.Black, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                    }
                }
            }
        }
    }
}

@Composable
fun ManageHabitsDialog(
    prefs: SharedPreferences,
    onDismiss: () -> Unit
) {
    val context = LocalContext.current
    val db = remember { MHealthDatabase.getInstance(context.applicationContext) }
    val baselineEntities by produceState<List<BaselineEntity>>(emptyList(), db) {
        value = try {
            db.baselineDao().getBaseline(DataRepository.userProfile.value?.email ?: "default_user")
        } catch (e: Exception) {
            emptyList()
        }
    }

    // Baseline stats
    val baseSteps = baselineEntities.firstOrNull { it.featureName == "dailyStepCount" }?.baselineValue ?: 3500f
    val baseScreen = baselineEntities.firstOrNull { it.featureName == "screenTimeHours" }?.baselineValue ?: 4.5f
    val baseSleepHour = baselineEntities.firstOrNull { it.featureName == "sleepTimeHour" }?.baselineValue ?: 23.5f
    val baseSocial = baselineEntities.firstOrNull { it.featureName == "socialAppRatio" }?.baselineValue ?: 0.25f
    val baseDaylight = baselineEntities.firstOrNull { it.featureName == "daylightExposureMinutes" }?.baselineValue ?: 20f

    // Recommended values
    val recSteps = (((baseSteps * 1.2f) / 500f).roundToInt() * 500).coerceIn(2000, 12000)
    val recScreen = ((baseScreen * 0.85f * 10f).roundToInt() / 10f).coerceIn(2.0f, 6.0f)
    val recBedHour = if (baseSleepHour > 23f || baseSleepHour < 5f) 23.0f else (baseSleepHour - 0.5f)
    val recSocial = ((baseSocial * 0.75f * 100f).roundToInt() / 100f).coerceIn(0.10f, 0.30f)
    val recDaylight = ((baseDaylight + 15f) / 5f).roundToInt() * 5

    // State
    var movementEnabled by remember { mutableStateOf(prefs.getBoolean("habit_movement_boost_enabled", false)) }
    var movementTarget by remember { mutableIntStateOf(prefs.getInt("habit_movement_boost_target", 6000)) }

    var screenLimitEnabled by remember { mutableStateOf(prefs.getBoolean("habit_screen_limit_enabled", false)) }
    var screenLimitTarget by remember { mutableFloatStateOf(prefs.getFloat("habit_screen_limit_target", 4.0f)) }

    var sunsetEnabled by remember { mutableStateOf(prefs.getBoolean("habit_digital_sunset_enabled", false)) }
    var sunsetTarget by remember { mutableIntStateOf(prefs.getInt("habit_digital_sunset_target", 30)) }

    var circadianEnabled by remember { mutableStateOf(prefs.getBoolean("habit_circadian_anchor_enabled", false)) }
    var circadianTarget by remember { mutableFloatStateOf(prefs.getFloat("habit_circadian_anchor_target", 23.0f)) }

    var focusEnabled by remember { mutableStateOf(prefs.getBoolean("habit_focus_mode_enabled", false)) }
    var focusTarget by remember { mutableFloatStateOf(prefs.getFloat("habit_focus_mode_target", 0.20f)) }

    var mindfulEnabled by remember { mutableStateOf(prefs.getBoolean("habit_mindful_pause_enabled", false)) }
    var mindfulTarget by remember { mutableIntStateOf(prefs.getInt("habit_mindful_pause_target", 1)) }

    var daylightEnabled by remember { mutableStateOf(prefs.getBoolean("habit_daylight_boost_enabled", false)) }
    var daylightTarget by remember { mutableIntStateOf(prefs.getInt("habit_daylight_boost_target", 30)) }

    // Notification toggles
    var notifProgress by remember { mutableStateOf(prefs.getBoolean("quest_progress_notifications_enabled", true)) }
    var notifStreak by remember { mutableStateOf(prefs.getBoolean("quest_streak_notifications_enabled", true)) }
    var notifMilestone by remember { mutableStateOf(prefs.getBoolean("quest_milestone_notifications_enabled", true)) }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false)
    ) {
        Surface(
            shape = RoundedCornerShape(24.dp),
            color = MaterialTheme.colorScheme.surface,
            modifier = Modifier
                .fillMaxWidth(0.94f)
                .fillMaxHeight(0.88f)
                .padding(vertical = 12.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Icon(Icons.Default.Tune, null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(22.dp))
                        Text(
                            text = "Configure Habit Quests",
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.onBackground
                        )
                    }
                    IconButton(onClick = onDismiss, modifier = Modifier.size(28.dp)) {
                        Icon(Icons.Default.Close, null, tint = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }

                LazyColumn(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    // Movement Boost
                    item {
                        QuestConfigItem(
                            title = "Movement Boost",
                            description = "Daily physical step goal",
                            enabled = movementEnabled,
                            onToggle = { movementEnabled = it },
                            recommendationText = "★ Recommended: ${String.format(Locale.US, "%,d", recSteps)} steps (+20% vs ${baseSteps.toInt()} norm)",
                            onApplyRecommendation = { movementTarget = recSteps; movementEnabled = true }
                        ) {
                            val stepOptions = listOf(2000, 3000, 5000, 6000, 8000, 10000, 12000)
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .horizontalScroll(rememberScrollState()),
                                horizontalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                stepOptions.forEach { steps ->
                                    val isSelected = movementTarget == steps
                                    FilterChip(
                                        selected = isSelected,
                                        onClick = { movementTarget = steps },
                                        label = { Text("${String.format(Locale.US, "%,d", steps)} steps", fontSize = 11.sp, fontFamily = Fredoka) },
                                        colors = FilterChipDefaults.filterChipColors(
                                            selectedContainerColor = MaterialTheme.colorScheme.primary,
                                            selectedLabelColor = Color.Black
                                        )
                                    )
                                }
                            }
                        }
                    }

                    // Daily Screen Time Limit
                    item {
                        QuestConfigItem(
                            title = "Screen Time Limit",
                            description = "Daily total screen hours cap",
                            enabled = screenLimitEnabled,
                            onToggle = { screenLimitEnabled = it },
                            recommendationText = "★ Recommended: ${String.format(Locale.US, "%.1f", recScreen)}h cap (-15% vs ${String.format(Locale.US, "%.1f", baseScreen)}h norm)",
                            onApplyRecommendation = { screenLimitTarget = recScreen; screenLimitEnabled = true }
                        ) {
                            val screenOptions = listOf(2.0f, 3.0f, 3.5f, 4.0f, 4.5f, 5.0f, 6.0f)
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .horizontalScroll(rememberScrollState()),
                                horizontalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                screenOptions.forEach { hours ->
                                    val isSelected = (screenLimitTarget * 10).toInt() == (hours * 10).toInt()
                                    FilterChip(
                                        selected = isSelected,
                                        onClick = { screenLimitTarget = hours },
                                        label = { Text("${hours}h limit", fontSize = 11.sp, fontFamily = Fredoka) },
                                        colors = FilterChipDefaults.filterChipColors(
                                            selectedContainerColor = MaterialTheme.colorScheme.primary,
                                            selectedLabelColor = Color.Black
                                        )
                                    )
                                }
                            }
                        }
                    }

                    // Digital Sunset
                    item {
                        QuestConfigItem(
                            title = "Digital Sunset",
                            description = "Screen-free gap before bedtime",
                            enabled = sunsetEnabled,
                            onToggle = { sunsetEnabled = it },
                            recommendationText = "★ Recommended: 30 min screen-free gap before sleep",
                            onApplyRecommendation = { sunsetTarget = 30; sunsetEnabled = true }
                        ) {
                            val sunsetOptions = listOf(15, 30, 45, 60)
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                sunsetOptions.forEach { mins ->
                                    val isSelected = sunsetTarget == mins
                                    FilterChip(
                                        selected = isSelected,
                                        onClick = { sunsetTarget = mins },
                                        label = { Text("${mins}m gap", fontSize = 11.sp, fontFamily = Fredoka) },
                                        colors = FilterChipDefaults.filterChipColors(
                                            selectedContainerColor = MaterialTheme.colorScheme.primary,
                                            selectedLabelColor = Color.Black
                                        ),
                                        modifier = Modifier.weight(1f)
                                    )
                                }
                            }
                        }
                    }

                    // Circadian Anchor
                    item {
                        val bedStr = if (recBedHour >= 24f || recBedHour < 1f) "12:00 AM" else if (recBedHour == 23.5f) "11:30 PM" else if (recBedHour == 23.0f) "11:00 PM" else if (recBedHour == 22.5f) "10:30 PM" else "10:00 PM"
                        QuestConfigItem(
                            title = "Circadian Bedtime Anchor",
                            description = "Consistent target bedtime boundary",
                            enabled = circadianEnabled,
                            onToggle = { circadianEnabled = it },
                            recommendationText = "★ Recommended: $bedStr (30m earlier than usual)",
                            onApplyRecommendation = { circadianTarget = recBedHour; circadianEnabled = true }
                        ) {
                            val bedChoices = listOf(22.0f to "10:00 PM", 22.5f to "10:30 PM", 23.0f to "11:00 PM", 23.5f to "11:30 PM", 24.0f to "12:00 AM")
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .horizontalScroll(rememberScrollState()),
                                horizontalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                bedChoices.forEach { (hourVal, label) ->
                                    val isSelected = (circadianTarget * 10).toInt() == (hourVal * 10).toInt()
                                    FilterChip(
                                        selected = isSelected,
                                        onClick = { circadianTarget = hourVal },
                                        label = { Text(label, fontSize = 11.sp, fontFamily = Fredoka) },
                                        colors = FilterChipDefaults.filterChipColors(
                                            selectedContainerColor = MaterialTheme.colorScheme.primary,
                                            selectedLabelColor = Color.Black
                                        )
                                    )
                                }
                            }
                        }
                    }

                    // Focus Mode
                    item {
                        QuestConfigItem(
                            title = "Focus Mode",
                            description = "Social app time ceiling ratio",
                            enabled = focusEnabled,
                            onToggle = { focusEnabled = it },
                            recommendationText = "★ Recommended: ${(recSocial * 100).toInt()}% ratio limit (-25% vs usual norm)",
                            onApplyRecommendation = { focusTarget = recSocial; focusEnabled = true }
                        ) {
                            val focusOptions = listOf(0.10f to "10%", 0.15f to "15%", 0.20f to "20%", 0.25f to "25%", 0.30f to "30%")
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                focusOptions.forEach { (ratio, label) ->
                                    val isSelected = (focusTarget * 100).toInt() == (ratio * 100).toInt()
                                    FilterChip(
                                        selected = isSelected,
                                        onClick = { focusTarget = ratio },
                                        label = { Text(label, fontSize = 11.sp, fontFamily = Fredoka) },
                                        colors = FilterChipDefaults.filterChipColors(
                                            selectedContainerColor = MaterialTheme.colorScheme.primary,
                                            selectedLabelColor = Color.Black
                                        ),
                                        modifier = Modifier.weight(1f)
                                    )
                                }
                            }
                        }
                    }

                    // Mindful Pause
                    item {
                        QuestConfigItem(
                            title = "Mindful Pause",
                            description = "Guided breathing reset daily sessions",
                            enabled = mindfulEnabled,
                            onToggle = { mindfulEnabled = it },
                            recommendationText = "★ Recommended: 1 session/day for nervous system reset",
                            onApplyRecommendation = { mindfulTarget = 1; mindfulEnabled = true }
                        ) {
                            val sessionOptions = listOf(1, 2, 3)
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                sessionOptions.forEach { sessions ->
                                    val isSelected = mindfulTarget == sessions
                                    FilterChip(
                                        selected = isSelected,
                                        onClick = { mindfulTarget = sessions },
                                        label = { Text("$sessions session${if (sessions > 1) "s" else ""}/day", fontSize = 11.sp, fontFamily = Fredoka) },
                                        colors = FilterChipDefaults.filterChipColors(
                                            selectedContainerColor = MaterialTheme.colorScheme.primary,
                                            selectedLabelColor = Color.Black
                                        ),
                                        modifier = Modifier.weight(1f)
                                    )
                                }
                            }
                        }
                    }

                    // Daylight Boost
                    item {
                        QuestConfigItem(
                            title = "Daylight Boost",
                            description = "Natural sunlight exposure duration",
                            enabled = daylightEnabled,
                            onToggle = { daylightEnabled = it },
                            recommendationText = "★ Recommended: ${recDaylight}m daylight (+15m vs norm)",
                            onApplyRecommendation = { daylightTarget = recDaylight; daylightEnabled = true }
                        ) {
                            val daylightOptions = listOf(20, 30, 45, 60)
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                daylightOptions.forEach { mins ->
                                    val isSelected = daylightTarget == mins
                                    FilterChip(
                                        selected = isSelected,
                                        onClick = { daylightTarget = mins },
                                        label = { Text("${mins}m daily", fontSize = 11.sp, fontFamily = Fredoka) },
                                        colors = FilterChipDefaults.filterChipColors(
                                            selectedContainerColor = MaterialTheme.colorScheme.primary,
                                            selectedLabelColor = Color.Black
                                        ),
                                        modifier = Modifier.weight(1f)
                                    )
                                }
                            }
                        }
                    }

                    // Notification Settings Section
                    item {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(16.dp),
                            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(0.35f)),
                            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
                        ) {
                            Column(
                                modifier = Modifier.padding(14.dp),
                                verticalArrangement = Arrangement.spacedBy(10.dp)
                            ) {
                                Text(
                                    text = "Quest Notification Alerts",
                                    fontSize = 14.sp,
                                    fontWeight = FontWeight.Bold,
                                    fontFamily = Fredoka,
                                    color = MaterialTheme.colorScheme.onBackground
                                )

                                ToggleRow(
                                    title = "Mid-day Progress Updates",
                                    subtitle = "Afternoon nudge (3 PM) on active quest progress",
                                    checked = notifProgress,
                                    color = MaterialTheme.colorScheme.primary,
                                    onToggle = { notifProgress = it }
                                )

                                ToggleRow(
                                    title = "Evening Streak-Saver Alerts",
                                    subtitle = "Evening alert (8:30 PM) to protect your active streaks",
                                    checked = notifStreak,
                                    color = MaterialTheme.colorScheme.primary,
                                    onToggle = { notifStreak = it }
                                )

                                ToggleRow(
                                    title = "Milestone & Badge Celebrations",
                                    subtitle = "Alerts when hitting 3, 7, 14, or 30-day streaks",
                                    checked = notifMilestone,
                                    color = MaterialTheme.colorScheme.primary,
                                    onToggle = { notifMilestone = it }
                                )
                            }
                        }
                    }
                }

                Button(
                    onClick = {
                        prefs.edit()
                            .putBoolean("habit_movement_boost_enabled", movementEnabled)
                            .putInt("habit_movement_boost_target", movementTarget)
                            .putBoolean("habit_screen_limit_enabled", screenLimitEnabled)
                            .putFloat("habit_screen_limit_target", screenLimitTarget)
                            .putBoolean("habit_digital_sunset_enabled", sunsetEnabled)
                            .putInt("habit_digital_sunset_target", sunsetTarget)
                            .putBoolean("habit_circadian_anchor_enabled", circadianEnabled)
                            .putFloat("habit_circadian_anchor_target", circadianTarget)
                            .putBoolean("habit_focus_mode_enabled", focusEnabled)
                            .putFloat("habit_focus_mode_target", focusTarget)
                            .putBoolean("habit_mindful_pause_enabled", mindfulEnabled)
                            .putInt("habit_mindful_pause_target", mindfulTarget)
                            .putBoolean("habit_daylight_boost_enabled", daylightEnabled)
                            .putInt("habit_daylight_boost_target", daylightTarget)
                            .putBoolean("quest_progress_notifications_enabled", notifProgress)
                            .putBoolean("quest_streak_notifications_enabled", notifStreak)
                            .putBoolean("quest_milestone_notifications_enabled", notifMilestone)
                            .apply()

                        Toast.makeText(context, "Quest configurations saved!", Toast.LENGTH_SHORT).show()
                        onDismiss()
                    },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
                ) {
                    Text("Save & Apply Goals", color = Color.Black, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                }
            }
        }
    }
}

@Composable
fun QuestConfigItem(
    title: String,
    description: String,
    enabled: Boolean,
    onToggle: (Boolean) -> Unit,
    recommendationText: String,
    onApplyRecommendation: () -> Unit,
    targetPicker: @Composable () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (enabled) MaterialTheme.colorScheme.surface else MaterialTheme.colorScheme.surfaceVariant.copy(0.2f)
        ),
        border = BorderStroke(
            1.dp,
            if (enabled) MaterialTheme.colorScheme.primary.copy(0.25f) else MaterialTheme.colorScheme.outline.copy(0.08f)
        )
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            ToggleRow(
                title = title,
                subtitle = description,
                checked = enabled,
                color = MaterialTheme.colorScheme.primary,
                onToggle = { onToggle(it) }
            )

            if (enabled) {
                targetPicker()

                Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { onApplyRecommendation() },
                    shape = RoundedCornerShape(10.dp),
                    color = MaterialTheme.colorScheme.primary.copy(0.1f),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(0.25f))
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            text = recommendationText,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.weight(1f)
                        )
                        Text(
                            text = "Apply",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                }
            }
        }
    }
}


@Composable
fun CalmLotusPulse(modifier: Modifier = Modifier) {
    val infiniteTransition = rememberInfiniteTransition(label = "LotusPulse")
    
    val scale by infiniteTransition.animateFloat(
        initialValue = 0.96f,
        targetValue = 1.04f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 3000, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "LotusScale"
    )
    
    val rippleRadius1 by infiniteTransition.animateFloat(
        initialValue = 0.8f,
        targetValue = 1.6f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 3000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "Ripple1"
    )
    val rippleAlpha1 by infiniteTransition.animateFloat(
        initialValue = 0.4f,
        targetValue = 0.0f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 3000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "RippleAlpha1"
    )

    val rippleRadius2 by infiniteTransition.animateFloat(
        initialValue = 0.8f,
        targetValue = 1.6f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 3000, delayMillis = 1500, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "Ripple2"
    )
    val rippleAlpha2 by infiniteTransition.animateFloat(
        initialValue = 0.4f,
        targetValue = 0.0f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 3000, delayMillis = 1500, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "RippleAlpha2"
    )

    val primaryColor = MaterialTheme.colorScheme.primary

    Box(
        modifier = modifier.size(200.dp),
        contentAlignment = Alignment.Center
    ) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val center = Offset(size.width / 2, size.height / 2)
            val baseRadius = 80.dp.toPx()
            
            drawCircle(
                color = primaryColor,
                radius = baseRadius * rippleRadius1,
                center = center,
                alpha = rippleAlpha1,
                style = Stroke(width = 2.dp.toPx())
            )
            
            drawCircle(
                color = primaryColor,
                radius = baseRadius * rippleRadius2,
                center = center,
                alpha = rippleAlpha2,
                style = Stroke(width = 2.dp.toPx())
            )
        }
        
        Box(
            modifier = Modifier
                .size(110.dp)
                .scale(scale)
                .clip(CircleShape)
                .background(
                    Brush.radialGradient(
                        colors = listOf(
                            primaryColor.copy(alpha = 0.25f),
                            primaryColor.copy(alpha = 0.05f)
                        )
                    )
                )
                .border(2.dp, primaryColor.copy(alpha = 0.6f), CircleShape),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = Icons.Default.Spa,
                contentDescription = null,
                tint = primaryColor,
                modifier = Modifier.size(48.dp)
            )
        }
    }
}

@Composable
fun DailyFocusCard() {
    val quotes = remember {
        listOf(
            "Your body is a clock; let it chime in harmony with the sun. (Circadian Sync)",
            "Consistency in small routines breeds great peace of mind. (Routine)",
            "The best of wellness is not speed, but natural rhythm. (Pacing)",
            "Step by step, day by day, we find our anchors. (Habits)",
            "You have power over your mind - not outside events. Realize this, and you will find strength. — Marcus Aurelius (Stoicism)",
            "We suffer more often in imagination than in reality. — Seneca (Stoicism)",
            "Difficulties strengthen the mind, as labor does the body. — Seneca (Stoicism)",
            "Talk to yourself like you would to someone you love. — Brené Brown (Self-Compassion)",
            "If your compassion does not include yourself, it is incomplete. — Jack Kornfield (Self-Compassion)",
            "You yourself, as much as anybody in the entire universe, deserve your love and affection. — Buddha (Self-Compassion)",
            "Be gentle with yourself. You are doing the best you can. (Self-Compassion)",
            "The present moment is filled with joy and happiness. If you are attentive, you will see it. — Thich Nhat Hanh (Mindfulness)",
            "Quiet the mind, and the soul will speak. — Ma Jaya Sati Bhagavati (Mindfulness)",
            "Slow down and everything you are chasing will come and catch you. — John De Paola (Mindfulness)",
            "Circadian rhythms are our ancient connection to the spinning Earth. Align with daylight. (Science)",
            "Consistent daily patterns of light, movement, and sleep are the biological pillars of mental well-being. (Science)",
            "The brain works in oscillations; finding your natural resonance is key to focus. (Science)",
            "Nature does not hurry, yet everything is accomplished. — Lao Tzu (Pacing)",
            "A small routine change today creates a completely different biological trajectory tomorrow. (Science)",
            "Control your perceptions. Direct your actions properly. Willingly accept what's outside your control. (Stoicism)",
            "Rule your mind or it will rule you. — Horace (Stoicism)",
            "Quiet the mind, and the patterns of wellness will speak. (Mindfulness)",
            "Rest is not idleness, but key to restoration. (Pacing)",
            "Allow yourself to breathe, to exist, and to just be. (Mindfulness)",
            "Small shifts in screen habits build massive changes in focus. (Digital Boundaries)",
            "Movement is the natural medicine for a cluttered mind. (Mobility)"
        )
    }
    val quoteIndex = remember { quotes.indices.random() }
    val quote = quotes[quoteIndex]
    val cleanQuote = remember(quote) { quote.replace(Regex("\\s*\\([^)]+\\)$"), "") }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(0.06f)),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.06f))
    ) {
        Column(modifier = Modifier.padding(18.dp)) {
            Text(
                text = "Daily Focus",
                fontSize = 11.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                color = MaterialTheme.colorScheme.primary,
                modifier = Modifier.padding(bottom = 6.dp)
            )
            Text(
                text = "\"$cleanQuote\"",
                fontSize = 13.sp,
                fontStyle = androidx.compose.ui.text.font.FontStyle.Italic,
                lineHeight = 18.sp,
                color = MaterialTheme.colorScheme.onBackground.copy(0.9f)
            )
        }
    }
}

@Composable
fun TelemetrySnapshotCard(features: List<PersonalityVector>, baseline: PersonalityVector?) {
    val latest = features.lastOrNull() ?: return
    
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Today's Routine Snapshot",
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                color = MaterialTheme.colorScheme.onBackground,
                modifier = Modifier.padding(bottom = 12.dp)
            )
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                val snapshotPillModifier = Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(12.dp))
                    .background(MaterialTheme.colorScheme.onSurface.copy(alpha = 0.03f))
                    .padding(vertical = 10.dp, horizontal = 8.dp)
                
                // Sleep Pill
                Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = snapshotPillModifier) {
                    Text("🌙 Sleep", fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurfaceVariant, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                    Text("%.1f h".format(latest.sleepDurationHours), fontSize = 13.sp, fontWeight = FontWeight.ExtraBold, color = MaterialTheme.colorScheme.primary, modifier = Modifier.padding(top = 4.dp))
                }
                
                // Steps Pill
                Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = snapshotPillModifier) {
                    Text("🏃 Steps", fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurfaceVariant, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                    Text("%.0f".format(latest.dailyStepCount), fontSize = 13.sp, fontWeight = FontWeight.ExtraBold, color = MaterialTheme.colorScheme.primary, modifier = Modifier.padding(top = 4.dp))
                }
                
                // Screen Pill
                Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = snapshotPillModifier) {
                    Text("📱 Screen", fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurfaceVariant, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                    Text("%.1f h".format(latest.screenTimeHours), fontSize = 13.sp, fontWeight = FontWeight.ExtraBold, color = MaterialTheme.colorScheme.primary, modifier = Modifier.padding(top = 4.dp))
                }
            }
        }
    }
}

class CalmingSoundSynthesizer {
    private var audioTrack: AudioTrack? = null
    private var isPlaying = false
    private var currentVolume = 0.0f
    private var targetVolume = 0.0f

    fun start() {
        if (isPlaying) return
        isPlaying = true
        kotlin.concurrent.thread {
            val sampleRate = 44100
            val bufferSize = AudioTrack.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            )
            try {
                val track = AudioTrack(
                    AudioManager.STREAM_MUSIC,
                    sampleRate,
                    AudioFormat.CHANNEL_OUT_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    bufferSize,
                    AudioTrack.MODE_STREAM
                )
                audioTrack = track
                track.play()

                val buffer = ShortArray(1024)
                var phaseAngle = 0.0
                val frequency = 432.0 // Soothing 432Hz sine wave

                while (isPlaying) {
                    val volStep = 0.02f
                    if (currentVolume < targetVolume) {
                        currentVolume = (currentVolume + volStep).coerceAtMost(targetVolume)
                    } else if (currentVolume > targetVolume) {
                        currentVolume = (currentVolume - volStep).coerceAtLeast(targetVolume)
                    }

                    for (i in buffer.indices) {
                        val angle = phaseAngle + (2.0 * Math.PI * frequency / sampleRate)
                        buffer[i] = (Math.sin(angle) * Short.MAX_VALUE * currentVolume).toInt().toShort()
                        phaseAngle = angle
                    }
                    track.write(buffer, 0, buffer.size)
                }
                try {
                    track.stop()
                } catch (ignored: Exception) {}
                track.release()
            } catch (e: Exception) {
                Log.e("Synthesizer", "Error in audio thread: ${e.message}")
            }
        }
    }

    fun setVolume(volume: Float) {
        targetVolume = volume.coerceIn(0f, 0.5f) // Cap volume to prevent loudness
    }

    fun stop() {
        isPlaying = false
    }
}

@Composable
fun FullScreenBreathingScreen(
    onDismiss: () -> Unit
) {
    var setupMode by remember { mutableStateOf(true) }
    var selectedMinutes by remember { mutableIntStateOf(1) }
    var enableSound by remember { mutableStateOf(true) }

    val haptic = LocalHapticFeedback.current
    val synth = remember { CalmingSoundSynthesizer() }

    if (setupMode) {
        val navBarPad = rememberNavBarPadding()
        Dialog(
            onDismissRequest = onDismiss,
            properties = DialogProperties(usePlatformDefaultWidth = false, decorFitsSystemWindows = false)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color(0xFF0B1F28)) // Rich Dark Teal
                    .statusBarsPadding()
                    .padding(start = 24.dp, end = 24.dp, top = 24.dp, bottom = (navBarPad.value + 24).dp),
                contentAlignment = Alignment.Center
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(24.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Spa,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(64.dp)
                    )
                    Text(
                        text = "Mindful Breathing Reset",
                        fontSize = 24.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color.White,
                        fontFamily = Fredoka
                    )
                    Text(
                        text = "Take a moment to align your focus. Box breathing (4s inhale, 4s hold, 4s exhale, 4s hold) reduces stress and anchors your nervous system.",
                        fontSize = 14.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center,
                        lineHeight = 20.sp,
                        modifier = Modifier.padding(horizontal = 16.dp)
                    )

                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Text(
                            text = "Duration",
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color.White,
                            fontFamily = Fredoka
                        )
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            listOf(1, 3, 5).forEach { min ->
                                val isSel = selectedMinutes == min
                                OutlinedButton(
                                    onClick = { selectedMinutes = min },
                                    colors = ButtonDefaults.outlinedButtonColors(
                                        containerColor = if (isSel) MaterialTheme.colorScheme.primary else Color.Transparent,
                                        contentColor = if (isSel) Color.Black else MaterialTheme.colorScheme.primary
                                    ),
                                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary),
                                    shape = RoundedCornerShape(12.dp)
                                ) {
                                    Text(
                                        text = "$min Min",
                                        fontWeight = FontWeight.Bold,
                                        fontFamily = Fredoka
                                    )
                                }
                            }
                        }
                    }

                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                        modifier = Modifier
                            .clip(RoundedCornerShape(16.dp))
                            .background(Color.White.copy(0.05f))
                            .padding(horizontal = 16.dp, vertical = 12.dp)
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = "Ambient Sound Bath",
                                fontSize = 14.sp,
                                fontWeight = FontWeight.Bold,
                                color = Color.White,
                                fontFamily = Fredoka
                            )
                            Text(
                                text = "Play calming 432Hz sine wave harmony",
                                fontSize = 11.sp,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        Switch(
                            checked = enableSound,
                            onCheckedChange = { enableSound = it },
                            colors = SwitchDefaults.colors(
                                checkedThumbColor = MaterialTheme.colorScheme.primary,
                                checkedTrackColor = MaterialTheme.colorScheme.primary.copy(0.4f)
                            )
                        )
                    }

                    Spacer(Modifier.height(16.dp))

                    Row(
                        horizontalArrangement = Arrangement.spacedBy(16.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        OutlinedButton(
                            onClick = onDismiss,
                            modifier = Modifier.weight(1f),
                            colors = ButtonDefaults.outlinedButtonColors(contentColor = Color.White),
                            border = BorderStroke(1.dp, Color.White.copy(0.3f)),
                            shape = RoundedCornerShape(16.dp)
                        ) {
                            Text("Cancel", fontFamily = Fredoka)
                        }
                        Button(
                            onClick = { setupMode = false },
                            modifier = Modifier.weight(1.5f),
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                            shape = RoundedCornerShape(16.dp)
                        ) {
                            Text("Start Session", color = Color.Black, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                        }
                    }
                }
            }
        }
    } else {
        Dialog(
            onDismissRequest = {
                synth.stop()
                onDismiss()
            },
            properties = DialogProperties(usePlatformDefaultWidth = false, decorFitsSystemWindows = false)
        ) {
            var activePhase by remember { mutableStateOf("Inhale") } // "Inhale", "Hold (In)", "Exhale", "Hold (Out)"
            var secondsLeft by remember { mutableIntStateOf(4) }
            var totalTimerSeconds by remember { mutableIntStateOf(selectedMinutes * 60) }

            DisposableEffect(Unit) {
                if (enableSound) {
                    synth.start()
                    synth.setVolume(0.1f)
                }
                onDispose {
                    synth.stop()
                }
            }

            LaunchedEffect(activePhase) {
                haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                if (enableSound) {
                    when (activePhase) {
                        "Inhale" -> synth.setVolume(0.4f)
                        "Hold (In)" -> synth.setVolume(0.4f)
                        "Exhale" -> synth.setVolume(0.02f)
                        "Hold (Out)" -> synth.setVolume(0.0f)
                    }
                }
            }

            LaunchedEffect(Unit) {
                while (totalTimerSeconds > 0) {
                    delay(1000L)
                    totalTimerSeconds -= 1
                    if (secondsLeft > 1) {
                        secondsLeft -= 1
                    } else {
                        activePhase = when (activePhase) {
                            "Inhale" -> "Hold (In)"
                            "Hold (In)" -> "Exhale"
                            "Exhale" -> "Hold (Out)"
                            else -> "Inhale"
                        }
                        secondsLeft = 4
                    }
                }
                haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                delay(500L)
                haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                synth.stop()
                onDismiss()
            }

            val navBarPad2 = rememberNavBarPadding()
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color(0xFF07141C))
                    .statusBarsPadding()
                    .navigationBarsPadding(),
                contentAlignment = Alignment.Center
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.SpaceBetween,
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(top = 32.dp, bottom = 24.dp, start = 24.dp, end = 24.dp)
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = "Reset Your Rhythm",
                            fontSize = 16.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = MaterialTheme.colorScheme.primary,
                            fontFamily = Fredoka
                        )
                        Spacer(Modifier.height(4.dp))
                        Text(
                            text = "Remaining: ${totalTimerSeconds / 60}:${String.format("%02d", totalTimerSeconds % 60)}",
                            fontSize = 14.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }

                    val animatedProgress = remember { Animatable(0f) }
                    LaunchedEffect(activePhase, secondsLeft) {
                        val targetVal = when (activePhase) {
                            "Inhale" -> 1.0f - (secondsLeft - 1) / 4f
                            "Hold (In)" -> 1f
                            "Exhale" -> (secondsLeft - 1) / 4f
                            else -> 0f // Hold (Out)
                        }
                        animatedProgress.animateTo(
                            targetValue = targetVal,
                            animationSpec = tween(durationMillis = 1000, easing = LinearEasing)
                        )
                    }

                    val scaleFactor = when (activePhase) {
                        "Inhale" -> 0.7f + (animatedProgress.value * 0.5f)
                        "Hold (In)" -> 1.2f
                        "Exhale" -> 0.7f + (animatedProgress.value * 0.5f)
                        else -> 0.7f // Hold (Out)
                    }

                    Box(
                        contentAlignment = Alignment.Center,
                        modifier = Modifier.size(300.dp)
                    ) {
                        val infiniteTransition = rememberInfiniteTransition(label = "Ripple")
                        val rippleScale by infiniteTransition.animateFloat(
                            initialValue = 1f,
                            targetValue = 1.15f,
                            animationSpec = infiniteRepeatable(
                                animation = tween(1500, easing = EaseInOutSine),
                                repeatMode = RepeatMode.Reverse
                            ),
                            label = "RippleScale"
                        )
                        
                        val primaryColor = MaterialTheme.colorScheme.primary
                        Canvas(
                            modifier = Modifier
                                .fillMaxSize()
                                .scale(scaleFactor)
                        ) {
                            drawCircle(
                                brush = Brush.radialGradient(
                                    colors = listOf(
                                        primaryColor.copy(alpha = 0.25f),
                                        primaryColor.copy(alpha = 0.0f)
                                    )
                                ),
                                radius = size.minDimension / 2 * rippleScale
                            )

                            drawCircle(
                                color = primaryColor,
                                style = Stroke(width = 4.dp.toPx(), cap = StrokeCap.Round),
                                radius = size.minDimension / 3
                            )

                            drawCircle(
                                color = primaryColor.copy(alpha = 0.15f),
                                radius = size.minDimension / 3 - 2.dp.toPx()
                            )
                        }

                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            val displayPhase = if (activePhase.startsWith("Hold")) "Hold" else activePhase
                            Text(
                                text = displayPhase.uppercase(),
                                fontSize = 24.sp,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.primary,
                                fontFamily = Fredoka,
                                letterSpacing = 2.sp
                            )
                            Spacer(Modifier.height(8.dp))
                            Text(
                                text = "$secondsLeft",
                                fontSize = 32.sp,
                                fontWeight = FontWeight.ExtraBold,
                                color = Color.White
                            )
                        }
                    }

                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        val instruction = when (activePhase) {
                            "Inhale" -> "Breathe in slowly, filling your lungs."
                            "Hold (In)" -> "Suspend your breath, rest in silence."
                            "Exhale" -> "Release the air gently, letting go."
                            else -> "Keep your lungs empty, wait for the cycle." // Hold (Out)
                        }
                        Text(
                            text = instruction,
                            fontSize = 14.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            textAlign = TextAlign.Center,
                            fontFamily = Fredoka,
                            modifier = Modifier.padding(horizontal = 24.dp)
                        )
                        OutlinedButton(
                            onClick = {
                                synth.stop()
                                onDismiss()
                            },
                            colors = ButtonDefaults.outlinedButtonColors(contentColor = AlertRose),
                            border = BorderStroke(1.dp, AlertRose.copy(0.5f)),
                            shape = RoundedCornerShape(16.dp),
                            modifier = Modifier.padding(top = 8.dp)
                        ) {
                            Text("Stop Session", fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun MindfulBreathingCard() {
    var showSession by remember { mutableStateOf(false) }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(modifier = Modifier.weight(1f), verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier
                        .size(40.dp)
                        .clip(CircleShape)
                        .background(MaterialTheme.colorScheme.primary.copy(0.1f)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(Icons.Default.Spa, null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(20.dp))
                }
                Spacer(Modifier.width(16.dp))
                Column {
                    Text(
                        text = "Mindful Breathing Pause",
                        fontSize = 15.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                    Text(
                        text = "Guided box breathing reset for your nervous system.",
                        fontSize = 12.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
            Button(
                onClick = { showSession = true },
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier.height(36.dp)
            ) {
                Text("Start", color = Color.Black, fontSize = 12.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
            }
        }
    }

    if (showSession) {
        FullScreenBreathingScreen(onDismiss = { showSession = false })
    }
}

@Composable
fun QuestsScreen(
    prefs: SharedPreferences,
    badges: List<BadgeEntity>,
    onBack: () -> Unit
) {
    val context = LocalContext.current
    var showManageHabits by remember { mutableStateOf(false) }
    var showCreateCustomHabit by remember { mutableStateOf(false) }

    val unlockedCount = remember(badges) { badges.count { it.isUnlocked } }

    val sunsetEnabled = remember { prefs.getBoolean("habit_digital_sunset_enabled", false) }
    val circadianEnabled = remember { prefs.getBoolean("habit_circadian_anchor_enabled", false) }
    val movementEnabled = remember { prefs.getBoolean("habit_movement_boost_enabled", false) }
    val screenLimitEnabled = remember { prefs.getBoolean("habit_screen_limit_enabled", false) }
    val focusEnabled = remember { prefs.getBoolean("habit_focus_mode_enabled", false) }
    val mindfulEnabled = remember { prefs.getBoolean("habit_mindful_pause_enabled", false) }
    val daylightEnabled = remember { prefs.getBoolean("habit_daylight_boost_enabled", false) }

    val sunsetStreak = remember { prefs.getInt("habit_digital_sunset_streak", prefs.getInt("streak_digital_sunset", 0)) }
    val circadianStreak = remember { prefs.getInt("habit_circadian_anchor_streak", prefs.getInt("streak_circadian_anchor", 0)) }
    val movementStreak = remember { prefs.getInt("habit_movement_boost_streak", prefs.getInt("streak_movement_boost", 0)) }
    val screenLimitStreak = remember { prefs.getInt("habit_screen_limit_streak", 0) }
    val focusStreak = remember { prefs.getInt("habit_focus_mode_streak", prefs.getInt("streak_focus_mode", 0)) }
    val mindfulStreak = remember { prefs.getInt("habit_mindful_pause_streak", 0) }
    val daylightStreak = remember { prefs.getInt("habit_daylight_boost_streak", 0) }

    val movementTarget = remember { prefs.getInt("habit_movement_boost_target", 6000) }
    val screenLimitTarget = remember { prefs.getFloat("habit_screen_limit_target", 4.0f) }
    val sunsetTarget = remember { prefs.getInt("habit_digital_sunset_target", 30) }
    val mindfulTarget = remember { prefs.getInt("habit_mindful_pause_target", 1) }
    val daylightTarget = remember { prefs.getInt("habit_daylight_boost_target", 30) }

    var customQuests by remember { mutableStateOf(loadCustomQuests(prefs)) }

    if (showManageHabits) {
        ManageHabitsDialog(prefs = prefs, onDismiss = {
            showManageHabits = false
            customQuests = loadCustomQuests(prefs)
        })
    }
    if (showCreateCustomHabit) {
        CreateCustomHabitDialog(
            onSave = { newQuest ->
                val updated = customQuests + newQuest
                saveCustomQuests(prefs, updated)
                customQuests = updated
                showCreateCustomHabit = false
                Toast.makeText(context, "Custom quest added!", Toast.LENGTH_SHORT).show()
            },
            onDismiss = { showCreateCustomHabit = false }
        )
    }

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentPadding = PaddingValues(20.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp)
    ) {
        // Header
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
                    text = "Habit Quests & Badges",
                    fontSize = 22.sp,
                    fontWeight = FontWeight.ExtraBold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
            }
        }

        // Quest Stats Summary Card
        val activeCount = listOf(
            sunsetEnabled, circadianEnabled, movementEnabled, screenLimitEnabled,
            focusEnabled, mindfulEnabled, daylightEnabled
        ).count { it } + customQuests.size

        val maxStreak = (listOf(
            sunsetStreak, circadianStreak, movementStreak, screenLimitStreak,
            focusStreak, mindfulStreak, daylightStreak
        ) + customQuests.map { it.streak }).maxOrNull() ?: 0

        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
            ) {
                Row(
                    modifier = Modifier.padding(18.dp),
                    horizontalArrangement = Arrangement.SpaceAround,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("Active Quests", fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Text(
                            "$activeCount",
                            fontSize = 20.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("Badges Unlocked", fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Text(
                            "$unlockedCount / ${badges.size}",
                            fontSize = 20.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("Max Streak", fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Text(
                            "$maxStreak d",
                            fontSize = 20.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                }
            }
        }

        // Action Row (+ Custom Habit, Configure Quests)
        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Button(
                    onClick = { showCreateCustomHabit = true },
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Icon(Icons.Default.Add, null, tint = Color.Black, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Add Custom Quest", color = Color.Black, fontWeight = FontWeight.Bold, fontFamily = Fredoka, fontSize = 12.sp)
                }

                Button(
                    onClick = { showManageHabits = true },
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Icon(Icons.Default.Tune, null, tint = MaterialTheme.colorScheme.onPrimaryContainer, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Configure Quests", color = MaterialTheme.colorScheme.onPrimaryContainer, fontWeight = FontWeight.Bold, fontFamily = Fredoka, fontSize = 12.sp)
                }
            }
        }

        // Active Quests Section
        item {
            Text("Active Quests", fontSize = 16.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
        }

        if (movementEnabled) {
            item {
                QuestRow(
                    title = "Movement Boost",
                    subtitle = "Goal: ${String.format(Locale.US, "%,d", movementTarget)} steps/day",
                    streak = movementStreak,
                    icon = Icons.Default.DirectionsRun
                )
            }
        }
        if (screenLimitEnabled) {
            item {
                QuestRow(
                    title = "Screen Time Limit",
                    subtitle = "Cap: ${String.format(Locale.US, "%.1f", screenLimitTarget)}h/day",
                    streak = screenLimitStreak,
                    icon = Icons.Default.PhoneAndroid
                )
            }
        }
        if (sunsetEnabled) {
            item {
                QuestRow(
                    title = "Digital Sunset",
                    subtitle = "Screen-free ${sunsetTarget}m before sleep",
                    streak = sunsetStreak,
                    icon = Icons.Default.NightsStay
                )
            }
        }
        if (circadianEnabled) {
            val bedHour = prefs.getFloat("habit_circadian_anchor_target", 23.0f)
            val bedStr = if (bedHour >= 24f || bedHour < 1f) "12:00 AM" else if (bedHour == 23.5f) "11:30 PM" else if (bedHour == 23.0f) "11:00 PM" else if (bedHour == 22.5f) "10:30 PM" else "10:00 PM"
            item {
                QuestRow(
                    title = "Circadian Anchor",
                    subtitle = "Bedtime boundary: $bedStr",
                    streak = circadianStreak,
                    icon = Icons.Default.Schedule
                )
            }
        }
        if (focusEnabled) {
            val focusTarget = prefs.getFloat("habit_focus_mode_target", 0.20f)
            item {
                QuestRow(
                    title = "Focus Mode",
                    subtitle = "Social apps below ${(focusTarget * 100).toInt()}% ratio",
                    streak = focusStreak,
                    icon = Icons.Default.CenterFocusStrong
                )
            }
        }
        if (mindfulEnabled) {
            item {
                QuestRow(
                    title = "Mindful Pause",
                    subtitle = "Target: $mindfulTarget breathing session/day",
                    streak = mindfulStreak,
                    icon = Icons.Default.SelfImprovement
                )
            }
        }
        if (daylightEnabled) {
            item {
                QuestRow(
                    title = "Daylight Boost",
                    subtitle = "Target: ${daylightTarget}m daylight exposure",
                    streak = daylightStreak,
                    icon = Icons.Default.WbSunny
                )
            }
        }

        // Custom Quests
        items(customQuests) { quest ->
            val catIcon = when (quest.category) {
                "Movement" -> Icons.Default.DirectionsRun
                "Screen" -> Icons.Default.PhoneAndroid
                "Sleep" -> Icons.Default.Bedtime
                "Mindfulness" -> Icons.Default.SelfImprovement
                "Hydration" -> Icons.Default.LocalDrink
                "Reading" -> Icons.Default.MenuBook
                else -> Icons.Default.Star
            }
            QuestRow(
                title = quest.title,
                subtitle = "${quest.description.ifEmpty { "Custom personal goal" }} (${quest.targetQuantity} ${quest.unit})",
                streak = quest.streak,
                icon = catIcon,
                progressFraction = if (quest.targetQuantity > 0) (quest.currentProgress.toFloat() / quest.targetQuantity).coerceIn(0f, 1f) else null,
                progressText = "${quest.currentProgress} / ${quest.targetQuantity} ${quest.unit}",
                actionButton = if (!quest.isAutoTracked) {
                    {
                        IconButton(
                            onClick = {
                                val updated = customQuests.map {
                                    if (it.id == quest.id) {
                                        val newProg = it.currentProgress + 1
                                        val completed = newProg >= it.targetQuantity
                                        val newStreak = if (completed && it.currentProgress < it.targetQuantity) it.streak + 1 else it.streak
                                        it.copy(currentProgress = newProg, streak = newStreak)
                                    } else it
                                }
                                saveCustomQuests(prefs, updated)
                                customQuests = updated
                                Toast.makeText(context, "Progress logged for ${quest.title}!", Toast.LENGTH_SHORT).show()
                            },
                            modifier = Modifier.size(32.dp)
                        ) {
                            Icon(Icons.Default.AddCircle, null, tint = MaterialTheme.colorScheme.primary)
                        }
                    }
                } else null
            )
        }

        // Achievement Badges & Sharing Section
        item {
            Text("Achievement Badges & Sharing", fontSize = 16.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
        }

        items(badges) { badge ->
            BadgeRowWithShare(badge = badge, onShare = { shareBadge(context, badge) })
        }
    }
}

@Composable
fun BadgeRowWithShare(badge: BadgeEntity, onShare: () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (badge.isUnlocked) MaterialTheme.colorScheme.primary.copy(0.12f) else MaterialTheme.colorScheme.surface
        ),
        border = BorderStroke(1.dp, if (badge.isUnlocked) MaterialTheme.colorScheme.primary.copy(0.3f) else MaterialTheme.colorScheme.outline.copy(0.1f))
    ) {
        Row(
            modifier = Modifier.padding(14.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(
                modifier = Modifier.weight(1f),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Icon(
                    imageVector = if (badge.isUnlocked) Icons.Default.WorkspacePremium else Icons.Default.Lock,
                    contentDescription = null,
                    tint = if (badge.isUnlocked) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant.copy(0.4f),
                    modifier = Modifier.size(28.dp)
                )
                Column {
                    Text(badge.badgeName, fontWeight = FontWeight.Bold, fontSize = 14.sp, fontFamily = Fredoka)
                    Text(badge.description, fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }

            if (badge.isUnlocked) {
                IconButton(onClick = onShare) {
                    Icon(Icons.Default.Share, "Share", tint = MaterialTheme.colorScheme.primary)
                }
            }
        }
    }
}

fun shareBadge(context: Context, badge: BadgeEntity) {
    try {
        val shareIntent = Intent("android.intent.action.SEND").apply {
            type = "text/plain"
            putExtra("android.intent.extra.SUBJECT", "Lumen Achievement: ${badge.badgeName}")
            putExtra("android.intent.extra.TEXT", "🎉 I unlocked the '${badge.badgeName}' badge on Lumen! ${badge.description}")
        }
        context.startActivity(Intent.createChooser(shareIntent, "Share Achievement Badge"))
    } catch (e: Exception) {
        Toast.makeText(context, "Cannot share badge", Toast.LENGTH_SHORT).show()
    }
}

@Composable
fun CreateCustomHabitDialog(
    onSave: (CustomQuest) -> Unit,
    onDismiss: () -> Unit
) {
    var title by remember { mutableStateOf("") }
    var desc by remember { mutableStateOf("") }
    var category by remember { mutableStateOf("Hydration") }
    var targetQtyText by remember { mutableStateOf("8") }
    var unit by remember { mutableStateOf("glasses") }
    var isAutoTracked by remember { mutableStateOf(false) }

    val categories = listOf(
        "Hydration" to ("glasses" to Icons.Default.LocalDrink),
        "Movement" to ("steps" to Icons.Default.DirectionsRun),
        "Screen" to ("minutes" to Icons.Default.PhoneAndroid),
        "Mindfulness" to ("sessions" to Icons.Default.SelfImprovement),
        "Reading" to ("pages" to Icons.Default.MenuBook),
        "Sleep" to ("hours" to Icons.Default.Bedtime),
        "General" to ("times" to Icons.Default.Star)
    )

    Dialog(onDismissRequest = onDismiss) {
        Surface(
            shape = RoundedCornerShape(24.dp),
            color = MaterialTheme.colorScheme.surface,
            modifier = Modifier.fillMaxWidth().padding(16.dp)
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp)
            ) {
                Text(
                    text = "Add Custom Quest",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )

                OutlinedTextField(
                    value = title,
                    onValueChange = { title = it },
                    label = { Text("Quest Title", fontSize = 12.sp) },
                    placeholder = { Text("e.g., Drink 8 Glasses of Water", fontSize = 12.sp) },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp)
                )

                OutlinedTextField(
                    value = desc,
                    onValueChange = { desc = it },
                    label = { Text("Description / Purpose", fontSize = 12.sp) },
                    placeholder = { Text("e.g., Maintain steady hydration throughout the day", fontSize = 12.sp) },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp)
                )

                // Category Chips
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("Category & Icon", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        categories.forEach { (catName, defaultUnitPair) ->
                            val isSelected = category == catName
                            FilterChip(
                                selected = isSelected,
                                onClick = {
                                    category = catName
                                    unit = defaultUnitPair.first
                                },
                                label = { Text(catName, fontSize = 11.sp, fontFamily = Fredoka) },
                                leadingIcon = { Icon(defaultUnitPair.second, null, modifier = Modifier.size(14.dp)) },
                                colors = FilterChipDefaults.filterChipColors(
                                    selectedContainerColor = MaterialTheme.colorScheme.primary,
                                    selectedLabelColor = Color.Black
                                )
                            )
                        }
                    }
                }

                // Quantity & Unit Row
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    OutlinedTextField(
                        value = targetQtyText,
                        onValueChange = { if (it.all { ch -> ch.isDigit() }) targetQtyText = it },
                        label = { Text("Target Qty", fontSize = 12.sp) },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(12.dp)
                    )

                    OutlinedTextField(
                        value = unit,
                        onValueChange = { unit = it },
                        label = { Text("Unit", fontSize = 12.sp) },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(12.dp)
                    )
                }

                // Tracking Mode Toggle if Movement / Screen
                if (category == "Movement" || category == "Screen" || category == "Daylight") {
                    ToggleRow(
                        title = "Auto-track with Sensors",
                        subtitle = "Automatically update progress from phone sensors",
                        checked = isAutoTracked,
                        color = MaterialTheme.colorScheme.primary,
                        onToggle = { isAutoTracked = it }
                    )
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    OutlinedButton(
                        onClick = onDismiss,
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Text("Cancel", fontFamily = Fredoka)
                    }
                    Button(
                        onClick = {
                            if (title.isNotBlank()) {
                                val qty = targetQtyText.toIntOrNull() ?: 1
                                val quest = CustomQuest(
                                    id = "quest_${System.currentTimeMillis()}",
                                    title = title.trim(),
                                    description = desc.trim(),
                                    category = category,
                                    targetQuantity = qty,
                                    unit = unit.trim().ifEmpty { "times" },
                                    isAutoTracked = isAutoTracked
                                )
                                onSave(quest)
                            }
                        },
                        modifier = Modifier.weight(1f),
                        enabled = title.isNotBlank(),
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
                    ) {
                        Text("Create Quest", color = Color.Black, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                    }
                }
            }
        }
    }
}
