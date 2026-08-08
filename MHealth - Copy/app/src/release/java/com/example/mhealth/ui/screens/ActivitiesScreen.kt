package com.example.mhealth.ui.screens

import android.content.Context
import android.content.SharedPreferences
import android.widget.Toast
import androidx.compose.animation.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import com.example.mhealth.logic.DataRepository
import com.example.mhealth.logic.db.BadgeEntity
import com.example.mhealth.ui.components.AlertWarning
import com.example.mhealth.ui.components.StaggeredFadeIn
import com.example.mhealth.ui.components.ToggleRow
import com.example.mhealth.ui.components.Fredoka

@Composable
fun ActivitiesScreen() {
    val context = LocalContext.current
    val prefs = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }
    val badges by DataRepository.badges.collectAsState()

    var showBadgeGallery by remember { mutableStateOf(false) }
    var showManageHabits by remember { mutableStateOf(false) }
    var showWindDownOverlay by remember { mutableStateOf(false) }
    var showDetoxOverlay by remember { mutableStateOf(false) }

    if (showBadgeGallery) {
        BadgeGalleryDialog(badges = badges, onDismiss = { showBadgeGallery = false })
    }
    if (showManageHabits) {
        ManageHabitsDialog(prefs = prefs, onDismiss = { showManageHabits = false })
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
                onGalleryClick = { showBadgeGallery = true }
            )
        }

        // Wind-Down Companion
        item {
            WindDownCard(onStartClick = { showWindDownOverlay = true })
        }

        // Digital Detox
        item {
            DigitalDetoxCard(onStartClick = { showDetoxOverlay = true })
        }
    }
}

@Composable
fun HabitQuestsCard(
    prefs: SharedPreferences,
    badges: List<BadgeEntity>,
    onManageClick: () -> Unit,
    onGalleryClick: () -> Unit
) {
    val unlockedCount = remember(badges) { badges.count { it.isUnlocked } }

    val sunsetEnabled = remember { prefs.getBoolean("habit_digital_sunset_enabled", false) }
    val circadianEnabled = remember { prefs.getBoolean("habit_circadian_anchor_enabled", false) }
    val movementEnabled = remember { prefs.getBoolean("habit_movement_boost_enabled", false) }
    val focusEnabled = remember { prefs.getBoolean("habit_focus_mode_enabled", false) }

    val sunsetStreak = remember { prefs.getInt("streak_digital_sunset", 0) }
    val circadianStreak = remember { prefs.getInt("streak_circadian_anchor", 0) }
    val movementStreak = remember { prefs.getInt("streak_movement_boost", 0) }
    val focusStreak = remember { prefs.getInt("streak_focus_mode", 0) }

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

            val activeCount = listOf(sunsetEnabled, circadianEnabled, movementEnabled, focusEnabled).count { it }
            if (activeCount == 0) {
                Text(
                    text = "No active habit quests configured. Tap settings to enable digital sunset, circadian anchor, or movement goals.",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    lineHeight = 17.sp
                )
            } else {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    if (sunsetEnabled) {
                        QuestRow(
                            title = "Digital Sunset",
                            subtitle = "Screen-free 30 min before sleep",
                            streak = sunsetStreak,
                            icon = Icons.Default.NightsStay
                        )
                    }
                    if (circadianEnabled) {
                        QuestRow(
                            title = "Circadian Anchor",
                            subtitle = "Consistent sleep window boundary",
                            streak = circadianStreak,
                            icon = Icons.Default.Schedule
                        )
                    }
                    if (movementEnabled) {
                        QuestRow(
                            title = "Movement Boost",
                            subtitle = "Physical step goal",
                            streak = movementStreak,
                            icon = Icons.Default.DirectionsRun
                        )
                    }
                    if (focusEnabled) {
                        QuestRow(
                            title = "Focus Mode",
                            subtitle = "Balanced social media ratio",
                            streak = focusStreak,
                            icon = Icons.Default.CenterFocusStrong
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
    icon: ImageVector
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp),
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
                    fontSize = 14.sp,
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

        Surface(
            shape = RoundedCornerShape(20.dp),
            color = MaterialTheme.colorScheme.primary.copy(0.15f)
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.LocalFireDepartment,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(14.dp)
                )
                Text(
                    text = "$streak d",
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.primary
                )
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
fun ManageHabitsDialog(
    prefs: SharedPreferences,
    onDismiss: () -> Unit
) {
    var sunset by remember { mutableStateOf(prefs.getBoolean("habit_digital_sunset_enabled", false)) }
    var circadian by remember { mutableStateOf(prefs.getBoolean("habit_circadian_anchor_enabled", false)) }
    var movement by remember { mutableStateOf(prefs.getBoolean("habit_movement_boost_enabled", false)) }
    var focus by remember { mutableStateOf(prefs.getBoolean("habit_focus_mode_enabled", false)) }

    Dialog(onDismissRequest = onDismiss) {
        Surface(
            shape = RoundedCornerShape(24.dp),
            color = MaterialTheme.colorScheme.surface,
            modifier = Modifier.fillMaxWidth().padding(16.dp)
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Text(
                    text = "Configure Habit Quests",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )

                ToggleRow("Digital Sunset", "Screen-free gap before sleep", sunset, MaterialTheme.colorScheme.primary) {
                    sunset = it
                    prefs.edit().putBoolean("habit_digital_sunset_enabled", it).apply()
                }
                ToggleRow("Circadian Anchor", "Consistent bedtime window", circadian, MaterialTheme.colorScheme.primary) {
                    circadian = it
                    prefs.edit().putBoolean("habit_circadian_anchor_enabled", it).apply()
                }
                ToggleRow("Movement Boost", "Daily physical step goal", movement, MaterialTheme.colorScheme.primary) {
                    movement = it
                    prefs.edit().putBoolean("habit_movement_boost_enabled", it).apply()
                }
                ToggleRow("Focus Mode", "Keep social app ratio healthy", focus, MaterialTheme.colorScheme.primary) {
                    focus = it
                    prefs.edit().putBoolean("habit_focus_mode_enabled", it).apply()
                }

                Button(
                    onClick = onDismiss,
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
                ) {
                    Text("Save & Close", color = Color.Black, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                }
            }
        }
    }
}
