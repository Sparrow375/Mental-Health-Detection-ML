package com.example.mhealth.ui.screens

import android.content.Context
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
import com.example.mhealth.models.PersonalityVector
import com.example.mhealth.ui.components.AlertWarning
import com.example.mhealth.ui.components.AlertRose
import com.example.mhealth.ui.components.StaggeredFadeIn
import com.example.mhealth.ui.components.ToggleRow
import com.example.mhealth.ui.components.Fredoka
import com.example.mhealth.ui.components.rememberNavBarPadding
import kotlinx.coroutines.delay

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
                    .statusBarsPadding(),
                contentAlignment = Alignment.Center
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.SpaceBetween,
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(top = 48.dp, bottom = (navBarPad2.value + 24).dp, start = 24.dp, end = 24.dp)
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

// =============================================================================
// Insights Screen Composable
