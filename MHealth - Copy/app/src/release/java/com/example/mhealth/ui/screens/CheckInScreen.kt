package com.example.mhealth.ui.screens

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.widget.Toast
import androidx.activity.compose.BackHandler
import androidx.compose.animation.*
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.mhealth.ui.components.Fredoka
import com.example.mhealth.logic.DataRepository
import androidx.compose.runtime.snapshots.SnapshotStateList
import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.sin
import kotlin.math.roundToInt

@Composable
fun CheckInScreen() {
    val context = LocalContext.current
    val prefs = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }
    
    var activeSubScreen by remember { mutableStateOf<String?>(null) } // "guided", "past", "monthly"

    BackHandler(enabled = activeSubScreen != null) {
        activeSubScreen = null
    }

    when (activeSubScreen) {
        "guided" -> GuidedReflectionScreen(prefs = prefs, onBack = { activeSubScreen = null })
        "past" -> PastReflectionsScreen(prefs = prefs, onBack = { activeSubScreen = null })
        "monthly" -> MonthlyWellnessScreen(prefs = prefs, onBack = { activeSubScreen = null })
        else -> CheckInHubScreen(
            prefs = prefs,
            onStartGuided = { activeSubScreen = "guided" },
            onViewPast = { activeSubScreen = "past" },
            onStartMonthly = { activeSubScreen = "monthly" }
        )
    }
}

@Composable
fun CheckInHubScreen(
    prefs: android.content.SharedPreferences,
    onStartGuided: () -> Unit,
    onViewPast: () -> Unit,
    onStartMonthly: () -> Unit
) {
    val todayStr = remember { SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date()) }
    val lastDate = remember(prefs) { prefs.getString("daily_checkin_date_last", "") ?: "" }
    val isDoneToday = lastDate == todayStr

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentPadding = PaddingValues(20.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp)
    ) {
        item {
            Column {
                Text(
                    text = "Lumen.",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.padding(bottom = 4.dp)
                )
                Text(
                    text = "Self Reflection",
                    fontSize = 26.sp,
                    fontWeight = FontWeight.ExtraBold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Text(
                    text = "A quiet space to check in with your mind and evening rhythm",
                    fontSize = 13.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 4.dp)
                )
            }
        }

        // Streak Banner Card
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(22.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primary.copy(0.12f)),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(0.3f))
            ) {
                Row(
                    modifier = Modifier.padding(20.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .size(48.dp)
                            .background(MaterialTheme.colorScheme.primary, CircleShape),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            imageVector = Icons.Default.Bedtime,
                            contentDescription = null,
                            tint = Color.Black,
                            modifier = Modifier.size(24.dp)
                        )
                    }
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = if (isDoneToday) "Today's Pause Completed" else "Evening Pause Ready",
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.onBackground
                        )
                        Text(
                            text = if (isDoneToday) "Your reflection for today is saved." else "Take 2 minutes to pause and journal your evening.",
                            fontSize = 12.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }
        }

        item {
            Text(
                text = "Reflection Actions",
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                color = MaterialTheme.colorScheme.onBackground
            )
        }

        // Card 1: Begin Evening Reflection
        item {
            HubActionCard(
                title = "Begin Evening Reflection",
                subtitle = "Guided 3-step calm check-in with mood, energy, and journal page",
                icon = Icons.Default.SelfImprovement,
                tagText = if (isDoneToday) "Completed" else "Daily",
                onClick = onStartGuided
            )
        }

        // Card 2: View Past Reflections
        item {
            HubActionCard(
                title = "View Past Reflections",
                subtitle = "Browse your calendar heat map and personal journal entries",
                icon = Icons.Default.MenuBook,
                tagText = "Journal History",
                onClick = onViewPast
            )
        }

        // Card 3: Monthly Wellness Check
        item {
            HubActionCard(
                title = "Monthly Wellness Check",
                subtitle = "Comprehensive monthly questionnaire for wellness trends",
                icon = Icons.Default.Assessment,
                tagText = "Monthly",
                onClick = onStartMonthly
            )
        }
    }
}

@Composable
fun HubActionCard(
    title: String,
    subtitle: String,
    icon: ImageVector,
    tagText: String,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick() },
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
    ) {
        Row(
            modifier = Modifier.padding(18.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(42.dp)
                    .background(MaterialTheme.colorScheme.primary.copy(0.12f), CircleShape),
                contentAlignment = Alignment.Center
            ) {
                Icon(icon, null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(22.dp))
            }

            Column(modifier = Modifier.weight(1f)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = title,
                        fontSize = 15.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground,
                        modifier = Modifier.weight(1f, fill = false)
                    )
                    Spacer(Modifier.width(8.dp))
                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = MaterialTheme.colorScheme.primary.copy(0.12f)
                    ) {
                        Text(
                            text = tagText,
                            fontSize = 10.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.primary,
                            maxLines = 1,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp)
                        )
                    }
                }
                Spacer(Modifier.height(4.dp))
                Text(
                    text = subtitle,
                    fontSize = 11.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    lineHeight = 15.sp
                )
            }
        }
    }
}

@Composable
fun GuidedReflectionScreen(
    prefs: android.content.SharedPreferences,
    onBack: () -> Unit
) {
    val context = LocalContext.current
    var currentStep by remember { mutableIntStateOf(1) } // Step 1, 2, 3

    var mood by remember { mutableIntStateOf(3) }
    var energy by remember { mutableIntStateOf(3) }
    var anxiety by remember { mutableIntStateOf(3) }
    var sleepQuality by remember { mutableIntStateOf(3) }
    var journalNote by remember { mutableStateOf("") }

    // Ambient Calm Synth Music Player
    var isMusicPlaying by remember { mutableStateOf(false) }
    val audioTrack = remember {
        try {
            val sampleRate = 22050
            val numSamples = sampleRate * 4
            val buffer = ShortArray(numSamples)
            val baseFreq = 174.0 // Solfeggio healing frequency 174 Hz
            val harmonics = doubleArrayOf(1.0, 1.5, 2.0)
            
            for (i in 0 until numSamples) {
                val t = i.toDouble() / sampleRate
                val env = sin(Math.PI * i / numSamples)
                var valD = 0.0
                for (h in harmonics) {
                    valD += sin(2.0 * Math.PI * baseFreq * h * t) * (0.3 / h)
                }
                buffer[i] = (valD * env * 32767).toInt().coerceIn(-32768, 32767).toShort()
            }
            
            AudioTrack.Builder()
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                        .build()
                )
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(sampleRate)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build()
                )
                .setBufferSizeInBytes(buffer.size * 2)
                .setTransferMode(AudioTrack.MODE_STATIC)
                .build().apply {
                    write(buffer, 0, buffer.size)
                    setLoopPoints(0, buffer.size, -1)
                }
        } catch (e: Exception) {
            null
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            try {
                audioTrack?.stop()
                audioTrack?.release()
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    val todayFullDate = remember {
        SimpleDateFormat("EEEE, MMMM d, yyyy", Locale.getDefault()).format(Date())
    }

    val todayShortDate = remember {
        SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date())
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .padding(20.dp),
        verticalArrangement = Arrangement.SpaceBetween
    ) {
        // Top Navigation & Music Bar
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                IconButton(onClick = onBack) {
                    Icon(Icons.Default.ArrowBack, "Back")
                }
                Text(
                    text = "Your Evening Pause",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.ExtraBold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
            }

            IconButton(
                onClick = {
                    if (isMusicPlaying) {
                        audioTrack?.pause()
                        isMusicPlaying = false
                    } else {
                        audioTrack?.play()
                        isMusicPlaying = true
                    }
                }
            ) {
                Icon(
                    imageVector = if (isMusicPlaying) Icons.Default.VolumeUp else Icons.Default.VolumeOff,
                    contentDescription = "Ambient Music Toggle",
                    tint = if (isMusicPlaying) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }

        // Progress indicator dots
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.CenterVertically
        ) {
            for (step in 1..3) {
                val isSel = step == currentStep
                Box(
                    modifier = Modifier
                        .padding(horizontal = 4.dp)
                        .size(if (isSel) 24.dp else 10.dp, 10.dp)
                        .clip(CircleShape)
                        .background(if (isSel) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant)
                )
            }
        }

        // Content per step
        Box(modifier = Modifier.weight(1f).padding(vertical = 16.dp)) {
            when (currentStep) {
                1 -> Step1MoodEnergy(mood = mood, energy = energy, onMoodChange = { mood = it }, onEnergyChange = { energy = it })
                2 -> Step2AnxietyRest(anxiety = anxiety, rest = sleepQuality, onAnxietyChange = { anxiety = it }, onRestChange = { sleepQuality = it })
                3 -> Step3JournalEntry(dateText = todayFullDate, note = journalNote, onNoteChange = { journalNote = it })
            }
        }

        // Bottom Navigation Buttons
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            if (currentStep > 1) {
                OutlinedButton(
                    onClick = { currentStep-- },
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Text("Previous", fontFamily = Fredoka, fontWeight = FontWeight.Bold)
                }
            } else {
                Spacer(Modifier.width(100.dp))
            }

            if (currentStep < 3) {
                Button(
                    onClick = { currentStep++ },
                    shape = RoundedCornerShape(12.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
                ) {
                    Text("Next", color = Color.Black, fontFamily = Fredoka, fontWeight = FontWeight.Bold)
                }
            } else {
                Button(
                    onClick = {
                        prefs.edit().putString("daily_checkin_date_last", todayShortDate).apply()
                        com.example.mhealth.saveCheckinToHistory(
                            prefs = prefs,
                            mood = mood,
                            energy = energy,
                            sleep = sleepQuality,
                            anxiety = anxiety,
                            note = journalNote
                        )
                        Toast.makeText(context, "Evening reflection saved to journal", Toast.LENGTH_SHORT).show()
                        onBack()
                    },
                    shape = RoundedCornerShape(12.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
                ) {
                    Text("Save Entry", color = Color.Black, fontFamily = Fredoka, fontWeight = FontWeight.Bold)
                }
            }
        }
    }
}

@Composable
fun Step1MoodEnergy(
    mood: Int,
    energy: Int,
    onMoodChange: (Int) -> Unit,
    onEnergyChange: (Int) -> Unit
) {
    Column(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(20.dp)
    ) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(22.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
        ) {
            Column(
                modifier = Modifier.padding(horizontal = 14.dp, vertical = 18.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Text(
                    text = "How has your mood felt today?",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )

                ThemedRatingSelector(
                    selectedLevel = mood,
                    labels = listOf("Challenging", "Low", "Balanced", "Good", "Great"),
                    icons = listOf(Icons.Default.DarkMode, Icons.Default.Cloud, Icons.Default.WbSunny, Icons.Default.LightMode, Icons.Default.AutoAwesome),
                    onSelect = onMoodChange
                )
            }
        }

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(22.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
        ) {
            Column(
                modifier = Modifier.padding(horizontal = 14.dp, vertical = 18.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Text(
                    text = "What was your energy level?",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )

                ThemedRatingSelector(
                    selectedLevel = energy,
                    labels = listOf("Depleted", "Low", "Steady", "High", "Vibrant"),
                    icons = listOf(Icons.Default.Battery1Bar, Icons.Default.Battery3Bar, Icons.Default.Battery5Bar, Icons.Default.BatteryFull, Icons.Default.Bolt),
                    onSelect = onEnergyChange
                )
            }
        }
    }
}

@Composable
fun Step2AnxietyRest(
    anxiety: Int,
    rest: Int,
    onAnxietyChange: (Int) -> Unit,
    onRestChange: (Int) -> Unit
) {
    Column(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(20.dp)
    ) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(22.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
        ) {
            Column(
                modifier = Modifier.padding(horizontal = 14.dp, vertical = 18.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Text(
                    text = "Did you experience tension or anxiety?",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )

                ThemedRatingSelector(
                    selectedLevel = anxiety,
                    labels = listOf("Calm", "Mild", "Moderate", "High", "Severe"),
                    icons = listOf(Icons.Default.Spa, Icons.Default.Air, Icons.Default.Grain, Icons.Default.Tsunami, Icons.Default.Storm),
                    onSelect = onAnxietyChange
                )
            }
        }

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(22.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
        ) {
            Column(
                modifier = Modifier.padding(horizontal = 14.dp, vertical = 18.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Text(
                    text = "How was your rest quality?",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )

                ThemedRatingSelector(
                    selectedLevel = rest,
                    labels = listOf("Restless", "Poor", "Fair", "Good", "Deep"),
                    icons = listOf(Icons.Default.NightsStay, Icons.Default.Bed, Icons.Default.SingleBed, Icons.Default.Hotel, Icons.Default.KingBed),
                    onSelect = onRestChange
                )
            }
        }
    }
}

@Composable
fun Step3JournalEntry(
    dateText: String,
    note: String,
    onNoteChange: (String) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxSize(),
        shape = RoundedCornerShape(24.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(0.25f))
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            Text(
                text = dateText,
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                color = MaterialTheme.colorScheme.primary
            )

            Text(
                text = "Journal Reflection Entry",
                fontSize = 18.sp,
                fontWeight = FontWeight.ExtraBold,
                fontFamily = Fredoka,
                color = MaterialTheme.colorScheme.onBackground
            )

            Text(
                text = "Write down any thoughts, highlights, or feelings from your day in your diary.",
                fontSize = 12.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            OutlinedTextField(
                value = note,
                onValueChange = onNoteChange,
                placeholder = {
                    Text(
                        text = "Dear Journal, today felt...",
                        fontSize = 13.sp,
                        fontStyle = FontStyle.Italic
                    )
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                shape = RoundedCornerShape(16.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = MaterialTheme.colorScheme.primary,
                    unfocusedBorderColor = MaterialTheme.colorScheme.outline.copy(0.2f)
                )
            )
        }
    }
}

@Composable
fun ThemedRatingSelector(
    selectedLevel: Int,
    labels: List<String>,
    icons: List<ImageVector>,
    onSelect: (Int) -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        for (i in 1..5) {
            val isSel = selectedLevel == i
            val label = labels.getOrElse(i - 1) { "" }
            val icon = icons.getOrElse(i - 1) { Icons.Default.Star }

            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(6.dp),
                modifier = Modifier
                    .weight(1f)
                    .clickable { onSelect(i) }
            ) {
                Box(
                    modifier = Modifier
                        .size(42.dp)
                        .clip(CircleShape)
                        .background(if (isSel) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant.copy(0.4f))
                        .border(1.5.dp, if (isSel) MaterialTheme.colorScheme.primary else Color.Transparent, CircleShape),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = icon,
                        contentDescription = label,
                        tint = if (isSel) Color.Black else MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.size(20.dp)
                    )
                }
                Text(
                    text = label,
                    fontSize = 9.sp,
                    fontWeight = if (isSel) FontWeight.Bold else FontWeight.Normal,
                    fontFamily = Fredoka,
                    color = if (isSel) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = TextAlign.Center,
                    maxLines = 1,
                    softWrap = false,
                    letterSpacing = (-0.2).sp
                )
            }
        }
    }
}

@Composable
fun PastReflectionsScreen(
    prefs: android.content.SharedPreferences,
    onBack: () -> Unit
) {
    val checkinHistoryStr = remember(prefs) { prefs.getString("daily_checkin_history", "[]") ?: "[]" }
    val records = remember(checkinHistoryStr) {
        val list = mutableListOf<CheckinRecord>()
        try {
            val arr = JSONArray(checkinHistoryStr)
            for (i in 0 until arr.length()) {
                val obj = arr.getJSONObject(i)
                list.add(
                    CheckinRecord(
                        date = obj.optString("date", ""),
                        mood = obj.optInt("mood", 3),
                        anxiety = obj.optInt("anxiety", 3),
                        energy = obj.optInt("energy", 3),
                        sleep = obj.optInt("sleep", 3),
                        note = obj.optString("note", "")
                    )
                )
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        list.reversed()
    }

    var selectedRecord by remember { mutableStateOf<CheckinRecord?>(null) }

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
                    text = "Past Reflections Journal",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.ExtraBold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
            }
        }

        if (selectedRecord != null) {
            item {
                JournalPageCard(record = selectedRecord!!, onClose = { selectedRecord = null })
            }
        }

        item {
            Text(
                text = "Reflection History Logs",
                fontSize = 15.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                color = MaterialTheme.colorScheme.onBackground
            )
        }

        if (records.isEmpty()) {
            item {
                Text(
                    text = "No journal reflections logged yet. Completed check-ins will appear here.",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        } else {
            items(records.size) { idx ->
                val rec = records[idx]
                val formattedDate = remember(rec.date) {
                    try {
                        val d = SimpleDateFormat("yyyy-MM-dd", Locale.US).parse(rec.date)
                        SimpleDateFormat("EEEE, MMMM d, yyyy", Locale.getDefault()).format(d!!)
                    } catch (e: Exception) {
                        rec.date
                    }
                }

                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { selectedRecord = rec },
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Text(
                            text = formattedDate,
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.primary
                        )
                        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            Surface(shape = RoundedCornerShape(6.dp), color = MaterialTheme.colorScheme.primary.copy(0.12f)) {
                                Text("Mood: ${rec.mood}/5", fontSize = 11.sp, fontFamily = Fredoka, modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp), color = MaterialTheme.colorScheme.primary)
                            }
                            Surface(shape = RoundedCornerShape(6.dp), color = MaterialTheme.colorScheme.surfaceVariant) {
                                Text("Energy: ${rec.energy}/5", fontSize = 11.sp, fontFamily = Fredoka, modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp))
                            }
                        }
                        if (rec.note.isNotBlank()) {
                            Text(
                                text = "\"${rec.note}\"",
                                fontSize = 12.sp,
                                fontStyle = FontStyle.Italic,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun JournalPageCard(
    record: CheckinRecord,
    onClose: () -> Unit
) {
    val formattedDate = remember(record.date) {
        try {
            val d = SimpleDateFormat("yyyy-MM-dd", Locale.US).parse(record.date)
            SimpleDateFormat("EEEE, MMMM d, yyyy", Locale.getDefault()).format(d!!)
        } catch (e: Exception) {
            record.date
        }
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(22.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.5.dp, MaterialTheme.colorScheme.primary.copy(0.4f))
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
                Text(
                    text = formattedDate,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.primary
                )
                IconButton(onClick = onClose) {
                    Icon(Icons.Default.Close, null)
                }
            }

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Surface(shape = RoundedCornerShape(8.dp), color = MaterialTheme.colorScheme.primary.copy(0.15f)) {
                    Text("Mood: ${record.mood}/5", fontSize = 12.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka, modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp), color = MaterialTheme.colorScheme.primary)
                }
                Surface(shape = RoundedCornerShape(8.dp), color = MaterialTheme.colorScheme.surfaceVariant) {
                    Text("Anxiety: ${record.anxiety}/5", fontSize = 12.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka, modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp))
                }
                Surface(shape = RoundedCornerShape(8.dp), color = MaterialTheme.colorScheme.surfaceVariant) {
                    Text("Rest: ${record.sleep}/5", fontSize = 12.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka, modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp))
                }
            }

            Surface(
                shape = RoundedCornerShape(14.dp),
                color = MaterialTheme.colorScheme.surfaceVariant.copy(0.3f),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "Journal Entry",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.padding(bottom = 6.dp)
                    )
                    Text(
                        text = if (record.note.isNotBlank()) "\"${record.note}\"" else "No journal note written for this entry.",
                        fontSize = 13.sp,
                        fontStyle = FontStyle.Italic,
                        lineHeight = 18.sp,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                }
            }
        }
    }
}

fun getMonthlyCooldownDays(prefs: android.content.SharedPreferences): Int {
    val lastDateStr = prefs.getString("monthly_checkin_last_date", "") ?: ""
    if (lastDateStr.isEmpty()) return 0
    return try {
        val sdf = SimpleDateFormat("yyyy-MM-dd", Locale.US)
        val lastDate = sdf.parse(lastDateStr) ?: return 0
        val today = Calendar.getInstance().apply {
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }.time
        val diffMs = today.time - lastDate.time
        val diffDays = diffMs / (1000 * 60 * 60 * 24)
        (30 - diffDays).toInt().coerceAtLeast(0)
    } catch (e: Exception) {
        0
    }
}

data class MonthlyCheckinEntry(
    val date: String,
    val phqScore: Int,
    val gadScore: Int,
    val lifeEventsCount: Int,
    val note: String
)

@Composable
fun MonthlyWellnessScreen(
    prefs: android.content.SharedPreferences,
    onBack: () -> Unit
) {
    val context = LocalContext.current
    var isWizardActive by remember { mutableStateOf(false) }
    var wizardStep by remember { mutableIntStateOf(1) } // 1: Mood, 2: Anxiety, 3: Stressors, 4: Notes
    
    val cooldownDays = remember(isWizardActive) { getMonthlyCooldownDays(prefs) }

    val phqAnswers = remember { mutableStateListOf(*Array(5) { -1 }) }
    val gadAnswers = remember { mutableStateListOf(*Array(5) { -1 }) }
    val selectedStressors = remember { mutableStateMapOf<Int, Boolean>() }
    var monthlyNote by remember { mutableStateOf("") }

    val phqQuestions = listOf(
        "Little interest or pleasure in doing daily activities",
        "Feeling down, low vitality, or discouraged",
        "Trouble falling or staying asleep, or sleeping excessively",
        "Feeling exhausted or having diminished physical energy",
        "Noticeable change in appetite or eating routine"
    )

    val gadQuestions = listOf(
        "Feeling restless, tense, or on edge",
        "Trouble unwinding or turning off persistent worries",
        "Worrying excessively about different aspects of routine",
        "Difficulty focusing or finding a calm mental space",
        "Becoming easily irritable or sensitive to small disruptions"
    )

    val optionsList = listOf("Not at all", "Several days", "More than half the days", "Nearly every day")

    val stressorsList = listOf(
        "Heavy work or academic deadlines / pressure",
        "Changes in home, living situation, or major travel",
        "Interpersonal, family, or relationship tension",
        "Health concern, physical illness, or fatigue recovery",
        "Irregular sleep schedule or major routine disruption",
        "Unexpected financial or personal lifestyle friction",
        "None of the above (smooth and steady routine)"
    )

    val historyStr = remember(isWizardActive) { prefs.getString("monthly_checkin_history", "[]") ?: "[]" }
    val monthlyHistory = remember(historyStr) {
        val list = mutableListOf<MonthlyCheckinEntry>()
        try {
            val arr = JSONArray(historyStr)
            for (i in 0 until arr.length()) {
                val obj = arr.getJSONObject(i)
                list.add(
                    MonthlyCheckinEntry(
                        date = obj.optString("date", ""),
                        phqScore = obj.optInt("phq", 0),
                        gadScore = obj.optInt("gad", 0),
                        lifeEventsCount = obj.optInt("events", 0),
                        note = obj.optString("note", "")
                    )
                )
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        list.reversed()
    }

    if (isWizardActive) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colorScheme.background)
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Wizard Header with Step Indicator
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    IconButton(onClick = {
                        if (wizardStep > 1) wizardStep-- else isWizardActive = false
                    }) {
                        Icon(Icons.Default.ArrowBack, "Back")
                    }
                    Text(
                        text = "Monthly Assessment",
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                }
                Surface(
                    shape = RoundedCornerShape(8.dp),
                    color = MaterialTheme.colorScheme.primary.copy(0.12f)
                ) {
                    Text(
                        text = "Step $wizardStep of 4",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
                    )
                }
            }

            LinearProgressIndicator(
                progress = { wizardStep / 4f },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(6.dp)
                    .clip(CircleShape),
                color = MaterialTheme.colorScheme.primary,
                trackColor = MaterialTheme.colorScheme.outline.copy(0.15f)
            )

            when (wizardStep) {
                1 -> {
                    ScreenerStepContent(
                        categoryTitle = "Mood & Vitality (PHQ)",
                        description = "Over the last 2 weeks, how often have you noticed the following?",
                        questions = phqQuestions,
                        answers = phqAnswers,
                        options = optionsList,
                        onNext = {
                            val allAnswered = phqAnswers.all { it >= 0 }
                            if (allAnswered) {
                                wizardStep = 2
                            } else {
                                Toast.makeText(context, "Please answer all questions to proceed", Toast.LENGTH_SHORT).show()
                            }
                        }
                    )
                }
                2 -> {
                    ScreenerStepContent(
                        categoryTitle = "Anxiety & Calm Rhythm (GAD)",
                        description = "Over the last 2 weeks, how often have you been bothered by the following?",
                        questions = gadQuestions,
                        answers = gadAnswers,
                        options = optionsList,
                        onNext = {
                            val allAnswered = gadAnswers.all { it >= 0 }
                            if (allAnswered) {
                                wizardStep = 3
                            } else {
                                Toast.makeText(context, "Please answer all questions to proceed", Toast.LENGTH_SHORT).show()
                            }
                        }
                    )
                }
                3 -> {
                    LifeEventsStepContent(
                        stressors = stressorsList,
                        selectedStressors = selectedStressors,
                        onNext = { wizardStep = 4 }
                    )
                }
                4 -> {
                    MonthlyReflectionNoteStep(
                        note = monthlyNote,
                        onNoteChange = { monthlyNote = it },
                        onSubmit = {
                            val totalPhq = (phqAnswers.sum() * 9f / 5f).roundToInt()
                            val totalGad = (gadAnswers.sum() * 7f / 5f).roundToInt()
                            val totalEvents = selectedStressors.count { it.value && it.key != (stressorsList.size - 1) }
                            val todayStr = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date())

                            DataRepository.saveScreenerScores(totalPhq, totalGad, totalEvents)
                            prefs.edit().putString("monthly_checkin_last_date", todayStr).apply()
                            prefs.edit().putString("monthly_checkin_notification_sent_date", todayStr).apply()

                            // Save to monthly history
                            try {
                                val currentArr = JSONArray(prefs.getString("monthly_checkin_history", "[]") ?: "[]")
                                val newEntry = JSONObject().apply {
                                    put("date", todayStr)
                                    put("phq", totalPhq)
                                    put("gad", totalGad)
                                    put("events", totalEvents)
                                    put("note", monthlyNote.trim())
                                }
                                currentArr.put(newEntry)
                                prefs.edit().putString("monthly_checkin_history", currentArr.toString()).apply()
                            } catch (e: Exception) {
                                e.printStackTrace()
                            }

                            Toast.makeText(context, "Monthly assessment recorded successfully!", Toast.LENGTH_SHORT).show()
                            isWizardActive = false
                            wizardStep = 1
                        }
                    )
                }
            }
        }
    } else {
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
                        text = "Monthly Wellness Check",
                        fontSize = 20.sp,
                        fontWeight = FontWeight.ExtraBold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                }
            }

            // Assessment Status Card
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(22.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = if (cooldownDays > 0) MaterialTheme.colorScheme.surface else MaterialTheme.colorScheme.primary.copy(0.12f)
                    ),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(if (cooldownDays > 0) 0.15f else 0.35f))
                ) {
                    Column(
                        modifier = Modifier.padding(20.dp),
                        verticalArrangement = Arrangement.spacedBy(14.dp)
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(44.dp)
                                    .background(MaterialTheme.colorScheme.primary.copy(0.2f), CircleShape),
                                contentAlignment = Alignment.Center
                            ) {
                                Icon(
                                    imageVector = if (cooldownDays > 0) Icons.Default.HourglassTop else Icons.Default.Assessment,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.primary,
                                    modifier = Modifier.size(24.dp)
                                )
                            }
                            Column {
                                Text(
                                    text = if (cooldownDays > 0) "Assessment Calibrated" else "Monthly Screener Ready",
                                    fontSize = 17.sp,
                                    fontWeight = FontWeight.Bold,
                                    fontFamily = Fredoka,
                                    color = MaterialTheme.colorScheme.onBackground
                                )
                                Text(
                                    text = if (cooldownDays > 0) "Next check-in in $cooldownDays days" else "Take 3 minutes to log your monthly trends",
                                    fontSize = 12.sp,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }

                        Text(
                            text = if (cooldownDays > 0)
                                "Lumen uses your monthly wellness assessments to calibrate tracking thresholds and align with your usual rhythm."
                            else
                                "Standard monthly reflection questionnaires provide a longitudinal picture of your mood and anxiety stability.",
                            fontSize = 12.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            lineHeight = 17.sp
                        )

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(10.dp)
                        ) {
                            Button(
                                onClick = {
                                    // Reset questionnaire state and begin
                                    phqAnswers.fill(-1)
                                    gadAnswers.fill(-1)
                                    selectedStressors.clear()
                                    monthlyNote = ""
                                    wizardStep = 1
                                    isWizardActive = true
                                },
                                shape = RoundedCornerShape(12.dp),
                                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                                modifier = Modifier.weight(1f)
                            ) {
                                Text(
                                    text = if (cooldownDays > 0) "Retake Assessment" else "Begin Assessment",
                                    color = Color.Black,
                                    fontWeight = FontWeight.Bold,
                                    fontFamily = Fredoka,
                                    fontSize = 13.sp
                                )
                            }
                        }
                    }
                }
            }

            // Historical Logs Header
            item {
                Text(
                    text = "Assessment History",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground,
                    modifier = Modifier.padding(top = 4.dp)
                )
            }

            if (monthlyHistory.isEmpty()) {
                item {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(16.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
                    ) {
                        Column(
                            modifier = Modifier.padding(20.dp),
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Text("📋", fontSize = 28.sp)
                            Text(
                                text = "No monthly check-ins recorded yet",
                                fontSize = 13.sp,
                                fontWeight = FontWeight.Bold,
                                fontFamily = Fredoka,
                                color = MaterialTheme.colorScheme.onBackground
                            )
                            Text(
                                text = "When you complete your 30-day assessment, your wellness trajectory and reflection notes will appear here.",
                                fontSize = 11.sp,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                textAlign = TextAlign.Center
                            )
                        }
                    }
                }
            } else {
                items(monthlyHistory.size) { idx ->
                    val entry = monthlyHistory[idx]
                    MonthlyHistoryCard(entry = entry)
                }
            }
        }
    }
}

@Composable
fun ScreenerStepContent(
    categoryTitle: String,
    description: String,
    questions: List<String>,
    answers: SnapshotStateList<Int>,
    options: List<String>,
    onNext: () -> Unit
) {
    Column(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        Column(modifier = Modifier.fillMaxWidth()) {
            Text(
                text = categoryTitle,
                fontSize = 17.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                color = MaterialTheme.colorScheme.primary
            )
            Text(
                text = description,
                fontSize = 12.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 2.dp)
            )
        }

        LazyColumn(
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            items(questions.size) { qIdx ->
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
                ) {
                    Column(
                        modifier = Modifier.padding(14.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Text(
                            text = "${qIdx + 1}. ${questions[qIdx]}",
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.onBackground
                        )

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            options.forEachIndexed { score, optText ->
                                val isSelected = answers[qIdx] == score
                                Surface(
                                    modifier = Modifier
                                        .weight(1f)
                                        .clickable { answers[qIdx] = score },
                                    shape = RoundedCornerShape(10.dp),
                                    color = if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant.copy(0.4f),
                                    border = BorderStroke(1.dp, if (isSelected) MaterialTheme.colorScheme.primary else Color.Transparent)
                                ) {
                                    Text(
                                        text = optText,
                                        fontSize = 9.sp,
                                        fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                                        fontFamily = Fredoka,
                                        color = if (isSelected) Color.Black else MaterialTheme.colorScheme.onSurfaceVariant,
                                        textAlign = TextAlign.Center,
                                        modifier = Modifier.padding(vertical = 8.dp, horizontal = 2.dp),
                                        maxLines = 2,
                                        lineHeight = 11.sp
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }

        Button(
            onClick = onNext,
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
        ) {
            Text("Next Step", color = Color.Black, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
        }
    }
}

@Composable
fun LifeEventsStepContent(
    stressors: List<String>,
    selectedStressors: androidx.compose.runtime.snapshots.SnapshotStateMap<Int, Boolean>,
    onNext: () -> Unit
) {
    Column(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        Column(modifier = Modifier.fillMaxWidth()) {
            Text(
                text = "Life Events & Stressors",
                fontSize = 17.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                color = MaterialTheme.colorScheme.primary
            )
            Text(
                text = "Have you experienced any of these during the past 14 to 30 days?",
                fontSize = 12.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 2.dp)
            )
        }

        LazyColumn(
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(stressors.size) { idx ->
                val label = stressors[idx]
                val selected = selectedStressors[idx] ?: false
                val isNoneOption = idx == stressors.size - 1

                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable {
                            if (isNoneOption) {
                                selectedStressors.clear()
                                selectedStressors[idx] = true
                            } else {
                                selectedStressors[stressors.size - 1] = false
                                selectedStressors[idx] = !selected
                            }
                        },
                    shape = RoundedCornerShape(12.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = if (selected) MaterialTheme.colorScheme.primary.copy(0.08f) else MaterialTheme.colorScheme.surface
                    ),
                    border = BorderStroke(
                        1.dp,
                        if (selected) MaterialTheme.colorScheme.primary.copy(0.4f) else MaterialTheme.colorScheme.outline.copy(0.12f)
                    )
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 14.dp, vertical = 12.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Checkbox(
                            checked = selected,
                            onCheckedChange = { isChecked ->
                                if (isNoneOption) {
                                    selectedStressors.clear()
                                    selectedStressors[idx] = isChecked
                                } else {
                                    selectedStressors[stressors.size - 1] = false
                                    selectedStressors[idx] = isChecked
                                }
                            },
                            colors = CheckboxDefaults.colors(checkedColor = MaterialTheme.colorScheme.primary)
                        )
                        Text(
                            text = label,
                            fontSize = 13.sp,
                            fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.onBackground
                        )
                    }
                }
            }
        }

        Button(
            onClick = onNext,
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
        ) {
            Text("Continue to Notes", color = Color.Black, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
        }
    }
}

@Composable
fun MonthlyReflectionNoteStep(
    note: String,
    onNoteChange: (String) -> Unit,
    onSubmit: () -> Unit
) {
    Column(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        Column(modifier = Modifier.fillMaxWidth()) {
            Text(
                text = "Monthly Reflection & Notes",
                fontSize = 17.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                color = MaterialTheme.colorScheme.primary
            )
            Text(
                text = "Add any personal highlights, routine changes, or reflection thoughts from this past month.",
                fontSize = 12.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 2.dp)
            )
        }

        Card(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.15f))
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp)
            ) {
                OutlinedTextField(
                    value = note,
                    onValueChange = { if (it.length <= 1000) onNoteChange(it) },
                    placeholder = {
                        Text(
                            text = "Reflecting on this month: my energy, routines, and headspace felt...",
                            fontSize = 13.sp,
                            fontStyle = FontStyle.Italic,
                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(0.6f)
                        )
                    },
                    modifier = Modifier.fillMaxSize(),
                    shape = RoundedCornerShape(12.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = MaterialTheme.colorScheme.primary,
                        unfocusedBorderColor = MaterialTheme.colorScheme.outline.copy(0.2f)
                    )
                )
            }
        }

        Button(
            onClick = onSubmit,
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
        ) {
            Icon(Icons.Default.Check, null, tint = Color.Black, modifier = Modifier.size(18.dp))
            Spacer(Modifier.width(6.dp))
            Text("Complete & Save Assessment", color = Color.Black, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
        }
    }
}

@Composable
fun MonthlyHistoryCard(entry: MonthlyCheckinEntry) {
    val formattedDate = remember(entry.date) {
        try {
            val d = SimpleDateFormat("yyyy-MM-dd", Locale.US).parse(entry.date)
            SimpleDateFormat("MMMM yyyy", Locale.getDefault()).format(d!!)
        } catch (e: Exception) {
            entry.date
        }
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
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
                    fontSize = 15.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.primary
                )
                Surface(
                    shape = RoundedCornerShape(6.dp),
                    color = MaterialTheme.colorScheme.primary.copy(0.12f)
                ) {
                    Text(
                        text = "Completed",
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp)
                    )
                }
            }

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Surface(
                    shape = RoundedCornerShape(8.dp),
                    color = MaterialTheme.colorScheme.surfaceVariant.copy(0.5f)
                ) {
                    Text(
                        text = "Mood Score: ${entry.phqScore}/27",
                        fontSize = 11.sp,
                        fontFamily = Fredoka,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Surface(
                    shape = RoundedCornerShape(8.dp),
                    color = MaterialTheme.colorScheme.surfaceVariant.copy(0.5f)
                ) {
                    Text(
                        text = "Anxiety Score: ${entry.gadScore}/21",
                        fontSize = 11.sp,
                        fontFamily = Fredoka,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                if (entry.lifeEventsCount > 0) {
                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = MaterialTheme.colorScheme.primary.copy(0.15f)
                    ) {
                        Text(
                            text = "${entry.lifeEventsCount} Stressors",
                            fontSize = 11.sp,
                            fontFamily = Fredoka,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                }
            }

            if (entry.note.isNotBlank()) {
                Text(
                    text = "\"${entry.note}\"",
                    fontSize = 12.sp,
                    fontStyle = FontStyle.Italic,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    lineHeight = 16.sp
                )
            }
        }
    }
}
