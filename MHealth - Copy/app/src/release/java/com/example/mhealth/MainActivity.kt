package com.example.mhealth

import android.Manifest
import android.app.Activity
import android.app.AppOpsManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Process
import android.provider.Settings
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ShowChart
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.snapshots.SnapshotStateList
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.mhealth.logic.DataRepository
import com.example.mhealth.logic.db.AnalysisResultEntity
import com.example.mhealth.logic.db.MHealthDatabase
import com.example.mhealth.logic.db.UserProfileEntity
import com.example.mhealth.models.DailyReport
import com.example.mhealth.models.PersonalityVector
import com.example.mhealth.services.MHealthAccessibilityService
import com.example.mhealth.services.MHealthNotificationListenerService
import com.example.mhealth.ui.charts.*
import com.example.mhealth.ui.theme.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        
        // Synchronously initialize the local data repository
        DataRepository.init(applicationContext)
        
        setContent {
            CoveTheme {
                LumenAppShell()
            }
        }
    }
}

// =============================================================================
// Fredoka Premium Rounded Typography
// =============================================================================
val Fredoka = FontFamily(
    Font(R.font.fredoka, FontWeight.Normal),
    Font(R.font.fredoka, FontWeight.Bold),
    Font(R.font.fredoka, FontWeight.Medium),
    Font(R.font.fredoka, FontWeight.SemiBold)
)

// =============================================================================
// App Navigation & Layout
// =============================================================================
enum class LumenDest(val label: String, val icon: ImageVector) {
    HOME("Sensors", Icons.Default.Sensors),
    MONITOR("Monitor", Icons.AutoMirrored.Filled.ShowChart),
    ANALYSIS("Analysis", Icons.Default.Analytics),
    INSIGHTS("Insights", Icons.Default.Lightbulb),
    SETTINGS("Settings", Icons.Default.Settings)
}

enum class LumenNavState {
    ONBOARDING,
    DASHBOARD
}

@Composable
fun LumenAppShell() {
    val firstLoginComplete by DataRepository.firstLoginComplete.collectAsState()
    
    var appState by remember {
        mutableStateOf(
            if (DataRepository.firstLoginComplete.value) LumenNavState.DASHBOARD else LumenNavState.ONBOARDING
        )
    }

    when (appState) {
        LumenNavState.ONBOARDING -> OnboardingWizard(onComplete = {
            appState = LumenNavState.DASHBOARD
        })
        LumenNavState.DASHBOARD -> MainLumenDashboard()
    }
}

// =============================================================================
// Onboarding Wizard (Profile + GPS + Screeners + Calibration)
// =============================================================================
@Composable
fun OnboardingWizard(onComplete: () -> Unit) {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    
    var step by remember { mutableIntStateOf(1) } // 1: Splash, 2: Demographics, 3: Home GPS, 4: PHQ-9, 5: GAD-7, 6: Stressors, 7: Finalize
    
    // Step 2 State
    var name by remember { mutableStateOf("") }
    var gender by remember { mutableStateOf("") }
    var age by remember { mutableStateOf("") }
    var profession by remember { mutableStateOf("") }
    var country by remember { mutableStateOf("") }
    var showErrors by remember { mutableStateOf(false) }

    // Step 3 State
    var homeCapturing by remember { mutableStateOf(false) }
    var homeSet by remember { mutableStateOf(DataRepository.homeLocation.value != null) }

    // Step 4: PHQ-9 Screener Answers (0-3 for each of the 9 items)
    val phq9Answers = remember { mutableStateListOf(*Array(9) { -1 }) }
    val phq9Questions = listOf(
        "Little interest or pleasure in doing things.",
        "Feeling down, depressed, or hopeless.",
        "Trouble falling or staying asleep, or sleeping too much.",
        "Feeling tired or having little energy.",
        "Poor appetite or overeating.",
        "Feeling bad about yourself — or that you are a failure or have let yourself or your family down.",
        "Trouble concentrating on things, such as reading the newspaper or watching television.",
        "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual.",
        "Thoughts that you would be better off dead or of hurting yourself in some way."
    )

    // Step 5: GAD-7 Screener Answers (0-3 for each of the 7 items)
    val gad7Answers = remember { mutableStateListOf(*Array(7) { -1 }) }
    val gad7Questions = listOf(
        "Feeling nervous, anxious, or on edge.",
        "Not being able to stop or control worrying.",
        "Worrying too much about different things.",
        "Trouble relaxing.",
        "Being so restless that it is hard to sit still.",
        "Becoming easily annoyed or irritable.",
        "Feeling afraid as if something awful might happen."
    )

    // Step 6: Life Events Checkbox counts
    val stressors = listOf(
        "Major health change or severe illness of family member/friend",
        "Loss of a close relative, family member, or friend",
        "Relocation, changing housing, or moving to a new city",
        "Job loss, career transition, or severe financial stress",
        "Severe relationship conflict, divorce, or academic setback",
        "None of the above / Stable environments"
    )
    val selectedStressors = remember { mutableStateMapOf<Int, Boolean>() }

    val optionsList = listOf("Not at all", "Several days", "More than half the days", "Nearly every day")

    Column(
        Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
    ) {
        // Calming Header
        Box(
            Modifier
                .fillMaxWidth()
                .background(
                    Brush.verticalGradient(
                        listOf(
                            MaterialTheme.colorScheme.primary,
                            MaterialTheme.colorScheme.primary.copy(0.6f)
                        )
                    )
                )
                .padding(horizontal = 24.dp, vertical = 24.dp)
        ) {
            Column {
                Spacer(Modifier.height(16.dp))
                Text(
                    text = when (step) {
                        1 -> "Welcome to Lumen."
                        2 -> "Personalize Your Profile"
                        3 -> "Set Your Home Anchor"
                        4 -> "Personal Health Questionnaire (PHQ-9)"
                        5 -> "Generalized Anxiety Screener (GAD-7)"
                        6 -> "Stressful Life Events"
                        else -> "Calibration Completed"
                    },
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = Color.White
                )
                Text(
                    text = when (step) {
                        1 -> "Passively mapping behavioral metrics to support wellness"
                        2 -> "These demographics remain entirely private on this device"
                        3 -> "Used locally to evaluate daily time spent at home"
                        4 -> "Answer honestly to establish your clinical baseline priors"
                        5 -> "Helps Lumen calibrate threshold sensitivities to keep you safe"
                        6 -> "Identifies transient life events that might mimic indicators"
                        else -> "Your idiographic, local baseline calibration is active!"
                    },
                    fontSize = 12.sp,
                    color = Color.White.copy(0.85f)
                )
            }
        }

        // Content Frame
        Box(Modifier.weight(1f)) {
            when (step) {
                1 -> {
                    // Splash Introduction
                    Column(
                        Modifier
                            .fillMaxSize()
                            .padding(24.dp),
                        verticalArrangement = Arrangement.Center,
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Box(
                            Modifier
                                .size(96.dp)
                                .clip(CircleShape)
                                .background(MaterialTheme.colorScheme.primary.copy(0.15f)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                Icons.Default.Lightbulb,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.size(48.dp)
                            )
                        }
                        Spacer(Modifier.height(24.dp))
                        Text(
                            "Guided Calming Biomarkers",
                            fontSize = 20.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.onBackground
                        )
                        Spacer(Modifier.height(12.dp))
                        Text(
                            "Lumen operates 100% locally and offline. It gathers passive behavioral telemetry—such as sleep patterns, physical movement, typing speeds, and social frequency—to construct a personalized Digital DNA and assist your assessment.",
                            fontSize = 13.sp,
                            color = MaterialTheme.colorScheme.onBackground.copy(0.7f),
                            textAlign = TextAlign.Center,
                            lineHeight = 19.sp
                        )
                        Spacer(Modifier.height(32.dp))
                        Button(
                            onClick = { step = 2 },
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(50.dp),
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Text("Begin Configuration", color = Color.White, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                        }
                    }
                }
                2 -> {
                    // Profile Demographics
                    LazyColumn(
                        Modifier
                            .fillMaxSize()
                            .padding(24.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        item {
                            OutlinedTextField(
                                value = name, onValueChange = { name = it },
                                label = { Text("Full Name") },
                                isError = showErrors && name.isBlank(),
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(10.dp)
                            )
                        }
                        item {
                            Text("Gender", fontSize = 13.sp, fontWeight = FontWeight.Medium, color = MaterialTheme.colorScheme.onBackground)
                            Column(
                                Modifier
                                    .fillMaxWidth()
                                    .border(
                                        1.dp,
                                        if (showErrors && gender.isBlank()) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.outline.copy(0.3f),
                                        RoundedCornerShape(10.dp)
                                    )
                                    .padding(8.dp)
                            ) {
                                listOf("Male", "Female", "Non-binary", "Prefer not to say").forEach { option ->
                                    Row(
                                        verticalAlignment = Alignment.CenterVertically,
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .clickable { gender = option }
                                            .padding(vertical = 4.dp)
                                    ) {
                                        RadioButton(
                                            selected = gender == option,
                                            onClick = { gender = option },
                                            colors = RadioButtonDefaults.colors(selectedColor = MaterialTheme.colorScheme.primary)
                                        )
                                        Text(option, fontSize = 13.sp, color = MaterialTheme.colorScheme.onBackground.copy(0.8f))
                                    }
                                }
                            }
                        }
                        item {
                            OutlinedTextField(
                                value = age, onValueChange = { age = it.filter { ch -> ch.isDigit() } },
                                label = { Text("Age") },
                                isError = showErrors && age.isBlank(),
                                keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(keyboardType = androidx.compose.ui.text.input.KeyboardType.Number),
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(10.dp)
                            )
                        }
                        item {
                            var expanded by remember { mutableStateOf(false) }
                            Box(Modifier.fillMaxWidth()) {
                                OutlinedTextField(
                                    value = profession, onValueChange = {},
                                    label = { Text("Profession") },
                                    readOnly = true,
                                    isError = showErrors && profession.isBlank(),
                                    trailingIcon = { IconButton(onClick = { expanded = !expanded }) { Icon(Icons.Default.ArrowDropDown, null) } },
                                    modifier = Modifier.fillMaxWidth(),
                                    shape = RoundedCornerShape(10.dp)
                                )
                                Box(
                                    Modifier
                                        .matchParentSize()
                                        .clickable { expanded = !expanded })
                                DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                                    listOf("Student", "Employed", "Self-employed", "Other").forEach { opt ->
                                        DropdownMenuItem(
                                            text = { Text(opt) },
                                            onClick = { profession = opt; expanded = false }
                                        )
                                    }
                                }
                            }
                        }
                        item {
                            OutlinedTextField(
                                value = country, onValueChange = { country = it },
                                label = { Text("Country") },
                                isError = showErrors && country.isBlank(),
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(10.dp)
                            )
                        }
                        item {
                            Button(
                                onClick = {
                                    if (name.isBlank() || gender.isBlank() || age.isBlank() || profession.isBlank() || country.isBlank()) {
                                        showErrors = true
                                    } else {
                                        val p = com.example.mhealth.models.UserProfile(
                                            email = "patient@lumen.health",
                                            name = name,
                                            gender = gender,
                                            age = age.toIntOrNull() ?: 0,
                                            profession = profession,
                                            country = country
                                        )
                                        DataRepository.saveUserProfile(p)
                                        step = 3
                                    }
                                },
                                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(50.dp)
                                    .padding(top = 8.dp),
                                shape = RoundedCornerShape(12.dp)
                            ) {
                                Text("Continue Setup", color = Color.White, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                            }
                        }
                    }
                }
                3 -> {
                    // Home Location Capture
                    Column(
                        Modifier
                            .fillMaxSize()
                            .padding(24.dp),
                        verticalArrangement = Arrangement.spacedBy(20.dp)
                    ) {
                        Card(
                            shape = RoundedCornerShape(12.dp),
                            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer.copy(0.2f)),
                            border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(0.2f))
                        ) {
                            Column(Modifier.padding(16.dp)) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Icon(Icons.Default.Home, null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(24.dp))
                                    Spacer(Modifier.width(8.dp))
                                    Text("Home Location Accuracy", fontWeight = FontWeight.Bold, fontSize = 14.sp, fontFamily = Fredoka)
                                }
                                Spacer(Modifier.height(8.dp))
                                Text(
                                    "Your coordinates are securely stored on-device to passively measure daily displacement and time spent home. Knowing where home sits is the mathematical anchor of your circadian entropy score.",
                                    fontSize = 12.sp,
                                    lineHeight = 17.sp,
                                    color = MaterialTheme.colorScheme.onBackground.copy(0.7f)
                                )
                            }
                        }

                        if (homeSet) {
                            val loc = DataRepository.homeLocation.value
                            Card(
                                shape = RoundedCornerShape(8.dp),
                                colors = CardDefaults.cardColors(containerColor = LumenTeal.copy(0.12f))
                            ) {
                                Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                                    Icon(Icons.Default.CheckCircle, null, tint = LumenTeal, modifier = Modifier.size(20.dp))
                                    Spacer(Modifier.width(8.dp))
                                    Text(
                                        "Coordinates Anchored: %.4f, %.4f".format(loc?.first ?: 0.0, loc?.second ?: 0.0),
                                        fontSize = 12.sp,
                                        fontWeight = FontWeight.SemiBold,
                                        color = LumenTeal
                                    )
                                }
                            }
                        }

                        Button(
                            onClick = {
                                homeCapturing = true
                                com.example.mhealth.logic.DataCollector(ctx).captureHomeLocation { success ->
                                    homeCapturing = false
                                    homeSet = success
                                    if (!success) {
                                        Toast.makeText(ctx, "GPS Timeout. Please make sure location access is enabled.", Toast.LENGTH_LONG).show()
                                    }
                                }
                            },
                            enabled = !homeCapturing,
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.secondary),
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(50.dp),
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            if (homeCapturing) {
                                CircularProgressIndicator(Modifier.size(20.dp), color = Color.White, strokeWidth = 2.dp)
                                Spacer(Modifier.width(8.dp))
                            }
                            Text(if (homeCapturing) "Acquiring GPS Signal..." else "📌 Capture Current GPS as Home", color = Color.White, fontFamily = Fredoka)
                        }

                        Spacer(Modifier.weight(1f))

                        Button(
                            onClick = { step = 4 },
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(50.dp),
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Text(if (homeSet) "Next: Clinical Screeners" else "Skip Home Location for Now", color = Color.White, fontFamily = Fredoka)
                        }
                    }
                }
                4 -> {
                    // PHQ-9 Screener questions
                    ScreenerWizard(
                        questions = phq9Questions,
                        answers = phq9Answers,
                        options = optionsList,
                        onCompleted = {
                            step = 5
                        }
                    )
                }
                5 -> {
                    // GAD-7 Screener questions
                    ScreenerWizard(
                        questions = gad7Questions,
                        answers = gad7Answers,
                        options = optionsList,
                        onCompleted = {
                            step = 6
                        }
                    )
                }
                6 -> {
                    // Life Events & Stressors Count
                    Column(
                        Modifier
                            .fillMaxSize()
                            .padding(24.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Text(
                            "Have you experienced any of the following events in the last 14 days?",
                            fontWeight = FontWeight.SemiBold,
                            fontSize = 14.sp
                        )

                        LazyColumn(
                            verticalArrangement = Arrangement.spacedBy(8.dp),
                            modifier = Modifier.weight(1f)
                        ) {
                            items(stressors.size) { idx ->
                                val label = stressors[idx]
                                val selected = selectedStressors[idx] ?: false
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .clickable {
                                            if (idx == stressors.size - 1) {
                                                selectedStressors.clear()
                                                selectedStressors[idx] = true
                                            } else {
                                                selectedStressors[stressors.size - 1] = false
                                                selectedStressors[idx] = !selected
                                            }
                                        }
                                        .border(
                                            1.dp,
                                            if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outline.copy(0.2f),
                                            RoundedCornerShape(8.dp)
                                        )
                                        .background(if (selected) MaterialTheme.colorScheme.primary.copy(0.04f) else Color.Transparent)
                                        .padding(horizontal = 12.dp, vertical = 10.dp)
                                ) {
                                    Checkbox(
                                        checked = selected,
                                        onCheckedChange = {
                                            if (idx == stressors.size - 1) {
                                                selectedStressors.clear()
                                                selectedStressors[idx] = it
                                            } else {
                                                selectedStressors[stressors.size - 1] = false
                                                selectedStressors[idx] = it
                                            }
                                        },
                                        colors = CheckboxDefaults.colors(checkedColor = MaterialTheme.colorScheme.primary)
                                    )
                                    Spacer(Modifier.width(8.dp))
                                    Text(label, fontSize = 12.5.sp, color = MaterialTheme.colorScheme.onBackground.copy(0.8f))
                                }
                            }
                        }

                        Button(
                            onClick = {
                                // Calculate Calibration Scores
                                val totalPhq = phq9Answers.sum()
                                val totalGad = gad7Answers.sum()
                                val totalEvents = selectedStressors.filter { it.value && it.key < stressors.size - 1 }.size

                                DataRepository.saveScreenerScores(totalPhq, totalGad, totalEvents)
                                
                                // Perform threshold recalculation locally
                                val localPref = ctx.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE)
                                localPref.edit().apply {
                                    putInt("screener_phq9", totalPhq)
                                    putInt("screener_gad7", totalGad)
                                    putInt("screener_life_events", totalEvents)
                                }.apply()

                                step = 7
                            },
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(50.dp),
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Text("Calibrate Thresholds", color = Color.White, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                        }
                    }
                }
                7 -> {
                    // Final onboarding display & Notification permissions
                    val checkinEnabled by DataRepository.checkinNotificationsEnabled.collectAsState()
                    val pScore = DataRepository.phq9Score.collectAsState()
                    val gScore = DataRepository.gad7Score.collectAsState()
                    val events = DataRepository.recentLifeEventsCount.collectAsState()

                    val isSensitive = pScore.value >= 10 || gScore.value >= 10

                    Column(
                        Modifier
                            .fillMaxSize()
                            .padding(24.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Box(
                            Modifier
                                .size(80.dp)
                                .clip(CircleShape)
                                .background(LumenTeal.copy(0.12f)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(Icons.Default.Check, null, tint = LumenTeal, modifier = Modifier.size(40.dp))
                        }
                        
                        Text(
                            "Thresholds Calibrated Successfully!",
                            fontWeight = FontWeight.Bold,
                            fontSize = 18.sp,
                            fontFamily = Fredoka,
                            textAlign = TextAlign.Center
                        )

                        Card(
                            Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(12.dp),
                            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(0.3f))
                        ) {
                            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Text("Calibration Summary", fontWeight = FontWeight.Bold, fontSize = 13.sp, fontFamily = Fredoka)
                                HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.2f))
                                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                    Text("Initial PHQ-9 Status:", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    Text("${pScore.value} (Moderate/Severe: ${pScore.value >= 10})", fontSize = 12.sp, fontWeight = FontWeight.Bold)
                                }
                                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                    Text("Initial GAD-7 Status:", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    Text("${gScore.value} (Moderate/Severe: ${gScore.value >= 10})", fontSize = 12.sp, fontWeight = FontWeight.Bold)
                                }
                                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                    Text("Life Events Filter Window:", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    Text(if (isSensitive) "14 Days (Extended)" else "10 Days (Standard)", fontSize = 12.sp, fontWeight = FontWeight.Bold)
                                }
                                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                    Text("S1 Anomaly Trigger floor:", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    Text(if (isSensitive) "0.32 (High Sensitivity)" else "0.38 (Standard)", fontSize = 12.sp, fontWeight = FontWeight.Bold)
                                }
                            }
                        }

                        // Notification Permissions & Toggle
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 8.dp)
                        ) {
                            Column(Modifier.weight(1f)) {
                                Text("Weekly Screening Reminders", fontWeight = FontWeight.Bold, fontSize = 14.sp)
                                Text("Remind me to complete weekly checks.", fontSize = 12.sp, color = MaterialTheme.colorScheme.onBackground.copy(0.6f))
                            }
                            Switch(
                                checked = checkinEnabled,
                                onCheckedChange = { DataRepository.setCheckinNotificationsEnabled(it) },
                                colors = SwitchDefaults.colors(checkedThumbColor = MaterialTheme.colorScheme.primary)
                            )
                        }

                        Spacer(Modifier.weight(1f))

                        Button(
                            onClick = {
                                // Finalize first login status
                                val localPref = ctx.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE)
                                localPref.edit().putBoolean("first_login_complete", true).apply()
                                scope.launch(Dispatchers.IO) {
                                    val db = MHealthDatabase.getInstance(ctx)
                                    db.userProfileDao().upsert(
                                        UserProfileEntity(
                                            userId = "patient@lumen.health",
                                            onboardingDate = System.currentTimeMillis().toString(),
                                            baselineReady = false,
                                            currentStatus = "Collecting"
                                        )
                                    )
                                    // Warm-start data repo
                                    withContext(Dispatchers.Main) {
                                        DataRepository.initWithDb(ctx, "patient@lumen.health")
                                        onComplete()
                                    }
                                }
                            },
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(50.dp),
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Text("Launch Lumen Dashboard", color = Color.White, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun ScreenerWizard(
    questions: List<String>,
    answers: SnapshotStateList<Int>,
    options: List<String>,
    onCompleted: () -> Unit
) {
    var qIndex by remember { mutableIntStateOf(0) }
    val currentAnswer = answers[qIndex]

    Column(
        Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Progress Indicator
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
            Text("Question ${qIndex + 1} of ${questions.size}", fontSize = 12.sp, color = MaterialTheme.colorScheme.onBackground.copy(0.6f))
            LinearProgressIndicator(
                progress = { (qIndex + 1).toFloat() / questions.size },
                modifier = Modifier
                    .width(120.dp)
                    .height(6.dp)
                    .clip(CircleShape),
                color = MaterialTheme.colorScheme.primary,
                trackColor = MaterialTheme.colorScheme.outline.copy(0.15f)
            )
        }

        Spacer(Modifier.height(8.dp))

        // Question text card
        Card(
            Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer.copy(0.1f)),
            border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(0.1f))
        ) {
            Text(
                text = questions[qIndex],
                fontSize = 15.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                modifier = Modifier.padding(18.dp),
                lineHeight = 22.sp
            )
        }

        Text(
            "Over the last 2 weeks, how often have you been bothered by this problem?",
            fontSize = 12.sp,
            color = MaterialTheme.colorScheme.onBackground.copy(0.5f),
            modifier = Modifier.padding(bottom = 8.dp)
        )

        // Answers options vertical list - Scrollable column to fix the bugged cut-off layout
        Column(
            modifier = Modifier
                .weight(1f)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            options.forEachIndexed { score, text ->
                val selected = currentAnswer == score
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { answers[qIndex] = score }
                        .border(
                            1.dp,
                            if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outline.copy(0.2f),
                            RoundedCornerShape(10.dp)
                        )
                        .background(if (selected) MaterialTheme.colorScheme.primary.copy(0.05f) else Color.Transparent)
                        .padding(horizontal = 16.dp, vertical = 12.dp)
                ) {
                    RadioButton(
                        selected = selected,
                        onClick = { answers[qIndex] = score },
                        colors = RadioButtonDefaults.colors(selectedColor = MaterialTheme.colorScheme.primary)
                    )
                    Spacer(Modifier.width(8.dp))
                    Text(text, fontSize = 13.5.sp, color = MaterialTheme.colorScheme.onBackground.copy(0.85f))
                }
            }
        }

        // Navigation row
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Button(
                onClick = { if (qIndex > 0) qIndex-- },
                enabled = qIndex > 0,
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.outline.copy(0.1f), contentColor = MaterialTheme.colorScheme.onBackground),
                shape = RoundedCornerShape(10.dp),
                modifier = Modifier.height(48.dp)
            ) {
                Text("Back", fontFamily = Fredoka)
            }

            Button(
                onClick = {
                    if (qIndex < questions.size - 1) {
                        qIndex++
                    } else {
                        onCompleted()
                    }
                },
                enabled = currentAnswer != -1,
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                shape = RoundedCornerShape(10.dp),
                modifier = Modifier
                    .width(120.dp)
                    .height(48.dp)
            ) {
                Text(if (qIndex == questions.size - 1) "Complete" else "Next", fontFamily = Fredoka)
            }
        }
    }
}

// =============================================================================
// Main Lumen Dashboard (5 Tabs: Home, Monitor, Analysis, Insights, Settings)
// =============================================================================
@Composable
fun MainLumenDashboard() {
    var selectedTab by remember { mutableStateOf(LumenDest.HOME) }
    var checkInActive by remember { mutableStateOf(false) }
    val context = LocalContext.current
    
    // Auto-request basic sensor permissions dynamically for background collection
    val perms = buildList {
        addAll(listOf(
            Manifest.permission.READ_CALL_LOG,
            Manifest.permission.READ_CONTACTS, 
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION, 
            Manifest.permission.READ_CALENDAR
        ))
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            add(Manifest.permission.POST_NOTIFICATIONS)
            add(Manifest.permission.READ_MEDIA_IMAGES)
            add(Manifest.permission.READ_MEDIA_VIDEO)
        } else {
            add(Manifest.permission.READ_EXTERNAL_STORAGE)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            add(Manifest.permission.ACTIVITY_RECOGNITION)
        }
    }

    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        startMonitoringService(context)
    }

    LaunchedEffect(Unit) {
        DataRepository.initWithDb(context.applicationContext, "patient@lumen.health")
        launcher.launch(perms.toTypedArray())
    }

    if (checkInActive) {
        // Full screen Check-In wizard
        Column(
            Modifier
                .fillMaxSize()
                .background(MaterialTheme.colorScheme.background)
        ) {
            Box(
                Modifier
                    .fillMaxWidth()
                    .background(MaterialTheme.colorScheme.primary)
                    .padding(vertical = 12.dp, horizontal = 24.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    IconButton(onClick = { checkInActive = false }) {
                        Icon(Icons.Default.ArrowBack, null, tint = Color.White)
                    }
                    Text("Weekly Clinical Screening", color = Color.White, fontWeight = FontWeight.Bold, fontFamily = Fredoka, fontSize = 16.sp)
                }
            }
            Box(Modifier.weight(1f)) {
                CheckInScreen(onCompleted = { checkInActive = false })
            }
        }
    } else {
        Scaffold(
            bottomBar = {
                NavigationBar(containerColor = MaterialTheme.colorScheme.surface, tonalElevation = 0.dp) {
                    LumenDest.entries.forEach { dest ->
                        val isSelected = selectedTab == dest
                        NavigationBarItem(
                            selected = isSelected,
                            onClick = { selectedTab = dest },
                            icon = { Icon(dest.icon, contentDescription = dest.label) },
                            label = { Text(dest.label, fontSize = 11.sp, fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal, fontFamily = Fredoka) },
                            colors = NavigationBarItemDefaults.colors(
                                selectedIconColor = MaterialTheme.colorScheme.primary,
                                selectedTextColor = MaterialTheme.colorScheme.primary,
                                unselectedIconColor = MaterialTheme.colorScheme.outline,
                                unselectedTextColor = MaterialTheme.colorScheme.outline,
                                indicatorColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.08f)
                            )
                        )
                    }
                }
            }
        ) { innerPadding ->
            Box(
                Modifier
                    .fillMaxSize()
                    .padding(innerPadding)
            ) {
                when (selectedTab) {
                    LumenDest.HOME -> HomeScreen()
                    LumenDest.MONITOR -> MonitorScreen()
                    LumenDest.ANALYSIS -> AnalysisScreen()
                    LumenDest.INSIGHTS -> InsightsScreen(onLaunchCheckIn = { checkInActive = true })
                    LumenDest.SETTINGS -> SettingsScreen()
                }
            }
        }
    }
}

// =============================================================================
// HomeScreen Tab (Clinical Multi-Sensor Progressive Dashboard)
// =============================================================================
@Composable
fun HomeScreen() {
    val vector by DataRepository.latestVector.collectAsState()
    val context = LocalContext.current

    // Reactive Service Checks
    val isAccessibilityEnabled = remember { MHealthAccessibilityService.isServiceEnabled(context) }
    val isNotificationEnabled = remember { MHealthNotificationListenerService.isServiceEnabled(context) }

    LazyColumn(
        Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
    ) {
        // Calming Header
        item {
            Box(
                Modifier
                    .fillMaxWidth()
                    .background(Brush.horizontalGradient(listOf(OceanBlue, SoftCyan)))
                    .padding(20.dp)
            ) {
                Column {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(Modifier.size(8.dp).clip(CircleShape).background(Color.White.copy(0.9f)))
                        Spacer(Modifier.width(6.dp))
                        Text(
                            "PASSIVE DIGITAL PHENOTYPING",
                            fontSize = 10.sp,
                            color = Color.White.copy(0.9f),
                            fontWeight = FontWeight.SemiBold,
                            letterSpacing = 1.sp,
                            fontFamily = Fredoka
                        )
                    }
                    Text("Lumen.", fontSize = 26.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka, color = Color.White)
                    Text("Passive Clinical Sensor Telemetry Dashboard", fontSize = 12.sp, color = Color.White.copy(0.8f))
                }
            }
        }

        // Section: Permission Warning Cards
        if (!isAccessibilityEnabled || !isNotificationEnabled) {
            item {
                Column(Modifier.padding(horizontal = 16.dp, vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    if (!isAccessibilityEnabled) {
                        Card(
                            onClick = {
                                context.startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
                            },
                            shape = RoundedCornerShape(12.dp),
                            colors = CardDefaults.cardColors(containerColor = AlertYellow.copy(0.08f)),
                            border = BorderStroke(1.dp, AlertYellow.copy(0.2f))
                        ) {
                            Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                                Icon(Icons.Default.Warning, null, tint = AlertYellow, modifier = Modifier.size(20.dp))
                                Spacer(Modifier.width(12.dp))
                                Column(Modifier.weight(1f)) {
                                    Text("Accessibility Service Disabled", fontWeight = FontWeight.Bold, fontSize = 12.5.sp, color = TextPrimary)
                                    Text("Typing metrics and scroll dynamics require accessibility permission to compute psychomotor speeds. Click here to enable.", fontSize = 11.sp, color = TextSecondary)
                                }
                            }
                        }
                    }

                    if (!isNotificationEnabled) {
                        Card(
                            onClick = {
                                context.startActivity(Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS))
                            },
                            shape = RoundedCornerShape(12.dp),
                            colors = CardDefaults.cardColors(containerColor = AlertYellow.copy(0.08f)),
                            border = BorderStroke(1.dp, AlertYellow.copy(0.2f))
                        ) {
                            Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                                Icon(Icons.Default.Warning, null, tint = AlertYellow, modifier = Modifier.size(20.dp))
                                Spacer(Modifier.width(12.dp))
                                Column(Modifier.weight(1f)) {
                                    Text("Notification Access Disabled", fontWeight = FontWeight.Bold, fontSize = 12.5.sp, color = TextPrimary)
                                    Text("Passive audio tracking and system music exposure require listener permission. Click here to enable.", fontSize = 11.sp, color = TextSecondary)
                                }
                            }
                        }
                    }
                }
            }
        }

        if (vector == null) {
            item {
                Box(Modifier.fillMaxWidth().height(300.dp), contentAlignment = Alignment.Center) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator(color = OceanBlue)
                        Spacer(Modifier.height(12.dp))
                        Text("Gathering passive digital phenotyping data...", color = TextSecondary, fontSize = 13.sp)
                    }
                }
            }
        } else {
            val v = checkNotNull(vector) { "Live telemetry missing" }

            // 1. Digital Wellbeing Card
            item {
                InfoCard("Digital Wellbeing Metrics", headerColor = OceanBlue) {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                        ArcProgressRing(v.screenTimeHours, 12f, OceanBlue, "Screen Time", "hrs")
                        ArcProgressRing(v.unlockCount, 100f, SoftCyan, "Unlocks", "")
                        ArcProgressRing(v.appLaunchCount, 200f, ChartRed, "App Opens", "")
                    }
                    Spacer(Modifier.height(16.dp))
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                        ArcProgressRing(v.notificationsToday, 200f, AlertOrange, "Notifs", "")
                        ArcProgressRing(v.socialAppRatio * 100f, 100f, ChartGreen, "Social", "%")
                        ArcProgressRing(v.upiTransactionsToday, 10f, ChartPurple, "UPI Opens", "")
                    }
                }
            }

            // 2. Movement & Location Card
            item {
                InfoCard("Movement & Location", headerColor = SoftCyan) {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                        ArcProgressRing(v.dailyDisplacementKm, 20f, ChartRed, "Distance", "km")
                        ArcProgressRing(v.locationEntropy, 3f, AlertOrange, "Loc. Entropy", "")
                        ArcProgressRing(v.homeTimeRatio * 100f, 100f, OceanBlue, "Home Stay", "%")
                    }
                    Spacer(Modifier.height(12.dp))
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                        MetricPill("🔀 Entropy Score", "%.2f".format(v.locationEntropy), AlertOrange)
                        MetricPill("📍 Visited Places", "${v.mediaCountToday.toInt()} items", ChartPurple) // placeholder/metric binding
                    }
                }
            }

            // 3. Communication & Media Card
            item {
                InfoCard("Communication & Relational", headerColor = SoftCyan) {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                        MetricPill("📞 Call Count", "${v.callsPerDay.toInt()} calls", SoftCyan)
                        MetricPill("⏱ Talk Duration", "${v.callDurationMinutes.toInt()}m", ChartRed)
                        MetricPill("👥 Unique Contacts", "${v.uniqueContacts.toInt()}", ChartPurple)
                        MetricPill("🎵 Music Time", "${v.musicTimeMinutes.toInt()}m", ChartGreen)
                    }
                }
            }

            // 4. Sleep Proxy Card
            item {
                InfoCard("Circadian Sleep Proxy", headerColor = ChartPurple) {
                    Column {
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                            ArcProgressRing(v.sleepDurationHours, 10f, ChartPurple, "Est. Sleep", "hrs")
                            ArcProgressRing(v.chargeDurationHours, 8f, AlertOrange, "Charging", "hrs")
                            ArcProgressRing(v.chargeRegularity * 100f, 100f, SoftCyan, "Regularity", "%")
                        }
                        Spacer(Modifier.height(16.dp))
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Text("Sleep Onset", fontSize = 11.sp, color = TextSecondary)
                                Text("%.0f:00".format(v.sleepTimeHour), fontSize = 18.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
                            }
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Text("Wake Up Time", fontSize = 11.sp, color = TextSecondary)
                                Text("%.0f:00".format(v.wakeTimeHour), fontSize = 18.sp, fontWeight = FontWeight.Bold, color = ChartPurple)
                            }
                        }
                    }
                }
            }

            // 5. Per-app usage breakdown
            if (v.appBreakdown.isNotEmpty() || v.appLaunchesBreakdown.isNotEmpty() || v.notificationBreakdown.isNotEmpty()) {
                item {
                    PerAppBreakdownCard(vector = v)
                }
            }

            if (v.bgAudioBreakdown.isNotEmpty()) {
                item {
                    BgAudioBreakdownCard(vector = v)
                }
            }

            // 6. Advanced Interaction Biomarkers
            item {
                InfoCard("Interaction Biomarkers", headerColor = AlertOrange) {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text("Keystroke Speed", fontSize = 12.sp, color = TextSecondary)
                            Text("%.1f char/sec".format(v.keystrokeSpeed), fontSize = 12.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
                        }
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text("Backspace Correction Ratio", fontSize = 12.sp, color = TextSecondary)
                            Text("%.1f%%".format(v.backspaceRatio * 100), fontSize = 12.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
                        }
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text("Scroll Velocity", fontSize = 12.sp, color = TextSecondary)
                            Text("%.0f pixels/sec".format(v.scrollVelocity), fontSize = 12.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
                        }
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text("Passive Lux Daylight Exposure", fontSize = 12.sp, color = TextSecondary)
                            Text("%.0f mins".format(v.daylightExposureMinutes), fontSize = 12.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
                        }
                    }
                }
            }
        }
        item { Spacer(Modifier.height(16.dp)) }
    }
}

// =============================================================================
// MonitorScreen Tab (Rhythm Stability & Longitudinal Trends)
// =============================================================================
@Composable
fun MonitorScreen() {
    val progress by DataRepository.baselineProgress.collectAsState()
    val isBuilding by DataRepository.isBuildingBaseline.collectAsState()
    val vector by DataRepository.latestVector.collectAsState()
    val baseline by DataRepository.baseline.collectAsState()
    val hourly by DataRepository.hourlySnapshots.collectAsState()
    val baselineVectors by DataRepository.collectedBaselineVectors.collectAsState()
    val analysisResult by DataRepository.latestAnalysisResult.collectAsState()
    val analysisHistory by DataRepository.analysisHistory.collectAsState()

    LazyColumn(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
        item {
            Box(
                Modifier.fillMaxWidth()
                    .background(Brush.horizontalGradient(listOf(SoftCyan, ChartPurple)))
                    .padding(20.dp)
            ) {
                Column {
                    Text("Baseline & Monitoring", fontSize = 24.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka, color = Color.White)
                    Text("Layers 2 & 3 — ${if (isBuilding) "Building Personal Normal" else "Continuous Tracking"}", fontSize = 13.sp, color = Color.White.copy(0.85f))
                }
            }
        }

        // 1. Progress Indicator
        item {
            InfoCard("Baseline Progress (P₀)", headerColor = SoftCyan) {
                if (isBuilding) {
                    val target = 28f
                    val frac = (progress.toFloat() / target).coerceIn(0f, 1f)
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        ArcProgressRing(progress.toFloat(), target, SoftCyan, "Days", "/ 28", size = 90.dp)
                        Spacer(Modifier.width(16.dp))
                        Column {
                            Text("Learning Your Unique Patterns", fontWeight = FontWeight.SemiBold, color = TextPrimary)
                            Text("Day $progress of 28 in mathematically establishing your scientific P₀ baseline. Collecting multidimensional behavioral data continuously for accuracy.", fontSize = 12.sp, color = TextSecondary, lineHeight = 16.sp)
                            Spacer(Modifier.height(6.dp))
                            LinearProgressIndicator(
                                progress = { frac },
                                color = SoftCyan,
                                trackColor = SoftCyan.copy(0.15f),
                                modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(4.dp))
                            )
                        }
                    }
                } else {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        val statusText = if (analysisResult != null) {
                            "Baseline Locked - ${analysisResult?.alertLevel?.uppercase()} Status"
                        } else {
                            "Scientific Baseline Established"
                        }
                        val statusColor = analysisResult?.let { alertColorForLevel(it.alertLevel) } ?: AlertGreen
                        val isHighRisk = analysisResult?.alertLevel?.lowercase() in listOf("orange", "red")
                        val icon = if (isHighRisk) Icons.Default.Warning else Icons.Default.CheckCircle
                        
                        Icon(icon, null, tint = statusColor, modifier = Modifier.size(40.dp))
                        Spacer(Modifier.width(12.dp))
                        Column {
                            Text(statusText, fontWeight = FontWeight.SemiBold, color = statusColor)
                            
                            val descriptionText = if (analysisResult != null) {
                                "Your current behavioral vector is being compared against your locked P₀ baseline. " + 
                                when(analysisResult?.alertLevel?.lowercase()) {
                                    "green" -> "Data indicates high alignment with your normal routines."
                                    "yellow" -> "Slight deviations from your baseline detected. Tracking for potential shifts."
                                    "orange" -> "Moderate departure from baseline established. Behavioral patterns show significant variance."
                                    "red" -> "Critical deviation from your established P₀. Immediate attention recommended."
                                    else -> "Continuous monitoring active."
                                }
                            } else {
                                "P₀ baseline vector is locked. Real-time multidimensional tracking is active."
                            }
                            Text(descriptionText, fontSize = 12.sp, color = TextSecondary, lineHeight = 16.sp)
                        }
                    }
                }
                
                if (baselineVectors.isNotEmpty()) {
                    Spacer(Modifier.height(20.dp))
                    Text(if (isBuilding) "Multi-Sensor Formation Trend" else "Composite Behavioral Index", fontSize = 13.sp, color = TextPrimary, fontWeight = FontWeight.Medium)
                    Spacer(Modifier.height(12.dp))
                    
                    val composite = baselineVectors.takeLast(28).map { v ->
                        val screen = (v.screenTimeHours / 12f).coerceIn(0f, 1f) * 40f
                        val move = (v.dailyDisplacementKm / 20f).coerceIn(0f, 1f) * 30f
                        val comms = (v.callsPerDay / 10f).coerceIn(0f, 1f) * 30f
                        screen + move + comms
                    }
                    
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.Bottom) {
                        Text("Activity Index (Last ${composite.size} Days)", fontSize = 11.sp, color = TextSecondary)
                        if (composite.isNotEmpty()) {
                            Text("%.0f".format(composite.last()), fontSize = 14.sp, fontWeight = FontWeight.Bold, color = SoftCyan)
                        }
                    }
                    Spacer(Modifier.height(4.dp))
                    SparklineChart(composite, SoftCyan, Modifier.fillMaxWidth().height(80.dp), showDots = true)
                }
            }
        }

        // 2. Intraday sparklines
        item {
            InfoCard("Today's Intraday Trends", headerColor = ChartPurple) {
                if (hourly.size < 2) {
                    Text("Collecting hourly snapshots…", color = TextSecondary, fontSize = 12.sp)
                } else {
                    val screenTimes = hourly.map { it.screenTimeHours }
                    val distances = hourly.map { it.dailyDisplacementKm }
                    SparklineLabel("Screen Time (hrs)", screenTimes, OceanBlue)
                    Spacer(Modifier.height(12.dp))
                    SparklineLabel("Distance (km)", distances, ChartRed)
                }
            }
        }

        // 3. Complete Baseline Table
        if (baseline != null && vector != null) {
            item {
                FeatureTableCard(baseline = checkNotNull(baseline), current = checkNotNull(vector))
            }
        }
        item { Spacer(Modifier.height(16.dp)) }
    }
}

// =============================================================================
// AnalysisScreen Tab (Composite Anomaly score & Radar Classifier)
// =============================================================================
@Composable
fun AnalysisScreen() {
    val latestResult by DataRepository.latestAnalysisResult.collectAsState()
    val provisional by DataRepository.provisionalAnalysis.collectAsState()
    val isBuilding by DataRepository.isBuildingBaseline.collectAsState()
    val baseline by DataRepository.baseline.collectAsState()
    val vector by DataRepository.latestVector.collectAsState()
    val analysisHistory by DataRepository.analysisHistory.collectAsState()
    
    val activeResult = provisional ?: latestResult

    LazyColumn(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
        item {
            Box(
                Modifier.fillMaxWidth()
                    .background(Brush.horizontalGradient(listOf(ChartRed, AlertOrange)))
                    .padding(20.dp)
            ) {
                Column {
                    Text("Anomaly Engine", fontSize = 24.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka, color = Color.White)
                    Text("System 1 & 2 — Deviation detection & pattern classification", fontSize = 12.sp, color = Color.White.copy(0.85f))
                }
            }
        }

        if (isBuilding) {
            item {
                InfoCard("Status", headerColor = SoftCyan) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(Modifier.size(32.dp), color = SoftCyan)
                        Spacer(Modifier.width(12.dp))
                        Text("Calibrating — baseline not yet ready.\nAnomaly detection begins after 28 days.", color = TextSecondary, fontSize = 12.sp)
                    }
                }
            }
        } else {
            // Anomaly Score Gauge
            item {
                val score = activeResult?.anomalyScore ?: 0f
                val isLive = provisional != null

                InfoCard("Anomaly Score", headerColor = ChartRed) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            if (isLive) {
                                Surface(
                                    color = AlertRed.copy(0.1f),
                                    shape = RoundedCornerShape(4.dp)
                                ) {
                                    Text(
                                        " LIVE UPDATE ",
                                        fontSize = 9.sp, fontWeight = FontWeight.Black,
                                        color = AlertRed, modifier = Modifier.padding(2.dp)
                                    )
                                }
                                Spacer(Modifier.width(8.dp))
                            }
                            Text(
                                if (isLive) "Current Day (Provisional)" else "Last Daily Report",
                                fontSize = 11.sp, color = TextSecondary
                            )
                        }
                        Spacer(Modifier.height(8.dp))
                        AnomalyScoreGauge(score, Modifier.fillMaxWidth().height(130.dp))
                        Spacer(Modifier.height(4.dp))
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                            Text("STABLE", fontSize = 10.sp, color = AlertGreen, fontWeight = FontWeight.Bold)
                            Text("MILD", fontSize = 10.sp, color = AlertYellow, fontWeight = FontWeight.Bold)
                            Text("MOD.", fontSize = 10.sp, color = AlertOrange, fontWeight = FontWeight.Bold)
                            Text("SEVERE", fontSize = 10.sp, color = AlertRed, fontWeight = FontWeight.Bold)
                        }
                        Spacer(Modifier.height(6.dp))
                        Text(
                            "Score: ${"%.3f".format(score)}",
                            fontSize = 18.sp, fontWeight = FontWeight.Bold, color = ChartRed, fontFamily = Fredoka
                        )
                        Text(
                            "Pattern: ${(activeResult?.patternType ?: "stable").replace("_", " ").uppercase()}",
                            fontSize = 12.sp, color = TextSecondary
                        )
                    }
                }
            }

            // Radar chart — with optional disorder prototype overlay
            if (baseline != null && vector != null) {
                item {
                    val PROTO_RADAR_ZSCORES: Map<String, List<Float>> = mapOf(
                        "depression_type_1"    to listOf(-0.60f, -0.03f,  0.04f, -0.43f, -0.87f, -0.81f),
                        "depression_type_2"    to listOf( 5.00f,  0.69f,  3.50f,  0.00f,  0.22f,  2.50f),
                        "depression_type_3"    to listOf( 5.00f,  2.20f,  1.04f,  5.00f,  0.80f,  1.84f),
                        "schizophrenia_type_1" to listOf(-0.08f, -0.16f,  1.04f,  0.04f,  0.35f,  0.18f),
                        "schizophrenia_type_2" to listOf( 4.01f,  3.24f,  2.68f,  1.67f,  3.11f,  3.25f),
                        "schizophrenia_type_3" to listOf( 5.00f,  1.14f,  1.17f,  0.99f, -2.92f,  1.46f)
                    )

                    fun zToRadar(z: Float): Float = ((z / 5f) * 0.5f + 0.5f).coerceIn(0f, 1f)

                    val matchedDisorder = latestResult?.prototypeMatch?.lowercase()?.trim()
                    val isRealMatch = matchedDisorder != null &&
                        matchedDisorder != "normal" &&
                        matchedDisorder != "situational" &&
                        !matchedDisorder.startsWith("healthy")

                    val protoVals: List<Float>? = if (isRealMatch) {
                        PROTO_RADAR_ZSCORES[matchedDisorder]?.map { zToRadar(it) }
                    } else null

                    InfoCard("Feature Deviation Radar", headerColor = ChartPurple) {
                        val b = checkNotNull(baseline); val v = checkNotNull(vector)
                        val radarLabels = listOf("Screen\nTime", "Social", "Places", "Location", "Sleep", "Comms")
                        val normalizeDev: (Float, Float) -> Float = { cur, base ->
                            if (base <= 0.01f) {
                                if (cur <= 0.01f) 0.5f else 1.0f
                            } else {
                                ((cur / base) * 0.5f).coerceIn(0f, 1f)
                            }
                        }
                        val curVals = listOf(
                            normalizeDev(v.screenTimeHours, b.screenTimeHours),
                            normalizeDev(v.socialAppRatio, b.socialAppRatio),
                            normalizeDev(v.mediaCountToday, b.mediaCountToday), // places proxy in this context
                            normalizeDev(v.dailyDisplacementKm, b.dailyDisplacementKm),
                            normalizeDev(v.sleepDurationHours, b.sleepDurationHours),
                            normalizeDev(v.conversationFrequency, b.conversationFrequency)
                        )
                        val baseVals = listOf(0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f)

                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) {
                            RadarChart(
                                labels          = radarLabels,
                                values          = curVals,
                                baseline        = baseVals,
                                color           = ChartPurple,
                                modifier        = Modifier.fillMaxWidth(0.9f).aspectRatio(1f).padding(vertical = 16.dp),
                                prototypeValues = protoVals
                            )
                        }

                        Spacer(Modifier.height(8.dp))

                        Row(
                            Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.Center,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Box(Modifier.size(12.dp).background(ChartPurple.copy(0.7f), CircleShape))
                            Text(" Current   ", fontSize = 11.sp, color = TextSecondary)
                            Box(Modifier.size(12.dp).background(SoftCyan.copy(0.5f), CircleShape))
                            Text(" Baseline", fontSize = 11.sp, color = TextSecondary)
                            if (protoVals != null) {
                                Spacer(Modifier.width(10.dp))
                                Canvas(Modifier.width(20.dp).height(12.dp)) {
                                    val dashEffect = PathEffect.dashPathEffect(floatArrayOf(6f, 4f), 0f)
                                    drawLine(
                                        color = Color(0xFFEF5350),
                                        start = Offset(0f, size.height / 2f),
                                        end   = Offset(size.width, size.height / 2f),
                                        strokeWidth = 2.5f,
                                        pathEffect = dashEffect
                                    )
                                }
                                Text(
                                    " ${latestResult?.prototypeMatch?.replace("_", " ")?.replaceFirstChar { it.uppercase() }}",
                                    fontSize = 11.sp,
                                    color = Color(0xFFEF5350)
                                )
                            }
                        }
                    }
                }
            }

            // Top deviations & Flagged Features
            if (activeResult != null) {
                item {
                    val flaggedList = remember(activeResult.flaggedFeatures) {
                        try {
                            val arr = org.json.JSONArray(activeResult.flaggedFeatures)
                            List(arr.length()) { idx -> arr.getString(idx) }
                        } catch (e: Exception) {
                            emptyList<String>()
                        }
                    }

                    InfoCard("Top Deviations (SD units)", headerColor = AlertOrange) {
                        if (flaggedList.isEmpty()) {
                            Text("No significant behavioral deviations flagged today.", color = TextSecondary, fontSize = 12.sp)
                        } else {
                            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                flaggedList.take(5).forEach { feat ->
                                    val parts = feat.split(" ")
                                    val valPart = parts.lastOrNull()?.replace("(", "")?.replace(")", "")?.toFloatOrNull() ?: 0f
                                    val namePart = feat.substringBefore(" (")
                                    DeviationRow(namePart.replace(Regex("([a-z])([A-Z])"), "$1 $2"), valPart)
                                }
                            }
                        }
                    }
                }

                item {
                    InfoCard("Temporal Pattern Info", headerColor = SoftCyan) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.Timeline, null, tint = SoftCyan)
                            Spacer(Modifier.width(8.dp))
                            Column {
                                Text(activeResult.patternType.replace("_", " ").replaceFirstChar { it.uppercase() }, fontWeight = FontWeight.SemiBold, color = TextPrimary)
                                Text("Sustained deviation days: ${activeResult.sustainedDays}", fontSize = 12.sp, color = TextSecondary)
                                Text("Evidence accumulated: ${"%.2f".format(activeResult.evidenceAccumulated)}", fontSize = 12.sp, color = TextSecondary)
                            }
                        }
                    }
                }
            }
        }
        item { Spacer(Modifier.height(16.dp)) }
    }
}

// =============================================================================
// InsightsScreen Tab (100% Qualitative summaries, crisis resources, check-in card)
// =============================================================================
@Composable
fun InsightsScreen(onLaunchCheckIn: () -> Unit) {
    val weeklyFeatures by DataRepository.weeklyFeatureHistory.collectAsState()
    val baseline by DataRepository.baseline.collectAsState()
    val isDnaReady by DataRepository.isDnaBaselineReady.collectAsState()

    LazyColumn(
        Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentPadding = PaddingValues(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            Column {
                Text("Insights Dashboard", fontSize = 22.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                Text("Demographics-anchored qualitative summaries", fontSize = 12.sp, color = MaterialTheme.colorScheme.outline)
            }
        }

        // Action: Self-Report Screener Card
        item {
            Card(
                onClick = onLaunchCheckIn,
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer.copy(0.12f)),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(0.2f))
            ) {
                Row(Modifier.padding(18.dp), verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.Favorite, null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(24.dp))
                    Spacer(Modifier.width(12.dp))
                    Column(Modifier.weight(1f)) {
                        Text("Weekly Mental Check-in", fontWeight = FontWeight.Bold, fontSize = 14.sp, color = MaterialTheme.colorScheme.primary, fontFamily = Fredoka)
                        Text("Complete a quick PHQ-9 and GAD-7 assessment to update your calibration baseline score.", fontSize = 11.5.sp, color = TextSecondary)
                    }
                    Icon(Icons.Default.ArrowForward, null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(20.dp))
                }
            }
        }

        if (!isDnaReady || weeklyFeatures.isEmpty() || baseline == null) {
            item {
                Card(
                    Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
                ) {
                    Column(
                        Modifier.padding(24.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        CircularProgressIndicator(color = MaterialTheme.colorScheme.primary, strokeWidth = 3.dp)
                        Text(
                            "Establishing Behavioral Anchors...",
                            fontWeight = FontWeight.SemiBold,
                            fontSize = 14.sp,
                            fontFamily = Fredoka
                        )
                        Text(
                            "Please continue using your device normally. Insights requires baseline aggregates to compute standard deviations and identify qualitative variations safely.",
                            fontSize = 12.sp,
                            color = MaterialTheme.colorScheme.outline,
                            textAlign = TextAlign.Center,
                            lineHeight = 17.sp
                        )
                    }
                }
            }
        } else {
            val latest = weeklyFeatures.firstOrNull()
            val base = baseline

            if (latest != null && base != null) {
                // 1. Line Chart representation (Relative bounds 0-100%, NO NUMBERS)
                item {
                    Card(
                        Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(16.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
                    ) {
                        Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                            Text("Longitudinal Rhythm Stability", fontWeight = FontWeight.Bold, fontSize = 14.sp, fontFamily = Fredoka)
                            Text("A rolling depiction of lifestyle rhythm consistency. A stable, flat trend highlights healthy routine adherence.", fontSize = 11.5.sp, color = MaterialTheme.colorScheme.outline)
                            
                            Box(
                                Modifier
                                    .fillMaxWidth()
                                    .height(130.dp)
                                    .padding(vertical = 8.dp)
                            ) {
                                QualitativeTrendChart(weeklyFeatures)
                            }
                            
                            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                Text("Previous 7 Days", fontSize = 10.sp, color = MaterialTheme.colorScheme.outline)
                                Text("Today", fontSize = 10.sp, color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Bold)
                            }
                        }
                    }
                }

                // 2. Qualitative Clinical Trend Summaries
                item {
                    Text("Behavioral Strata Analysis", fontWeight = FontWeight.Bold, fontSize = 14.sp, fontFamily = Fredoka, modifier = Modifier.padding(top = 8.dp, bottom = 4.dp))
                }

                // Sleep Card
                item {
                    val sleepDiff = latest.sleepDurationHours - base.sleepDurationHours
                    QualitativeInsightCard(
                        title = "Sleep & Circadian Alignment",
                        icon = Icons.Default.NightsStay,
                        badgeText = when {
                            sleepDiff > 1.5f -> "Elongated"
                            sleepDiff < -1.5f -> "Contracted"
                            else -> "Balanced"
                        },
                        badgeColor = when {
                            Math.abs(sleepDiff) > 1.5f -> LumenAmber
                            else -> LumenTeal
                        },
                        description = when {
                            sleepDiff > 1.5f -> "Your sleep duration is significantly longer than your locked baseline. This might represent vegetative hypersomnia or withdrawal state."
                            sleepDiff < -1.5f -> "Your sleep cycle is compressed. Restricting rest or waking too early can decay cognitive resilience."
                            else -> "Sleep durations and DND silent gaps are stable, demonstrating strong circular time consistency."
                        }
                    )
                }

                // Steps Card
                item {
                    val stepRatio = if (base.dailyStepCount > 0) latest.dailyStepCount / base.dailyStepCount else 1.0f
                    QualitativeInsightCard(
                        title = "Physical Mobility",
                        icon = Icons.Default.DirectionsRun,
                        badgeText = when {
                            stepRatio < 0.6f -> "Reduced"
                            stepRatio > 1.4f -> "Elevated"
                            else -> "Stable"
                        },
                        badgeColor = when {
                            stepRatio < 0.6f -> LumenRose
                            stepRatio > 1.4f -> LumenAmber
                            else -> LumenTeal
                        },
                        description = when {
                            stepRatio < 0.6f -> "Physical steps are notably lower than your locked routine. Consider introducing small active windows to promote circulation."
                            stepRatio > 1.4f -> "Physical step count is highly elevated. Active pacing or energetic exercise has been registered."
                            else -> "Mobility levels and physical tracking features match your standard behavior metrics."
                        }
                    )
                }

                // Communication Card
                item {
                    val callDiff = latest.callsPerDay - base.callsPerDay
                    QualitativeInsightCard(
                        title = "Relational Frequency",
                        icon = Icons.Default.Call,
                        badgeText = when {
                            callDiff < -2.0f -> "Low Engagement"
                            else -> "Stable Outbound"
                        },
                        badgeColor = when {
                            callDiff < -2.0f -> LumenRose
                            else -> LumenTeal
                        },
                        description = when {
                            callDiff < -2.0f -> "We observed a significant retraction in calls and contact frequencies. Maintaining active connection guards against emotional withdrawal."
                            else -> "Call logs, unique contacts, and conversation frequency variables remain steady."
                        }
                    )
                }

                // Screen Card
                item {
                    val screenDiff = latest.screenTimeHours - base.screenTimeHours
                    QualitativeInsightCard(
                        title = "Digital Interaction Dynamics",
                        icon = Icons.Default.Smartphone,
                        badgeText = when {
                            screenDiff > 2.0f -> "Increased Screen"
                            screenDiff < -2.0f -> "Reduced Screen"
                            else -> "Within Norms"
                        },
                        badgeColor = when {
                            Math.abs(screenDiff) > 2.0f -> LumenAmber
                            else -> LumenTeal
                        },
                        description = when {
                            screenDiff > 2.0f -> "Digital interaction is elevated. Extended evening engagement or quick unlock pickup bursts can indicate restlessness."
                            screenDiff < -2.0f -> "Screen interactions have contracted. This highlights lower digital dependence or reduced social apps ratio."
                            else -> "Daily screen time hours, lock counts, and app session metrics are steady."
                        }
                    )
                }
            }
        }
    }
}

@Composable
fun QualitativeInsightCard(
    title: String,
    icon: ImageVector,
    badgeText: String,
    badgeColor: Color,
    description: String
) {
    Card(
        Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.08f))
    ) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        Modifier
                            .size(32.dp)
                            .clip(CircleShape)
                            .background(MaterialTheme.colorScheme.primary.copy(0.08f)),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(icon, null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(16.dp))
                    }
                    Spacer(Modifier.width(10.dp))
                    Text(title, fontWeight = FontWeight.Bold, fontSize = 13.5.sp)
                }
                
                Box(
                    Modifier
                        .clip(RoundedCornerShape(6.dp))
                        .background(badgeColor.copy(0.12f))
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                ) {
                    Text(badgeText, fontSize = 10.sp, color = badgeColor, fontWeight = FontWeight.Bold)
                }
            }

            Text(
                description,
                fontSize = 12.sp,
                lineHeight = 17.sp,
                color = MaterialTheme.colorScheme.onBackground.copy(0.7f)
            )
        }
    }
}

@Composable
fun QualitativeTrendChart(features: List<PersonalityVector>) {
    val primary = MaterialTheme.colorScheme.primary
    val lineBrush = Brush.horizontalGradient(listOf(primary.copy(0.5f), primary))
    val reversed = features.take(7).reversed()

    Canvas(Modifier.fillMaxSize()) {
        if (reversed.size < 2) return@Canvas

        val maxVal = reversed.map { it.screenTimeHours }.maxOrNull() ?: 1f
        val minVal = reversed.map { it.screenTimeHours }.minOrNull() ?: 0f
        val range = if (maxVal - minVal > 0f) maxVal - minVal else 1f

        val path = Path()
        val spacing = size.width / (reversed.size - 1)

        reversed.forEachIndexed { idx, item ->
            val relativeY = (item.screenTimeHours - minVal) / range
            val x = idx * spacing
            val y = size.height - (relativeY * (size.height - 20.dp.toPx())) - 10.dp.toPx()

            if (idx == 0) {
                path.moveTo(x, y)
            } else {
                path.lineTo(x, y)
            }
        }

        drawPath(
            path = path,
            brush = lineBrush,
            style = Stroke(width = 3.dp.toPx(), cap = StrokeCap.Round)
        )

        reversed.forEachIndexed { idx, item ->
            val relativeY = (item.screenTimeHours - minVal) / range
            val x = idx * spacing
            val y = size.height - (relativeY * (size.height - 20.dp.toPx())) - 10.dp.toPx()
            
            drawCircle(
                color = primary,
                radius = 4.dp.toPx(),
                center = Offset(x, y)
            )
        }
    }
}

// =============================================================================
// CheckInScreen Tab (Manual weekly self-report screening wizard)
// =============================================================================
@Composable
fun CheckInScreen(onCompleted: () -> Unit) {
    val phq9Answers = remember { mutableStateListOf(*Array(9) { -1 }) }
    val gad7Answers = remember { mutableStateListOf(*Array(7) { -1 }) }
    val optionsList = listOf("Not at all", "Several days", "More than half the days", "Nearly every day")

    val phq9Questions = listOf(
        "Little interest or pleasure in doing things.",
        "Feeling down, depressed, or hopeless.",
        "Trouble falling or staying asleep, or sleeping too much.",
        "Feeling tired or having little energy.",
        "Poor appetite or overeating.",
        "Feeling bad about yourself — or that you are a failure or have let yourself or your family down.",
        "Trouble concentrating on things, such as reading the newspaper or watching television.",
        "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual.",
        "Thoughts that you would be better off dead or of hurting yourself in some way."
    )

    val gad7Questions = listOf(
        "Feeling nervous, anxious, or on edge.",
        "Not being able to stop or control worrying.",
        "Worrying too much about different things.",
        "Trouble relaxing.",
        "Being so restless that it is hard to sit still.",
        "Becoming easily annoyed or irritable.",
        "Feeling afraid as if something awful might happen."
    )

    var screenerStep by remember { mutableIntStateOf(1) } // 1: PHQ, 2: GAD

    if (screenerStep == 1) {
        ScreenerWizard(
            questions = phq9Questions,
            answers = phq9Answers,
            options = optionsList,
            onCompleted = { screenerStep = 2 }
        )
    } else {
        ScreenerWizard(
            questions = gad7Questions,
            answers = gad7Answers,
            options = optionsList,
            onCompleted = {
                val totalPhq = phq9Answers.sum()
                val totalGad = gad7Answers.sum()
                
                // Save updated scores to trigger recalibration
                DataRepository.saveScreenerScores(totalPhq, totalGad, DataRepository.recentLifeEventsCount.value)
                
                onCompleted()
            }
        )
    }
}

// =============================================================================
// SettingsScreen Tab (Demographics + JSON Backup Export/Import + Switches)
// =============================================================================
@Composable
fun SettingsScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    
    val profile by DataRepository.userProfile.collectAsState()
    val checkinEnabled by DataRepository.checkinNotificationsEnabled.collectAsState()
    val homeLocation by DataRepository.homeLocation.collectAsState()
    var homeCapturing by remember { mutableStateOf(false) }

    // Persistent collection toggles state inside local SharedPreferences
    val localPref = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }
    var masterTracking by remember { mutableStateOf(localPref.getBoolean("master_tracking_enabled", true)) }
    var locationTracking by remember { mutableStateOf(localPref.getBoolean("location_tracking_enabled", true)) }
    var communicationLogs by remember { mutableStateOf(localPref.getBoolean("communication_logs_enabled", true)) }

    // Document pickers for export and import
    val exportLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("application/json"),
        onResult = { uri ->
            if (uri != null) {
                exportDataToUri(context, uri)
            }
        }
    )

    val importLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument(),
        onResult = { uri ->
            if (uri != null) {
                importBackupDataFromJson(context, uri)
            }
        }
    )

    LazyColumn(
        Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentPadding = PaddingValues(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            Column {
                Text("Profile & Settings", fontSize = 22.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                Text("Manage your localized settings and reports securely", fontSize = 12.sp, color = MaterialTheme.colorScheme.outline)
            }
        }

        // Section 1: Demographics info card
        item {
            Card(
                Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(14.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.08f))
            ) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Patient Identity Metadata", fontWeight = FontWeight.Bold, fontSize = 13.5.sp, fontFamily = Fredoka)
                    HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.1f))
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("Patient Name:", fontSize = 12.5.sp, color = MaterialTheme.colorScheme.outline)
                        Text(profile?.name ?: "Lumen User", fontSize = 12.5.sp, fontWeight = FontWeight.SemiBold)
                    }
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("Age / Profession:", fontSize = 12.5.sp, color = MaterialTheme.colorScheme.outline)
                        Text("${profile?.age ?: 0} / ${profile?.profession ?: "N/A"}", fontSize = 12.5.sp, fontWeight = FontWeight.SemiBold)
                    }
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("Registered Country:", fontSize = 12.5.sp, color = MaterialTheme.colorScheme.outline)
                        Text(profile?.country ?: "N/A", fontSize = 12.5.sp, fontWeight = FontWeight.SemiBold)
                    }
                }
            }
        }

        // Section 2: Restored Data Collection Controls & Toggles
        item {
            InfoCard("Data Collection Toggles", headerColor = TextSecondary) {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    ToggleRow(
                        title = "Master Tracking",
                        subtitle = "Enable all sensor logs passively in the background",
                        checked = masterTracking,
                        color = OceanBlue,
                        onToggle = {
                            masterTracking = it
                            localPref.edit().putBoolean("master_tracking_enabled", it).apply()
                            Toast.makeText(context, if (it) "Background tracking resumed" else "Background tracking paused", Toast.LENGTH_SHORT).show()
                        }
                    )
                    HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.1f))
                    ToggleRow(
                        title = "Location GPS Tracking",
                        subtitle = "Saves spatial displacement and location variety entropy",
                        checked = locationTracking,
                        color = SoftCyan,
                        onToggle = {
                            locationTracking = it
                            localPref.edit().putBoolean("location_tracking_enabled", it).apply()
                        }
                    )
                    HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.1f))
                    ToggleRow(
                        title = "Communication Logs",
                        subtitle = "Passively log outbound call length and contact counts",
                        checked = communicationLogs,
                        color = AlertOrange,
                        onToggle = {
                            communicationLogs = it
                            localPref.edit().putBoolean("communication_logs_enabled", it).apply()
                        }
                    )
                }
            }
        }

        // Section 3: Home Location capture card
        item {
            InfoCard("Home Location GPS Anchor", headerColor = SoftCyan) {
                Column(Modifier.fillMaxWidth()) {
                    if (homeLocation != null) {
                        val (lat, lon) = checkNotNull(homeLocation)
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.Home, null, tint = OceanBlue, modifier = Modifier.size(18.dp))
                            Spacer(Modifier.width(8.dp))
                            Text(
                                text = "✓ Home GPS set: %.4f, %.4f".format(lat, lon),
                                fontSize = 13.sp, color = OceanBlue, fontWeight = FontWeight.Medium
                            )
                        }
                    } else {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.LocationOff, null, tint = AlertRed, modifier = Modifier.size(18.dp))
                            Spacer(Modifier.width(8.dp))
                            Text("Home coordinate anchor is not set yet.", fontSize = 13.sp, color = TextSecondary)
                        }
                    }
                    Spacer(Modifier.height(12.dp))
                    Button(
                        onClick = {
                            homeCapturing = true
                            com.example.mhealth.logic.DataCollector(context).captureHomeLocation { success ->
                                homeCapturing = false
                                if (success) {
                                    Toast.makeText(context, "🏠 Home location coordinate saved!", Toast.LENGTH_SHORT).show()
                                } else {
                                    Toast.makeText(context, "❌ Location Timeout. Check GPS settings.", Toast.LENGTH_SHORT).show()
                                }
                            }
                        },
                        enabled = !homeCapturing,
                        colors = ButtonDefaults.buttonColors(containerColor = SoftCyan),
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        if (homeCapturing) {
                            CircularProgressIndicator(Modifier.size(18.dp), color = Color.White, strokeWidth = 2.dp)
                            Spacer(Modifier.width(8.dp))
                        }
                        Text(if (homeCapturing) "Getting GPS fix..." else "📌 Reset Current Location as Home", color = Color.White, fontSize = 13.sp, fontFamily = Fredoka)
                    }
                }
            }
        }

        // Section 4: JSON Backup & Import Sharing Portal
        item {
            Card(
                Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(14.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer.copy(0.1f)),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(0.15f))
            ) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text("Clinician Encrypted Sharing Portal", fontWeight = FontWeight.Bold, fontSize = 14.sp, color = MaterialTheme.colorScheme.primary, fontFamily = Fredoka)
                    Text(
                        "Lumen runs 100% on-device. No cloud servers host your reports. Export a clean, structured JSON file locally to backup your historical baseline data or share it directly with your healthcare provider.",
                        fontSize = 12.sp,
                        lineHeight = 17.sp,
                        color = MaterialTheme.colorScheme.onPrimaryContainer.copy(0.8f)
                    )

                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(
                            onClick = {
                                val dateStr = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
                                exportLauncher.launch("Lumen_backup_$dateStr.json")
                            },
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                            modifier = Modifier
                                .weight(1f)
                                .height(48.dp),
                            shape = RoundedCornerShape(10.dp)
                        ) {
                            Icon(Icons.Default.Download, null, tint = Color.White, modifier = Modifier.size(18.dp))
                            Spacer(Modifier.width(8.dp))
                            Text("Export Backup", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 12.sp, fontFamily = Fredoka)
                        }

                        Button(
                            onClick = {
                                importLauncher.launch(arrayOf("application/json"))
                            },
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.secondary),
                            modifier = Modifier
                                .weight(1f)
                                .height(48.dp),
                            shape = RoundedCornerShape(10.dp)
                        ) {
                            Icon(Icons.Default.Upload, null, tint = Color.White, modifier = Modifier.size(18.dp))
                            Spacer(Modifier.width(8.dp))
                            Text("Import Backup", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 12.sp, fontFamily = Fredoka)
                        }
                    }
                }
            }
        }

        // Section 5: Safety Reset
        item {
            Button(
                onClick = {
                    scope.launch(Dispatchers.IO) {
                        val db = MHealthDatabase.getInstance(context)
                        db.clearAllTables()
                        val localPrefWipe = context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE)
                        localPrefWipe.edit().clear().apply()
                        
                        withContext(Dispatchers.Main) {
                            Toast.makeText(context, "Local DB Wiped. Please restart app to re-onboard.", Toast.LENGTH_LONG).show()
                            (context as Activity).finishAffinity()
                        }
                    }
                },
                colors = ButtonDefaults.buttonColors(containerColor = LumenRose.copy(0.12f), contentColor = LumenRose),
                border = BorderStroke(1.dp, LumenRose.copy(0.3f)),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(48.dp),
                shape = RoundedCornerShape(10.dp)
            ) {
                Text("Wipe & Reset Local Databases", fontWeight = FontWeight.SemiBold, fontFamily = Fredoka)
            }
        }
    }
}

// =============================================================================
// Helper Components & Compatibility Variables
// =============================================================================

@Composable
fun ToggleRow(title: String, subtitle: String, checked: Boolean, color: Color, onToggle: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(title, fontSize = 13.5.sp, fontWeight = FontWeight.Medium, color = TextPrimary)
            Text(subtitle, fontSize = 11.sp, color = TextSecondary)
        }
        Switch(
            checked = checked, onCheckedChange = onToggle,
            colors = SwitchDefaults.colors(checkedThumbColor = Color.White, checkedTrackColor = color, uncheckedTrackColor = TextMuted.copy(0.3f))
        )
    }
}

@Composable
fun InfoCard(
    title: String,
    headerColor: Color = OceanBlue,
    modifier: Modifier = Modifier,
    content: @Composable ColumnScope.() -> Unit
) {
    Card(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = CardWhite),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(Modifier.padding(16.dp)) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.padding(bottom = 12.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(4.dp, 16.dp)
                        .clip(CircleShape)
                        .background(headerColor)
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    title,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                    color = TextPrimary,
                    fontFamily = Fredoka
                )
            }
            content()
        }
    }
}

@Composable
fun MetricPill(
    label: String,
    value: String,
    color: Color,
    modifier: Modifier = Modifier
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = modifier.padding(4.dp)
    ) {
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(12.dp))
                .background(color.copy(alpha = 0.1f))
                .padding(horizontal = 12.dp, vertical = 6.dp)
        ) {
            Text(
                value,
                fontSize = 14.sp,
                fontWeight = FontWeight.ExtraBold,
                color = color
            )
        }
        Spacer(Modifier.height(4.dp))
        Text(
            label,
            fontSize = 10.sp,
            color = TextSecondary,
            fontWeight = FontWeight.Medium
        )
    }
}

@Composable
fun SparklineLabel(
    label: String,
    history: List<Float>,
    color: Color,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(label, fontSize = 12.sp, color = TextSecondary, fontWeight = FontWeight.Medium)
            Spacer(Modifier.height(6.dp))
            SparklineChart(history, color, Modifier.fillMaxWidth().height(45.dp), showDots = false)
        }
    }
}

@Composable
fun PerAppBreakdownCard(vector: PersonalityVector) {
    val pm = LocalContext.current.packageManager
    val topApps = remember(vector) {
        vector.appBreakdown
            .filterKeys { it.isNotBlank() }
            .toList()
            .sortedByDescending { it.second }
            .take(7)
    }

    if (topApps.isEmpty() && vector.bgAudioBreakdown.isEmpty()) return

    if (topApps.isNotEmpty()) {
        InfoCard("Per-App Screen Breakdown", headerColor = ChartPurple) {
            Row(Modifier.fillMaxWidth().padding(bottom = 6.dp)) {
                Text("App",      fontSize = 11.sp, fontWeight = FontWeight.Bold, color = TextSecondary, modifier = Modifier.weight(2.5f))
                Text("Screen",   fontSize = 11.sp, fontWeight = FontWeight.Bold, color = TextSecondary, modifier = Modifier.weight(1.5f))
                Text("Launches", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = TextSecondary, modifier = Modifier.weight(1.3f))
                Text("Notifs",   fontSize = 11.sp, fontWeight = FontWeight.Bold, color = TextSecondary, modifier = Modifier.weight(1.2f))
            }
            HorizontalDivider(color = TextSecondary.copy(alpha = 0.15f), thickness = 0.5.dp)
            Spacer(Modifier.height(4.dp))

            topApps.forEach { (pkg, minutes) ->
                val appName = try {
                    pm.getApplicationLabel(pm.getApplicationInfo(pkg, 0)).toString()
                } catch (e: Exception) { pkg.substringAfterLast(".") }
                val launches = vector.appLaunchesBreakdown[pkg] ?: 0
                val notifs   = vector.notificationBreakdown[pkg] ?: 0
                val hrs  = minutes / 60L
                val mins = minutes % 60L
                val timeStr = if (hrs > 0) "${hrs}h ${mins}m" else "${mins}m"

                Row(Modifier.fillMaxWidth().padding(vertical = 5.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text(appName,   fontSize = 11.sp, color = TextPrimary,   modifier = Modifier.weight(2.5f),
                        maxLines = 1, overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis)
                    Text(timeStr,   fontSize = 11.sp, color = TextSecondary, modifier = Modifier.weight(1.5f))
                    Text("$launches", fontSize = 11.sp, color = TextSecondary, modifier = Modifier.weight(1.3f))
                    Text("$notifs",   fontSize = 11.sp,
                        color = if (notifs > 30) AlertOrange else TextSecondary,
                        fontWeight = if (notifs > 30) FontWeight.Bold else FontWeight.Normal,
                        modifier = Modifier.weight(1.2f))
                }
                HorizontalDivider(color = TextSecondary.copy(alpha = 0.08f), thickness = 0.5.dp)
            }
        }
    }
}

@Composable
fun BgAudioBreakdownCard(vector: PersonalityVector) {
    val pm = LocalContext.current.packageManager
    val audioApps = remember(vector) {
        vector.bgAudioBreakdown
            .filterKeys { it.isNotBlank() }
            .toList()
            .sortedByDescending { it.second }
            .take(5)
    }

    if (audioApps.isEmpty()) return

    InfoCard("Background Audio Breakdown", headerColor = OceanBlue) {
        Row(Modifier.fillMaxWidth().padding(bottom = 6.dp)) {
            Text("Music App", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = TextSecondary, modifier = Modifier.weight(3f))
            Text("Duration", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = TextSecondary, modifier = Modifier.weight(1f))
        }
        HorizontalDivider(color = TextSecondary.copy(alpha = 0.15f), thickness = 0.5.dp)
        Spacer(Modifier.height(4.dp))

        audioApps.forEach { (pkg, ms) ->
            val appName = try {
                if (pkg == "unknown_music_app") "Other Audio"
                else pm.getApplicationLabel(pm.getApplicationInfo(pkg, 0)).toString()
            } catch (e: Exception) { pkg.substringAfterLast(".") }
            
            val totalSec = ms / 1000
            val minutes = totalSec / 60
            val seconds = totalSec % 60
            val timeStr = if (minutes > 0) "${minutes}m ${seconds}s" else "${seconds}s"

            Row(Modifier.fillMaxWidth().padding(vertical = 5.dp), verticalAlignment = Alignment.CenterVertically) {
                Text(appName, fontSize = 11.sp, color = TextPrimary, modifier = Modifier.weight(3f),
                    maxLines = 1, overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis)
                Text(timeStr, fontSize = 11.sp, color = TextSecondary, modifier = Modifier.weight(1f))
            }
            HorizontalDivider(color = TextSecondary.copy(alpha = 0.08f), thickness = 0.5.dp)
        }
    }
}

@Composable
fun DeviationRow(feature: String, sd: Float) {
    val color = when {
        kotlin.math.abs(sd) > 3f -> AlertRed
        kotlin.math.abs(sd) > 2f -> AlertOrange
        else                     -> AlertYellow
    }
    Row(Modifier.fillMaxWidth().padding(vertical = 3.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(feature, fontSize = 12.sp, color = TextSecondary, modifier = Modifier.weight(1f))
        Box(
            Modifier.clip(RoundedCornerShape(12.dp)).background(color.copy(0.12f)).padding(horizontal = 8.dp, vertical = 2.dp)
        ) {
            Text("${"%.2f".format(sd)} SD", fontSize = 11.sp, color = color, fontWeight = FontWeight.Bold)
        }
    }
}

private data class FeatureRow(
    val label: String,
    val unit: String,
    val mean: Float,
    val std: Float,
    val current: Float
)

private val featureLabels: Map<String, Pair<String, String>> = linkedMapOf(
    "screenTimeHours"      to Pair("Screen Time",         "hrs"),
    "unlockCount"          to Pair("Phone Unlocks",        ""),
    "appLaunchCount"       to Pair("App Launches",         ""),
    "notificationsToday"   to Pair("Notifications",        ""),
    "socialAppRatio"       to Pair("Social App Ratio",     "%"),
    "callsPerDay"          to Pair("Calls / Day",          ""),
    "callDurationMinutes"  to Pair("Call Duration",        "min"),
    "uniqueContacts"       to Pair("Unique Contacts",      ""),
    "conversationFrequency" to Pair("Conversation Freq.", ""),
    "dailyDisplacementKm" to Pair("Displacement",          "km"),
    "locationEntropy"      to Pair("Location Entropy",     ""),
    "homeTimeRatio"        to Pair("Home Time Ratio",      "%"),
    "wakeTimeHour"         to Pair("Wake Time",            "hr"),
    "sleepTimeHour"        to Pair("Sleep Time",           "hr"),
    "sleepDurationHours"   to Pair("Sleep Duration",       "hrs"),
    "dailyStepCount"       to Pair("Step Count",           "steps"),
    "activeMinutes"        to Pair("Active Minutes",       "min"),
    "keystrokeSpeed"       to Pair("Keystroke Speed",      "char/s"),
    "backspaceRatio"       to Pair("Backspace Ratio",      "%"),
    "scrollVelocity"       to Pair("Scroll Velocity",      "px/s"),
    "daylightExposureMinutes" to Pair("Daylight Exposure", "min"),
    "chargeRegularity"     to Pair("Charge Regularity",    "%"),
    "chargeDurationHours"  to Pair("Charging Time",        "hrs"),
    "upiTransactionsToday" to Pair("UPI / Payments",        ""),
    "appUninstallsToday"   to Pair("App Uninstalls",         ""),
    "appInstallsToday"     to Pair("App Installs",          ""),
    "calendarEventsToday"  to Pair("Calendar Events",        ""),
    "mediaCountToday"      to Pair("Media Files",           ""),
    "downloadsToday"       to Pair("Downloads Today",       ""),
    "musicTimeMinutes"     to Pair("Music Time",           "min")
)

@Composable
fun FeatureTableCard(
    baseline: PersonalityVector,
    current: PersonalityVector
) {
    val rows = remember(baseline, current) {
        val baselineMap = baseline.toMap()
        val currentMap = current.toMap()
        val RATIO_FEATURES = setOf("socialAppRatio", "homeTimeRatio")

        featureLabels.mapNotNull { (key, labelUnit) ->
            val meanRaw = baselineMap[key] ?: return@mapNotNull null
            val stdRaw = baseline.variances[key] ?: 0f
            val curRaw = currentMap[key] ?: 0f
            val scale = if (key in RATIO_FEATURES) 100f else 1f
            FeatureRow(labelUnit.first, labelUnit.second, meanRaw * scale, stdRaw * scale, curRaw * scale)
        }
    }

    InfoCard("Full Baseline Reference", headerColor = SoftCyan) {
        Row(
            Modifier.fillMaxWidth().padding(bottom = 6.dp),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text("Feature",            fontSize = 11.sp, fontWeight = FontWeight.Bold, color = TextSecondary, modifier = Modifier.weight(2f))
            Text("Baseline (μ ± σ)",   fontSize = 11.sp, fontWeight = FontWeight.Bold, color = TextSecondary, modifier = Modifier.weight(2.5f))
            Text("Now",                fontSize = 11.sp, fontWeight = FontWeight.Bold, color = TextSecondary, modifier = Modifier.weight(1.2f))
            Text("Flag",               fontSize = 11.sp, fontWeight = FontWeight.Bold, color = TextSecondary, modifier = Modifier.weight(1.5f))
        }
        HorizontalDivider(color = TextSecondary.copy(alpha = 0.15f), thickness = 0.5.dp)
        Spacer(Modifier.height(4.dp))

        rows.forEach { row ->
            val std = row.std.takeIf { it > 0f } ?: (row.mean * 0.15f).coerceAtLeast(0.01f)
            val zScore = (row.current - row.mean) / std
            val (flagText, flagColor, flagIcon) = when {
                kotlin.math.abs(zScore) < 1.0f  -> Triple("Normal",    AlertGreen,  Icons.Default.Check)
                zScore > 0f                     -> Triple("Elevated",  AlertOrange, Icons.Default.ArrowUpward)
                else                            -> Triple("Decreased", SoftCyan,     Icons.Default.ArrowDownward)
            }
            val unitSuffix = if (row.unit.isNotEmpty()) " ${row.unit}" else ""
            val fmtMean    = if (row.mean < 100f) "%.1f" else "%.0f"
            val fmtStd     = if (row.std  < 100f) "%.1f" else "%.0f"
            val fmtCur     = if (row.current < 100f) "%.1f" else "%.0f"

            Row(
                Modifier.fillMaxWidth().padding(vertical = 5.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(row.label,          fontSize = 11.sp, color = TextPrimary, modifier = Modifier.weight(2f))
                Text(
                    "${fmtMean.format(row.mean)} ± ${fmtStd.format(row.std)}$unitSuffix",
                    fontSize = 11.sp, color = TextSecondary,
                    modifier = Modifier.weight(2.5f)
                )
                Text(
                    "${fmtCur.format(row.current)}$unitSuffix",
                    fontSize = 11.sp, fontWeight = FontWeight.SemiBold, color = TextPrimary,
                    modifier = Modifier.weight(1.2f)
                )
                Surface(
                    color = flagColor.copy(alpha = 0.12f),
                    shape = RoundedCornerShape(10.dp),
                    modifier = Modifier.weight(1.5f)
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.padding(horizontal = 5.dp, vertical = 3.dp)
                    ) {
                        Icon(flagIcon, null, tint = flagColor, modifier = Modifier.size(11.dp))
                        Spacer(Modifier.width(3.dp))
                        Text(flagText, fontSize = 9.sp, color = flagColor, fontWeight = FontWeight.Bold)
                    }
                }
            }
            HorizontalDivider(color = TextSecondary.copy(alpha = 0.08f), thickness = 0.5.dp)
        }
    }
}

// =============================================================================
// JSON Data Export & Import Systems (Woven into Settings Screen)
// =============================================================================

private fun exportDataToUri(context: Context, uri: android.net.Uri) {
    if (context !is ComponentActivity) return
    context.lifecycleScope.launch(Dispatchers.IO) {
        try {
            val db = MHealthDatabase.getInstance(context)
            val userId = DataRepository.userProfile.value?.email ?: "local_patient@lumen.health"
            
            // 1. Fetch All Historical Data
            val dailyHistory = db.dailyFeaturesDao().getAllFeatures(userId)
            val baselineRows = db.baselineDao().getBaseline(userId)
            val analysisReports = db.analysisResultDao().getAll(userId)
            val profile = db.userProfileDao().getProfile(userId)
            
            // 2. Construct Master JSON
            val masterJson = org.json.JSONObject()
            
            // A. Identity & Profile
            masterJson.put("profile", org.json.JSONObject().apply {
                put("userId", userId)
                put("baselineReady", profile?.baselineReady ?: false)
                put("onboardingDate", profile?.onboardingDate ?: "")
                put("currentStatus", profile?.currentStatus ?: "Collecting")
            })

            // B. Baseline (P₀)
            val baselineArr = org.json.JSONArray()
            baselineRows.forEach { row ->
                baselineArr.put(org.json.JSONObject().apply {
                    put("feature", row.featureName)
                    put("mean", row.baselineValue)
                    put("std", row.stdDeviation)
                    put("start", row.baselineStart)
                    put("end", row.baselineEnd)
                    put("contaminated", row.isContaminated)
                })
            }
            masterJson.put("baseline", baselineArr)

            // C. Daily Behavioral History
            val scoreByDate: Map<String, Float> = analysisReports.associate { it.date to it.effectiveScore }

            val historyArr = org.json.JSONArray()
            dailyHistory.forEach { day ->
                val dayObj = org.json.JSONObject()
                dayObj.put("date", day.date)
                dayObj.put("isSimulated", day.isSimulated)
                dayObj.put("anomaly_score", scoreByDate[day.date] ?: -1.0)

                val features = org.json.JSONObject().apply {
                    put("screenTimeHours", day.screenTimeHours)
                    put("unlockCount", day.unlockCount)
                    put("appLaunchCount", day.appLaunchCount)
                    put("notificationsToday", day.notificationsToday)
                    put("socialAppRatio", day.socialAppRatio)
                    put("callsPerDay", day.callsPerDay)
                    put("callDurationMinutes", day.callDurationMinutes)
                    put("uniqueContacts", day.uniqueContacts)
                    put("dailyDisplacementKm", day.dailyDisplacementKm)
                    put("locationEntropy", day.locationEntropy)
                    put("homeTimeRatio", day.homeTimeRatio)
                    put("wakeTimeHour", day.wakeTimeHour)
                    put("sleepTimeHour", day.sleepTimeHour)
                    put("sleepDurationHours", day.sleepDurationHours)
                    put("dailyStepCount", day.dailyStepCount)
                    put("activeMinutes", day.activeMinutes)
                    put("keystrokeSpeed", day.keystrokeSpeed)
                    put("backspaceRatio", day.backspaceRatio)
                    put("scrollVelocity", day.scrollVelocity)
                    put("daylightExposureMinutes", day.daylightExposureMinutes)
                    put("chargeRegularity", day.chargeRegularity)
                    put("chargeDurationHours", day.chargeDurationHours)
                    put("upiTransactionsToday", day.upiTransactionsToday)
                    put("appUninstallsToday", day.appUninstallsToday)
                    put("appInstallsToday", day.appInstallsToday)
                    put("calendarEventsToday", day.calendarEventsToday)
                    put("mediaCountToday", day.mediaCountToday)
                    put("downloadsToday", day.downloadsToday)
                    put("musicTimeMinutes", day.musicTimeMinutes)
                }
                dayObj.put("metrics", features)

                dayObj.put("detailed_logs", org.json.JSONObject().apply {
                    put("app_breakdown", org.json.JSONObject(day.appBreakdownJson))
                    put("notifications_breakdown", org.json.JSONObject(day.notificationBreakdownJson))
                    put("app_launches_breakdown", org.json.JSONObject(day.appLaunchesBreakdownJson))
                    put("bg_audio_breakdown", org.json.JSONObject(day.bgAudioBreakdownJson))
                })
                
                historyArr.put(dayObj)
            }
            masterJson.put("daily_history", historyArr)

            // D. Today's LIVE snapshot
            val liveVector = DataRepository.latestVector.value
            if (liveVector != null) {
                val todayStr = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()).format(Date())
                val todayObj = org.json.JSONObject()
                todayObj.put("date", todayStr)
                todayObj.put("is_live_snapshot", true)
                todayObj.put("isSimulated", false)
                todayObj.put("metrics", org.json.JSONObject().apply {
                    put("screenTimeHours", liveVector.screenTimeHours)
                    put("unlockCount", liveVector.unlockCount)
                    put("appLaunchCount", liveVector.appLaunchCount)
                    put("notificationsToday", liveVector.notificationsToday)
                    put("socialAppRatio", liveVector.socialAppRatio)
                    put("callsPerDay", liveVector.callsPerDay)
                    put("callDurationMinutes", liveVector.callDurationMinutes)
                    put("uniqueContacts", liveVector.uniqueContacts)
                    put("dailyDisplacementKm", liveVector.dailyDisplacementKm)
                    put("locationEntropy", liveVector.locationEntropy)
                    put("homeTimeRatio", liveVector.homeTimeRatio)
                    put("wakeTimeHour", liveVector.wakeTimeHour)
                    put("sleepTimeHour", liveVector.sleepTimeHour)
                    put("sleepDurationHours", liveVector.sleepDurationHours)
                    put("dailyStepCount", liveVector.dailyStepCount)
                    put("activeMinutes", liveVector.activeMinutes)
                    put("keystrokeSpeed", liveVector.keystrokeSpeed)
                    put("backspaceRatio", liveVector.backspaceRatio)
                    put("scrollVelocity", liveVector.scrollVelocity)
                    put("daylightExposureMinutes", liveVector.daylightExposureMinutes)
                    put("chargeRegularity", liveVector.chargeRegularity)
                    put("chargeDurationHours", liveVector.chargeDurationHours)
                    put("upiTransactionsToday", liveVector.upiTransactionsToday)
                    put("appUninstallsToday", liveVector.appUninstallsToday)
                    put("appInstallsToday", liveVector.appInstallsToday)
                    put("calendarEventsToday", liveVector.calendarEventsToday)
                    put("mediaCountToday", liveVector.mediaCountToday)
                    put("downloadsToday", liveVector.downloadsToday)
                    put("musicTimeMinutes", liveVector.musicTimeMinutes)
                })
                
                todayObj.put("location_snapshots", DataRepository.locationSnapshots.value.joinToString(";") { "${it.lat},${it.lon},${it.timeMs}" })
                todayObj.put("charge_hours", DataRepository.accumulatedChargeHours.value)
                todayObj.put("bg_audio_ms", DataRepository.accumulatedBgAudioMs.value)
                todayObj.put("step_baseline", DataRepository.stepBaseline.value ?: -1f)
                
                masterJson.put("today_live", todayObj)
            }

            // E. Analysis History (Anomaly detections)
            val reportsArr = org.json.JSONArray()
            analysisReports.forEach { report ->
                reportsArr.put(org.json.JSONObject().apply {
                    put("date", report.date)
                    put("anomalyDetected", report.anomalyDetected)
                    put("anomalyScore", report.anomalyScore)
                    put("effectiveScore", report.effectiveScore)
                    put("l2Modifier", report.l2Modifier)
                    put("coherence", report.coherence)
                    put("rhythmDissolution", report.rhythmDissolution)
                    put("sessionIncoherence", report.sessionIncoherence)
                    put("evidenceAccumulated", report.evidenceAccumulated)
                    put("anomalyMessage", report.anomalyMessage)
                    put("alertLevel", report.alertLevel)
                    put("sustainedDays", report.sustainedDays)
                    put("prototypeMatch", report.prototypeMatch)
                    put("matchMessage", report.matchMessage)
                    put("prototypeConfidence", report.prototypeConfidence)
                    put("gateResults", org.json.JSONObject(report.gateResults))
                })
            }
            masterJson.put("analysis_reports", reportsArr)

            // 3. Write directly to Uri selected by User via CreateDocument
            context.contentResolver.openOutputStream(uri)?.use { outputStream ->
                outputStream.write(masterJson.toString(4).toByteArray())
            }

            withContext(Dispatchers.Main) {
                Toast.makeText(context, "✅ Backup exported successfully!", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            e.printStackTrace()
            withContext(Dispatchers.Main) {
                Toast.makeText(context, "❌ Export failed: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }
    }
}

private fun importBackupDataFromJson(context: Context, uri: android.net.Uri) {
    if (context !is ComponentActivity) return
    
    Toast.makeText(context, "Importing backup...", Toast.LENGTH_SHORT).show()

    context.lifecycleScope.launch(Dispatchers.IO) {
        try {
            val contentResolver = context.contentResolver
            val inputStream = contentResolver.openInputStream(uri) ?: throw Exception("Cannot open file")
            val jsonString = inputStream.bufferedReader().use { it.readText() }
            val masterJson = org.json.JSONObject(jsonString)
            
            val db = MHealthDatabase.getInstance(context)
            
            // Parse Profile
            if (masterJson.has("profile")) {
                val profileObj = masterJson.getJSONObject("profile")
                val userId = profileObj.optString("userId", "patient@lumen.health")
                val isReady = profileObj.optBoolean("baselineReady", false)
                val onboarding = profileObj.optString("onboardingDate", "")
                val status = profileObj.optString("currentStatus", "Collecting")
                
                db.userProfileDao().upsert(UserProfileEntity(
                    userId = userId,
                    baselineReady = isReady,
                    onboardingDate = onboarding,
                    currentStatus = status
                ))
            }
            
            val userId = DataRepository.userProfile.value?.email ?: "patient@lumen.health"
            
            // Parse Baseline
            if (masterJson.has("baseline")) {
                val baselineArr = masterJson.getJSONArray("baseline")
                val entities = mutableListOf<com.example.mhealth.logic.db.BaselineEntity>()
                for (i in 0 until baselineArr.length()) {
                    val obj = baselineArr.getJSONObject(i)
                    entities.add(com.example.mhealth.logic.db.BaselineEntity(
                        userId = userId,
                        featureName = obj.optString("feature"),
                        baselineValue = obj.optDouble("mean", 0.0).toFloat(),
                        stdDeviation = obj.optDouble("std", 0.0).toFloat(),
                        baselineStart = obj.optString("start", ""),
                        baselineEnd = obj.optString("end", ""),
                        isContaminated = obj.optBoolean("contaminated", false)
                    ))
                }
                if (entities.isNotEmpty()) {
                    db.baselineDao().insertAll(entities)
                }
            }
            
            // Parse Daily History
            if (masterJson.has("daily_history")) {
                val historyArr = masterJson.getJSONArray("daily_history")
                for (i in 0 until historyArr.length()) {
                    val dayObj = historyArr.getJSONObject(i)
                    val date = dayObj.optString("date")
                    val isSim = dayObj.optBoolean("isSimulated", false)
                    
                    val metrics = dayObj.optJSONObject("metrics") ?: continue
                    val logs = dayObj.optJSONObject("detailed_logs")
                    
                    val entity = com.example.mhealth.logic.db.DailyFeaturesEntity(
                        userId = userId,
                        date = date,
                        isSimulated = isSim,
                        screenTimeHours = metrics.optDouble("screenTimeHours", 0.0).toFloat(),
                        unlockCount = metrics.optDouble("unlockCount", 0.0).toFloat(),
                        appLaunchCount = metrics.optDouble("appLaunchCount", 0.0).toFloat(),
                        notificationsToday = metrics.optDouble("notificationsToday", 0.0).toFloat(),
                        socialAppRatio = metrics.optDouble("socialAppRatio", 0.0).toFloat(),
                        callsPerDay = metrics.optDouble("callsPerDay", 0.0).toFloat(),
                        callDurationMinutes = metrics.optDouble("callDurationMinutes", 0.0).toFloat(),
                        uniqueContacts = metrics.optDouble("uniqueContacts", 0.0).toFloat(),
                        conversationFrequency = metrics.optDouble("conversationFrequency", 0.0).toFloat(),
                        dailyDisplacementKm = metrics.optDouble("dailyDisplacementKm", 0.0).toFloat(),
                        locationEntropy = metrics.optDouble("locationEntropy", 0.0).toFloat(),
                        homeTimeRatio = metrics.optDouble("homeTimeRatio", 0.0).toFloat(),
                        wakeTimeHour = metrics.optDouble("wakeTimeHour", 0.0).toFloat(),
                        sleepTimeHour = metrics.optDouble("sleepTimeHour", 0.0).toFloat(),
                        sleepDurationHours = metrics.optDouble("sleepDurationHours", 0.0).toFloat(),
                        dailyStepCount = metrics.optDouble("dailyStepCount", 0.0).toFloat(),
                        activeMinutes = metrics.optDouble("activeMinutes", 0.0).toFloat(),
                        keystrokeSpeed = metrics.optDouble("keystrokeSpeed", 0.0).toFloat(),
                        backspaceRatio = metrics.optDouble("backspaceRatio", 0.0).toFloat(),
                        scrollVelocity = metrics.optDouble("scrollVelocity", 0.0).toFloat(),
                        daylightExposureMinutes = metrics.optDouble("daylightExposureMinutes", 0.0).toFloat(),
                        chargeRegularity = metrics.optDouble("chargeRegularity", 0.0).toFloat(),
                        chargeDurationHours = metrics.optDouble("chargeDurationHours", 0.0).toFloat(),
                        upiTransactionsToday = metrics.optDouble("upiTransactionsToday", 0.0).toFloat(),
                        appUninstallsToday = metrics.optDouble("appUninstallsToday", 0.0).toFloat(),
                        appInstallsToday = metrics.optDouble("appInstallsToday", 0.0).toFloat(),
                        calendarEventsToday = metrics.optDouble("calendarEventsToday", 0.0).toFloat(),
                        mediaCountToday = metrics.optDouble("mediaCountToday", 0.0).toFloat(),
                        downloadsToday = metrics.optDouble("downloadsToday", 0.0).toFloat(),
                        musicTimeMinutes = metrics.optDouble("musicTimeMinutes", 0.0).toFloat(),
                        appBreakdownJson = logs?.optJSONObject("app_breakdown")?.toString() ?: "{}",
                        notificationBreakdownJson = logs?.optJSONObject("notifications_breakdown")?.toString() ?: "{}",
                        appLaunchesBreakdownJson = logs?.optJSONObject("app_launches_breakdown")?.toString() ?: "{}",
                        bgAudioBreakdownJson = logs?.optJSONObject("bg_audio_breakdown")?.toString() ?: "{}"
                    )
                    db.dailyFeaturesDao().insert(entity)
                }
            }
            
            // Rehydrate Live Accumulators
            if (masterJson.has("today_live")) {
                val liveObj = masterJson.getJSONObject("today_live")
                val locStr = liveObj.optString("location_snapshots", "")
                val locs = if (locStr.isNotEmpty()) {
                    locStr.split(";").filter { it.isNotBlank() }.map { 
                        val parts = it.split(",")
                        com.example.mhealth.models.LatLonPoint(
                            parts[0].toDouble(), 
                            parts[1].toDouble(), 
                            parts[2].toLong(),
                            if (parts.size > 3) parts[3].toFloat() else 0f
                        )
                    }
                } else emptyList()
                
                val chargeHrs = liveObj.optDouble("charge_hours", 0.0).toFloat()
                val bgAudio = liveObj.optLong("bg_audio_ms", 0L)
                val stepBase = liveObj.optDouble("step_baseline", -1.0).toFloat()
                
                DataRepository.restoreTodayState(locs, chargeHrs, bgAudio, stepBase)
            }
            
            // Parse Analysis Reports
            if (masterJson.has("analysis_reports")) {
                val reportsArr = masterJson.getJSONArray("analysis_reports")
                for (i in 0 until reportsArr.length()) {
                    val reportObj = reportsArr.getJSONObject(i)
                    val r = AnalysisResultEntity(
                        userId = userId,
                        date = reportObj.optString("date"),
                        anomalyDetected = reportObj.optBoolean("anomalyDetected"),
                        anomalyScore = reportObj.optDouble("anomalyScore", 0.0).toFloat(),
                        anomalyMessage = reportObj.optString("anomalyMessage", ""),
                        alertLevel = reportObj.optString("alertLevel", "Normal"),
                        sustainedDays = reportObj.optInt("sustainedDays", 0),
                        prototypeMatch = reportObj.optString("prototypeMatch", "Normal"),
                        matchMessage = reportObj.optString("matchMessage", ""),
                        prototypeConfidence = reportObj.optDouble("prototypeConfidence", 0.0).toFloat(),
                        gateResults = reportObj.optJSONObject("gateResults")?.toString() ?: "{}",
                        l2Modifier = reportObj.optDouble("l2Modifier", 1.0).toFloat(),
                        coherence = reportObj.optDouble("coherence", 0.0).toFloat(),
                        rhythmDissolution = reportObj.optDouble("rhythmDissolution", 0.0).toFloat(),
                        sessionIncoherence = reportObj.optDouble("sessionIncoherence", 0.0).toFloat(),
                        effectiveScore = reportObj.optDouble("effectiveScore", 0.0).toFloat(),
                        evidenceAccumulated = reportObj.optDouble("evidenceAccumulated", 0.0).toFloat()
                    )
                    db.analysisResultDao().insert(r)
                }
            }
            
            withContext(Dispatchers.Main) {
                Toast.makeText(context, "✅ Backup imported successfully. Please restart Lumen to view imported data.", Toast.LENGTH_LONG).show()
            }
        } catch (e: Exception) {
            e.printStackTrace()
            withContext(Dispatchers.Main) {
                Toast.makeText(context, "❌ Import failed: Invalid backup file", Toast.LENGTH_SHORT).show()
            }
        }
    }
}

private fun startMonitoringService(context: Context) {
    val intent = Intent(context, com.example.mhealth.services.MonitoringService::class.java)
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        context.startForegroundService(intent)
    } else {
        context.startService(intent)
    }
}

private fun hasUsageStatsPermission(context: Context): Boolean {
    val appOps = context.getSystemService(Context.APP_OPS_SERVICE) as? AppOpsManager ?: return false
    val mode = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
        appOps.unsafeCheckOpNoThrow(
            AppOpsManager.OPSTR_GET_USAGE_STATS,
            Process.myUid(),
            context.packageName
        )
    } else {
        @Suppress("DEPRECATION")
        appOps.checkOpNoThrow(
            AppOpsManager.OPSTR_GET_USAGE_STATS,
            Process.myUid(),
            context.packageName
        )
    }
    return mode == AppOpsManager.MODE_ALLOWED
}

private fun alertColorForLevel(level: String): Color = when (level.lowercase()) {
    "green" -> AlertGreen
    "yellow" -> AlertYellow
    "orange" -> AlertOrange
    "red" -> AlertRed
    else -> AlertGreen
}
