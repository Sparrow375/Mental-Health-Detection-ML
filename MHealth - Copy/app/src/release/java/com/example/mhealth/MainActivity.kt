package com.example.mhealth

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
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
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
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
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.core.content.ContextCompat
import com.example.mhealth.logic.DataRepository
import com.example.mhealth.logic.ReportGenerator
import com.example.mhealth.logic.db.AnalysisResultEntity
import com.example.mhealth.logic.db.MHealthDatabase
import com.example.mhealth.logic.db.UserProfileEntity
import com.example.mhealth.models.DailyReport
import com.example.mhealth.models.PersonalityVector
import com.example.mhealth.ui.theme.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.*

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
// App Navigation & Layout
// =============================================================================
enum class LumenDest(val label: String, val icon: ImageVector) {
    HOME("Breathe", Icons.Default.Sensors),
    INSIGHTS("Insights", Icons.Default.Lightbulb),
    CHECK_IN("Check In", Icons.Default.FavoriteBorder),
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
                        1 -> "Welcome to Lumen"
                        2 -> "Personalize Your Profile"
                        3 -> "Set Your Home Anchor"
                        4 -> "Personal Health Questionnaire (PHQ-9)"
                        5 -> "Generalized Anxiety Screener (GAD-7)"
                        6 -> "Stressful Life Events"
                        else -> "Calibration Completed"
                    },
                    fontSize = 22.sp,
                    fontWeight = FontWeight.Bold,
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
                            color = MaterialTheme.colorScheme.onBackground
                        )
                        Spacer(Modifier.height(12.dp))
                        Text(
                            "Lumen operates 100% locally and offline. It gathers passive behavioral telemetry—such as sleep patterns, physical movement, typing speeds, and social frequency—to construct a personalized Digital DNA and assist your clinician's assessment.",
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
                            Text("Begin Configuration", color = Color.White, fontWeight = FontWeight.Bold)
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
                                Text("Continue Setup", color = Color.White, fontWeight = FontWeight.Bold)
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
                                    Text("Home Location Accuracy", fontWeight = FontWeight.Bold, fontSize = 14.sp)
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
                            Text(if (homeCapturing) "Acquiring GPS Signal..." else "📌 Capture Current GPS as Home", color = Color.White)
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
                            Text(if (homeSet) "Next: Clinical Screeners" else "Skip Home Location for Now", color = Color.White)
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
                            Text("Calibrate Thresholds", color = Color.White, fontWeight = FontWeight.Bold)
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
                            textAlign = TextAlign.Center
                        )

                        Card(
                            Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(12.dp),
                            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(0.3f))
                        ) {
                            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Text("Calibration Summary", fontWeight = FontWeight.Bold, fontSize = 13.sp)
                                Divider(color = MaterialTheme.colorScheme.outline.copy(0.2f))
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
                            Text("Launch Lumen Dashboard", color = Color.White, fontWeight = FontWeight.Bold)
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
                progress = (qIndex + 1).toFloat() / questions.size,
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

        // Answers options vertical list
        Column(
            modifier = Modifier.weight(1f),
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
                Text("Back")
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
                Text(if (qIndex == questions.size - 1) "Complete" else "Next")
            }
        }
    }
}

// =============================================================================
// Main Lumen Dashboard (4 Tabs: Home, Insights, Check In, Settings)
// =============================================================================
@Composable
fun MainLumenDashboard() {
    var selectedTab by remember { mutableStateOf(LumenDest.HOME) }
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
        // Boot up service once permissions are sorted
        startMonitoringService(context)
    }

    LaunchedEffect(Unit) {
        DataRepository.initWithDb(context.applicationContext, "patient@lumen.health")
        launcher.launch(perms.toTypedArray())
    }

    Scaffold(
        bottomBar = {
            NavigationBar(containerColor = MaterialTheme.colorScheme.surface, tonalElevation = 0.dp) {
                LumenDest.entries.forEach { dest ->
                    val isSelected = selectedTab == dest
                    NavigationBarItem(
                        selected = isSelected,
                        onClick = { selectedTab = dest },
                        icon = { Icon(dest.icon, contentDescription = dest.label) },
                        label = { Text(dest.label, fontSize = 11.sp, fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal) },
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
                LumenDest.INSIGHTS -> InsightsScreen()
                LumenDest.CHECK_IN -> CheckInScreen()
                LumenDest.SETTINGS -> SettingsScreen()
            }
        }
    }
}

// =============================================================================
// HomeScreen Tab (Pulsing guided breathing + Status Indicator + Mood logger)
// =============================================================================
@Composable
fun HomeScreen() {
    val analysisResult by DataRepository.latestAnalysisResult.collectAsState()
    val isDnaReady by DataRepository.isDnaBaselineReady.collectAsState()
    val baselineDays by DataRepository.baselineProgress.collectAsState()
    
    val alertLevel = analysisResult?.alertLevel ?: "stable"
    val isSensitive = (DataRepository.phq9Score.collectAsState().value >= 10 || DataRepository.gad7Score.collectAsState().value >= 10)

    LazyColumn(
        Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentPadding = PaddingValues(20.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp)
    ) {
        // App header title
        item {
            Row(
                Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column {
                    Text("LUMEN", fontSize = 24.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary)
                    Text("PASSIVE CLINICAL BIOMARKERS", fontSize = 11.sp, fontWeight = FontWeight.SemiBold, color = MaterialTheme.colorScheme.outline)
                }
                Box(
                    Modifier
                        .size(36.dp)
                        .clip(CircleShape)
                        .background(LumenTeal.copy(0.1f)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(Icons.Default.Shield, null, tint = LumenTeal, modifier = Modifier.size(18.dp))
                }
            }
        }

        // Section: Guided Breathing Animation Ripple (Canvas)
        item {
            Card(
                Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
            ) {
                Column(
                    Modifier.padding(20.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Text("Guided Calm Breathing", fontWeight = FontWeight.Bold, fontSize = 14.sp)
                    Text("Match the expanding aura pattern below to calm your nervous system.", fontSize = 12.sp, color = MaterialTheme.colorScheme.outline, textAlign = TextAlign.Center)

                    Box(
                        Modifier
                            .height(180.dp)
                            .fillMaxWidth(),
                        contentAlignment = Alignment.Center
                    ) {
                        BreathingRippleCanvas()
                    }
                }
            }
        }

        // Section: Status Card
        item {
            Card(
                Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = when (alertLevel.lowercase()) {
                        "orange", "red" -> LumenRose.copy(0.08f)
                        "yellow" -> LumenAmber.copy(0.08f)
                        else -> LumenTeal.copy(0.08f)
                    }
                ),
                border = BorderStroke(
                    1.dp,
                    when (alertLevel.lowercase()) {
                        "orange", "red" -> LumenRose.copy(0.2f)
                        "yellow" -> LumenAmber.copy(0.2f)
                        else -> LumenTeal.copy(0.2f)
                    }
                )
            ) {
                Row(Modifier.padding(18.dp), verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        Modifier
                            .size(12.dp)
                            .clip(CircleShape)
                            .background(
                                when (alertLevel.lowercase()) {
                                    "orange", "red" -> LumenRose
                                    "yellow" -> LumenAmber
                                    else -> LumenTeal
                                }
                            )
                    )
                    Spacer(Modifier.width(12.dp))
                    Column(Modifier.weight(1f)) {
                        Text(
                            text = when {
                                !isDnaReady -> "Collecting Baseline (Day $baselineDays)"
                                alertLevel.lowercase() in listOf("orange", "red") -> "Noticeable Behavioral Variations"
                                alertLevel.lowercase() == "yellow" -> "Minor Lifestyle Shift Detected"
                                else -> "Behavioral Rhythms are Steady"
                            },
                            fontWeight = FontWeight.Bold,
                            fontSize = 14.sp
                        )
                        Text(
                            text = when {
                                !isDnaReady -> "Lumen is learning your personal digital habits. Continue using your device normally to establish anchors."
                                alertLevel.lowercase() in listOf("orange", "red") -> "Passive sensors noticed marked changes in sleep, steps, or screen habits. A check-in with a practitioner might be helpful."
                                alertLevel.lowercase() == "yellow" -> "Some minor fluctuations found, but well within expected baseline parameters. Keep tracking."
                                else -> "Your location entropy, physical mobility, sleep sync, and typing metrics are close to your locked personal DNA profile."
                            },
                            fontSize = 12.sp,
                            lineHeight = 17.sp,
                            color = MaterialTheme.colorScheme.onBackground.copy(0.6f)
                        )
                    }
                }
            }
        }

        // Section: 5-Emoji Mood check-in
        item {
            Card(
                Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
            ) {
                var selectedMood by remember { mutableStateOf<Int?>(null) }
                val moods = listOf("😢", "😔", "😐", "🙂", "✨")
                val labels = listOf("Low", "Mild", "Steady", "Good", "Serene")
                
                Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text("Quick Mood Check-in", fontWeight = FontWeight.Bold, fontSize = 14.sp)
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        moods.forEachIndexed { idx, emoji ->
                            val isSelected = selectedMood == idx
                            Column(
                                horizontalAlignment = Alignment.CenterHorizontally,
                                modifier = Modifier
                                    .clickable { selectedMood = idx }
                                    .border(
                                        1.dp,
                                        if (isSelected) MaterialTheme.colorScheme.primary else Color.Transparent,
                                        RoundedCornerShape(8.dp)
                                    )
                                    .background(if (isSelected) MaterialTheme.colorScheme.primary.copy(0.05f) else Color.Transparent)
                                    .padding(vertical = 8.dp, horizontal = 6.dp)
                                    .width(48.dp)
                            ) {
                                Text(emoji, fontSize = 22.sp)
                                Spacer(Modifier.height(4.dp))
                                Text(labels[idx], fontSize = 10.sp, fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal, color = if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outline)
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun BreathingRippleCanvas() {
    val infiniteTransition = rememberInfiniteTransition()
    
    // Wave ripple animation parameters
    val scalePulse by infiniteTransition.animateFloat(
        initialValue = 1.0f,
        targetValue = 2.0f,
        animationSpec = infiniteRepeatable(
            animation = tween(4000, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Restart
        )
    )
    val alphaPulse by infiniteTransition.animateFloat(
        initialValue = 0.4f,
        targetValue = 0.0f,
        animationSpec = infiniteRepeatable(
            animation = tween(4000, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Restart
        )
    )

    // Breathing phase: 16 second infinite loop (4 inhale, 4 hold, 4 exhale, 4 hold)
    val clockValue = infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 16f,
        animationSpec = infiniteRepeatable(
            animation = tween(16000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        )
    ).value

    val phaseValue = clockValue % 16f
    
    val breathingText = when {
        phaseValue < 4f -> "Inhale..."
        phaseValue < 8f -> "Hold..."
        phaseValue < 12f -> "Exhale..."
        else -> "Hold..."
    }

    val baseScale = when {
        phaseValue < 4f -> 1f + (phaseValue / 4f) * 0.5f // Expand 1.0 -> 1.5
        phaseValue < 8f -> 1.5f // Hold
        phaseValue < 12f -> 1.5f - ((phaseValue - 8f) / 4f) * 0.5f // Contract 1.5 -> 1.0
        else -> 1.0f // Hold
    }

    val primaryColor = MaterialTheme.colorScheme.primary
    val calmingCyan = softCyanAlias

    Box(contentAlignment = Alignment.Center) {
        Canvas(modifier = Modifier.size(160.dp)) {
            val center = Offset(size.width / 2, size.height / 2)
            val outerRadius = 36.dp.toPx()

            // 1. Draw expanding background ripples
            drawCircle(
                color = calmingCyan,
                radius = outerRadius * scalePulse,
                center = center,
                alpha = alphaPulse,
                style = Stroke(width = 2.dp.toPx())
            )

            // 2. Draw pulsing breathing circle aura
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(
                        primaryColor.copy(alpha = 0.35f),
                        calmingCyan.copy(alpha = 0.02f)
                    ),
                    center = center,
                    radius = outerRadius * baseScale * 1.5f
                ),
                radius = outerRadius * baseScale * 1.5f,
                center = center
            )

            // 3. Draw core guided breathing physical circle
            drawCircle(
                color = primaryColor,
                radius = outerRadius * baseScale,
                center = center,
                alpha = 0.85f
            )
        }

        // Inner breathing guided prompt text
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = breathingText,
                color = Color.White,
                fontWeight = FontWeight.Bold,
                fontSize = 13.sp
            )
        }
    }
}

// =============================================================================
// InsightsScreen Tab (100% Qualitative, No absolute metrics or raw scores)
// =============================================================================
@Composable
fun InsightsScreen() {
    val weeklyFeatures by DataRepository.weeklyFeatureHistory.collectAsState()
    val baseline by DataRepository.baseline.collectAsState()
    val isDnaReady by DataRepository.isDnaBaselineReady.collectAsState()
    val analysisResult by DataRepository.latestAnalysisResult.collectAsState()

    LazyColumn(
        Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentPadding = PaddingValues(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            Column {
                Text("Insights Dashboard", fontSize = 22.sp, fontWeight = FontWeight.Bold)
                Text("Demographics-anchored qualitative summaries", fontSize = 12.sp, color = MaterialTheme.colorScheme.outline)
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
                            fontSize = 14.sp
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
                            Text("Longitudinal Rhythm Stability", fontWeight = FontWeight.Bold, fontSize = 14.sp)
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
                    Text("Behavioral Strata Analysis", fontWeight = FontWeight.Bold, fontSize = 14.sp, modifier = Modifier.padding(top = 8.dp, bottom = 4.dp))
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

                // Location Card
                item {
                    val homeDiff = latest.homeTimeRatio - base.homeTimeRatio
                    QualitativeInsightCard(
                        title = "Geographical Entropy",
                        icon = Icons.Default.Map,
                        badgeText = when {
                            homeDiff > 0.15f -> "Confinement"
                            else -> "Active Displacement"
                        },
                        badgeColor = when {
                            homeDiff > 0.15f -> LumenRose
                            else -> LumenTeal
                        },
                        description = when {
                            homeDiff > 0.15f -> "Confinement to your home anchor is highly elevated. Transitioning between grid cells outdoors enhances mood variance."
                            else -> "Your displacement distance and grid-cell entropy metrics are well balanced."
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
    
    // Reverse features to display chronologically (left to right)
    val reversed = features.take(7).reversed()

    Canvas(Modifier.fillMaxSize()) {
        if (reversed.size < 2) return@Canvas

        val maxVal = reversed.map { it.screenTimeHours }.maxOrNull() ?: 1f
        val minVal = reversed.map { it.screenTimeHours }.minOrNull() ?: 0f
        val range = if (maxVal - minVal > 0f) maxVal - minVal else 1f

        val path = Path()
        val spacing = size.width / (reversed.size - 1)

        reversed.forEachIndexed { idx, item ->
            val relativeY = (item.screenTimeHours - minVal) / range // 0..1
            val x = idx * spacing
            // Flip coordinate because y starts at top
            val y = size.height - (relativeY * (size.height - 20.dp.toPx())) - 10.dp.toPx()

            if (idx == 0) {
                path.moveTo(x, y)
            } else {
                path.lineTo(x, y)
            }
        }

        // Draw path line
        drawPath(
            path = path,
            brush = lineBrush,
            style = Stroke(width = 3.dp.toPx(), cap = StrokeCap.Round)
        )

        // Draw dot points
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
fun CheckInScreen() {
    var inProgress by remember { mutableStateOf(false) }
    var completeMessage by remember { mutableStateOf<String?>(null) }
    
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

    if (completeMessage != null) {
        Column(
            Modifier
                .fillMaxSize()
                .padding(24.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Box(
                Modifier
                    .size(80.dp)
                    .clip(CircleShape)
                    .background(LumenTeal.copy(0.12f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(Icons.Default.Check, null, tint = LumenTeal, modifier = Modifier.size(36.dp))
            }
            Spacer(Modifier.height(16.dp))
            Text("Check-in Completed", fontWeight = FontWeight.Bold, fontSize = 18.sp)
            Spacer(Modifier.height(8.dp))
            Text(completeMessage!!, fontSize = 12.sp, color = MaterialTheme.colorScheme.outline, textAlign = TextAlign.Center, lineHeight = 18.sp)
            Spacer(Modifier.height(24.dp))
            Button(
                onClick = { 
                    completeMessage = null 
                    inProgress = false 
                },
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                shape = RoundedCornerShape(10.dp)
            ) {
                Text("Return to Check-In")
            }
        }
    } else if (inProgress) {
        var screenerStep by remember { mutableIntStateOf(1) } // 1: PHQ, 2: GAD

        Column(Modifier.fillMaxSize()) {
            Box(
                Modifier
                    .fillMaxWidth()
                    .background(MaterialTheme.colorScheme.primary)
                    .padding(vertical = 12.dp, horizontal = 24.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    IconButton(onClick = { inProgress = false }) {
                        Icon(Icons.Default.ArrowBack, null, tint = Color.White)
                    }
                    Text("Weekly Screening", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 16.sp)
                }
            }
            Box(Modifier.weight(1f)) {
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
                            
                            completeMessage = "Your scores (PHQ-9: $totalPhq, GAD-7: $totalGad) were saved locally. Your on-device engines have adjusted their calibration sensitivity accordingly to protect your diagnostic accuracy."
                        }
                    )
                }
            }
        }
    } else {
        Column(
            Modifier
                .fillMaxSize()
                .padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                Text("Weekly Mental Check-In", fontSize = 20.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary, textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth())
                
                Card(
                    shape = RoundedCornerShape(12.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(0.3f))
                ) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.Info, null, tint = MaterialTheme.colorScheme.primary)
                            Spacer(Modifier.width(8.dp))
                            Text("Why do screeners matter?", fontWeight = FontWeight.Bold, fontSize = 13.sp)
                        }
                        Text(
                            "Standardized screeners provide high-value demographic anchor calibration. Passive phone telemetry maps your physiological signals, while these surveys map your psychological baseline, generating a comprehensive, locked clinical correlation profile.",
                            fontSize = 12.sp,
                            lineHeight = 17.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(0.8f)
                        )
                    }
                }
                
                Text(
                    "This screening wizard takes approximately 2 minutes to complete and is saved safely in the local Room database to secure your clinician exports.",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onBackground.copy(0.6f),
                    textAlign = TextAlign.Center
                )
            }

            Button(
                onClick = {
                    phq9Answers.fill(-1)
                    gad7Answers.fill(-1)
                    inProgress = true
                },
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(52.dp),
                shape = RoundedCornerShape(12.dp)
            ) {
                Text("Start Weekly Check-In", color = Color.White, fontWeight = FontWeight.Bold)
            }
        }
    }
}

// =============================================================================
// SettingsScreen Tab (Demographics + Notification switches + Share Sheets)
// =============================================================================
@Composable
fun SettingsScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    
    val profile by DataRepository.userProfile.collectAsState()
    val checkinEnabled by DataRepository.checkinNotificationsEnabled.collectAsState()
    val phqScore = DataRepository.phq9Score.collectAsState()
    val gadScore = DataRepository.gad7Score.collectAsState()

    var showPinDialog by remember { mutableStateOf(false) }
    var shareLoading by remember { mutableStateOf(false) }

    LazyColumn(
        Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentPadding = PaddingValues(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            Column {
                Text("Profile & Settings", fontSize = 22.sp, fontWeight = FontWeight.Bold)
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
                    Text("Patient Identity Metadata", fontWeight = FontWeight.Bold, fontSize = 13.5.sp)
                    Divider(color = MaterialTheme.colorScheme.outline.copy(0.1f))
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

        // Section 2: Weekly check-in notification toggles
        item {
            Card(
                Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(14.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.08f))
            ) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("Notifications & Reminders", fontWeight = FontWeight.Bold, fontSize = 13.5.sp)
                    Divider(color = MaterialTheme.colorScheme.outline.copy(0.1f))
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text("Weekly Screener Alert", fontWeight = FontWeight.SemiBold, fontSize = 12.5.sp)
                            Text("Trigger reminder notifications for check-ins.", fontSize = 11.sp, color = MaterialTheme.colorScheme.outline)
                        }
                        Switch(
                            checked = checkinEnabled,
                            onCheckedChange = { DataRepository.setCheckinNotificationsEnabled(it) },
                            colors = SwitchDefaults.colors(checkedThumbColor = MaterialTheme.colorScheme.primary)
                        )
                    }
                }
            }
        }

        // Section 3: Share Report Card
        item {
            Card(
                Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(14.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer.copy(0.1f)),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(0.15f))
            ) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text("Clinician Sharing Portal", fontWeight = FontWeight.Bold, fontSize = 14.sp, color = MaterialTheme.colorScheme.primary)
                    Text(
                        "Lumen runs 100% on-device. No cloud servers host your reports. Compile a secure, password-encrypted PDF and a structured behavioral JSON locally to share with your healthcare provider.",
                        fontSize = 12.sp,
                        lineHeight = 17.sp,
                        color = MaterialTheme.colorScheme.onPrimaryContainer.copy(0.8f)
                    )

                    Button(
                        onClick = { showPinDialog = true },
                        colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(48.dp),
                        shape = RoundedCornerShape(10.dp)
                    ) {
                        if (shareLoading) {
                            CircularProgressIndicator(Modifier.size(20.dp), color = Color.White, strokeWidth = 2.dp)
                        } else {
                            Text("Share Clinical Data", color = Color.White, fontWeight = FontWeight.Bold)
                        }
                    }
                }
            }
        }

        // Section 4: Safety Reset
        item {
            Button(
                onClick = {
                    scope.launch(Dispatchers.IO) {
                        val db = MHealthDatabase.getInstance(context)
                        db.clearAllTables()
                        val localPref = context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE)
                        localPref.edit().clear().apply()
                        
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
                Text("Wipe & Reset Local Databases", fontWeight = FontWeight.SemiBold)
            }
        }
    }

    // Encrypted PIN share dialog
    if (showPinDialog) {
        var pinText by remember { mutableStateOf("") }
        var pinConfirm by remember { mutableStateOf("") }
        var errorMessage by remember { mutableStateOf<String?>(null) }

        Dialog(onDismissRequest = { showPinDialog = false }) {
            Card(
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp)
            ) {
                Column(
                    modifier = Modifier.padding(20.dp),
                    verticalArrangement = Arrangement.spacedBy(14.dp)
                ) {
                    Text("Secure PDF Encryption PIN", fontWeight = FontWeight.Bold, fontSize = 16.sp)
                    Text("Set a temporary 4-digit PIN. Lumen will password-protect your generated behavioral PDF. Only individuals with this PIN will be able to read the shared document.", fontSize = 12.sp, color = MaterialTheme.colorScheme.outline, lineHeight = 17.sp)

                    OutlinedTextField(
                        value = pinText,
                        onValueChange = { if (it.length <= 8) pinText = it.filter { ch -> ch.isDigit() } },
                        label = { Text("Enter Numeric PIN") },
                        keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(keyboardType = androidx.compose.ui.text.input.KeyboardType.NumberPassword),
                        visualTransformation = PasswordVisualTransformation(),
                        modifier = Modifier.fillMaxWidth()
                    )

                    OutlinedTextField(
                        value = pinConfirm,
                        onValueChange = { if (it.length <= 8) pinConfirm = it.filter { ch -> ch.isDigit() } },
                        label = { Text("Confirm PIN") },
                        keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(keyboardType = androidx.compose.ui.text.input.KeyboardType.NumberPassword),
                        visualTransformation = PasswordVisualTransformation(),
                        modifier = Modifier.fillMaxWidth()
                    )

                    if (errorMessage != null) {
                        Text(errorMessage!!, color = MaterialTheme.colorScheme.error, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
                    }

                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End, verticalAlignment = Alignment.CenterVertically) {
                        TextButton(onClick = { showPinDialog = false }) {
                            Text("Cancel")
                        }
                        Spacer(Modifier.width(8.dp))
                        Button(
                            onClick = {
                                when {
                                    pinText.length < 4 -> errorMessage = "PIN must be at least 4 digits."
                                    pinText != pinConfirm -> errorMessage = "PIN entries do not match."
                                    else -> {
                                        errorMessage = null
                                        showPinDialog = false
                                        shareLoading = true
                                        scope.launch {
                                            ReportGenerator.generateAndShareReport(context, pinText) { success, err ->
                                                shareLoading = false
                                                if (success) {
                                                    Toast.makeText(context, "Report compiled and shared successfully!", Toast.LENGTH_LONG).show()
                                                } else {
                                                    Toast.makeText(context, "❌ Compile Failed: $err", Toast.LENGTH_LONG).show()
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
                        ) {
                            Text("Encrypt & Share", color = Color.White)
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// Helper Components & Compatibility Variables
// =============================================================================
val softCyanAlias = Color(0xFF4ECDC4)

val Activity.softCyanAlias: Color
    get() = Color(0xFF4ECDC4)

private fun startMonitoringService(context: Context) {
    val intent = Intent(context, com.example.mhealth.services.MonitoringService::class.java)
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        context.startForegroundService(intent)
    } else {
        context.startService(intent)
    }
}

private fun hasUsageStatsPermission(context: Context): Boolean {
    val appOps = context.getSystemService(Context.APP_OPS_SERVICE) as? android.app.AppOpsManager ?: return false
    val mode = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
        appOps.unsafeCheckOpNoThrow(
            android.app.AppOpsManager.OPSTR_GET_USAGE_STATS,
            android.os.Process.myUid(),
            context.packageName
        )
    } else {
        @Suppress("DEPRECATION")
        appOps.checkOpNoThrow(
            android.app.AppOpsManager.OPSTR_GET_USAGE_STATS,
            android.os.Process.myUid(),
            context.packageName
        )
    }
    return mode == android.app.AppOpsManager.MODE_ALLOWED
}
