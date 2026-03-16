package com.example.mhealth

import android.Manifest
import android.app.AppOpsManager
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.os.Process
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.mhealth.logic.DataRepository
import com.example.mhealth.models.DailyReport
import com.example.mhealth.models.PersonalityVector
import com.example.mhealth.services.MonitoringService
import com.example.mhealth.ui.charts.*
import com.example.mhealth.ui.theme.*

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent { MHealthTheme { MHealthApp() } }
    }
}

// =============================================================================
// App Shell & Navigation
// =============================================================================
enum class AppDest(val label: String, val icon: ImageVector) {
    HOME("Sensors", Icons.Default.Sensors),
    MONITOR("Monitor", Icons.Default.ShowChart),
    ANALYSIS("Analysis", Icons.Default.Analytics),
    INSIGHTS("Insights", Icons.Default.Lightbulb),
    SETTINGS("Settings", Icons.Default.Settings)
}

enum class NavState {
    LOGIN,
    QUESTIONNAIRE,
    DASHBOARD
}

@Composable
fun MHealthApp() {
    val firstLoginComplete by DataRepository.firstLoginComplete.collectAsState()
    var appState by remember { mutableStateOf(NavState.LOGIN) }

    // Once collected, skip login if already onboarded (this assumes "login" remembers local state for returning users)
    LaunchedEffect(firstLoginComplete) {
        if (firstLoginComplete && appState == NavState.LOGIN) {
            appState = NavState.DASHBOARD
        }
    }

    when (appState) {
        NavState.LOGIN -> LoginScreen(onLoginSuccess = {
            appState = if (firstLoginComplete) NavState.DASHBOARD else NavState.QUESTIONNAIRE
        })
        NavState.QUESTIONNAIRE -> QuestionnaireScreen(onComplete = {
            appState = NavState.DASHBOARD
        })
        NavState.DASHBOARD -> MainDashboard()
    }
}

@Composable
fun MainDashboard() {
    var current by remember { mutableStateOf(AppDest.HOME) }
    val context = LocalContext.current

    val perms = buildList {
        addAll(listOf(
            Manifest.permission.READ_CALL_LOG, Manifest.permission.READ_SMS,
            Manifest.permission.READ_CONTACTS, Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION, Manifest.permission.READ_CALENDAR
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
            add(Manifest.permission.ACCESS_BACKGROUND_LOCATION)
        }
    }

    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { _ ->
        if (hasUsageStatsPermission(context)) startMonitoringService(context)
    }
    LaunchedEffect(Unit) {
        launcher.launch(perms.toTypedArray())
        if (!hasUsageStatsPermission(context)) {
            context.startActivity(Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS))
        } else {
            startMonitoringService(context)
        }
    }

    Scaffold(
        bottomBar = {
            NavigationBar(containerColor = CardWhite, tonalElevation = 0.dp) {
                AppDest.entries.forEach { dest ->
                    NavigationBarItem(
                        icon = { Icon(dest.icon, dest.label) },
                        label = { Text(dest.label, fontSize = 10.sp) },
                        selected = dest == current,
                        onClick = { current = dest },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = MintGreen,
                            selectedTextColor = MintGreen,
                            indicatorColor = MintGreen.copy(alpha = 0.15f),
                            unselectedIconColor = TextSecondary,
                            unselectedTextColor = TextSecondary
                        )
                    )
                }
            }
        },
        containerColor = BackgroundWhite
    ) { padding ->
        Box(
            Modifier.fillMaxSize().padding(padding)
        ) {
            when (current) {
                AppDest.HOME     -> HomeScreen()
                AppDest.MONITOR  -> MonitorScreen()
                AppDest.ANALYSIS -> AnalysisScreen()
                AppDest.INSIGHTS -> InsightsScreen()
                AppDest.SETTINGS -> SettingsScreen()
            }
        }
    }
}

// =============================================================================
// AUTH & ONBOARDING SCREENS
// =============================================================================
@Composable
fun LoginScreen(onLoginSuccess: () -> Unit) {
    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var emailError by remember { mutableStateOf(false) }
    var passError by remember { mutableStateOf(false) }

    Column(
        Modifier.fillMaxSize().background(BackgroundWhite).padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Box(
            Modifier.size(80.dp).clip(CircleShape).background(MintGreen.copy(0.15f)),
            contentAlignment = Alignment.Center
        ) {
            Icon(Icons.Default.HealthAndSafety, "Logo", tint = MintGreen, modifier = Modifier.size(48.dp))
        }
        Spacer(Modifier.height(24.dp))
        Text("Welcome to MHealth", fontSize = 24.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
        Text("Sign in to continue your mental health journey", fontSize = 14.sp, color = TextSecondary, textAlign = androidx.compose.ui.text.style.TextAlign.Center)
        Spacer(Modifier.height(32.dp))

        OutlinedTextField(
            value = email, onValueChange = { email = it; emailError = false },
            label = { Text("Email", color = TextSecondary) },
            isError = emailError,
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
            colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = MintGreen, focusedLabelColor = MintGreen, cursorColor = MintGreen)
        )
        if (emailError) Text("Please enter a valid email", color = AlertRed, fontSize = 11.sp, modifier = Modifier.fillMaxWidth().padding(start = 4.dp, top = 2.dp))
        Spacer(Modifier.height(16.dp))

        OutlinedTextField(
            value = password, onValueChange = { password = it; passError = false },
            label = { Text("Password", color = TextSecondary) },
            isError = passError,
            singleLine = true,
            visualTransformation = androidx.compose.ui.text.input.PasswordVisualTransformation(),
            modifier = Modifier.fillMaxWidth(),
            colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = MintGreen, focusedLabelColor = MintGreen, cursorColor = MintGreen)
        )
        if (passError) Text("Password is required", color = AlertRed, fontSize = 11.sp, modifier = Modifier.fillMaxWidth().padding(start = 4.dp, top = 2.dp))
        Spacer(Modifier.height(32.dp))

        Button(
            onClick = {
                val emailValid = android.util.Patterns.EMAIL_ADDRESS.matcher(email).matches()
                if (!emailValid) emailError = true
                if (password.isBlank()) passError = true
                if (emailValid && password.isNotBlank()) onLoginSuccess()
            },
            colors = ButtonDefaults.buttonColors(containerColor = MintGreen),
            modifier = Modifier.fillMaxWidth().height(50.dp),
            shape = RoundedCornerShape(12.dp)
        ) {
            Text("Sign In", color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold)
        }
        Spacer(Modifier.height(16.dp))
        TextButton(onClick = { /* Sign Up Logic Placeholder */ }) {
            Text("Don't have an account? ", color = TextSecondary, fontSize = 14.sp)
            Text("Sign Up", color = MintGreen, fontSize = 14.sp, fontWeight = FontWeight.Bold)
        }
    }
}

@Composable
fun QuestionnaireScreen(onComplete: () -> Unit) {
    var name by remember { mutableStateOf("") }
    var gender by remember { mutableStateOf("") }
    var age by remember { mutableStateOf("") }
    var profession by remember { mutableStateOf("") }
    var country by remember { mutableStateOf("") }
    var showErrors by remember { mutableStateOf(false) }

    val genderOptions = listOf("Male", "Female", "Non-binary", "Prefer not to say")

    Column(Modifier.fillMaxSize().background(BackgroundWhite)) {
        Box(Modifier.fillMaxWidth().background(Brush.horizontalGradient(listOf(MintGreen, SkyBlue))).padding(horizontal = 24.dp, vertical = 24.dp)) {
            Column {
                Spacer(Modifier.height(16.dp))
                Text("Let's personalize MHealth", fontSize = 24.sp, fontWeight = FontWeight.Bold, color = Color.White)
                Text("Just a few details to set up your baseline metrics", fontSize = 14.sp, color = Color.White.copy(0.85f))
            }
        }

        LazyColumn(Modifier.weight(1f).padding(24.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            item {
                OutlinedTextField(
                    value = name, onValueChange = { name = it },
                    label = { Text("Full Name") },
                    isError = showErrors && name.isBlank(),
                    modifier = Modifier.fillMaxWidth(),
                    colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = MintGreen, cursorColor = MintGreen)
                )
            }
            item {
                Text("Gender", fontSize = 14.sp, color = TextPrimary, fontWeight = FontWeight.Medium)
                Column(Modifier.fillMaxWidth().border(1.dp, if (showErrors && gender.isBlank()) AlertRed else SurfaceBlue, RoundedCornerShape(8.dp)).padding(8.dp)) {
                    genderOptions.forEach { option ->
                        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth().clickable { gender = option }.padding(vertical = 4.dp)) {
                            RadioButton(
                                selected = gender == option,
                                onClick = { gender = option },
                                colors = RadioButtonDefaults.colors(selectedColor = MintGreen)
                            )
                            Text(option, fontSize = 14.sp, color = TextSecondary)
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
                    colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = MintGreen, cursorColor = MintGreen)
                )
            }
            item {
                var expanded by remember { mutableStateOf(false) }
                val profOptions = listOf("Student", "Employed", "Self-employed", "Other")
                Box(Modifier.fillMaxWidth()) {
                    OutlinedTextField(
                        value = profession, onValueChange = {},
                        label = { Text("Profession") },
                        isError = showErrors && profession.isBlank(),
                        readOnly = true,
                        trailingIcon = { IconButton(onClick = { expanded = !expanded }) { Icon(Icons.Default.ArrowDropDown, null) } },
                        modifier = Modifier.fillMaxWidth(),
                        colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = MintGreen, cursorColor = MintGreen)
                    )
                    DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                        profOptions.forEach { opt ->
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
                    colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = MintGreen, cursorColor = MintGreen)
                )
            }
            item { Spacer(Modifier.height(16.dp)) }
        }

        Box(Modifier.fillMaxWidth().padding(24.dp)) {
            Button(
                onClick = {
                    if (name.isBlank() || gender.isBlank() || age.isBlank() || profession.isBlank() || country.isBlank()) {
                        showErrors = true
                    } else {
                        val profile = com.example.mhealth.models.UserProfile(
                            name = name,
                            gender = gender,
                            age = age.toIntOrNull() ?: 0,
                            profession = profession,
                            country = country
                        )
                        DataRepository.saveUserProfile(profile)
                        onComplete()
                    }
                },
                colors = ButtonDefaults.buttonColors(containerColor = MintGreen),
                modifier = Modifier.fillMaxWidth().height(52.dp),
                shape = RoundedCornerShape(12.dp)
            ) {
                Text("Complete Setup", color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold)
            }
        }
    }
}

// =============================================================================
// Shared UI components
// =============================================================================
@Composable
fun ScreenHeader(title: String, subtitle: String, icon: ImageVector, iconTint: Color = MintGreen) {
    Row(
        Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            Modifier.size(44.dp).clip(RoundedCornerShape(12.dp))
                .background(iconTint.copy(0.12f)),
            contentAlignment = Alignment.Center
        ) { Icon(icon, null, tint = iconTint, modifier = Modifier.size(24.dp)) }
        Spacer(Modifier.width(12.dp))
        Column {
            Text(title, fontSize = 20.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
            Text(subtitle, fontSize = 12.sp, color = TextSecondary)
        }
    }
}

@Composable
fun InfoCard(
    title: String,
    modifier: Modifier = Modifier,
    headerColor: Color = MintGreen,
    content: @Composable ColumnScope.() -> Unit
) {
    Card(
        modifier = modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 6.dp),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(2.dp),
        colors = CardDefaults.cardColors(containerColor = CardWhite)
    ) {
        Column(Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.width(4.dp).height(18.dp).clip(RoundedCornerShape(2.dp)).background(headerColor))
                Spacer(Modifier.width(8.dp))
                Text(title, fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = TextPrimary)
            }
            Spacer(Modifier.height(14.dp))
            content()
        }
    }
}

fun alertColor(level: String) = when (level.lowercase()) {
    "green", "stable" -> AlertGreen
    "yellow"          -> AlertYellow
    "orange"          -> AlertOrange
    "red"             -> AlertRed
    else              -> TextSecondary
}

// =============================================================================
// HOME SCREEN — Layer 1: Data Collection Infographic
// =============================================================================
@Composable
fun HomeScreen() {
    val vector by DataRepository.latestVector.collectAsState()
    val context = LocalContext.current

    LazyColumn(Modifier.fillMaxSize()) {
        item {
            // Gradient header banner
            Box(
                Modifier.fillMaxWidth()
                    .background(Brush.horizontalGradient(listOf(MintGreen, SkyBlue)))
                    .padding(20.dp)
            ) {
                Column {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(Modifier.size(10.dp).clip(CircleShape).background(Color.White.copy(0.9f)))
                        Spacer(Modifier.width(6.dp))
                        Text("LIVE MONITORING", fontSize = 11.sp, color = Color.White.copy(0.9f), fontWeight = FontWeight.SemiBold, letterSpacing = 1.sp)
                    }
                    Text("Device Sensors", fontSize = 24.sp, fontWeight = FontWeight.Bold, color = Color.White)
                    Text("Layer 1 — Real-time data collection", fontSize = 13.sp, color = Color.White.copy(0.8f))
                }
            }
        }

        if (vector == null) {
            item {
                Box(Modifier.fillMaxWidth().height(300.dp), contentAlignment = Alignment.Center) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator(color = MintGreen)
                        Spacer(Modifier.height(12.dp))
                        Text("Collecting sensor data…", color = TextSecondary)
                    }
                }
            }
        } else {
            val v = vector!!

            // Digital Wellbeing primary metrics — 6-up (matches DW dashboard exactly)
            item {
                InfoCard("Digital Wellbeing Metrics", headerColor = MintGreen) {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                        ArcProgressRing(v.screenTimeHours, 12f, MintGreen, "Screen Time", "hrs")
                        ArcProgressRing(v.unlockCount, 100f, SkyBlue, "Unlocks", "")
                        ArcProgressRing(v.appLaunchCount, 200f, CoralPink, "App Opens", "")
                    }
                    Spacer(Modifier.height(16.dp))
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                        ArcProgressRing(v.notificationsToday, 200f, AlertOrange, "Notifs", "")
                        ArcProgressRing(v.placesVisited, 10f, LavenderPurple, "Places Vis.", "")
                        ArcProgressRing(v.socialAppRatio, 1f, ChartGreen, "Social", "")
                    }
                }
            }

            // Movement & Location row
            item {
                InfoCard("Movement & Location", headerColor = SkyBlue) {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                        MetricPill("🏃 Displacement", "%.2f km".format(v.dailyDisplacementKm), SkyBlue)
                        MetricPill("🌐 Entropy", "%.2f".format(v.locationEntropy), LavenderPurple)
                    }
                    Spacer(Modifier.height(12.dp))
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                        MetricPill("🏠 Home Time", "%.0f%%".format(v.homeTimeRatio * 100), MintGreen)
                    }
                }
            }


            // Communication row
            item {
                InfoCard("Communication", headerColor = SkyBlue) {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                        MetricPill("📞 Calls", "${v.callsPerDay.toInt()}", SkyBlue)
                        MetricPill("⏱ Talk Time", "${v.callDurationMinutes.toInt()}m", CoralPink)
                        MetricPill("⭐ Favs", "${v.uniqueContacts.toInt()}", LavenderPurple)
                    }
                }
            }

            // Movement
            item {
                InfoCard("Movement & Location", headerColor = CoralPink) {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                        ArcProgressRing(v.dailyDisplacementKm, 20f, CoralPink, "Distance", "km")
                        ArcProgressRing(v.locationEntropy, 3f, AlertOrange, "Location\nVariety", "")
                        ArcProgressRing(v.homeTimeRatio, 1f, AlertYellow, "Home\nTime", "")
                    }
                }
            }

            // Sleep proxy
            item {
                InfoCard("Sleep Proxy", headerColor = LavenderPurple) {
                    Column {
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                            ArcProgressRing(v.sleepDurationHours, 10f, LavenderPurple, "Est. Sleep", "hrs")
                            ArcProgressRing(v.darkDurationHours, 12f, SkyBlue.copy(0.7f), "Dark Hours", "hrs")
                            ArcProgressRing(v.chargeDurationHours, 6f, AlertOrange, "Charge", "hrs")
                        }
                        Spacer(Modifier.height(16.dp))
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Text("Sleep Time", fontSize = 11.sp, color = TextSecondary)
                                Text("%.0f:00".format(v.sleepTimeHour), fontSize = 18.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
                            }
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Text("Wake Time", fontSize = 11.sp, color = TextSecondary)
                                Text("%.0f:00".format(v.wakeTimeHour), fontSize = 18.sp, fontWeight = FontWeight.Bold, color = LavenderPurple)
                            }
                        }
                    }
                }
            }

            // App usage bar chart
            if (v.appBreakdown.isNotEmpty()) {
                item {
                    InfoCard("App Usage Breakdown", headerColor = ChartOrange) {
                        val top = v.appBreakdown.entries.sortedByDescending { it.value }.take(6)
                            .map { it.key to it.value.toFloat() }
                        val max = top.firstOrNull()?.second ?: 1f
                        HorizontalBarChart(top, max, ChartOrange, unitSuffix = "m")
                    }
                }
            }

            // App Launches bar chart
            if (v.appLaunchesBreakdown.isNotEmpty()) {
                item {
                    InfoCard("Top App Launches (Times Opened)", headerColor = LavenderPurple) {
                        val top = v.appLaunchesBreakdown.entries.sortedByDescending { it.value }.take(6)
                            .map { it.key to it.value.toFloat() }
                        val max = top.firstOrNull()?.second ?: 1f
                        HorizontalBarChart(top, max, LavenderPurple)
                    }
                }
            }

            // Notifications bar chart
            if (v.notificationBreakdown.isNotEmpty()) {
                item {
                    InfoCard("Top Notifications", headerColor = ChartOrange.copy(alpha = 0.8f)) {
                        val top = v.notificationBreakdown.entries.sortedByDescending { it.value }.take(6)
                            .map { it.key to it.value.toFloat() }
                        val max = top.firstOrNull()?.second ?: 1f
                        HorizontalBarChart(top, max, ChartOrange.copy(alpha = 0.8f))
                    }
                }
            }

            // System stats row
            item {
                InfoCard("System", headerColor = ChartBlue) {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                        MetricPill("📶 WiFi", "%.0fMB".format(v.networkWifiMB), ChartBlue)
                    }
                }
            }

            item { Spacer(Modifier.height(12.dp)) }
        }
    }
}

@Composable
fun MetricPill(label: String, value: String, color: Color) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(label, fontSize = 11.sp, color = TextSecondary)
        Box(
            Modifier.padding(top = 4.dp).clip(RoundedCornerShape(20.dp)).background(color.copy(0.12f)).padding(horizontal = 12.dp, vertical = 6.dp)
        ) {
            Text(value, fontSize = 14.sp, fontWeight = FontWeight.Bold, color = color)
        }
    }
}

// =============================================================================
// MONITOR SCREEN — Layers 2 & 3: Baseline & Continuous Monitoring
// =============================================================================
@Composable
fun MonitorScreen() {
    val progress by DataRepository.baselineProgress.collectAsState()
    val isBuilding by DataRepository.isBuildingBaseline.collectAsState()
    val vector by DataRepository.latestVector.collectAsState()
    val baseline by DataRepository.baseline.collectAsState()
    val hourly by DataRepository.hourlySnapshots.collectAsState()
    val reports by DataRepository.reports.collectAsState()

    LazyColumn(Modifier.fillMaxSize()) {
        item {
            Box(
                Modifier.fillMaxWidth()
                    .background(Brush.horizontalGradient(listOf(SkyBlue, LavenderPurple)))
                    .padding(20.dp)
            ) {
                Column {
                    Text("Baseline & Monitoring", fontSize = 24.sp, fontWeight = FontWeight.Bold, color = Color.White)
                    Text("Layers 2 & 3 — ${if (isBuilding) "Building Personal Normal" else "Continuous Tracking"}", fontSize = 13.sp, color = Color.White.copy(0.85f))
                }
            }
        }

        // Baseline progress arc
        item {
            InfoCard("Baseline Progress (P₀)", headerColor = SkyBlue) {
                if (isBuilding) {
                    val frac = progress / 28f
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        ArcProgressRing(progress.toFloat(), 28f, SkyBlue, "Days", "/ 28", size = 90.dp)
                        Spacer(Modifier.width(16.dp))
                        Column {
                            Text("Building your personal baseline", fontWeight = FontWeight.SemiBold, color = TextPrimary)
                            Text("Weeks 1–4: Establishing P₀ vector.\nData is collected continuously.", fontSize = 12.sp, color = TextSecondary)
                            Spacer(Modifier.height(6.dp))
                            LinearProgressIndicator(
                                progress = { frac },
                                color = SkyBlue,
                                trackColor = SkyBlue.copy(0.15f),
                                modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(4.dp))
                            )
                        }
                    }
                } else {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.CheckCircle, null, tint = AlertGreen, modifier = Modifier.size(40.dp))
                        Spacer(Modifier.width(12.dp))
                        Column {
                            Text("Baseline Established", fontWeight = FontWeight.SemiBold, color = AlertGreen)
                            Text("28-day P₀ vector locked. Monitoring active.", fontSize = 12.sp, color = TextSecondary)
                        }
                    }
                }
            }
        }

        // Intraday sparklines
        item {
            InfoCard("Today's Intraday Trends", headerColor = LavenderPurple) {
                if (hourly.size < 2) {
                    Text("Collecting hourly snapshots…", color = TextSecondary, fontSize = 12.sp)
                } else {
                    val screenTimes = hourly.map { it.screenTimeHours }
                    val places = hourly.map { it.placesVisited }
                    SparklineLabel("Screen Time (hrs)", screenTimes, MintGreen)
                    Spacer(Modifier.height(12.dp))
                    SparklineLabel("Places Visited", places, LavenderPurple)
                }
            }
        }

        // Current vs Baseline comparison (only available post-baseline)
        if (!isBuilding && baseline != null && vector != null) {
            item {
                InfoCard("Current vs Baseline", headerColor = MintGreen) {
                    val v = vector!!; val b = baseline!!
                    val rows = listOf(
                        Triple("Screen Time", v.screenTimeHours, b.screenTimeHours),
                        Triple("Places Visited", v.placesVisited, b.placesVisited),
                        Triple("Calls/Day", v.callsPerDay, b.callsPerDay),
                        Triple("Social Ratio %", v.socialAppRatio * 100, b.socialAppRatio * 100),
                        Triple("Sleep Hours", v.sleepDurationHours, b.sleepDurationHours),
                        Triple("Displacement (km)", v.dailyDisplacementKm, b.dailyDisplacementKm)
                    )
                    rows.forEach { (label, cur, base) ->
                        ComparisonRow(label, cur, base)
                    }
                }
            }
        }

        // Sliding window selector (display only)
        item {
            InfoCard("Sliding Analysis Windows", headerColor = ChartOrange) {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                    WindowChip("24h", "Acute", ChartOrange)
                    WindowChip("7d", "Trend", SkyBlue)
                    WindowChip("28d", "Persistent", LavenderPurple)
                }
                Spacer(Modifier.height(8.dp))
                Text("System uses all three windows simultaneously to detect short-term spikes vs. gradual drift.", fontSize = 11.sp, color = TextSecondary)
            }
        }

        item { Spacer(Modifier.height(12.dp)) }
    }
}

@Composable
fun SparklineLabel(label: String, values: List<Float>, color: Color) {
    Text(label, fontSize = 11.sp, color = TextSecondary)
    Spacer(Modifier.height(4.dp))
    SparklineChart(values, color, Modifier.fillMaxWidth().height(50.dp))
}

@Composable
fun ComparisonRow(label: String, current: Float, baseline: Float) {
    val delta = if (baseline > 0) ((current - baseline) / baseline * 100) else 0f
    val deltaColor = if (kotlin.math.abs(delta) < 10f) TextSecondary else if (delta > 0) AlertOrange else SkyBlue
    Row(Modifier.fillMaxWidth().padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(label, fontSize = 12.sp, color = TextSecondary, modifier = Modifier.weight(1f))
        Text("%.1f".format(current), fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = TextPrimary)
        Spacer(Modifier.width(8.dp))
        Text(
            "${if (delta >= 0) "+" else ""}%.0f%%".format(delta),
            fontSize = 11.sp, color = deltaColor, fontWeight = FontWeight.SemiBold
        )
    }
}

@Composable
fun WindowChip(period: String, label: String, color: Color) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Box(
            Modifier.clip(RoundedCornerShape(24.dp)).background(color.copy(0.12f))
                .padding(horizontal = 16.dp, vertical = 8.dp),
            contentAlignment = Alignment.Center
        ) { Text(period, fontSize = 14.sp, fontWeight = FontWeight.Bold, color = color) }
        Text(label, fontSize = 10.sp, color = TextSecondary)
    }
}

// =============================================================================
// ANALYSIS SCREEN — System 1 & 2: Anomaly Detection Engine
// =============================================================================
@Composable
fun AnalysisScreen() {
    val reports by DataRepository.reports.collectAsState()
    val isBuilding by DataRepository.isBuildingBaseline.collectAsState()
    val baseline by DataRepository.baseline.collectAsState()
    val vector by DataRepository.latestVector.collectAsState()
    val last = reports.lastOrNull()

    LazyColumn(Modifier.fillMaxSize()) {
        item {
            Box(
                Modifier.fillMaxWidth()
                    .background(Brush.horizontalGradient(listOf(CoralPink, AlertOrange)))
                    .padding(20.dp)
            ) {
                Column {
                    Text("Anomaly Engine", fontSize = 24.sp, fontWeight = FontWeight.Bold, color = Color.White)
                    Text("System 1 & 2 — Deviation detection & pattern classification", fontSize = 12.sp, color = Color.White.copy(0.85f))
                }
            }
        }

        if (isBuilding) {
            item {
                InfoCard("Status", headerColor = SkyBlue) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(Modifier.size(32.dp), color = SkyBlue)
                        Spacer(Modifier.width(12.dp))
                        Text("Calibrating — baseline not yet ready.\nAnomaly detection begins after 28 days.", color = TextSecondary, fontSize = 12.sp)
                    }
                }
            }
        } else {
            // Anomaly Score Gauge
            item {
                InfoCard("Anomaly Score", headerColor = CoralPink) {
                    val score = last?.anomalyScore ?: 0f
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
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
                            fontSize = 18.sp, fontWeight = FontWeight.Bold, color = CoralPink
                        )
                        Text(
                            "Pattern: ${(last?.patternType ?: "stable").replace("_", " ").uppercase()}",
                            fontSize = 12.sp, color = TextSecondary
                        )
                    }
                }
            }

            // Radar chart
            if (baseline != null && vector != null) {
                item {
                    InfoCard("Feature Deviation Radar", headerColor = LavenderPurple) {
                        val b = baseline!!; val v = vector!!
                        val radarLabels = listOf("Screen\nTime", "Social", "Places", "Location", "Sleep", "Comms")
                        val normalize: (Float, Float, Float) -> Float = { cur, base, maxVal ->
                            ((cur / maxVal.coerceAtLeast(0.01f)).coerceIn(0f, 1f))
                        }
                        val curVals = listOf(
                            normalize(v.screenTimeHours, b.screenTimeHours, 12f),
                            normalize(v.socialAppRatio, b.socialAppRatio, 1f),
                            normalize(v.placesVisited, b.placesVisited, 20f),
                            normalize(v.dailyDisplacementKm, b.dailyDisplacementKm, 20f),
                            normalize(v.sleepDurationHours, b.sleepDurationHours, 10f),
                            normalize(v.conversationFrequency, b.conversationFrequency, 50f)
                        )
                        val baseVals = listOf(
                            normalize(b.screenTimeHours, b.screenTimeHours, 12f),
                            normalize(b.socialAppRatio, b.socialAppRatio, 1f),
                            normalize(b.placesVisited, b.placesVisited, 20f),
                            normalize(b.dailyDisplacementKm, b.dailyDisplacementKm, 20f),
                            normalize(b.sleepDurationHours, b.sleepDurationHours, 10f),
                            normalize(b.conversationFrequency, b.conversationFrequency, 50f)
                        )
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) {
                            RadarChart(radarLabels, curVals, baseVals, LavenderPurple, Modifier.size(220.dp))
                        }
                        Spacer(Modifier.height(8.dp))
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center, verticalAlignment = Alignment.CenterVertically) {
                            Box(Modifier.size(12.dp).background(LavenderPurple.copy(0.7f), CircleShape))
                            Text(" Current   ", fontSize = 11.sp, color = TextSecondary)
                            Box(Modifier.size(12.dp).background(SkyBlue.copy(0.5f), CircleShape))
                            Text(" Baseline", fontSize = 11.sp, color = TextSecondary)
                        }
                    }
                }
            }

            // Top deviations
            last?.let { report ->
                item {
                    InfoCard("Top Deviations (SD units)", headerColor = AlertOrange) {
                        if (report.topDeviations.isEmpty()) {
                            Text("No significant deviations detected.", color = TextSecondary, fontSize = 12.sp)
                        } else {
                            report.topDeviations.entries.sortedByDescending { kotlin.math.abs(it.value) }.take(5).forEach { (feature, sd) ->
                                DeviationRow(feature.replace(Regex("([a-z])([A-Z])"), "$1 $2"), sd)
                            }
                        }
                    }
                }
                item {
                    InfoCard("Temporal Pattern", headerColor = SkyBlue) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.Timeline, null, tint = SkyBlue)
                            Spacer(Modifier.width(8.dp))
                            Column {
                                Text(report.patternType.replace("_", " ").replaceFirstChar { it.uppercase() }, fontWeight = FontWeight.SemiBold, color = TextPrimary)
                                Text("Sustained deviation days: ${report.sustainedDeviationDays}", fontSize = 12.sp, color = TextSecondary)
                                Text("Evidence accumulated: ${"%.2f".format(report.evidenceAccumulated)}", fontSize = 12.sp, color = TextSecondary)
                            }
                        }
                    }
                }
            }
        }

        item { Spacer(Modifier.height(12.dp)) }
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

// =============================================================================
// INSIGHTS SCREEN — Layer 5: Alert & Output System
// =============================================================================
@Composable
fun InsightsScreen() {
    val reports by DataRepository.reports.collectAsState()
    val moodScore by DataRepository.moodScore.collectAsState()
    val isBuilding by DataRepository.isBuildingBaseline.collectAsState()
    val last = reports.lastOrNull()
    val alertLvl = last?.alertLevel ?: "green"
    val aColor = alertColor(alertLvl)

    LazyColumn(Modifier.fillMaxSize()) {
        item {
            Box(
                Modifier.fillMaxWidth()
                    .background(Brush.horizontalGradient(listOf(AlertGreen, MintGreen)))
                    .padding(20.dp)
            ) {
                Column {
                    Text("Insights & Alerts", fontSize = 24.sp, fontWeight = FontWeight.Bold, color = Color.White)
                    Text("Layer 5 — Alert system & user dashboard", fontSize = 13.sp, color = Color.White.copy(0.85f))
                }
            }
        }

        // Alert status card
        item {
            Card(
                Modifier.fillMaxWidth().padding(16.dp),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = aColor.copy(0.1f)),
                border = BorderStroke(2.dp, aColor.copy(0.4f))
            ) {
                Column(Modifier.padding(20.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            when (alertLvl.lowercase()) {
                                "red"    -> Icons.Default.Warning
                                "orange" -> Icons.Default.Warning
                                "yellow" -> Icons.Default.Info
                                else     -> Icons.Default.CheckCircle
                            },
                            null, tint = aColor, modifier = Modifier.size(32.dp)
                        )
                        Spacer(Modifier.width(12.dp))
                        Text(alertLvl.uppercase(), fontSize = 28.sp, fontWeight = FontWeight.ExtraBold, color = aColor)
                    }
                    Spacer(Modifier.height(8.dp))
                    Text(
                        when (alertLvl.lowercase()) {
                            "green"  -> "Your patterns are within normal range. Keep it up!"
                            "yellow" -> "Mild changes detected. Take a moment to check in with yourself."
                            "orange" -> "Moderate deviations observed. Consider self-care activities."
                            "red"    -> "Significant pattern changes detected. Professional consultation recommended."
                            else     -> "Collecting data to establish baseline."
                        },
                        fontSize = 13.sp, color = TextSecondary, lineHeight = 18.sp
                    )
                    last?.notes?.let { notes ->
                        if (notes.isNotBlank() && notes != "Normal operation") {
                            Spacer(Modifier.height(8.dp))
                            Text(notes, fontSize = 12.sp, color = TextSecondary)
                        }
                    }
                }
            }
        }

        // Alert history timeline
        if (reports.size > 1) {
            item {
                InfoCard("Alert History", headerColor = ChartBlue) {
                    reports.takeLast(10).reversed().forEach { report ->
                        Row(Modifier.fillMaxWidth().padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                            Box(Modifier.size(10.dp).clip(CircleShape).background(alertColor(report.alertLevel)))
                            Spacer(Modifier.width(10.dp))
                            Text("Day ${report.dayNumber}", fontSize = 12.sp, color = TextPrimary, modifier = Modifier.weight(1f))
                            Text(report.alertLevel.uppercase(), fontSize = 11.sp, color = alertColor(report.alertLevel), fontWeight = FontWeight.Bold)
                        }
                    }
                }
            }
        }

        // Crisis resources (only on red)
        if (alertLvl.lowercase() == "red") {
            item {
                Card(
                    Modifier.fillMaxWidth().padding(16.dp),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(containerColor = AlertRed.copy(0.08f)),
                    border = BorderStroke(1.dp, AlertRed.copy(0.3f))
                ) {
                    Column(Modifier.padding(16.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.Favorite, null, tint = AlertRed)
                            Spacer(Modifier.width(8.dp))
                            Text("Crisis Resources", fontWeight = FontWeight.Bold, color = AlertRed)
                        }
                        Spacer(Modifier.height(8.dp))
                        Text("iCall India: 9152987821", fontSize = 12.sp, color = TextPrimary)
                        Text("Vandrevala Foundation: 1860-2662-345", fontSize = 12.sp, color = TextPrimary)
                        Text("iHeal: 9990966684", fontSize = 12.sp, color = TextPrimary)
                    }
                }
            }
        }

        item { Spacer(Modifier.height(12.dp)) }
    }
}

// =============================================================================
// SETTINGS SCREEN — Layer 4: Baseline Adaptation Logic
// =============================================================================
@Composable
fun SettingsScreen() {
    val progress by DataRepository.baselineProgress.collectAsState()
    val isBuilding by DataRepository.isBuildingBaseline.collectAsState()
    var dataCollectionEnabled by remember { mutableStateOf(true) }
    var locationEnabled by remember { mutableStateOf(true) }
    var commsEnabled by remember { mutableStateOf(true) }

    // Dev Setting States
    val baselineDaysReq by DataRepository.baselineDaysRequired.collectAsState()
    val intervalMins by DataRepository.monitoringIntervalMinutes.collectAsState()

    LazyColumn(Modifier.fillMaxSize().padding(horizontal = 20.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        item { ScreenHeader("Settings", "Adaptation layer configurations", Icons.Default.Settings) }

        // Logic Infographic
        item {
            InfoCard("Adaptation Logic", headerColor = TextSecondary) {
                Column(Modifier.fillMaxWidth()) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(Modifier.size(8.dp).clip(CircleShape).background(MintGreen))
                        Spacer(Modifier.width(8.dp))
                        Text("Stable state (> 28d) → Adapts baseline", fontSize = 12.sp, color = TextPrimary)
                    }
                    Spacer(Modifier.height(8.dp))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(Modifier.size(8.dp).clip(CircleShape).background(AlertRed))
                        Spacer(Modifier.width(8.dp))
                        Text("Fluctuating state → Flags anomaly", fontSize = 12.sp, color = TextPrimary)
                    }
                }
            }
        }

        // Baseline Status
        item {
            InfoCard("Baseline Status", headerColor = MintGreen) {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                    Column {
                        Text(if (isBuilding) "Building Baseline" else "Active Monitoring", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
                        Text("Day $progress / $baselineDaysReq established", fontSize = 12.sp, color = TextSecondary)
                    }
                    if (!isBuilding) {
                        Button(onClick = { /* Reset logic */ }, colors = ButtonDefaults.buttonColors(containerColor = AlertRed)) {
                            Text("Reset", fontSize = 12.sp)
                        }
                    }
                }
            }
        }
        
        // Developer Settings Block
        item {
            InfoCard("Developer Testing Tools", headerColor = ChartOrange) {
                Column(Modifier.fillMaxWidth()) {
                    Text("⚠️ Modifying these will immediately affect the running background engine.", 
                         fontSize = 11.sp, color = TextMuted, lineHeight = 14.sp)
                    Spacer(Modifier.height(16.dp))
                    
                    Text("Baseline Time Period: $baselineDaysReq days", fontSize = 13.sp, color = TextPrimary, fontWeight = FontWeight.Bold)
                    Slider(
                        value = baselineDaysReq.toFloat(),
                        onValueChange = { DataRepository.setBaselineDaysRequired(it.toInt()) },
                        valueRange = 1f..35f,
                        steps = 34,
                        colors = SliderDefaults.colors(thumbColor = ChartOrange, activeTrackColor = ChartOrange)
                    )
                    
                    Spacer(Modifier.height(8.dp))
                    
                    Text("Polling Interval: $intervalMins mins", fontSize = 13.sp, color = TextPrimary, fontWeight = FontWeight.Bold)
                    Slider(
                        value = intervalMins.toFloat(),
                        onValueChange = { DataRepository.setMonitoringIntervalMinutes(it.toLong()) },
                        valueRange = 1f..60f,
                        steps = 59,
                        colors = SliderDefaults.colors(thumbColor = ChartOrange, activeTrackColor = ChartOrange)
                    )
                    
                    Spacer(Modifier.height(16.dp))
                    
                    Button(
                        onClick = { DataRepository.triggerNewDay() },
                        colors = ButtonDefaults.buttonColors(containerColor = LavenderPurple),
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text("Simulate Midnight (Force New Day)", color = Color.White)
                    }
                }
            }
        }

        // Collection Toggles
        item {
            InfoCard("Data Collection", headerColor = TextSecondary) {
                Column {
                    ToggleRow("Master Collection", "Enable all data logging", dataCollectionEnabled, TextSecondary) { dataCollectionEnabled = it }
                    Divider(Modifier.padding(vertical = 8.dp), color = SurfaceBlue)
                    ToggleRow("Location Tracking (GPS)", "Displacement & entropy tracking", locationEnabled, MintGreen) { locationEnabled = it }
                    ToggleRow("Communication Logs", "Call and SMS tracking", commsEnabled, ChartOrange) { commsEnabled = it }
                }
            }
        }

        // Action Toggles
        item {
            val context = LocalContext.current
            Card(Modifier.fillMaxWidth(), shape = RoundedCornerShape(16.dp), colors = CardDefaults.cardColors(containerColor = CardWhite), elevation = CardDefaults.cardElevation(2.dp)) {
                Column {
                    Row(
                        Modifier.fillMaxWidth().clickable {
                            exportDataAsJson(context)
                        }.padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(Icons.Default.Download, null, tint = SkyBlue)
                        Spacer(Modifier.width(16.dp))
                        Text("Export Local Data (JSON)", color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Medium)
                    }
                    Divider(color = SurfaceBlue)
                    Row(Modifier.fillMaxWidth().clickable {}.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.Delete, null, tint = AlertRed)
                        Spacer(Modifier.width(16.dp))
                        Text("Delete All Data", color = AlertRed, fontSize = 14.sp, fontWeight = FontWeight.Medium)
                    }
                }
            }
        }

        item { Spacer(Modifier.height(24.dp)) }
    }
}

private fun exportDataAsJson(context: Context) {
    try {
        val data = org.json.JSONObject().apply {
            val latestRaw = DataRepository.latestVector.value
            
            put("voice", org.json.JSONObject().apply {
                put("pitch_mean", 0.0)
                put("pitch_std", 0.0)
                put("energy_mean", 0.0)
                put("speaking_rate", 0.0)
                put("pause_rate", 0.0)
            })

            put("activity", org.json.JSONObject().apply {
                put("screen_time_daily", latestRaw?.screenTimeHours ?: 0.0)
                put("unlock_frequency", latestRaw?.unlockCount ?: 0.0)
                put("social_app_ratio", latestRaw?.socialAppRatio ?: 0.0)
                put("calls_per_day", latestRaw?.callsPerDay ?: 0.0)
                put("texts_per_day", 0.0) // SMS historically restricted by Google Play policy
                put("unique_contacts", latestRaw?.uniqueContacts?.toInt() ?: 0)
                put("avg_response_time", 0.0)
            })

            put("movement", org.json.JSONObject().apply {
                put("daily_displacement", latestRaw?.dailyDisplacementKm ?: 0.0)
                put("location_entropy", latestRaw?.locationEntropy ?: 0.0)
                put("home_time_ratio", latestRaw?.homeTimeRatio ?: 0.0)
                put("places_visited", latestRaw?.placesVisited?.toInt() ?: 0)
            })

            put("circadian", org.json.JSONObject().apply {
                put("wake_time_mean", latestRaw?.wakeTimeHour ?: 0.0)
                put("wake_time_std", 0.0)
                put("sleep_time_mean", latestRaw?.sleepTimeHour ?: 0.0)
                put("sleep_duration", latestRaw?.sleepDurationHours ?: 0.0)
            })

            put("regularity", org.json.JSONObject().apply {
                put("daily_routine_variance", 0.0) // Aggregated over 28-day baseline
                put("week_to_week_consistency", 0.0)
            })
        }

        val file = java.io.File(context.cacheDir, "mhealth_export_${System.currentTimeMillis()}.json")
        file.writeText(data.toString(4))

        val uri = androidx.core.content.FileProvider.getUriForFile(
            context,
            "${context.packageName}.provider",
            file
        )

        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "application/json"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        
        context.startActivity(Intent.createChooser(intent, "Export Data"))
    } catch (e: Exception) {
        e.printStackTrace()
        android.widget.Toast.makeText(context, "Export failed: ${e.message}", android.widget.Toast.LENGTH_SHORT).show()
    }
}




@Composable
fun ToggleRow(title: String, subtitle: String, checked: Boolean, color: Color, onToggle: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(title, fontSize = 13.sp, fontWeight = FontWeight.Medium, color = TextPrimary)
            Text(subtitle, fontSize = 11.sp, color = TextSecondary)
        }
        Switch(
            checked = checked, onCheckedChange = onToggle,
            colors = SwitchDefaults.colors(checkedThumbColor = Color.White, checkedTrackColor = color, uncheckedTrackColor = TextMuted.copy(0.3f))
        )
    }
}

// =============================================================================
// Utility helpers
// =============================================================================
fun hasUsageStatsPermission(context: Context): Boolean {
    val appOps = context.getSystemService(Context.APP_OPS_SERVICE) as AppOpsManager
    val mode = appOps.checkOpNoThrow(AppOpsManager.OPSTR_GET_USAGE_STATS, Process.myUid(), context.packageName)
    return mode == AppOpsManager.MODE_ALLOWED
}

fun startMonitoringService(context: Context) {
    val intent = Intent(context, MonitoringService::class.java)
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        context.startForegroundService(intent)
    } else {
        context.startService(intent)
    }
}
