package com.example.mhealth.ui.screens

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.provider.Settings
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import com.example.mhealth.HelplineType
import com.example.mhealth.LocationDisclosureDialog
import com.example.mhealth.ResearchContributionDialog
import com.example.mhealth.exportDataAsJson
import com.example.mhealth.exportDataToUri
import com.example.mhealth.getHelplinesByCountry
import com.example.mhealth.importBackupDataFromJson
import com.example.mhealth.logic.DataCollector
import com.example.mhealth.logic.DataRepository
import com.example.mhealth.services.MHealthNotificationListenerService
import com.example.mhealth.ui.components.Fredoka
import com.example.mhealth.ui.components.InfoCard
import com.example.mhealth.ui.components.PermissionSettingRow
import com.example.mhealth.ui.components.ToggleRow
import java.text.SimpleDateFormat
import java.util.*

@Composable
fun SettingsScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    val profile by DataRepository.userProfile.collectAsState()
    val homeLocation by DataRepository.homeLocation.collectAsState()
    val isBuilding by DataRepository.isBuildingBaseline.collectAsState()
    val progress by DataRepository.baselineProgress.collectAsState()
    var homeCapturing by remember { mutableStateOf(false) }

    val prefs = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }

    var themeMode by remember { mutableStateOf(prefs.getString("app_theme_mode", "dark") ?: "dark") }
    var showThemeDialog by remember { mutableStateOf(false) }

    var dailyReminders by remember { mutableStateOf(prefs.getBoolean("daily_reminders_enabled", true)) }
    var monthlyReminders by remember { mutableStateOf(prefs.getBoolean("monthly_reminders_enabled", true)) }
    var streakRemindersEnabled by remember { mutableStateOf(prefs.getBoolean("settings_streak_reminders_enabled", true)) }
    var autoBackupEnabled by remember { mutableStateOf(prefs.getBoolean("auto_backup_enabled", true)) }
    var weeklySummaryEnabled by remember { mutableStateOf(prefs.getBoolean("weekly_summary_notifications_enabled", true)) }

    // Reactive Permission States
    var isNotificationAccessGranted by remember {
        mutableStateOf(MHealthNotificationListenerService.isServiceEnabled(context))
    }
    var isLocationPermissionGranted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED)
    }
    var isBackgroundLocationGranted by remember {
        mutableStateOf(
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_BACKGROUND_LOCATION) == PackageManager.PERMISSION_GRANTED
            } else true
        )
    }
    var isUsageStatsGranted by remember { mutableStateOf(hasUsageStatsPermission(context)) }
    var showLocationDisclosure by remember { mutableStateOf(false) }
    val isHomeSet = homeLocation != null
    var isReminderDismissed by remember { mutableStateOf(prefs.getBoolean("home_permissions_reminder_dismissed", false)) }

    val lifecycleOwner = LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME) {
                isNotificationAccessGranted = MHealthNotificationListenerService.isServiceEnabled(context)
                isLocationPermissionGranted = ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
                isBackgroundLocationGranted = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_BACKGROUND_LOCATION) == PackageManager.PERMISSION_GRANTED
                } else true
                isUsageStatsGranted = hasUsageStatsPermission(context)
                isReminderDismissed = prefs.getBoolean("home_permissions_reminder_dismissed", false)
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }

    val bgLocationLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        isBackgroundLocationGranted = granted
    }

    val locPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        val fineGranted = results[Manifest.permission.ACCESS_FINE_LOCATION] == true
        val coarseGranted = results[Manifest.permission.ACCESS_COARSE_LOCATION] == true
        isLocationPermissionGranted = fineGranted || coarseGranted
        if (isLocationPermissionGranted && Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            bgLocationLauncher.launch(Manifest.permission.ACCESS_BACKGROUND_LOCATION)
        } else {
            isBackgroundLocationGranted = true
        }
    }

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
                    text = "Profile & Settings",
                    fontSize = 26.sp,
                    fontWeight = FontWeight.ExtraBold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Text(
                    text = "Manage your localized settings and reports securely",
                    fontSize = 13.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 4.dp)
                )
            }
        }

        // Patient Identity Metadata Card
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
            ) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text(
                        text = "Patient Identity Metadata",
                        fontWeight = FontWeight.Bold,
                        fontSize = 14.sp,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                    HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.1f))

                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("Patient Name:", fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Text(profile?.name ?: "Lumen User", fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                    }
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("Age / Profession:", fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Text("${profile?.age ?: 0} / ${profile?.profession ?: "N/A"}", fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                    }
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("Registered Country:", fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Text(profile?.country ?: "N/A", fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                    }
                }
            }
        }

        // Theme Appearance Card
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { showThemeDialog = true }
                        .padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Column {
                        Text("Theme Appearance", fontWeight = FontWeight.Bold, fontSize = 14.sp, fontFamily = Fredoka)
                        Text("Active: ${themeMode.uppercase()}", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    Icon(Icons.Default.ChevronRight, null, tint = MaterialTheme.colorScheme.primary)
                }
            }
        }

        // System Permissions Card
        item {
            InfoCard("System Permissions", headerColor = MaterialTheme.colorScheme.primary) {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    PermissionSettingRow(
                        title = "GPS Location Permission",
                        subtitle = "Required to track daily movement",
                        isGranted = isLocationPermissionGranted,
                        icon = Icons.Default.GpsFixed,
                        onClick = {
                            if (!isLocationPermissionGranted) {
                                showLocationDisclosure = true
                            } else {
                                Toast.makeText(context, "Location permission is enabled.", Toast.LENGTH_SHORT).show()
                            }
                        }
                    )
                    HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.1f))
                    PermissionSettingRow(
                        title = "Notification Listener Access",
                        subtitle = "Required for notification rates & music",
                        isGranted = isNotificationAccessGranted,
                        icon = Icons.Default.Notifications,
                        onClick = {
                            if (!isNotificationAccessGranted) {
                                context.startActivity(Intent("android.settings.ACTION_NOTIFICATION_LISTENER_SETTINGS"))
                            } else {
                                Toast.makeText(context, "Notification listener access is enabled.", Toast.LENGTH_SHORT).show()
                            }
                        }
                    )
                    HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.1f))
                    PermissionSettingRow(
                        title = "Usage Stats Access",
                        subtitle = "Required for app screen time telemetry",
                        isGranted = isUsageStatsGranted,
                        icon = Icons.Default.BarChart,
                        onClick = {
                            if (!isUsageStatsGranted) {
                                context.startActivity(Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS))
                            } else {
                                Toast.makeText(context, "Usage stats access is enabled.", Toast.LENGTH_SHORT).show()
                            }
                        }
                    )
                    HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.1f))
                    PermissionSettingRow(
                        title = "Home Location Anchor",
                        subtitle = if (isHomeSet) "Home coordinates configured" else "Tap to lock current GPS as Home",
                        isGranted = isHomeSet,
                        icon = Icons.Default.Home,
                        onClick = {
                            if (!homeCapturing) {
                                homeCapturing = true
                                DataCollector(context).captureHomeLocation { success ->
                                    homeCapturing = false
                                    val msg = if (success) "Home location saved" else "Location timeout"
                                    Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                                }
                            }
                        }
                    )
                }
            }
        }

        // Data Management & Backup Card
        item {
            InfoCard("Data Management & Backup", headerColor = MaterialTheme.colorScheme.primary) {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    ToggleRow(
                        title = "Automatic Daily Backup",
                        subtitle = "Backs up data to public Downloads daily",
                        checked = autoBackupEnabled,
                        color = MaterialTheme.colorScheme.primary,
                        onToggle = {
                            autoBackupEnabled = it
                            prefs.edit().putBoolean("auto_backup_enabled", it).apply()
                        }
                    )

                    HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.1f))

                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(
                            onClick = {
                                val dateStr = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
                                exportLauncher.launch("Lumen_backup_$dateStr.json")
                            },
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                            modifier = Modifier
                                .weight(1f)
                                .height(44.dp),
                            shape = RoundedCornerShape(10.dp)
                        ) {
                            Icon(Icons.Default.Download, null, tint = Color.Black, modifier = Modifier.size(18.dp))
                            Spacer(Modifier.width(6.dp))
                            Text("Export", color = Color.Black, fontWeight = FontWeight.Bold, fontSize = 12.sp, fontFamily = Fredoka)
                        }

                        Button(
                            onClick = {
                                importLauncher.launch(arrayOf("application/json"))
                            },
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
                            modifier = Modifier
                                .weight(1f)
                                .height(44.dp),
                            shape = RoundedCornerShape(10.dp)
                        ) {
                            Icon(Icons.Default.Upload, null, tint = MaterialTheme.colorScheme.onPrimaryContainer, modifier = Modifier.size(18.dp))
                            Spacer(Modifier.width(6.dp))
                            Text("Import", color = MaterialTheme.colorScheme.onPrimaryContainer, fontWeight = FontWeight.Bold, fontSize = 12.sp, fontFamily = Fredoka)
                        }
                    }
                }
            }
        }

        // Notifications Preferences Card
        item {
            InfoCard("Notifications", headerColor = MaterialTheme.colorScheme.primary) {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    ToggleRow(
                        title = "Daily Reflection Reminders",
                        subtitle = "Gentle alerts when daily logs are pending",
                        checked = dailyReminders,
                        color = MaterialTheme.colorScheme.primary,
                        onToggle = {
                            dailyReminders = it
                            prefs.edit().putBoolean("daily_reminders_enabled", it).apply()
                        }
                    )
                    HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.1f))
                    ToggleRow(
                        title = "Monthly Screener Reminders",
                        subtitle = "Notifications for detailed wellness assessments",
                        checked = monthlyReminders,
                        color = MaterialTheme.colorScheme.primary,
                        onToggle = {
                            monthlyReminders = it
                            prefs.edit().putBoolean("monthly_reminders_enabled", it).apply()
                        }
                    )
                    HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.1f))
                    ToggleRow(
                        title = "Weekly Insights Summary",
                        subtitle = "Qualitative summary of weekly rhythms on Sunday evening",
                        checked = weeklySummaryEnabled,
                        color = MaterialTheme.colorScheme.primary,
                        onToggle = {
                            weeklySummaryEnabled = it
                            prefs.edit().putBoolean("weekly_summary_notifications_enabled", it).apply()
                        }
                    )
                    HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.1f))
                    ToggleRow(
                        title = "Streak Reminders",
                        subtitle = "Evening push notification if a quest streak is at risk",
                        checked = streakRemindersEnabled,
                        color = MaterialTheme.colorScheme.primary,
                        onToggle = {
                            streakRemindersEnabled = it
                            prefs.edit().putBoolean("settings_streak_reminders_enabled", it).apply()
                        }
                    )
                }
            }
        }

        // Support & Crisis Resources
        item {
            val countryName = profile?.country ?: "N/A"
            val helplines = remember(countryName) { getHelplinesByCountry(countryName) }

            InfoCard("Support & Crisis Resources", headerColor = MaterialTheme.colorScheme.primary) {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text(
                        text = "Resources for ${if (countryName != "N/A") countryName else "Worldwide"}:",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    helplines.forEach { helpline ->
                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable {
                                    try {
                                        val intent = if (helpline.type == HelplineType.PHONE) {
                                            Intent(Intent.ACTION_DIAL, Uri.parse("tel:${helpline.number}"))
                                        } else {
                                            Intent(Intent.ACTION_VIEW, Uri.parse(helpline.number))
                                        }
                                        context.startActivity(intent)
                                    } catch (e: Exception) {
                                        Toast.makeText(context, "Could not open action", Toast.LENGTH_SHORT).show()
                                    }
                                },
                            shape = RoundedCornerShape(12.dp),
                            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer.copy(0.3f)),
                            border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(0.2f))
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(12.dp),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Column(modifier = Modifier.weight(1f)) {
                                    Text(helpline.name, fontWeight = FontWeight.Bold, fontSize = 13.sp, fontFamily = Fredoka)
                                    Text(helpline.availability, fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                }
                                Icon(
                                    imageVector = if (helpline.type == HelplineType.PHONE) Icons.Default.Call else Icons.Default.OpenInNew,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.primary,
                                    modifier = Modifier.size(18.dp)
                                )
                            }
                        }
                    }
                }
            }
        }

        // Research Contribution
        item {
            var showResearchDialogSettings by remember { mutableStateOf(false) }
            if (showResearchDialogSettings) {
                ResearchContributionDialog(onDismiss = { showResearchDialogSettings = false })
            }
            InfoCard("Research Project Contribution", headerColor = MaterialTheme.colorScheme.primary) {
                Column(modifier = Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        text = "Contribute to mental health research by sharing anonymized behavioral telemetry. All date timelines and PII are stripped, and differential privacy noise is added to protect your identity.",
                        fontSize = 12.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Button(
                        onClick = { showResearchDialogSettings = true },
                        colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text("🔬 Share Anonymized Telemetry Data", color = Color.Black, fontSize = 13.sp, fontFamily = Fredoka, fontWeight = FontWeight.Bold)
                    }
                }
            }
        }

        // System Setup & Reset
        item {
            var showHardResetDialog by remember { mutableStateOf(false) }

            if (showHardResetDialog) {
                AlertDialog(
                    onDismissRequest = { showHardResetDialog = false },
                    title = {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.Warning, null, tint = MaterialTheme.colorScheme.error, modifier = Modifier.size(20.dp))
                            Spacer(Modifier.width(8.dp))
                            Text("Clear All Data?", fontWeight = FontWeight.Bold, fontSize = 16.sp, fontFamily = Fredoka)
                        }
                    },
                    text = {
                        Column {
                            Text(
                                "This will permanently delete ALL collected telemetry data — daily features, app sessions, notification logs, behavioral norms, and analysis history.",
                                fontSize = 13.sp, lineHeight = 18.sp, color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            Spacer(Modifier.height(10.dp))
                            Text(
                                "✅ Your data will be automatically backed up as a JSON file in your Downloads directory before deletion.",
                                fontSize = 12.sp, lineHeight = 16.sp, color = MaterialTheme.colorScheme.primary,
                                fontWeight = FontWeight.Medium
                            )
                            Spacer(Modifier.height(6.dp))
                            Text(
                                "Tracking will restart from Day 1 setup in sync.",
                                fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    },
                    confirmButton = {
                        Button(
                            onClick = {
                                showHardResetDialog = false
                                exportDataAsJson(context, filePrefix = "mhealth_backup_before_reset_")
                                DataRepository.triggerHardReset()
                            },
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error),
                            shape = RoundedCornerShape(8.dp)
                        ) {
                            Text("Clear All Data", color = Color.White, fontSize = 13.sp, fontFamily = Fredoka, fontWeight = FontWeight.Bold)
                        }
                    },
                    dismissButton = {
                        TextButton(
                            onClick = { showHardResetDialog = false }
                        ) {
                            Text("Cancel", color = MaterialTheme.colorScheme.primary, fontSize = 13.sp, fontFamily = Fredoka, fontWeight = FontWeight.Medium)
                        }
                    },
                    shape = RoundedCornerShape(24.dp),
                    containerColor = MaterialTheme.colorScheme.surface
                )
            }

            InfoCard("System Setup & Reset", headerColor = MaterialTheme.colorScheme.primary) {
                Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                        Column {
                            Text(if (isBuilding) "Learning Phase" else "Active Monitoring", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onBackground, fontFamily = Fredoka)
                            Text("Day $progress completed", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                        Button(
                            onClick = {
                                DataRepository.triggerReset()
                                Toast.makeText(context, "Recalculating all wellness scores...", Toast.LENGTH_SHORT).show()
                            },
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                            shape = RoundedCornerShape(8.dp)
                        ) {
                            Text("Soft Reset", fontSize = 11.sp, color = Color.Black, fontFamily = Fredoka, fontWeight = FontWeight.Bold)
                        }
                    }

                    HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.1f))

                    Button(
                        onClick = { showHardResetDialog = true },
                        colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error.copy(0.15f), contentColor = MaterialTheme.colorScheme.error),
                        modifier = Modifier.fillMaxWidth(),
                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.error),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Icon(Icons.Default.Warning, null, tint = MaterialTheme.colorScheme.error, modifier = Modifier.size(16.dp))
                        Spacer(Modifier.width(8.dp))
                        Text("Clear All Data (Fresh Start)", fontSize = 13.sp, color = MaterialTheme.colorScheme.error, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                    }
                    Text(
                        "Exports a local JSON backup, wipes the database, and restarts setup from Day 1.",
                        fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurfaceVariant.copy(0.7f), lineHeight = 13.sp
                    )
                }
            }
        }

        // Data & On-Device Storage
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
            ) {
                Column(
                    modifier = Modifier.padding(18.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Text(
                        text = "Data & On-Device Storage",
                        fontSize = 15.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                    Text(
                        text = "All telemetry and behavioral vectors are processed 100% on-device. No raw logs leave your phone.",
                        fontSize = 12.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        lineHeight = 17.sp
                    )
                }
            }
        }
    }

    if (showLocationDisclosure) {
        LocationDisclosureDialog(
            onDismiss = { showLocationDisclosure = false },
            onConfirm = {
                showLocationDisclosure = false
                locPermissionLauncher.launch(
                    arrayOf(
                        Manifest.permission.ACCESS_FINE_LOCATION,
                        Manifest.permission.ACCESS_COARSE_LOCATION
                    )
                )
            }
        )
    }

    if (showThemeDialog) {
        AlertDialog(
            onDismissRequest = { showThemeDialog = false },
            title = { Text("Select Theme", fontFamily = Fredoka, fontWeight = FontWeight.Bold) },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf("dark" to "Dark Mode (Recommended)", "light" to "Light Mode", "system" to "System Default").forEach { (modeKey, modeLabel) ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable {
                                    themeMode = modeKey
                                    prefs.edit().putString("app_theme_mode", modeKey).apply()
                                    showThemeDialog = false
                                }
                                .padding(vertical = 8.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            RadioButton(
                                selected = (themeMode == modeKey),
                                onClick = {
                                    themeMode = modeKey
                                    prefs.edit().putString("app_theme_mode", modeKey).apply()
                                    showThemeDialog = false
                                }
                            )
                            Spacer(Modifier.width(8.dp))
                            Text(modeLabel, fontSize = 13.sp, fontFamily = Fredoka)
                        }
                    }
                }
            },
            confirmButton = {},
            dismissButton = {
                TextButton(onClick = { showThemeDialog = false }) {
                    Text("Close", fontFamily = Fredoka)
                }
            }
        )
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
