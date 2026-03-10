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
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.material3.adaptive.navigationsuite.NavigationSuiteScaffold
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.mhealth.logic.DataRepository
import com.example.mhealth.models.DailyReport
import com.example.mhealth.models.PersonalityVector
import com.example.mhealth.services.MonitoringService
import com.example.mhealth.ui.theme.MHealthTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MHealthTheme {
                MHealthApp()
            }
        }
    }
}

@Composable
fun MHealthApp() {
    var currentDestination by remember { mutableStateOf(AppDestinations.HOME) }
    val context = LocalContext.current
    
    val permissionsToRequest = mutableListOf(
        Manifest.permission.READ_CALL_LOG,
        Manifest.permission.READ_SMS,
        Manifest.permission.READ_CONTACTS,
        Manifest.permission.ACCESS_FINE_LOCATION,
        Manifest.permission.ACCESS_COARSE_LOCATION
    ).apply {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            add(Manifest.permission.POST_NOTIFICATIONS)
        }
    }

    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.values.all { it }
        if (allGranted && hasUsageStatsPermission(context)) {
            startMonitoringService(context)
        }
    }

    LaunchedEffect(Unit) {
        launcher.launch(permissionsToRequest.toTypedArray())
        if (!hasUsageStatsPermission(context)) {
            context.startActivity(Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS))
        } else {
            startMonitoringService(context)
        }
    }

    NavigationSuiteScaffold(
        navigationSuiteItems = {
            AppDestinations.entries.forEach {
                item(
                    icon = { Icon(it.icon, contentDescription = it.label) },
                    label = { Text(it.label) },
                    selected = it == currentDestination,
                    onClick = { currentDestination = it }
                )
            }
        }
    ) {
        Box(modifier = Modifier.fillMaxSize().padding(16.dp)) {
            when (currentDestination) {
                AppDestinations.HOME -> HomeScreen()
                AppDestinations.METRICS -> MetricsScreen()
                AppDestinations.ANOMALY -> AnomalyScreen()
                AppDestinations.SETTINGS -> SettingsScreen()
            }
        }
    }
}

@Composable
fun HomeScreen() {
    val latestVector by DataRepository.latestVector.collectAsState()
    val isBuilding by DataRepository.isBuildingBaseline.collectAsState()
    val progress by DataRepository.baselineProgress.collectAsState()
    
    Column(modifier = Modifier.fillMaxSize().padding(top = 32.dp)) {
        Text("Real-time Data Collection", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
        
        if (isBuilding) {
            BaselineProgressCard(progress)
        }

        Spacer(modifier = Modifier.height(16.dp))
        
        if (latestVector == null) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CircularProgressIndicator()
                    Text("Collecting window data (00:00 - Now)...", modifier = Modifier.padding(16.dp))
                }
            }
        } else {
            FeatureList(latestVector!!)
        }
    }
}

@Composable
fun BaselineProgressCard(days: Int) {
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
        modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Building Baseline Personality Profile", fontWeight = FontWeight.Bold)
            Text("Day $days of 28", style = MaterialTheme.typography.bodySmall)
            LinearProgressIndicator(
                progress = { days / 28f },
                modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
            )
            Text("Detection will start after baseline is established.", 
                style = MaterialTheme.typography.labelSmall, modifier = Modifier.padding(top = 4.dp))
        }
    }
}

@Composable
fun FeatureList(vector: PersonalityVector) {
    val features = vector.toMap()
    LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        items(features.toList()) { (name, value) ->
            Card(modifier = Modifier.fillMaxWidth()) {
                Row(modifier = Modifier.padding(16.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text(name.replaceFirstChar { it.uppercase() }, fontWeight = FontWeight.Medium)
                    Text("%.2f".format(value), color = MaterialTheme.colorScheme.primary)
                }
            }
        }
    }
}

@Composable
fun MetricsScreen() {
    val reports by DataRepository.reports.collectAsState()
    val isBuilding by DataRepository.isBuildingBaseline.collectAsState()
    
    Column(modifier = Modifier.fillMaxSize().padding(top = 32.dp)) {
        Text("Stability Metrics", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(16.dp))
        
        if (isBuilding) {
            Text("Metrics will be available after the 28-day baseline period.")
        } else if (reports.isEmpty()) {
            Text("No reports generated yet.")
        } else {
            LazyColumn {
                items(reports.reversed()) { report ->
                    MetricItem(report)
                }
            }
        }
    }
}

@Composable
fun MetricItem(report: DailyReport) {
    Card(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(horizontalArrangement = Arrangement.SpaceBetween, modifier = Modifier.fillMaxWidth()) {
                Text("Day ${report.dayNumber}", fontWeight = FontWeight.Bold)
                Text(report.alertLevel.uppercase(), color = getAlertColor(report.alertLevel))
            }
            Text("Anomaly Score: ${"%.3f".format(report.anomalyScore)}")
            Text("Pattern: ${report.patternType}")
        }
    }
}

@Composable
fun AnomalyScreen() {
    val reports by DataRepository.reports.collectAsState()
    val isBuilding by DataRepository.isBuildingBaseline.collectAsState()
    val baseline by DataRepository.baseline.collectAsState()
    val lastReport = reports.lastOrNull()
    
    Column(modifier = Modifier.fillMaxSize().padding(top = 32.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        Text("Anomaly Detection", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(32.dp))
        
        if (isBuilding) {
            StatusIndicator("building")
            Spacer(modifier = Modifier.height(24.dp))
            Text("Currently in baseline accumulation phase.", style = MaterialTheme.typography.titleMedium)
        } else {
            StatusIndicator(lastReport?.alertLevel ?: "green")
            Spacer(modifier = Modifier.height(24.dp))
            
            lastReport?.let {
                Text("Evidence: ${"%.2f".format(it.evidenceAccumulated)}", style = MaterialTheme.typography.titleLarge)
                Text("Sustained Days: ${it.sustainedDeviationDays}", style = MaterialTheme.typography.bodyLarge)
                Spacer(modifier = Modifier.height(16.dp))
                Text("Notes: ${it.notes}", modifier = Modifier.padding(16.dp))
            }
        }

        if (baseline != null) {
            Spacer(modifier = Modifier.height(32.dp))
            Text("Established Baseline Profile:", fontWeight = FontWeight.Bold)
            BaselineSummary(baseline!!)
        }
    }
}

@Composable
fun BaselineSummary(vector: PersonalityVector) {
    Card(modifier = Modifier.fillMaxWidth().padding(top = 8.dp)) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Avg Screen Time: %.1f hrs".format(vector.screenTimeHours))
            Text("Avg Unlocks: %.0f".format(vector.unlockCount))
            Text("Avg Social Ratio: %.2f".format(vector.socialAppRatio))
        }
    }
}

@Composable
fun StatusIndicator(level: String) {
    val color = when(level) {
        "building" -> MaterialTheme.colorScheme.secondary
        else -> getAlertColor(level)
    }
    Surface(
        modifier = Modifier.size(200.dp),
        shape = androidx.compose.foundation.shape.CircleShape,
        color = color.copy(alpha = 0.2f),
        border = androidx.compose.foundation.BorderStroke(4.dp, color)
    ) {
        Box(contentAlignment = Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(level.uppercase(), style = MaterialTheme.typography.headlineLarge, color = color, fontWeight = FontWeight.ExtraBold)
                Text("PHASE", style = MaterialTheme.typography.labelLarge, color = color)
            }
        }
    }
}

@Composable
fun SettingsScreen() {
    val progress by DataRepository.baselineProgress.collectAsState()
    Column(modifier = Modifier.fillMaxSize().padding(top = 32.dp)) {
        Text("Settings", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(16.dp))
        
        ListItem(
            headlineContent = { Text("Baseline Period") },
            supportingContent = { Text("Current progress: $progress/28 days") },
            trailingContent = { TextButton(onClick = {}) { Text("Reset") } }
        )
        ListItem(
            headlineContent = { Text("Anomaly Sensitivity") },
            supportingContent = { Text("Medium (Standard deviation > 1.5)") },
            trailingContent = { Icon(Icons.Default.Settings, contentDescription = null) }
        )
    }
}

fun getAlertColor(level: String): Color = when (level.lowercase()) {
    "green" -> Color(0xFF4CAF50)
    "yellow" -> Color(0xFFFBC02D)
    "orange" -> Color(0xFFFF9800)
    "red" -> Color(0xFFF44336)
    else -> Color.Gray
}

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

enum class AppDestinations(val label: String, val icon: ImageVector) {
    HOME("Data", Icons.Default.Home),
    METRICS("Metrics", Icons.Default.Info),
    ANOMALY("Detection", Icons.Default.Warning),
    SETTINGS("Settings", Icons.Default.Settings)
}
