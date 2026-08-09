package com.example.mhealth.ui.screens

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
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
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import com.example.mhealth.logic.DataCollector
import com.example.mhealth.logic.DataRepository
import com.example.mhealth.logic.db.BaselineEntity
import com.example.mhealth.logic.db.MHealthDatabase
import com.example.mhealth.models.PersonalityVector
import android.content.Intent
import androidx.compose.ui.graphics.vector.ImageVector
import com.example.mhealth.services.*
import com.example.mhealth.ui.components.Fredoka
import com.example.mhealth.ui.components.PermissionReminderRow
import com.example.mhealth.ui.components.WeeklyDigestDialog
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.abs

@Composable
fun HomeScreen(
    onNavigateToInsights: () -> Unit,
    onNavigateToActivities: () -> Unit,
    onNavigateToCheckIn: () -> Unit
) {
    val context = LocalContext.current
    val userProfile by DataRepository.userProfile.collectAsState()
    val latestResult by DataRepository.latestAnalysisResult.collectAsState()
    val provisional by DataRepository.provisionalAnalysis.collectAsState()
    val isBuilding by DataRepository.isBuildingBaseline.collectAsState()
    
    val activeResult = provisional ?: latestResult
    val score = activeResult?.effectiveScore ?: -1f
    
    val name = (userProfile?.name ?: "").trim()
    val greeting = remember { getGreeting() }
    
    val weeklyFeatures by DataRepository.weeklyFeatureHistory.collectAsState()
    val isDnaReady by DataRepository.isDnaBaselineReady.collectAsState()
    
    val db = remember { MHealthDatabase.getInstance(context.applicationContext) }
    val baselineEntities by produceState<List<BaselineEntity>>(emptyList(), db) {
        val userId = DataRepository.userProfile.value?.email ?: "patient@lumen.health"
        value = db.baselineDao().getBaseline(userId)
    }

    val prefs = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }
    val todayStr = remember { SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date()) }
    var lastCheckinDate by remember { mutableStateOf<String>(prefs.getString("daily_checkin_date_last", "") ?: "") }
    val alreadyCheckedIn = lastCheckinDate == todayStr

    var showWeeklyDigest by remember { mutableStateOf<Boolean>(false) }

    // Ambient breathing pulse animation
    val infiniteTransition = rememberInfiniteTransition(label = "AmbientGlow")
    val glowAlpha by infiniteTransition.animateFloat(
        initialValue = 0.04f,
        targetValue = 0.15f,
        animationSpec = infiniteRepeatable(
            animation = tween(4000, easing = LinearOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "GlowAlpha"
    )

    // Permission checks
    var isNotificationAccessGranted by remember {
        mutableStateOf<Boolean>(MHealthNotificationListenerService.isServiceEnabled(context))
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
    var isReminderDismissed by remember { mutableStateOf(prefs.getBoolean("home_permissions_reminder_dismissed", false)) }
    var homeCapturing by remember { mutableStateOf(false) }

    val homeLocationState by DataRepository.homeLocation.collectAsState()
    val isHomeSet = homeLocationState != null

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
                lastCheckinDate = prefs.getString("daily_checkin_date_last", "") ?: ""
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    val bgLocationLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        isBackgroundLocationGranted = granted
    }
    val locPermissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { results ->
        val fineGranted = results[Manifest.permission.ACCESS_FINE_LOCATION] == true
        val coarseGranted = results[Manifest.permission.ACCESS_COARSE_LOCATION] == true
        isLocationPermissionGranted = fineGranted || coarseGranted
        if (isLocationPermissionGranted && Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            bgLocationLauncher.launch(Manifest.permission.ACCESS_BACKGROUND_LOCATION)
        } else {
            isBackgroundLocationGranted = true
        }
    }

    val hasMissingPermissions = !isNotificationAccessGranted || !isLocationPermissionGranted || !isBackgroundLocationGranted || !isHomeSet || !isUsageStatsGranted
    val hasCriticalMissingPermissions = !isLocationPermissionGranted || !isUsageStatsGranted
    val showBanner = hasCriticalMissingPermissions || (!isReminderDismissed && hasMissingPermissions)

    if (showWeeklyDigest && weeklyFeatures.isNotEmpty() && baselineEntities.isNotEmpty()) {
        val baseVec = PersonalityVector(
            screenTimeHours = baselineEntities.firstOrNull { it.featureName == "screenTimeHours" }?.baselineValue ?: 4f,
            dailyStepCount = baselineEntities.firstOrNull { it.featureName == "dailyStepCount" }?.baselineValue ?: 3000f,
            sleepDurationHours = baselineEntities.firstOrNull { it.featureName == "sleepDurationHours" }?.baselineValue ?: 7f,
            socialAppRatio = baselineEntities.firstOrNull { it.featureName == "socialAppRatio" }?.baselineValue ?: 0.2f
        )
        WeeklyDigestDialog(
            weeklyFeatures = weeklyFeatures,
            baseline = baseVec,
            onDismiss = { showWeeklyDigest = false }
        )
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
    ) {
        // Ambient glow background
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(300.dp)
                .alpha(glowAlpha)
                .background(
                    Brush.verticalGradient(
                        colors = listOf(
                            MaterialTheme.colorScheme.primary,
                            Color.Transparent
                        )
                    )
                )
        )

        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally,
            contentPadding = PaddingValues(24.dp),
            verticalArrangement = Arrangement.spacedBy(20.dp)
        ) {
            // Header
            item {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalAlignment = Alignment.Start
                ) {
                    Text(
                        text = "Lumen.",
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.padding(bottom = 6.dp)
                    )
                    Text(
                        text = if (name.isNotBlank()) "$greeting,\n$name." else "$greeting.",
                        fontSize = 28.sp,
                        fontWeight = FontWeight.ExtraBold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                    Text(
                        text = SimpleDateFormat("EEEE, MMMM d", Locale.getDefault()).format(Date()),
                        fontSize = 14.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 4.dp)
                    )
                }
            }

            // Permissions Banner (if needed)
            if (showBanner) {
                item {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(20.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.15f)),
                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(0.3f))
                    ) {
                        Column(
                            modifier = Modifier.padding(16.dp),
                            verticalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Icon(
                                        imageVector = Icons.Default.Warning,
                                        contentDescription = null,
                                        tint = MaterialTheme.colorScheme.primary,
                                        modifier = Modifier.size(20.dp)
                                    )
                                    Spacer(Modifier.width(8.dp))
                                    Text(
                                        text = "Setup Permissions",
                                        fontSize = 16.sp,
                                        fontWeight = FontWeight.Bold,
                                        fontFamily = Fredoka,
                                        color = MaterialTheme.colorScheme.onBackground
                                    )
                                }
                                if (!hasCriticalMissingPermissions) {
                                    IconButton(
                                        onClick = {
                                            prefs.edit().putBoolean("home_permissions_reminder_dismissed", true).apply()
                                            isReminderDismissed = true
                                        },
                                        modifier = Modifier.size(24.dp)
                                    ) {
                                        Icon(
                                            imageVector = Icons.Default.Close,
                                            contentDescription = "Dismiss",
                                            tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                                            modifier = Modifier.size(16.dp)
                                        )
                                    }
                                }
                            }

                            Text(
                                text = "Lumen needs permissions to passively monitor wellness telemetry.",
                                fontSize = 12.sp,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                lineHeight = 17.sp
                            )

                            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                                if (!isLocationPermissionGranted || !isBackgroundLocationGranted) {
                                    PermissionReminderRow(
                                        name = "GPS Location Permission",
                                        buttonText = "Grant",
                                        onClick = { locPermissionLauncher.launch(arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION)) }
                                    )
                                }
                                if (isLocationPermissionGranted && !isHomeSet) {
                                    PermissionReminderRow(
                                        name = "Home Location Anchor",
                                        buttonText = if (homeCapturing) "Acquiring..." else "Set Home",
                                        enabled = !homeCapturing,
                                        onClick = {
                                            homeCapturing = true
                                            DataCollector(context).captureHomeLocation { success ->
                                                homeCapturing = false
                                                val msg = if (success) "Home location saved" else "Location timeout"
                                                Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                                            }
                                        }
                                    )
                                }
                                if (!isNotificationAccessGranted) {
                                    PermissionReminderRow(
                                        name = "Notification Access",
                                        buttonText = "Enable",
                                        onClick = { context.startActivity(Intent("android.settings.ACTION_NOTIFICATION_LISTENER_SETTINGS")) }
                                    )
                                }
                            }
                        }
                    }
                }
            }

            // Wellness Pulse Hero Card
            item {
                val isElevated = score > 0.38f
                val dotColor = if (isElevated) Color(0xFFF59E0B) else MaterialTheme.colorScheme.primary
                val summaryText = remember(isBuilding, score, isDnaReady) {
                    when {
                        isBuilding || !isDnaReady -> "Lumen is establishing your personal lifestyle baseline."
                        isElevated -> "We've noticed some quiet shifts in your routine flow recently."
                        else -> "Your daily rhythm has been steady and balanced this week."
                    }
                }

                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { onNavigateToInsights() },
                    shape = RoundedCornerShape(22.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(0.2f))
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
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                Box(
                                    modifier = Modifier
                                        .size(10.dp)
                                        .background(dotColor, CircleShape)
                                )
                                Text(
                                    text = "Wellness Pulse",
                                    fontSize = 13.sp,
                                    fontWeight = FontWeight.Bold,
                                    fontFamily = Fredoka,
                                    color = MaterialTheme.colorScheme.primary
                                )
                            }
                            
                            Icon(
                                imageVector = Icons.Default.ArrowForward,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.size(18.dp)
                            )
                        }

                        Text(
                            text = summaryText,
                            fontSize = 17.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.onBackground,
                            lineHeight = 23.sp
                        )

                        // Mini Preview Chips (FlowRow prevents character squishing on narrow screens)
                        @OptIn(ExperimentalLayoutApi::class)
                        FlowRow(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            RhythmChip(label = "Sleep", status = "Aligned", color = MaterialTheme.colorScheme.primary)
                            RhythmChip(label = "Movement", status = "Steady", color = MaterialTheme.colorScheme.primary)
                            RhythmChip(label = "Social", status = "Flowing", color = MaterialTheme.colorScheme.primary)
                        }
                    }
                }
            }

            // Daily Quote / Reflection Card
            item {
                val dailyQuotes = remember {
                    listOf(
                        "Peace comes from within. Do not seek it without." to "Buddha",
                        "Rest is not idleness, and to lie on the grass under trees is by no means a waste of time." to "John Lubbock",
                        "Almost everything will work again if you unplug it for a few minutes, including you." to "Anne Lamott",
                        "You don't have to control your thoughts. You just have to stop letting them control you." to "Dan Millman",
                        "Mindfulness is about being fully awake in our lives." to "Jon Kabat-Zinn",
                        "Feelings come and go like clouds in a windy sky. Conscious breathing is my anchor." to "Thich Nhat Hanh",
                        "Small steps in the right direction can turn out to be the biggest step of your life." to "Wisdom Tradition",
                        "Quiet the mind, and the soul will speak." to "Ma Jaya Sati Bhagavati",
                        "Within you, there is a stillness and a sanctuary to which you can retreat at any time." to "Hermann Hesse"
                    )
                }
                val dayOfYear = remember { Calendar.getInstance().get(Calendar.DAY_OF_YEAR) }
                val (quoteText, quoteAuthor) = dailyQuotes[dayOfYear % dailyQuotes.size]

                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(18.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.15f)),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.FormatQuote,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.size(18.dp)
                            )
                            Text(
                                text = "Daily Reflection Quote",
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Bold,
                                fontFamily = Fredoka,
                                color = MaterialTheme.colorScheme.primary
                            )
                        }
                        Text(
                            text = "\"$quoteText\"",
                            fontSize = 13.sp,
                            fontStyle = androidx.compose.ui.text.font.FontStyle.Italic,
                            color = MaterialTheme.colorScheme.onBackground,
                            lineHeight = 18.sp
                        )
                        Text(
                            text = "— $quoteAuthor",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.align(Alignment.End)
                        )
                    }
                }
            }

            // Quick Check-In Prompt Card (if not checked in today)
            if (!alreadyCheckedIn) {
                item {
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { onNavigateToCheckIn() },
                        shape = RoundedCornerShape(18.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primary.copy(0.08f)),
                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(0.2f))
                    ) {
                        Row(
                            modifier = Modifier.padding(16.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.EditNote,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.primary,
                                    modifier = Modifier.size(24.dp)
                                )
                                Column {
                                    Text(
                                        text = "Daily Reflection",
                                        fontSize = 15.sp,
                                        fontWeight = FontWeight.Bold,
                                        fontFamily = Fredoka,
                                        color = MaterialTheme.colorScheme.onBackground
                                    )
                                    Text(
                                        text = "How are you feeling today?",
                                        fontSize = 12.sp,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                            }

                            Icon(
                                imageVector = Icons.Default.ChevronRight,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.primary
                            )
                        }
                    }
                }
            }

            // Navigation Cards Grid (Equalized Heights)
            item {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text(
                        text = "Explore",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )

                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(IntrinsicSize.Max),
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        NavTile(
                            modifier = Modifier.weight(1f),
                            title = "Activities & Quests",
                            subtitle = "Habits, Wind Down, Detox",
                            icon = Icons.Default.CheckCircle,
                            onClick = onNavigateToActivities
                        )
                        NavTile(
                            modifier = Modifier.weight(1f),
                            title = "Weekly Digest",
                            subtitle = "Sunday routine summary",
                            icon = Icons.Default.Assessment,
                            onClick = { showWeeklyDigest = true }
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun RhythmChip(label: String, status: String, color: Color) {
    Surface(
        shape = RoundedCornerShape(8.dp),
        color = color.copy(alpha = 0.1f),
        border = BorderStroke(1.dp, color.copy(alpha = 0.2f))
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Box(modifier = Modifier.size(6.dp).background(color, CircleShape))
            Text(
                text = "$label: $status",
                fontSize = 11.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                color = color
            )
        }
    }
}

@Composable
private fun NavTile(
    modifier: Modifier = Modifier,
    title: String,
    subtitle: String,
    icon: ImageVector,
    onClick: () -> Unit
) {
    Card(
        modifier = modifier
            .fillMaxHeight()
            .clickable { onClick() },
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
    ) {
        Column(
            modifier = Modifier
                .fillMaxHeight()
                .padding(16.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(36.dp)
                    .background(MaterialTheme.colorScheme.primary.copy(0.12f), CircleShape),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(20.dp)
                )
            }
            Spacer(Modifier.height(12.dp))
            Text(
                text = title,
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                color = MaterialTheme.colorScheme.onBackground,
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(Modifier.height(4.dp))
            Text(
                text = subtitle,
                fontSize = 11.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                lineHeight = 14.sp,
                modifier = Modifier.fillMaxWidth()
            )
        }
    }
}

private fun getGreeting(): String {
    val hour = Calendar.getInstance().get(Calendar.HOUR_OF_DAY)
    return when (hour) {
        in 5..11 -> "Good Morning"
        in 12..16 -> "Good Afternoon"
        else -> "Good Evening"
    }
}

private fun hasUsageStatsPermission(context: Context): Boolean {
    val appOps = context.getSystemService(Context.APP_OPS_SERVICE) as android.app.AppOpsManager
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
