package com.example.mhealth

import android.Manifest
import android.util.Log
import android.app.Activity
import android.app.AppOpsManager
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.net.Uri
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
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.material3.TabRowDefaults.tabIndicatorOffset
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
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.drawText
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.lifecycle.lifecycleScope
import com.example.mhealth.logic.DataRepository
import com.example.mhealth.logic.db.AnalysisResultEntity
import com.example.mhealth.logic.db.MHealthDatabase
import com.example.mhealth.logic.db.UserProfileEntity
import com.example.mhealth.models.DailyReport
import com.example.mhealth.models.PersonalityVector
import com.example.mhealth.services.MHealthAccessibilityService
import com.example.mhealth.services.MHealthNotificationListenerService
import com.example.mhealth.ui.theme.Typography
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.roundToInt
import kotlin.math.abs
import android.location.Geocoder
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.graphics.toArgb
import androidx.compose.material.icons.automirrored.filled.ShowChart
import com.google.android.play.core.appupdate.AppUpdateManagerFactory
import com.google.android.play.core.appupdate.AppUpdateManager
import com.google.android.play.core.install.model.UpdateAvailability
import com.google.android.play.core.install.model.AppUpdateType
import androidx.compose.ui.window.DialogProperties
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.content.ContextWrapper
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.platform.LocalDensity

/**
 * Reads the real navigation-bar height from the HOST Activity's decor view.
 * This is needed because inside a Dialog window (decorFitsSystemWindows = false),
 * navigationBarsPadding() resolves to 0 — the dialog does not receive insets.
 * Falls back to 48.dp on older APIs or if the activity cannot be found.
 */
@Composable
fun rememberNavBarPadding(): Dp {
    val context = LocalContext.current
    val density = LocalDensity.current
    return remember(density) {
        var ctx: Context = context
        var activity: Activity? = null
        while (ctx is ContextWrapper) {
            if (ctx is Activity) { activity = ctx; break }
            ctx = ctx.baseContext
        }
        val bottomPx = try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                activity?.window?.decorView?.rootWindowInsets
                    ?.getInsets(android.view.WindowInsets.Type.navigationBars())?.bottom ?: 0
            } else {
                @Suppress("DEPRECATION")
                activity?.window?.decorView?.rootWindowInsets?.systemWindowInsetBottom ?: 0
            }
        } catch (_: Exception) { 0 }
        with(density) { bottomPx.toDp() }.coerceAtLeast(0.dp)
    }
}

class MainActivity : ComponentActivity() {
    private lateinit var appUpdateManager: AppUpdateManager
    private val UPDATE_REQUEST_CODE = 999

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        
        // Synchronously initialize the local data repository
        DataRepository.init(applicationContext)
        
        intent?.getStringExtra("navigate_to")?.let {
            DataRepository.setNavigationRoute(it)
            intent.removeExtra("navigate_to")
        }

        // Start checking for updates in release build
        checkForUpdates()
        
        setContent {
            LumenAppShell()
        }
    }

    private fun checkForUpdates() {
        appUpdateManager = AppUpdateManagerFactory.create(this)
        val appUpdateInfoTask = appUpdateManager.appUpdateInfo

        appUpdateInfoTask.addOnSuccessListener { appUpdateInfo ->
            if (appUpdateInfo.updateAvailability() == UpdateAvailability.UPDATE_AVAILABLE
                && appUpdateInfo.isUpdateTypeAllowed(AppUpdateType.IMMEDIATE)
            ) {
                try {
                    appUpdateManager.startUpdateFlowForResult(
                        appUpdateInfo,
                        AppUpdateType.IMMEDIATE,
                        this,
                        UPDATE_REQUEST_CODE
                    )
                } catch (e: Exception) {
                    Log.e("MainActivity", "Failed to start update flow: ${e.message}")
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        if (::appUpdateManager.isInitialized) {
            appUpdateManager.appUpdateInfo.addOnSuccessListener { appUpdateInfo ->
                if (appUpdateInfo.updateAvailability() == UpdateAvailability.DEVELOPER_TRIGGERED_UPDATE_IN_PROGRESS) {
                    try {
                        appUpdateManager.startUpdateFlowForResult(
                            appUpdateInfo,
                            AppUpdateType.IMMEDIATE,
                            this,
                            UPDATE_REQUEST_CODE
                        )
                    } catch (e: Exception) {
                        Log.e("MainActivity", "Failed to resume update flow: ${e.message}")
                    }
                }
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == UPDATE_REQUEST_CODE) {
            if (resultCode != Activity.RESULT_OK) {
                Log.e("MainActivity", "Update flow failed or was cancelled by user. Result code: $resultCode")
                Toast.makeText(this, "A critical update is required to continue using Lumen.", Toast.LENGTH_LONG).show()
                finishAffinity()
            }
        }
    }

    override fun onNewIntent(intent: android.content.Intent) {
        super.onNewIntent(intent)
        intent.getStringExtra("navigate_to")?.let {
            DataRepository.setNavigationRoute(it)
            intent.removeExtra("navigate_to")
        }
    }
}

// =============================================================================
// Premium Typography & Color Palettes
// =============================================================================
val Fredoka = FontFamily.SansSerif
val BrandingFont = FontFamily(
    Font(R.font.fredoka, FontWeight.Normal)
)

// Calming Premium Theme colors
val NavyBackground = Color(0xFF0D1117)
val NavySurface = Color(0xFF161B22)
val NavyCard = Color(0xFF21262D)
val TealAccent = Color(0xFF2DD4BF)
val TealDark = Color(0xFF14B8A6)
val GrayTextPrimary = Color(0xFFE8EDF2)
val GrayTextSecondary = Color(0xFF8B9BB4)
val AlertWarning = Color(0xFFF59E0B)
val AlertRose = Color(0xFFF43F5E)

@Composable
fun LumenTheme(
    themeMode: String = "dark",
    content: @Composable () -> Unit
) {
    val darkTheme = when (themeMode) {
        "light" -> false
        "dark" -> true
        else -> isSystemInDarkTheme()
    }
    
    val colorScheme = if (darkTheme) {
        darkColorScheme(
            primary = TealAccent,
            onPrimary = Color.Black,
            primaryContainer = NavySurface,
            onPrimaryContainer = GrayTextPrimary,
            secondary = TealDark,
            onSecondary = Color.White,
            background = NavyBackground,
            onBackground = GrayTextPrimary,
            surface = NavySurface,
            onSurface = GrayTextPrimary,
            surfaceVariant = NavyCard,
            onSurfaceVariant = GrayTextSecondary,
            outline = Color(0xFF30363D),
            error = AlertRose,
            onError = Color.White
        )
    } else {
        lightColorScheme(
            primary = Color(0xFF0D9488),
            onPrimary = Color.White,
            primaryContainer = Color(0xFFCCFBF1),
            onPrimaryContainer = Color(0xFF0F172A),
            secondary = Color(0xFF14B8A6),
            onSecondary = Color.White,
            background = Color(0xFFF6FAF9),
            onBackground = Color(0xFF0F172A),
            surface = Color.White,
            onSurface = Color(0xFF0F172A),
            surfaceVariant = Color(0xFFE2E8F0),
            onSurfaceVariant = Color(0xFF64748B),
            outline = Color(0xFFCBD5E1),
            error = Color(0xFFE11D48),
            onError = Color.White
        )
    }

    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.background.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !darkTheme
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography
    ) {
        Surface(
            color = colorScheme.background,
            contentColor = colorScheme.onBackground
        ) {
            content()
        }
    }
}

// =============================================================================
// App Shell & Navigation skeleton
// =============================================================================
enum class LumenDest(val label: String, val icon: ImageVector) {
    HOME("Home", Icons.Default.Home),
    INSIGHTS("Insights", Icons.Default.Timeline),
    HISTORY("History", Icons.Default.History),
    CHECKIN("Check In", Icons.Default.Favorite),
    SETTINGS("Settings", Icons.Default.Settings)
}

enum class LumenNavState {
    ONBOARDING,
    DASHBOARD
}

@Composable
fun LumenAppShell() {
    val ctx = LocalContext.current
    val firstLoginComplete by DataRepository.firstLoginComplete.collectAsState()
    
    val prefs = remember(ctx) { ctx.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }
    var themeMode by remember { mutableStateOf(prefs.getString("app_theme_mode", "dark") ?: "dark") }
    
    // Reactive Theme preferences listener
    val listener = remember {
        SharedPreferences.OnSharedPreferenceChangeListener { _, key ->
            if (key == "app_theme_mode") {
                themeMode = prefs.getString("app_theme_mode", "dark") ?: "dark"
            }
        }
    }
    
    DisposableEffect(prefs) {
        prefs.registerOnSharedPreferenceChangeListener(listener)
        onDispose { prefs.unregisterOnSharedPreferenceChangeListener(listener) }
    }

    var appState by remember {
        mutableStateOf(
            if (DataRepository.firstLoginComplete.value) LumenNavState.DASHBOARD else LumenNavState.ONBOARDING
        )
    }

    LaunchedEffect(firstLoginComplete) {
        appState = if (firstLoginComplete) LumenNavState.DASHBOARD else LumenNavState.ONBOARDING
    }

    LumenTheme(themeMode = themeMode) {
        when (appState) {
            LumenNavState.ONBOARDING -> OnboardingWizard(onComplete = {
                appState = LumenNavState.DASHBOARD
            })
            LumenNavState.DASHBOARD -> MainLumenDashboard()
        }
    }
}

@Composable
fun AmbientCoolingBackground() {
    val infiniteTransition = rememberInfiniteTransition(label = "ambient_cooling")
    val shiftX by infiniteTransition.animateFloat(
        initialValue = 0.2f,
        targetValue = 0.8f,
        animationSpec = infiniteRepeatable(
            animation = tween(12000, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "shift_x"
    )
    val shiftY by infiniteTransition.animateFloat(
        initialValue = 0.8f,
        targetValue = 0.2f,
        animationSpec = infiniteRepeatable(
            animation = tween(16000, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "shift_y"
    )
    val radiusScale by infiniteTransition.animateFloat(
        initialValue = 0.6f,
        targetValue = 1.2f,
        animationSpec = infiniteRepeatable(
            animation = tween(14000, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "radius"
    )

    Canvas(modifier = Modifier.fillMaxSize()) {
        val width = size.width
        val height = size.height
        val baseColor = Color(0xFF070B19)
        val glowColor1 = Color(0xFF0F2B48)
        val glowColor2 = Color(0xFF1B0F3A)

        drawRect(color = baseColor)

        drawCircle(
            brush = Brush.radialGradient(
                colors = listOf(glowColor1.copy(alpha = 0.45f), Color.Transparent),
                center = Offset(width * shiftX, height * shiftY),
                radius = width * radiusScale
            ),
            center = Offset(width * shiftX, height * shiftY),
            radius = width * radiusScale
        )

        drawCircle(
            brush = Brush.radialGradient(
                colors = listOf(glowColor2.copy(alpha = 0.35f), Color.Transparent),
                center = Offset(width * (1f - shiftX), height * (1f - shiftY)),
                radius = width * (radiusScale * 0.9f)
            ),
            center = Offset(width * (1f - shiftX), height * (1f - shiftY)),
            radius = width * (radiusScale * 0.9f)
        )
    }
}

@Composable
fun MainLumenDashboard() {
    var selectedTab by remember { mutableStateOf(LumenDest.HOME) }
    val context = LocalContext.current
    
    val navRoute by DataRepository.navigationRouteTrigger.collectAsState()
    LaunchedEffect(navRoute) {
        if (navRoute == "insights") {
            selectedTab = LumenDest.INSIGHTS
            DataRepository.setNavigationRoute(null)
        }
    }
    
    LaunchedEffect(Unit) {
        val prefs = context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE)
        if (!prefs.contains("app_install_timestamp")) {
            prefs.edit().putLong("app_install_timestamp", System.currentTimeMillis()).apply()
        }
        DataRepository.initWithDb(context.applicationContext, "patient@lumen.health")
        val hasLoc = ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
        if (hasLoc && hasUsageStatsPermission(context)) {
            startMonitoringService(context)
        }
    }

    Scaffold(
        containerColor = Color.Transparent,
        bottomBar = {
            NavigationBar(containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.85f), tonalElevation = 0.dp) {
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
                            unselectedIconColor = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                            unselectedTextColor = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                            indicatorColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.15f)
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
            AmbientCoolingBackground()
            AnimatedContent(
                targetState = selectedTab,
                transitionSpec = {
                    if (targetState.ordinal > initialState.ordinal) {
                        (slideInHorizontally { width -> width } + fadeIn()).togetherWith(
                            slideOutHorizontally { width -> -width } + fadeOut()
                        )
                    } else {
                        (slideInHorizontally { width -> -width } + fadeIn()).togetherWith(
                            slideOutHorizontally { width -> width } + fadeOut()
                        )
                    }
                },
                label = "TabTransition"
            ) { targetTab ->
                when (targetTab) {
                    LumenDest.HOME -> HomeScreen(onNavigateToCheckIn = { selectedTab = LumenDest.CHECKIN })
                    LumenDest.INSIGHTS -> InsightsScreen()
                    LumenDest.HISTORY -> HistoryScreen()
                    LumenDest.CHECKIN -> CheckInScreen()
                    LumenDest.SETTINGS -> SettingsScreen()
                }
            }
        }
    }
}

// =============================================================================
// Contextual Check-In Section
// =============================================================================
@Composable
fun ContextualCheckinSection(
    onSave: (mood: Int, anxiety: Int) -> Unit
) {
    var mood by remember { mutableIntStateOf(3) }
    var anxiety by remember { mutableIntStateOf(3) }
    
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        HorizontalDivider(thickness = 0.5.dp, color = MaterialTheme.colorScheme.outline.copy(alpha = 0.2f))
        
        Text(
            text = "Rate your mood & stress to complete validation:",
            fontSize = 13.sp,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary,
            fontFamily = Fredoka
        )
        
        // Mood Slider
        Column {
            Text(
                text = "Mood: ${listOf("Very Low", "Low", "Neutral", "Good", "Great")[mood - 1]}",
                fontSize = 12.sp,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onBackground
            )
            Slider(
                value = mood.toFloat(),
                onValueChange = { mood = it.roundToInt() },
                valueRange = 1f..5f,
                steps = 3,
                colors = SliderDefaults.colors(
                    activeTrackColor = MaterialTheme.colorScheme.primary,
                    inactiveTrackColor = MaterialTheme.colorScheme.outline.copy(0.2f),
                    thumbColor = MaterialTheme.colorScheme.primary,
                    activeTickColor = Color.Transparent,
                    inactiveTickColor = Color.Transparent
                )
            )
        }
        
        // Anxiety Slider
        Column {
            Text(
                text = "Stress / Anxiety: ${listOf("Tense", "Anxious", "Neutral", "Calm", "Very Peaceful")[anxiety - 1]}",
                fontSize = 12.sp,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onBackground
            )
            Slider(
                value = anxiety.toFloat(),
                onValueChange = { anxiety = it.roundToInt() },
                valueRange = 1f..5f,
                steps = 3,
                colors = SliderDefaults.colors(
                    activeTrackColor = MaterialTheme.colorScheme.primary,
                    inactiveTrackColor = MaterialTheme.colorScheme.outline.copy(0.2f),
                    thumbColor = MaterialTheme.colorScheme.primary,
                    activeTickColor = Color.Transparent,
                    inactiveTickColor = Color.Transparent
                )
            )
        }
        
        Button(
            onClick = { onSave(mood, anxiety) },
            modifier = Modifier.fillMaxWidth().height(38.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = MaterialTheme.colorScheme.primary,
                contentColor = Color.Black
            ),
            shape = RoundedCornerShape(10.dp)
        ) {
            Text(
                text = "Confirm & Complete",
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka
            )
        }
    }
}

// =============================================================================
// Home Screen Composable
// =============================================================================
@Composable
fun ToolTrayCard(
    title: String,
    description: String,
    icon: ImageVector,
    onClick: () -> Unit
) {
    Card(
        onClick = onClick,
        modifier = Modifier
            .width(140.dp)
            .height(115.dp),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.4f)
        ),
        border = BorderStroke(
            1.dp,
            Brush.horizontalGradient(
                colors = listOf(
                    MaterialTheme.colorScheme.primary.copy(alpha = 0.2f),
                    MaterialTheme.colorScheme.primary.copy(alpha = 0.05f)
                )
            )
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(12.dp),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            Box(
                modifier = Modifier
                    .size(32.dp)
                    .clip(CircleShape)
                    .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.1f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(18.dp)
                )
            }
            Column {
                Text(
                    text = title,
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onBackground,
                    maxLines = 1,
                    overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis
                )
                Spacer(Modifier.height(2.dp))
                Text(
                    text = description,
                    fontSize = 10.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.8f),
                    maxLines = 2,
                    lineHeight = 12.sp,
                    overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis
                )
            }
        }
    }
}

@Composable
fun HomeScreen(onNavigateToCheckIn: () -> Unit) {
    val context = LocalContext.current
    val userProfile by DataRepository.userProfile.collectAsState()
    val latestResult by DataRepository.latestAnalysisResult.collectAsState()
    val provisional by DataRepository.provisionalAnalysis.collectAsState()
    val isBuilding by DataRepository.isBuildingBaseline.collectAsState()
    val latestObservation by DataRepository.latestObservation.collectAsState()

    var configSeq by remember { mutableStateOf(0) }
    var showBreathingDialog by remember { mutableStateOf(false) }
    var showDetoxDialog by remember { mutableStateOf(false) }
    var showWindDownDialog by remember { mutableStateOf(false) }
    var showHabitsDialog by remember { mutableStateOf(false) }
    
    val activeResult = provisional ?: latestResult
    val score = activeResult?.effectiveScore ?: -1f
    
    val name = (userProfile?.name ?: "").trim()
    val greeting = getGreeting()
    
    val weeklyFeatures by DataRepository.weeklyFeatureHistory.collectAsState()
    val isDnaReady by DataRepository.isDnaBaselineReady.collectAsState()
    val baseline by DataRepository.baseline.collectAsState()
    
    val db = remember { com.example.mhealth.logic.db.MHealthDatabase.getInstance(context.applicationContext) }
    val baselineEntities by produceState<List<com.example.mhealth.logic.db.BaselineEntity>>(emptyList(), db) {
        val userId = DataRepository.userProfile.value?.email ?: "patient@lumen.health"
        value = db.baselineDao().getBaseline(userId)
    }
    
    val statusText = remember(isBuilding, score, weeklyFeatures, baselineEntities, isDnaReady) {
        generateBehavioralSummary(isBuilding, score, weeklyFeatures, baselineEntities, isDnaReady)
    }
    
    val prefs = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }
    val sleepTarget = remember(configSeq) {
        val anchorOverride = prefs.getFloat("habit_circadian_anchor_target", -1f)
        if (anchorOverride > 0f) {
            anchorOverride
        } else {
            val historyStr = prefs.getString("checkin_history", null)
            if (!historyStr.isNullOrBlank()) {
                try {
                    val arr = org.json.JSONArray(historyStr)
                    val sleepTimes = mutableListOf<Float>()
                    for (i in 0 until minOf(arr.length(), 14)) {
                        val obj = arr.optJSONObject(i)
                        val st = obj?.optDouble("sleep_time", -1.0)?.toFloat() ?: -1f
                        if (st > 0f) sleepTimes.add(st)
                    }
                    if (sleepTimes.size >= 3) {
                        sleepTimes.average().toFloat()
                    } else 23f
                } catch (_: Exception) { 23f }
            } else 23f
        }
    }
    val activeStreak = remember(prefs) { getActiveStreak(prefs) }
    
    val todayStr = remember { SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date()) }
    var lastCheckinDate by remember { mutableStateOf(prefs.getString("daily_checkin_date_last", "") ?: "") }
    val alreadyCheckedIn = lastCheckinDate == todayStr
    
    val todayMood = remember(alreadyCheckedIn) { prefs.getInt("daily_checkin_mood", 3) }
    val todayAnxiety = remember(alreadyCheckedIn) { prefs.getInt("daily_checkin_anxiety", 3) }
    val todayNote = remember(alreadyCheckedIn) {
        val list = getCheckinHistoryList(prefs)
        list.firstOrNull { it.optString("date") == todayStr }?.optString("note") ?: ""
    }
    
    val animatedStreak by animateIntAsState(
        targetValue = activeStreak,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioMediumBouncy,
            stiffness = Spring.StiffnessLow
        ),
        label = "StreakAnimation"
    )

    // Reactive Permission Checks
    var isNotificationAccessGranted by remember {
        mutableStateOf(com.example.mhealth.services.MHealthNotificationListenerService.isServiceEnabled(context))
    }
    var isLocationPermissionGranted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED)
    }
    var isBackgroundLocationGranted by remember {
        mutableStateOf(
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_BACKGROUND_LOCATION) == PackageManager.PERMISSION_GRANTED
            } else {
                true
            }
        )
    }
    var isUsageStatsGranted by remember {
        mutableStateOf(hasUsageStatsPermission(context))
    }
    var showLocationDisclosure by remember { mutableStateOf(false) }
    val homeLocationState by DataRepository.homeLocation.collectAsState()
    val isHomeSet = homeLocationState != null
    val isHomeSetAutomatically = remember(prefs, homeLocationState) {
        prefs.getBoolean("home_location_set_automatically", false)
    }

    var isReminderDismissed by remember {
        mutableStateOf(prefs.getBoolean("home_permissions_reminder_dismissed", false))
    }

    var homeCapturing by remember { mutableStateOf(false) }

    val lifecycleOwner = androidx.compose.ui.platform.LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val observer = androidx.lifecycle.LifecycleEventObserver { _, event ->
            if (event == androidx.lifecycle.Lifecycle.Event.ON_RESUME) {
                isNotificationAccessGranted = com.example.mhealth.services.MHealthNotificationListenerService.isServiceEnabled(context)
                isLocationPermissionGranted = ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
                isBackgroundLocationGranted = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_BACKGROUND_LOCATION) == PackageManager.PERMISSION_GRANTED
                } else {
                    true
                }
                isUsageStatsGranted = hasUsageStatsPermission(context)
                isReminderDismissed = prefs.getBoolean("home_permissions_reminder_dismissed", false)
                lastCheckinDate = prefs.getString("daily_checkin_date_last", "") ?: ""
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

    val hasMissingPermissions = !isNotificationAccessGranted || !isLocationPermissionGranted || !isBackgroundLocationGranted || !isHomeSet || !isUsageStatsGranted
    val hasCriticalMissingPermissions = !isLocationPermissionGranted || !isUsageStatsGranted
    val showBanner = hasCriticalMissingPermissions || (!isReminderDismissed && hasMissingPermissions)

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Transparent),
        horizontalAlignment = Alignment.CenterHorizontally,
        contentPadding = PaddingValues(start = 24.dp, top = 24.dp, end = 24.dp, bottom = 96.dp),
        verticalArrangement = Arrangement.spacedBy(24.dp)
    ) {
        item {
            Column(
                modifier = Modifier.fillMaxWidth(),
                horizontalAlignment = Alignment.Start
            ) {
                Text(
                    text = "Lumen.",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = BrandingFont,
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
        if (!isHomeSet || isHomeSetAutomatically) {
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = if (isHomeSet) MaterialTheme.colorScheme.primary.copy(alpha = 0.08f) else AlertWarning.copy(alpha = 0.08f)
                    ),
                    border = BorderStroke(
                        1.dp,
                        if (isHomeSet) MaterialTheme.colorScheme.primary.copy(alpha = 0.25f) else AlertWarning.copy(alpha = 0.25f)
                    )
                ) {
                    Row(
                        modifier = Modifier.padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = if (isHomeSet) Icons.Default.Home else Icons.Default.LocationOn,
                            contentDescription = null,
                            tint = if (isHomeSet) MaterialTheme.colorScheme.primary else AlertWarning,
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(Modifier.width(12.dp))
                        Text(
                            text = if (isHomeSet) {
                                "Home location has been set automatically based on where you spend the most time. If this is incorrect, you can change it anytime in Settings."
                            } else {
                                "Home location is not set yet. We'll automatically set it based on your location history (2 days of data), or you can set it now."
                            },
                            fontSize = 12.sp,
                            lineHeight = 16.sp,
                            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.85f),
                            modifier = Modifier.weight(1f)
                        )
                    }
                }
            }
        }

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
                                    text = "Complete Setup",
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
                            text = "Lumen needs a few system permissions to passively monitor your wellness telemetry.",
                            fontSize = 12.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            lineHeight = 17.sp
                        )

                        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                            if (!isLocationPermissionGranted || !isBackgroundLocationGranted) {
                                PermissionReminderRow(
                                    name = "GPS Location Permission",
                                    buttonText = "Grant",
                                    onClick = {
                                        showLocationDisclosure = true
                                    }
                                )
                            }
                            if (isLocationPermissionGranted && !isHomeSet) {
                                PermissionReminderRow(
                                    name = "Home Location Anchor",
                                    buttonText = if (homeCapturing) "Acquiring..." else "Set Home",
                                    enabled = !homeCapturing,
                                    onClick = {
                                        homeCapturing = true
                                        com.example.mhealth.logic.DataCollector(context).captureHomeLocation { success ->
                                            homeCapturing = false
                                            if (success) {
                                                Toast.makeText(context, "🏠 Home location coordinates saved!", Toast.LENGTH_SHORT).show()
                                            } else {
                                                Toast.makeText(context, "❌ Location Timeout. Check GPS settings.", Toast.LENGTH_SHORT).show()
                                            }
                                        }
                                    }
                                )
                            }
                            if (!isNotificationAccessGranted) {
                                PermissionReminderRow(
                                    name = "Notification Access",
                                    buttonText = "Enable",
                                    onClick = {
                                        context.startActivity(Intent("android.settings.ACTION_NOTIFICATION_LISTENER_SETTINGS"))
                                    }
                                )
                            }
                            if (!isUsageStatsGranted) {
                                PermissionReminderRow(
                                    name = "Usage Stats Access",
                                    buttonText = "Enable",
                                    onClick = {
                                        context.startActivity(Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS))
                                    }
                                )
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
                    }
                }
            }
        }
          item {
            val scope = rememberCoroutineScope()
            val userId = userProfile?.email ?: "patient@lumen.health"
            val targetDate = latestObservation?.date ?: activeResult?.date ?: todayStr
            val feedbackState = latestObservation?.feedbackState ?: activeResult?.userFeedbackState ?: "unresolved"
            val feedbackCategory = latestObservation?.feedbackCategory ?: activeResult?.userFeedbackCategory ?: ""
            val feedbackNotes = latestObservation?.feedbackNotes ?: activeResult?.userFeedbackNotes ?: ""
            val storyText = latestObservation?.narrative?.takeIf { it.isNotBlank() }
                ?: activeResult?.observationStory?.takeIf { it.isNotBlank() }
                ?: if (latestObservation?.isQuietDay == true || activeResult?.effectiveScore == 0f) {
                    "Your rhythm is stable and consistent today — steady as it's been."
                } else if (isBuilding) {
                    "Lumen is mapping your unique behavioral rhythms. Waking up, sleeping patterns, and app usages are being structured to build your personalized circadian baseline."
                } else {
                    "Your daily circadian rhythm is within standard operating parameters."
                }

            // Local state for interactive editing
            var showCorrectionFlow by remember(targetDate, feedbackState) { mutableStateOf(false) }
            var showNoteFlow by remember(targetDate, feedbackState) { mutableStateOf(false) }
            var noteInput by remember(targetDate) { mutableStateOf("") }
            var selectedCategory by remember { mutableStateOf("none") }
            var showContextualSliders by remember { mutableStateOf(false) }
            var pendingFeedbackState by remember { mutableStateOf("") }
            var pendingFeedbackCategory by remember { mutableStateOf("none") }
            var pendingFeedbackNotes by remember { mutableStateOf("") }

            val haptic = LocalHapticFeedback.current

            StaggeredFadeIn(index = 1) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .graphicsLayer {
                            clip = true
                            shape = RoundedCornerShape(24.dp)
                        },
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.4f)
                    ),
                    border = BorderStroke(
                        1.dp,
                        Brush.horizontalGradient(
                            colors = listOf(
                                MaterialTheme.colorScheme.primary.copy(alpha = 0.25f),
                                MaterialTheme.colorScheme.primary.copy(alpha = 0.05f)
                            )
                        )
                    )
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(20.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        // Header Row
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(
                                    imageVector = Icons.Default.Waves,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.primary,
                                    modifier = Modifier.size(20.dp)
                                )
                                Spacer(Modifier.width(8.dp))
                                Text(
                                    text = "Daily Rhythm Story",
                                    fontSize = 16.sp,
                                    fontWeight = FontWeight.Bold,
                                    fontFamily = Fredoka,
                                    color = MaterialTheme.colorScheme.primary
                                )
                            }
                            Text(
                                text = "AI Insights",
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(0.7f),
                                fontFamily = Fredoka,
                                modifier = Modifier
                                    .background(MaterialTheme.colorScheme.surfaceVariant.copy(0.4f), RoundedCornerShape(4.dp))
                                    .padding(horizontal = 6.dp, vertical = 2.dp)
                            )
                        }

                        // Main Narrative Text
                        Text(
                            text = storyText,
                            fontSize = 14.sp,
                            lineHeight = 22.sp,
                            color = MaterialTheme.colorScheme.onBackground.copy(0.9f)
                        )

                        // Divider if interactive
                        if (feedbackState == "unresolved") {
                            HorizontalDivider(
                                modifier = Modifier.fillMaxWidth(),
                                thickness = 0.5.dp,
                                color = MaterialTheme.colorScheme.outline.copy(alpha = 0.2f)
                            )

                            if (showContextualSliders) {
                                ContextualCheckinSection(
                                    onSave = { moodVal, anxietyVal ->
                                        recordDailyCheckin(prefs, moodVal, 3, 3, anxietyVal)
                                        saveCheckinToHistory(prefs, moodVal, 3, 3, anxietyVal, pendingFeedbackNotes)
                                        lastCheckinDate = todayStr
                                        
                                        DataRepository.updateFeedback(
                                            context,
                                            userId,
                                            targetDate,
                                            pendingFeedbackState,
                                            pendingFeedbackCategory,
                                            pendingFeedbackNotes
                                        )
                                        showContextualSliders = false
                                    }
                                )
                            } else if (!showCorrectionFlow && !showNoteFlow) {
                                // Primary Interpretation Chips
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    // That Tracks Button
                                    Button(
                                        onClick = {
                                            haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                                            if (!alreadyCheckedIn) {
                                                pendingFeedbackState = "confirmed"
                                                pendingFeedbackCategory = "none"
                                                pendingFeedbackNotes = ""
                                                showContextualSliders = true
                                            } else {
                                                DataRepository.updateFeedback(
                                                    context,
                                                    userId,
                                                    targetDate,
                                                    "confirmed",
                                                    "none",
                                                    ""
                                                )
                                            }
                                        },
                                        modifier = Modifier.weight(1f).height(38.dp),
                                        colors = ButtonDefaults.buttonColors(
                                            containerColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.1f),
                                            contentColor = MaterialTheme.colorScheme.primary
                                        ),
                                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.3f)),
                                        shape = RoundedCornerShape(10.dp),
                                        contentPadding = PaddingValues(0.dp)
                                    ) {
                                        Text(
                                            "That Tracks",
                                            fontSize = 12.sp,
                                            fontWeight = FontWeight.Bold,
                                            fontFamily = Fredoka
                                        )
                                    }

                                    // Not Quite Button
                                    Button(
                                        onClick = {
                                            haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                                            showCorrectionFlow = true
                                        },
                                        modifier = Modifier.weight(1f).height(38.dp),
                                        colors = ButtonDefaults.buttonColors(
                                            containerColor = MaterialTheme.colorScheme.surface,
                                            contentColor = MaterialTheme.colorScheme.onSurfaceVariant
                                        ),
                                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.2f)),
                                        shape = RoundedCornerShape(10.dp),
                                        contentPadding = PaddingValues(0.dp)
                                    ) {
                                        Text(
                                            "Not Quite",
                                            fontSize = 12.sp,
                                            fontWeight = FontWeight.Bold,
                                            fontFamily = Fredoka
                                        )
                                    }

                                    // Tell Me More Button
                                    Button(
                                        onClick = {
                                            haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                                            showNoteFlow = true
                                        },
                                        modifier = Modifier.weight(1f).height(38.dp),
                                        colors = ButtonDefaults.buttonColors(
                                            containerColor = MaterialTheme.colorScheme.surface,
                                            contentColor = MaterialTheme.colorScheme.onSurfaceVariant
                                        ),
                                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.2f)),
                                        shape = RoundedCornerShape(10.dp),
                                        contentPadding = PaddingValues(0.dp)
                                    ) {
                                        Text(
                                            "Tell me more",
                                            fontSize = 12.sp,
                                            fontWeight = FontWeight.Bold,
                                            fontFamily = Fredoka
                                        )
                                    }
                                }
                            } else if (showCorrectionFlow) {
                                // Correction Options Micro-Flow
                                Column(
                                    modifier = Modifier.fillMaxWidth(),
                                    verticalArrangement = Arrangement.spacedBy(8.dp)
                                ) {
                                    Text(
                                        "Help us adapt. What describes this shift best?",
                                        fontSize = 12.sp,
                                        fontWeight = FontWeight.SemiBold,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )

                                    val categories = listOf(
                                        "schedule_shift" to "Schedule shift / Late night",
                                        "travel" to "Travel / Out of town",
                                        "illness" to "Illness or fatigue",
                                        "stress" to "Temporary high stress",
                                        "none" to "Other benign routine change"
                                    )

                                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                                        categories.forEach { (cat, label) ->
                                            Row(
                                                modifier = Modifier
                                                    .fillMaxWidth()
                                                    .clickable { selectedCategory = cat }
                                                    .padding(vertical = 4.dp),
                                                verticalAlignment = Alignment.CenterVertically
                                            ) {
                                                RadioButton(
                                                    selected = selectedCategory == cat,
                                                    onClick = { selectedCategory = cat }
                                                )
                                                Spacer(Modifier.width(8.dp))
                                                Text(
                                                    text = label,
                                                    fontSize = 13.sp,
                                                    color = MaterialTheme.colorScheme.onBackground
                                                )
                                            }
                                        }
                                    }

                                    Row(
                                        modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                                    ) {
                                        Button(
                                            onClick = { showCorrectionFlow = false },
                                            modifier = Modifier.weight(1f).height(36.dp),
                                            colors = ButtonDefaults.buttonColors(
                                                containerColor = MaterialTheme.colorScheme.surface,
                                                contentColor = MaterialTheme.colorScheme.onSurfaceVariant
                                            ),
                                            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.2f)),
                                            shape = RoundedCornerShape(8.dp)
                                        ) {
                                            Text("Cancel", fontSize = 12.sp, fontFamily = Fredoka)
                                        }

                                        Button(
                                            onClick = {
                                                haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                                                if (!alreadyCheckedIn) {
                                                    pendingFeedbackState = "corrected"
                                                    pendingFeedbackCategory = selectedCategory
                                                    pendingFeedbackNotes = ""
                                                    showContextualSliders = true
                                                } else {
                                                    DataRepository.updateFeedback(
                                                        context,
                                                        userId,
                                                        targetDate,
                                                        "corrected",
                                                        selectedCategory,
                                                        ""
                                                    )
                                                }
                                                showCorrectionFlow = false
                                            },
                                            modifier = Modifier.weight(1f).height(36.dp),
                                            colors = ButtonDefaults.buttonColors(
                                                containerColor = MaterialTheme.colorScheme.primary,
                                                contentColor = Color.Black
                                            ),
                                            shape = RoundedCornerShape(8.dp)
                                        ) {
                                            Text("Update Model", fontSize = 12.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                                        }
                                    }
                                }
                            } else if (showNoteFlow) {
                                // Add Custom Note Flow
                                Column(
                                    modifier = Modifier.fillMaxWidth(),
                                    verticalArrangement = Arrangement.spacedBy(8.dp)
                                ) {
                                    Text(
                                        "Log notes for your secure, offline reflection",
                                        fontSize = 12.sp,
                                        fontWeight = FontWeight.SemiBold,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )

                                    OutlinedTextField(
                                        value = noteInput,
                                        onValueChange = { noteInput = it },
                                        placeholder = { Text("What caused this shift? (e.g. coffee, workout, exam)", fontSize = 13.sp) },
                                        modifier = Modifier.fillMaxWidth(),
                                        maxLines = 3,
                                        shape = RoundedCornerShape(8.dp),
                                        colors = OutlinedTextFieldDefaults.colors(
                                            focusedBorderColor = MaterialTheme.colorScheme.primary,
                                            unfocusedBorderColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)
                                        )
                                    )

                                    Row(
                                        modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                                    ) {
                                        Button(
                                            onClick = { showNoteFlow = false },
                                            modifier = Modifier.weight(1f).height(36.dp),
                                            colors = ButtonDefaults.buttonColors(
                                                containerColor = MaterialTheme.colorScheme.surface,
                                                contentColor = MaterialTheme.colorScheme.onSurfaceVariant
                                            ),
                                            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.2f)),
                                            shape = RoundedCornerShape(8.dp)
                                        ) {
                                            Text("Cancel", fontSize = 12.sp, fontFamily = Fredoka)
                                        }

                                        Button(
                                            onClick = {
                                                haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                                                if (!alreadyCheckedIn) {
                                                    pendingFeedbackState = "noted"
                                                    pendingFeedbackCategory = "none"
                                                    pendingFeedbackNotes = noteInput
                                                    showContextualSliders = true
                                                } else {
                                                    DataRepository.updateFeedback(
                                                        context,
                                                        userId,
                                                        targetDate,
                                                        "noted",
                                                        "none",
                                                        noteInput
                                                    )
                                                }
                                                showNoteFlow = false
                                            },
                                            modifier = Modifier.weight(1f).height(36.dp),
                                            colors = ButtonDefaults.buttonColors(
                                                containerColor = MaterialTheme.colorScheme.primary,
                                                contentColor = Color.Black
                                            ),
                                            shape = RoundedCornerShape(8.dp)
                                        ) {
                                            Text("Save Details", fontSize = 12.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                                        }
                                    }
                                }
                            }
                        } else {
                            // Feedback Submitted State Banner
                            HorizontalDivider(
                                modifier = Modifier.fillMaxWidth(),
                                thickness = 0.5.dp,
                                color = MaterialTheme.colorScheme.outline.copy(alpha = 0.2f)
                            )

                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.08f))
                                    .padding(horizontal = 14.dp, vertical = 10.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Icon(
                                        imageVector = Icons.Default.CheckCircle,
                                        contentDescription = null,
                                        tint = MaterialTheme.colorScheme.primary,
                                        modifier = Modifier.size(16.dp)
                                    )
                                    Spacer(Modifier.width(8.dp))
                                    val labelText = when (feedbackState) {
                                        "confirmed" -> "That Tracks (Feedback saved. Parameters refined.)"
                                        "corrected" -> {
                                            val displayCat = when (feedbackCategory) {
                                                "schedule_shift" -> "Schedule Shift"
                                                "travel" -> "Travel"
                                                "illness" -> "Illness/Fatigue"
                                                "stress" -> "Stress"
                                                else -> "Routine Shift"
                                            }
                                            "Noted: $displayCat (Weights adapted.)"
                                        }
                                        "noted" -> "Details logged to offline reflection history."
                                        else -> "Feedback saved."
                                    }
                                    Text(
                                        text = labelText,
                                        fontSize = 12.sp,
                                        color = MaterialTheme.colorScheme.primary,
                                        fontWeight = FontWeight.Medium
                                    )
                                }

                                Text(
                                    text = "Undo",
                                    fontSize = 12.sp,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    fontWeight = FontWeight.Bold,
                                    fontFamily = Fredoka,
                                    modifier = Modifier
                                        .clickable {
                                            DataRepository.updateFeedback(
                                                context,
                                                userId,
                                                targetDate,
                                                "unresolved",
                                                "",
                                                ""
                                            )
                                        }
                                        .padding(4.dp)
                                )
                            }
                        }
                    }
                }
            }
        }

        // T76: Collapse Breathing, Detox, Wind-Down, and Habits into a bottom horizontal tray
        item {
            StaggeredFadeIn(index = 2) {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Text(
                        text = "Today's Tools",
                        fontSize = 15.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground,
                        modifier = Modifier.padding(horizontal = 4.dp)
                    )
                    
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        // Tool 1: Mindful Breathing
                        ToolTrayCard(
                            title = "Breathing",
                            description = "Box breathing reset",
                            icon = Icons.Default.Spa,
                            onClick = { showBreathingDialog = true }
                        )
                        // Tool 2: Digital Detox
                        ToolTrayCard(
                            title = "Digital Detox",
                            description = "Pause notifications",
                            icon = Icons.Default.HourglassEmpty,
                            onClick = { showDetoxDialog = true }
                        )
                        // Tool 3: Wind-Down
                        ToolTrayCard(
                            title = "Wind-Down",
                            description = "Sleep prep guide",
                            icon = Icons.Default.Bedtime,
                            onClick = { showWindDownDialog = true }
                        )
                        // Tool 4: Habit Quests
                        ToolTrayCard(
                            title = "Habit Quests",
                            description = "Track goals",
                            icon = Icons.Default.TrendingUp,
                            onClick = { showHabitsDialog = true }
                        )
                    }
                }
            }
        }

        if (weeklyFeatures.isNotEmpty() && baseline != null) {
            item {
                StaggeredFadeIn(index = 3) {
                    TelemetrySnapshotCard(features = weeklyFeatures, baseline = baseline)
                }
            }
        }

        item {
            StaggeredFadeIn(index = 4) {
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
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(
                                modifier = Modifier
                                    .size(48.dp)
                                    .clip(CircleShape)
                                    .background(MaterialTheme.colorScheme.primary.copy(0.1f)),
                                contentAlignment = Alignment.Center
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Star,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.primary,
                                    modifier = Modifier.size(24.dp)
                                )
                            }
                            Spacer(Modifier.width(16.dp))
                            Column {
                                Text(
                                    text = if (animatedStreak > 0) "$animatedStreak-Day Check-in Streak" else "Start a New Streak",
                                    fontSize = 16.sp,
                                    fontWeight = FontWeight.Bold,
                                    fontFamily = Fredoka,
                                    color = MaterialTheme.colorScheme.onBackground
                                )
                                Text(
                                    text = if (animatedStreak > 0) "You're building healthy habits." else "Complete a daily check-in to start.",
                                    fontSize = 12.sp,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    }
                }
            }
        }

        // 30 days research contribution prompt card (Task 17)
        item {
            val installDate = remember(prefs) { prefs.getLong("app_install_timestamp", System.currentTimeMillis()) }
            val daysPassed = remember(installDate) { ((System.currentTimeMillis() - installDate) / (1000L * 3600 * 24)).toInt() }
            val isResearchShareCompleted = remember(prefs) { prefs.getBoolean("research_share_completed", false) }

            if (daysPassed >= 30 && !isResearchShareCompleted) {
                var showResearchDialog by remember { mutableStateOf(false) }
                if (showResearchDialog) {
                    ResearchContributionDialog(onDismiss = { showResearchDialog = false })
                }
                StaggeredFadeIn(index = 5) {
                    Card(
                        modifier = Modifier.fillMaxWidth().clickable { showResearchDialog = true },
                        shape = RoundedCornerShape(20.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primary.copy(0.08f)),
                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(0.15f))
                    ) {
                        Row(
                            modifier = Modifier.padding(20.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(40.dp)
                                    .clip(CircleShape)
                                    .background(MaterialTheme.colorScheme.primary.copy(0.1f)),
                                contentAlignment = Alignment.Center
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Science,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.primary,
                                    modifier = Modifier.size(24.dp)
                                )
                            }
                            Spacer(Modifier.width(16.dp))
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    text = "Support Mental Health Research",
                                    fontSize = 15.sp,
                                    fontWeight = FontWeight.Bold,
                                    fontFamily = Fredoka,
                                    color = MaterialTheme.colorScheme.primary
                                )
                                Text(
                                    text = "You have completed 30 days of tracking! Tap to share anonymized data with researchers.",
                                    fontSize = 12.sp,
                                    color = MaterialTheme.colorScheme.onBackground.copy(0.8f),
                                    modifier = Modifier.padding(top = 2.dp)
                                )
                            }
                            Icon(Icons.Default.ChevronRight, null, tint = MaterialTheme.colorScheme.primary)
                        }
                    }
                }
            }
        }

        item {
            StaggeredFadeIn(index = 6) {
                DailyFocusCard()
            }
        }

        item {
            StaggeredFadeIn(index = 7) {
                MilestoneCard(prefs, weeklyFeatures)
            }
        }
    }

    if (showBreathingDialog) {
        FullScreenBreathingScreen(onDismiss = { showBreathingDialog = false })
    }
    if (showDetoxDialog) {
        DigitalDetoxTimerOverlay(durationMinutes = 15, onDismiss = { showDetoxDialog = false; configSeq++ })
    }
    if (showWindDownDialog) {
        WindDownOverlay(sleepTarget = sleepTarget, onDismiss = { showWindDownDialog = false; configSeq++ })
    }
    if (showHabitsDialog) {
        ManageHabitsDialog(onDismiss = { showHabitsDialog = false; configSeq++ })
    }
}

@Composable
fun PermissionReminderRow(
    name: String,
    buttonText: String,
    enabled: Boolean = true,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.5f))
            .padding(horizontal = 12.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = name,
            fontSize = 13.sp,
            fontWeight = FontWeight.Medium,
            color = MaterialTheme.colorScheme.onBackground
        )
        Button(
            onClick = onClick,
            enabled = enabled,
            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
            contentPadding = PaddingValues(horizontal = 16.dp, vertical = 0.dp),
            modifier = Modifier.height(32.dp),
            shape = RoundedCornerShape(6.dp)
        ) {
            Text(
                text = buttonText,
                fontSize = 11.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                color = Color.Black
            )
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
            
            // Ripple 1 layered glows
            drawCircle(
                color = primaryColor,
                radius = baseRadius * rippleRadius1,
                center = center,
                alpha = rippleAlpha1 * 0.2f,
                style = Stroke(width = 8.dp.toPx())
            )
            drawCircle(
                color = primaryColor,
                radius = baseRadius * rippleRadius1,
                center = center,
                alpha = rippleAlpha1 * 0.5f,
                style = Stroke(width = 4.dp.toPx())
            )
            drawCircle(
                color = primaryColor,
                radius = baseRadius * rippleRadius1,
                center = center,
                alpha = rippleAlpha1,
                style = Stroke(width = 1.5f.dp.toPx())
            )
            
            // Ripple 2 layered glows
            drawCircle(
                color = primaryColor,
                radius = baseRadius * rippleRadius2,
                center = center,
                alpha = rippleAlpha2 * 0.2f,
                style = Stroke(width = 8.dp.toPx())
            )
            drawCircle(
                color = primaryColor,
                radius = baseRadius * rippleRadius2,
                center = center,
                alpha = rippleAlpha2 * 0.5f,
                style = Stroke(width = 4.dp.toPx())
            )
            drawCircle(
                color = primaryColor,
                radius = baseRadius * rippleRadius2,
                center = center,
                alpha = rippleAlpha2,
                style = Stroke(width = 1.5f.dp.toPx())
            )
        }
        
        // Central breathing glow halo
        Box(
            modifier = Modifier
                .size(125.dp)
                .scale(scale)
                .clip(CircleShape)
                .background(
                    Brush.radialGradient(
                        colors = listOf(
                            primaryColor.copy(alpha = 0.3f),
                            primaryColor.copy(alpha = 0.0f)
                        )
                    )
                )
        )
        
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
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Bedtime,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(0.7f),
                            modifier = Modifier.size(12.dp)
                        )
                        Text("Sleep", fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurfaceVariant, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                    }
                    Text("%.1f h".format(latest.sleepDurationHours), fontSize = 13.sp, fontWeight = FontWeight.ExtraBold, color = MaterialTheme.colorScheme.primary, modifier = Modifier.padding(top = 4.dp))
                }
                
                // Steps Pill
                Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = snapshotPillModifier) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.DirectionsRun,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(0.7f),
                            modifier = Modifier.size(12.dp)
                        )
                        Text("Steps", fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurfaceVariant, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                    }
                    Text("%.0f".format(latest.dailyStepCount), fontSize = 13.sp, fontWeight = FontWeight.ExtraBold, color = MaterialTheme.colorScheme.primary, modifier = Modifier.padding(top = 4.dp))
                }
                
                // Screen Pill
                Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = snapshotPillModifier) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.PhoneAndroid,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(0.7f),
                            modifier = Modifier.size(12.dp)
                        )
                        Text("Screen", fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurfaceVariant, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                    }
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
                    .padding(start = 24.dp, end = 24.dp, top = 24.dp, bottom = navBarPad + 24.dp),
                contentAlignment = Alignment.Center
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(24.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Spa,
                        contentDescription = null,
                        tint = TealAccent,
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
                        color = GrayTextSecondary,
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
                                        containerColor = if (isSel) TealAccent else Color.Transparent,
                                        contentColor = if (isSel) Color.Black else TealAccent
                                    ),
                                    border = BorderStroke(1.dp, TealAccent),
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
                                color = GrayTextSecondary
                            )
                        }
                        Switch(
                            checked = enableSound,
                            onCheckedChange = { enableSound = it },
                            colors = SwitchDefaults.colors(
                                checkedThumbColor = TealAccent,
                                checkedTrackColor = TealAccent.copy(0.4f)
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
                            colors = ButtonDefaults.buttonColors(containerColor = TealAccent),
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
                        .padding(top = 48.dp, bottom = navBarPad2 + 24.dp, start = 24.dp, end = 24.dp)
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = "Reset Your Rhythm",
                            fontSize = 16.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = TealAccent,
                            fontFamily = Fredoka
                        )
                        Spacer(Modifier.height(4.dp))
                        Text(
                            text = "Remaining: ${totalTimerSeconds / 60}:${String.format("%02d", totalTimerSeconds % 60)}",
                            fontSize = 14.sp,
                            color = GrayTextSecondary
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
                        
                        Canvas(
                            modifier = Modifier
                                .fillMaxSize()
                                .scale(scaleFactor)
                        ) {
                            drawCircle(
                                brush = Brush.radialGradient(
                                    colors = listOf(
                                        TealAccent.copy(alpha = 0.25f),
                                        TealAccent.copy(alpha = 0.0f)
                                    )
                                ),
                                radius = size.minDimension / 2 * rippleScale
                            )

                            drawCircle(
                                color = TealAccent,
                                style = Stroke(width = 4.dp.toPx(), cap = StrokeCap.Round),
                                radius = size.minDimension / 3
                            )

                            drawCircle(
                                color = TealAccent.copy(alpha = 0.15f),
                                radius = size.minDimension / 3 - 2.dp.toPx()
                            )
                        }

                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            val displayPhase = if (activePhase.startsWith("Hold")) "Hold" else activePhase
                            Text(
                                text = displayPhase.uppercase(),
                                fontSize = 24.sp,
                                fontWeight = FontWeight.Bold,
                                color = TealAccent,
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
                            color = GrayTextPrimary,
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
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun InsightsScreen() {
    val weeklyFeatures by DataRepository.weeklyFeatureHistory.collectAsState()
    val baseline by DataRepository.baseline.collectAsState()
    val isDnaReady by DataRepository.isDnaBaselineReady.collectAsState()
    val analysisHistory by DataRepository.analysisHistory.collectAsState()
    val stories = remember(analysisHistory) {
        analysisHistory.filter { it.observationStory.isNotBlank() }
    }
    val context = LocalContext.current
    
    val prefs = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }
    val showInsights = weeklyFeatures.size >= 2

    var activeDetailSector by remember { mutableStateOf<String?>(null) }
    var activeDetailIcon by remember { mutableStateOf<ImageVector>(Icons.Default.Info) }

    androidx.activity.compose.BackHandler(enabled = activeDetailSector != null) {
        activeDetailSector = null
    }

    val db = remember { com.example.mhealth.logic.db.MHealthDatabase.getInstance(context.applicationContext) }
    val baselineEntities by produceState<List<com.example.mhealth.logic.db.BaselineEntity>>(emptyList(), db) {
        val userId = DataRepository.userProfile.value?.email ?: "patient@lumen.health"
        value = db.baselineDao().getBaseline(userId)
    }
    val checkinHistory = remember(prefs) { getCheckinHistoryList(prefs) }

    if (activeDetailSector != null) {
        val sector = activeDetailSector!!
        if (sector == "Overall Rhythm") {
            val base = if (isDnaReady && baseline != null) baseline else {
                val count = weeklyFeatures.size
                if (count > 0) {
                    PersonalityVector(
                        screenTimeHours = weeklyFeatures.map { it.screenTimeHours }.sum() / count,
                        unlockCount = weeklyFeatures.map { it.unlockCount }.sum() / count,
                        appLaunchCount = weeklyFeatures.map { it.appLaunchCount }.sum() / count,
                        notificationsToday = weeklyFeatures.map { it.notificationsToday }.sum() / count,
                        socialAppRatio = weeklyFeatures.map { it.socialAppRatio }.sum() / count,
                        callsPerDay = weeklyFeatures.map { it.callsPerDay }.sum() / count,
                        callDurationMinutes = weeklyFeatures.map { it.callDurationMinutes }.sum() / count,
                        uniqueContacts = weeklyFeatures.map { it.uniqueContacts }.sum() / count,
                        conversationFrequency = weeklyFeatures.map { it.conversationFrequency }.sum() / count,
                        dailyDisplacementKm = weeklyFeatures.map { it.dailyDisplacementKm }.sum() / count,
                        locationEntropy = weeklyFeatures.map { it.locationEntropy }.sum() / count,
                        homeTimeRatio = weeklyFeatures.map { it.homeTimeRatio }.sum() / count,
                        wakeTimeHour = weeklyFeatures.map { it.wakeTimeHour }.sum() / count,
                        sleepTimeHour = weeklyFeatures.map { it.sleepTimeHour }.sum() / count,
                        sleepDurationHours = weeklyFeatures.map { it.sleepDurationHours }.sum() / count,
                        dailyStepCount = weeklyFeatures.map { it.dailyStepCount }.sum() / count,
                        activeMinutes = weeklyFeatures.map { it.activeMinutes }.sum() / count,
                        keystrokeSpeed = weeklyFeatures.map { it.keystrokeSpeed }.sum() / count,
                        backspaceRatio = weeklyFeatures.map { it.backspaceRatio }.sum() / count,
                        scrollVelocity = weeklyFeatures.map { it.scrollVelocity }.sum() / count,
                        daylightExposureMinutes = weeklyFeatures.map { it.daylightExposureMinutes }.sum() / count,
                        chargeRegularity = weeklyFeatures.map { it.chargeRegularity }.sum() / count,
                        chargeDurationHours = weeklyFeatures.map { it.chargeDurationHours }.sum() / count
                    )
                } else null
            }
            RhythmDetailScreen(
                features = weeklyFeatures,
                baseline = base,
                checkinHistory = checkinHistory,
                onBack = { activeDetailSector = null }
            )
        } else {
            SectorDetailScreen(
                sectorName = sector,
                sectorIcon = activeDetailIcon,
                features = weeklyFeatures,
                baselineEntities = baselineEntities,
                checkinHistory = checkinHistory,
                onBack = { activeDetailSector = null }
            )
        }
        return
    }
    
    var showMetricsDrawer by remember { mutableStateOf(false) }
    val userProfile by DataRepository.userProfile.collectAsState()
    val userId = userProfile?.email ?: "patient@lumen.health"

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Transparent),
        contentPadding = PaddingValues(start = 24.dp, top = 24.dp, end = 24.dp, bottom = 96.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            Column(modifier = Modifier.fillMaxWidth()) {
                Text(
                    text = "Lumen.",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = BrandingFont,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.padding(bottom = 6.dp)
                )
                Text(
                    text = "Your Discoveries",
                    fontSize = 24.sp,
                    fontWeight = FontWeight.ExtraBold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Text(
                    text = "Daily routines, deviations, and detailed telemetry analysis",
                    fontSize = 13.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 4.dp)
                )
            }
        }
        
        if (!showInsights) {
            item {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 48.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        CircularProgressIndicator(color = MaterialTheme.colorScheme.primary, strokeWidth = 3.dp)
                        Text(
                            text = "Lumen is still learning your patterns. Check back after a few days.",
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Medium,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            textAlign = TextAlign.Center,
                            modifier = Modifier.padding(horizontal = 24.dp)
                        )
                    }
                }
            }
        } else {
            val latest = weeklyFeatures.lastOrNull()
            val base = if (isDnaReady && baseline != null) {
                baseline
            } else {
                val count = weeklyFeatures.size
                if (count > 0) {
                    PersonalityVector(
                        screenTimeHours = weeklyFeatures.map { it.screenTimeHours }.sum() / count,
                        unlockCount = weeklyFeatures.map { it.unlockCount }.sum() / count,
                        appLaunchCount = weeklyFeatures.map { it.appLaunchCount }.sum() / count,
                        notificationsToday = weeklyFeatures.map { it.notificationsToday }.sum() / count,
                        socialAppRatio = weeklyFeatures.map { it.socialAppRatio }.sum() / count,
                        callsPerDay = weeklyFeatures.map { it.callsPerDay }.sum() / count,
                        callDurationMinutes = weeklyFeatures.map { it.callDurationMinutes }.sum() / count,
                        uniqueContacts = weeklyFeatures.map { it.uniqueContacts }.sum() / count,
                        conversationFrequency = weeklyFeatures.map { it.conversationFrequency }.sum() / count,
                        dailyDisplacementKm = weeklyFeatures.map { it.dailyDisplacementKm }.sum() / count,
                        locationEntropy = weeklyFeatures.map { it.locationEntropy }.sum() / count,
                        homeTimeRatio = weeklyFeatures.map { it.homeTimeRatio }.sum() / count,
                        wakeTimeHour = weeklyFeatures.map { it.wakeTimeHour }.sum() / count,
                        sleepTimeHour = weeklyFeatures.map { it.sleepTimeHour }.sum() / count,
                        sleepDurationHours = weeklyFeatures.map { it.sleepDurationHours }.sum() / count,
                        dailyStepCount = weeklyFeatures.map { it.dailyStepCount }.sum() / count,
                        activeMinutes = weeklyFeatures.map { it.activeMinutes }.sum() / count,
                        keystrokeSpeed = weeklyFeatures.map { it.keystrokeSpeed }.sum() / count,
                        backspaceRatio = weeklyFeatures.map { it.backspaceRatio }.sum() / count,
                        scrollVelocity = weeklyFeatures.map { it.scrollVelocity }.sum() / count,
                        daylightExposureMinutes = weeklyFeatures.map { it.daylightExposureMinutes }.sum() / count,
                        chargeRegularity = weeklyFeatures.map { it.chargeRegularity }.sum() / count,
                        chargeDurationHours = weeklyFeatures.map { it.chargeDurationHours }.sum() / count,
                        upiTransactionsToday = weeklyFeatures.map { it.upiTransactionsToday }.sum() / count,
                        appUninstallsToday = weeklyFeatures.map { it.appUninstallsToday }.sum() / count,
                        appInstallsToday = weeklyFeatures.map { it.appInstallsToday }.sum() / count,
                        calendarEventsToday = weeklyFeatures.map { it.calendarEventsToday }.sum() / count,
                        mediaCountToday = weeklyFeatures.map { it.mediaCountToday }.sum() / count,
                        downloadsToday = weeklyFeatures.map { it.downloadsToday }.sum() / count,
                        musicTimeMinutes = weeklyFeatures.map { it.musicTimeMinutes }.sum() / count
                    )
                } else null
            }
            
            if (latest != null && base != null) {
                val currentScore = {
                    val deviations = listOf(
                        safeDev(latest.sleepDurationHours, base.sleepDurationHours, 1.5f),
                        safeDev(latest.dailyStepCount, base.dailyStepCount, base.dailyStepCount.coerceAtLeast(500f)),
                        safeDev(latest.callsPerDay, base.callsPerDay, base.callsPerDay.coerceAtLeast(3f)),
                        safeDev(latest.screenTimeHours, base.screenTimeHours, base.screenTimeHours.coerceAtLeast(1f)),
                        safeDev(latest.locationEntropy, base.locationEntropy, base.locationEntropy.coerceAtLeast(0.1f)),
                        safeDev(latest.homeTimeRatio, base.homeTimeRatio, base.homeTimeRatio.coerceAtLeast(0.05f))
                    )
                    val avgDev = deviations.average().toFloat().coerceIn(0f, 2f)
                    ((1f - avgDev / 2f) * 100f).coerceIn(0f, 100f)
                }()

                val topDeviations = {
                    val devs = listOf(
                        "Sleep Duration" to (latest.sleepDurationHours - base.sleepDurationHours) / 1.5f,
                        "Daily Steps" to (latest.dailyStepCount - base.dailyStepCount) / base.dailyStepCount.coerceAtLeast(500f),
                        "Phone Calls" to (latest.callsPerDay - base.callsPerDay) / base.callsPerDay.coerceAtLeast(3f),
                        "Screen Time" to (latest.screenTimeHours - base.screenTimeHours) / base.screenTimeHours.coerceAtLeast(1f),
                        "Location Variance" to (latest.locationEntropy - base.locationEntropy) / base.locationEntropy.coerceAtLeast(0.1f),
                        "Time at Home" to (latest.homeTimeRatio - base.homeTimeRatio) / base.homeTimeRatio.coerceAtLeast(0.05f)
                    )
                    devs.filter { Math.abs(it.second) >= 0.35f }
                        .sortedByDescending { Math.abs(it.second) }
                        .take(3)
                }()

                if (!isDnaReady) {
                    item {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(16.dp),
                            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer.copy(0.15f)),
                            border = BorderStroke(1.dp, MaterialTheme.colorScheme.error.copy(0.2f))
                        ) {
                            Row(
                                modifier = Modifier.padding(16.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Info,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.onErrorContainer,
                                    modifier = Modifier.size(24.dp)
                                )
                                Spacer(Modifier.width(16.dp))
                                Text(
                                    text = "Lumen is in early stages of learning your rhythms — insights may be less accurate during this calibration period. Allow a few more days for precision.",
                                    fontSize = 12.sp,
                                    color = MaterialTheme.colorScheme.onErrorContainer,
                                    lineHeight = 17.sp,
                                    fontFamily = Fredoka
                                )
                            }
                        }
                    }
                }

                // 1. Daily Rhythm Score Gauge
                item {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(20.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
                    ) {
                        Column(
                            modifier = Modifier.padding(20.dp),
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(16.dp)
                        ) {
                            Text(
                                text = "Today's Rhythm Consistency",
                                fontSize = 15.sp,
                                fontWeight = FontWeight.Bold,
                                fontFamily = Fredoka,
                                color = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.align(Alignment.Start)
                            )
                            RhythmConsistencyGauge(score = currentScore)
                            Text(
                                text = when {
                                    currentScore >= 85f -> "Your circadian boundaries are extremely stable today. Excellent job staying in harmony with your natural routine!"
                                    currentScore >= 70f -> "Your routine is mostly stable. Only minor deviations in your typical behaviors were detected."
                                    else -> "We detected notable shifts in your routine boundaries today. Check the routine deviations below to align your habits."
                                },
                                fontSize = 12.sp,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                textAlign = TextAlign.Center,
                                lineHeight = 16.sp
                            )
                        }
                    }
                }

                // 2. Summary of Top Deviated Features Card
                item {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(20.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
                    ) {
                        Column(
                            modifier = Modifier.padding(20.dp),
                            verticalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            Text(
                                text = "Routine Deviations",
                                fontSize = 15.sp,
                                fontWeight = FontWeight.Bold,
                                fontFamily = Fredoka,
                                color = MaterialTheme.colorScheme.primary
                            )
                            if (topDeviations.isEmpty()) {
                                Text(
                                    text = "Your behaviors today are beautifully aligned with your typical baseline patterns. No significant routine deviations detected.",
                                    fontSize = 12.sp,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            } else {
                                topDeviations.forEach { (name, devRatio) ->
                                    val pct = Math.abs(devRatio * 100f).roundToInt()
                                    val dir = if (devRatio > 0f) "higher" else "lower"
                                    val color = if (Math.abs(devRatio) > 0.8f) AlertWarning else MaterialTheme.colorScheme.secondary

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
                                                    .size(6.dp)
                                                    .clip(CircleShape)
                                                    .background(color)
                                            )
                                            Text(
                                                text = name,
                                                fontSize = 13.sp,
                                                fontWeight = FontWeight.Medium,
                                                color = MaterialTheme.colorScheme.onBackground
                                            )
                                        }
                                        Text(
                                            text = "$pct% $dir",
                                            fontSize = 12.sp,
                                            fontWeight = FontWeight.Bold,
                                            color = color
                                        )
                                    }
                                }
                            }
                        }
                    }
                }

                // 3. Location Entropy & Home Time Ratio Card
                item {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(20.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
                    ) {
                        Column(
                            modifier = Modifier.padding(20.dp),
                            verticalArrangement = Arrangement.spacedBy(16.dp)
                        ) {
                            Text(
                                text = "Spatial Mobility & Bounds",
                                fontSize = 15.sp,
                                fontWeight = FontWeight.Bold,
                                fontFamily = Fredoka,
                                color = MaterialTheme.colorScheme.primary
                            )
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(16.dp)
                            ) {
                                // Location Entropy
                                Column(
                                    modifier = Modifier.weight(1f),
                                    verticalArrangement = Arrangement.spacedBy(4.dp)
                                ) {
                                    Text(
                                        text = "Location Entropy",
                                        fontSize = 11.sp,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        fontWeight = FontWeight.SemiBold
                                    )
                                    Text(
                                        text = "%.2f".format(latest.locationEntropy),
                                        fontSize = 20.sp,
                                        fontWeight = FontWeight.ExtraBold,
                                        color = MaterialTheme.colorScheme.onBackground
                                    )
                                    Text(
                                        text = "Baseline: %.2f".format(base.locationEntropy),
                                        fontSize = 10.sp,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
                                    )
                                }
                                // Home Time Ratio
                                Column(
                                    modifier = Modifier.weight(1f),
                                    verticalArrangement = Arrangement.spacedBy(4.dp)
                                ) {
                                    Text(
                                        text = "Time at Home",
                                        fontSize = 11.sp,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        fontWeight = FontWeight.SemiBold
                                    )
                                    Text(
                                        text = "%.0f%%".format(latest.homeTimeRatio * 100f),
                                        fontSize = 20.sp,
                                        fontWeight = FontWeight.ExtraBold,
                                        color = MaterialTheme.colorScheme.onBackground
                                    )
                                    Text(
                                        text = "Baseline: %.0f%%".format(base.homeTimeRatio * 100f),
                                        fontSize = 10.sp,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
                                    )
                                }
                            }
                        }
                    }
                }

                // 4. Explore everything in more detail (collapsible bottom drawer trigger button)
                item {
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { showMetricsDrawer = true },
                        shape = RoundedCornerShape(16.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.08f)),
                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.2f))
                    ) {
                        Row(
                            modifier = Modifier.padding(16.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(40.dp)
                                    .clip(CircleShape)
                                    .background(MaterialTheme.colorScheme.primary.copy(0.1f)),
                                contentAlignment = Alignment.Center
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Timeline,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.primary,
                                    modifier = Modifier.size(24.dp)
                                )
                            }
                            Spacer(Modifier.width(16.dp))
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    text = "View Detailed Telemetry",
                                    fontSize = 15.sp,
                                    fontWeight = FontWeight.Bold,
                                    fontFamily = Fredoka,
                                    color = MaterialTheme.colorScheme.primary
                                )
                                Text(
                                    text = "Analyze consistency trends, Sleep, Social, Screen, & Mobility details.",
                                    fontSize = 12.sp,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                            Icon(Icons.Default.ChevronRight, null, tint = MaterialTheme.colorScheme.primary)
                        }
                    }
                }

                // Weekly Digest Report Card (Task 16)
                item {
                    StaggeredFadeIn(index = 0) {
                        var showWeeklyDigest by remember { mutableStateOf(false) }
                        if (showWeeklyDigest) {
                            WeeklyDigestDialog(
                                weeklyFeatures = weeklyFeatures,
                                baseline = base,
                                onDismiss = { showWeeklyDigest = false }
                            )
                        }
                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable { showWeeklyDigest = true },
                            shape = RoundedCornerShape(16.dp),
                            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.5f)),
                            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.12f))
                        ) {
                            Row(
                                modifier = Modifier.padding(16.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Box(
                                    modifier = Modifier
                                        .size(40.dp)
                                        .clip(CircleShape)
                                        .background(MaterialTheme.colorScheme.onSurface.copy(alpha = 0.05f)),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.Assessment,
                                        contentDescription = null,
                                        tint = MaterialTheme.colorScheme.onSurfaceVariant,
                                        modifier = Modifier.size(24.dp)
                                    )
                                }
                                Spacer(Modifier.width(16.dp))
                                Column(modifier = Modifier.weight(1f)) {
                                    Text(
                                        text = "Weekly Digest Report Card",
                                        fontSize = 15.sp,
                                        fontWeight = FontWeight.Bold,
                                        fontFamily = Fredoka,
                                        color = MaterialTheme.colorScheme.onBackground
                                    )
                                    Text(
                                        text = "Your Sunday comprehensive routine summary is ready.",
                                        fontSize = 12.sp,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        modifier = Modifier.padding(top = 2.dp)
                                    )
                                }
                                Icon(Icons.Default.ChevronRight, null, tint = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                        }
                    }
                }

                // 5. Narrative Daily Discoveries Header
                item {
                    Text(
                        text = "Rhythm Stories Feed",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground,
                        modifier = Modifier.padding(top = 8.dp)
                    )
                }

                if (stories.isEmpty()) {
                    item {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(16.dp),
                            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.5f)),
                            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.1f))
                        ) {
                            Box(
                                modifier = Modifier.fillMaxWidth().padding(24.dp),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text = "Your daily rhythm stories will appear here as they are discovered.",
                                    fontSize = 12.sp,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    textAlign = TextAlign.Center
                                )
                            }
                        }
                    }
                } else {
                    items(stories) { result ->
                        val feedbackState = result.userFeedbackState
                        val feedbackCategory = result.userFeedbackCategory
                        val storyText = result.observationStory
                        val dateStr = result.date
                        val formattedDate = remember(dateStr) {
                            try {
                                val d = SimpleDateFormat("yyyy-MM-dd", Locale.US).parse(dateStr)
                                if (d != null) SimpleDateFormat("EEEE, MMMM d").format(d) else dateStr
                            } catch (e: Exception) { dateStr }
                        }

                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(18.dp),
                            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.1f))
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
                                        fontSize = 13.sp,
                                        fontWeight = FontWeight.Bold,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )

                                    if (feedbackState != "unresolved") {
                                        Surface(
                                            shape = RoundedCornerShape(6.dp),
                                            color = when (feedbackState) {
                                                "confirmed" -> MaterialTheme.colorScheme.primary.copy(alpha = 0.12f)
                                                "corrected" -> MaterialTheme.colorScheme.secondary.copy(alpha = 0.12f)
                                                else -> MaterialTheme.colorScheme.primary.copy(alpha = 0.08f)
                                            },
                                            border = BorderStroke(
                                                0.5.dp,
                                                when (feedbackState) {
                                                    "confirmed" -> MaterialTheme.colorScheme.primary
                                                    "corrected" -> MaterialTheme.colorScheme.secondary
                                                    else -> MaterialTheme.colorScheme.primary.copy(alpha = 0.5f)
                                                }
                                            )
                                        ) {
                                            Row(
                                                modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                                                verticalAlignment = Alignment.CenterVertically
                                            ) {
                                                Icon(
                                                    imageVector = Icons.Default.CheckCircle,
                                                    contentDescription = null,
                                                    tint = when (feedbackState) {
                                                        "confirmed" -> MaterialTheme.colorScheme.primary
                                                        "corrected" -> MaterialTheme.colorScheme.secondary
                                                        else -> MaterialTheme.colorScheme.primary
                                                    },
                                                    modifier = Modifier.size(12.dp)
                                                )
                                                Spacer(Modifier.width(4.dp))
                                                Text(
                                                    text = when (feedbackState) {
                                                        "confirmed" -> "Validated by You"
                                                        "corrected" -> {
                                                            val disp = when (feedbackCategory) {
                                                                "schedule_shift" -> "Schedule Shift"
                                                                "travel" -> "Travel"
                                                                "illness" -> "Illness/Fatigue"
                                                                "stress" -> "Stress"
                                                                else -> "Routine Shift"
                                                            }
                                                            "Refined: $disp"
                                                        }
                                                        else -> "Logged Note"
                                                    },
                                                    fontSize = 10.sp,
                                                    fontWeight = FontWeight.Bold,
                                                    color = when (feedbackState) {
                                                        "confirmed" -> MaterialTheme.colorScheme.primary
                                                        "corrected" -> MaterialTheme.colorScheme.secondary
                                                        else -> MaterialTheme.colorScheme.primary
                                                    }
                                                )
                                            }
                                        }
                                    }
                                }

                                Text(
                                    text = storyText,
                                    fontSize = 13.sp,
                                    lineHeight = 18.sp,
                                    color = MaterialTheme.colorScheme.onBackground
                                )

                                if (feedbackState == "unresolved") {
                                    var showLocalCorrection by remember { mutableStateOf(false) }
                                    if (showLocalCorrection) {
                                        Row(
                                            modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            listOf(
                                                "travel" to "Travel",
                                                "schedule_shift" to "Schedule",
                                                "stress" to "Stress",
                                                "illness" to "Fatigue"
                                            ).forEach { (cat, label) ->
                                                Button(
                                                    onClick = {
                                                        DataRepository.updateFeedback(
                                                            context,
                                                            userId,
                                                            dateStr,
                                                            "corrected",
                                                            cat,
                                                            ""
                                                        )
                                                        showLocalCorrection = false
                                                    },
                                                    modifier = Modifier.weight(1f).height(30.dp),
                                                    contentPadding = PaddingValues(0.dp),
                                                    colors = ButtonDefaults.buttonColors(
                                                        containerColor = MaterialTheme.colorScheme.secondary.copy(alpha = 0.8f),
                                                        contentColor = Color.White
                                                    ),
                                                    shape = RoundedCornerShape(6.dp)
                                                ) {
                                                    Text(label, fontSize = 10.sp, fontWeight = FontWeight.Bold)
                                                }
                                            }
                                        }
                                    } else {
                                        Row(
                                            modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                                            horizontalArrangement = Arrangement.spacedBy(12.dp)
                                        ) {
                                            Button(
                                                onClick = {
                                                    DataRepository.updateFeedback(
                                                        context,
                                                        userId,
                                                        dateStr,
                                                        "confirmed",
                                                        "none",
                                                        ""
                                                    )
                                                },
                                                modifier = Modifier.weight(1f).height(32.dp),
                                                colors = ButtonDefaults.buttonColors(
                                                    containerColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.1f),
                                                    contentColor = MaterialTheme.colorScheme.primary
                                                ),
                                                border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.3f)),
                                                shape = RoundedCornerShape(8.dp)
                                            ) {
                                                Icon(Icons.Default.Check, null, modifier = Modifier.size(12.dp))
                                                Spacer(Modifier.width(4.dp))
                                                Text("That Tracks", fontSize = 11.sp, fontWeight = FontWeight.Bold)
                                            }

                                            Button(
                                                onClick = {
                                                    showLocalCorrection = true
                                                },
                                                modifier = Modifier.weight(1f).height(32.dp),
                                                colors = ButtonDefaults.buttonColors(
                                                    containerColor = MaterialTheme.colorScheme.secondary.copy(alpha = 0.1f),
                                                    contentColor = MaterialTheme.colorScheme.secondary
                                                ),
                                                border = BorderStroke(1.dp, MaterialTheme.colorScheme.secondary.copy(alpha = 0.3f)),
                                                shape = RoundedCornerShape(8.dp)
                                            ) {
                                                Icon(Icons.Default.Close, null, modifier = Modifier.size(12.dp))
                                                Spacer(Modifier.width(4.dp))
                                                Text("Not Quite", fontSize = 11.sp, fontWeight = FontWeight.Bold)
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
            }
        }
    }
}

    // Modal Bottom Sheet Drawer for Explore Metrics (T78)
    if (showMetricsDrawer && showInsights) {
        val latest = weeklyFeatures.lastOrNull()
        val base = if (isDnaReady && baseline != null) {
            baseline
        } else {
            val count = weeklyFeatures.size
            if (count > 0) {
                PersonalityVector(
                    screenTimeHours = weeklyFeatures.map { it.screenTimeHours }.sum() / count,
                    unlockCount = weeklyFeatures.map { it.unlockCount }.sum() / count,
                    appLaunchCount = weeklyFeatures.map { it.appLaunchCount }.sum() / count,
                    notificationsToday = weeklyFeatures.map { it.notificationsToday }.sum() / count,
                    socialAppRatio = weeklyFeatures.map { it.socialAppRatio }.sum() / count,
                    callsPerDay = weeklyFeatures.map { it.callsPerDay }.sum() / count,
                    callDurationMinutes = weeklyFeatures.map { it.callDurationMinutes }.sum() / count,
                    uniqueContacts = weeklyFeatures.map { it.uniqueContacts }.sum() / count,
                    conversationFrequency = weeklyFeatures.map { it.conversationFrequency }.sum() / count,
                    dailyDisplacementKm = weeklyFeatures.map { it.dailyDisplacementKm }.sum() / count,
                    locationEntropy = weeklyFeatures.map { it.locationEntropy }.sum() / count,
                    homeTimeRatio = weeklyFeatures.map { it.homeTimeRatio }.sum() / count,
                    wakeTimeHour = weeklyFeatures.map { it.wakeTimeHour }.sum() / count,
                    sleepTimeHour = weeklyFeatures.map { it.sleepTimeHour }.sum() / count,
                    sleepDurationHours = weeklyFeatures.map { it.sleepDurationHours }.sum() / count,
                    dailyStepCount = weeklyFeatures.map { it.dailyStepCount }.sum() / count,
                    activeMinutes = weeklyFeatures.map { it.activeMinutes }.sum() / count,
                    keystrokeSpeed = weeklyFeatures.map { it.keystrokeSpeed }.sum() / count,
                    backspaceRatio = weeklyFeatures.map { it.backspaceRatio }.sum() / count,
                    scrollVelocity = weeklyFeatures.map { it.scrollVelocity }.sum() / count,
                    daylightExposureMinutes = weeklyFeatures.map { it.daylightExposureMinutes }.sum() / count,
                    chargeRegularity = weeklyFeatures.map { it.chargeRegularity }.sum() / count,
                    chargeDurationHours = weeklyFeatures.map { it.chargeDurationHours }.sum() / count
                )
            } else null
        }

        if (latest != null && base != null) {
            val currentScore = {
                val deviations = listOf(
                    safeDev(latest.sleepDurationHours, base.sleepDurationHours, 1.5f),
                    safeDev(latest.dailyStepCount, base.dailyStepCount, base.dailyStepCount.coerceAtLeast(500f)),
                    safeDev(latest.callsPerDay, base.callsPerDay, base.callsPerDay.coerceAtLeast(3f)),
                    safeDev(latest.screenTimeHours, base.screenTimeHours, base.screenTimeHours.coerceAtLeast(1f))
                )
                val avgDev = deviations.average().toFloat().coerceIn(0f, 2f)
                ((1f - avgDev / 2f) * 100f).coerceIn(0f, 100f)
            }()

            ModalBottomSheet(
                onDismissRequest = { showMetricsDrawer = false },
                sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true),
                containerColor = MaterialTheme.colorScheme.background,
                dragHandle = { BottomSheetDefaults.DragHandle() }
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .verticalScroll(rememberScrollState())
                        .navigationBarsPadding()
                        .padding(horizontal = 24.dp, vertical = 8.dp),
                    verticalArrangement = Arrangement.spacedBy(20.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "Explore Metrics & Baselines",
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.align(Alignment.Start)
                    )

                    Text(
                        text = "Circadian Consistency",
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.align(Alignment.Start),
                        fontFamily = Fredoka
                    )
                    RhythmConsistencyGauge(score = currentScore)

                    HorizontalDivider(thickness = 0.5.dp, color = MaterialTheme.colorScheme.outline.copy(alpha = 0.2f))
                    Text(
                        text = "Behavioral Fingerprint Radar",
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.align(Alignment.Start),
                        fontFamily = Fredoka
                    )
                    BehavioralFingerprintRadar(latest = latest, base = base)

                    HorizontalDivider(thickness = 0.5.dp, color = MaterialTheme.colorScheme.outline.copy(alpha = 0.2f))
                    Text(
                        text = "Consistency Trend",
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.align(Alignment.Start),
                        fontFamily = Fredoka
                    )
                    QualitativeTrendChart(features = weeklyFeatures, baseline = base)

                    HorizontalDivider(thickness = 0.5.dp, color = MaterialTheme.colorScheme.outline.copy(alpha = 0.2f))
                    Text(
                        text = "Sensor Telemetry Details",
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.align(Alignment.Start),
                        fontFamily = Fredoka
                    )

                    // Sleep Card
                    val sleepDiff = latest.sleepDurationHours - base.sleepDurationHours
                    val sleepBadge = when {
                        sleepDiff > 1.5f -> "Rest Extended"
                        sleepDiff < -1.5f -> "Rest Shortened"
                        else -> "Balanced Rest"
                    }
                    val sleepBadgeColor = when {
                        Math.abs(sleepDiff) > 1.5f -> AlertWarning
                        else -> MaterialTheme.colorScheme.primary
                    }
                    val sleepDesc = when {
                        sleepDiff > 1.5f -> "Your sleep window is notably longer than typical. Allow yourself the extra rest, but try to ease back into your active daily rhythm with gentle daylight."
                        sleepDiff < -1.5f -> "Your sleep duration is shorter today. Creating a calm, screen-free wind-down routine tonight can help restore your energy."
                        else -> "Your sleep duration and bedtime boundaries are beautifully aligned with your typical rest rhythms."
                    }
                    QualitativeInsightCard(
                        title = "Sleep & Circadian Alignment",
                        icon = Icons.Default.NightsStay,
                        badgeText = sleepBadge,
                        badgeColor = sleepBadgeColor,
                        description = sleepDesc,
                        onClick = {
                            activeDetailSector = "Sleep"
                            activeDetailIcon = Icons.Default.NightsStay
                            showMetricsDrawer = false
                        }
                    )

                    // Movement Card
                    val stepRatio = if (base.dailyStepCount > 0) latest.dailyStepCount / base.dailyStepCount else 1.0f
                    val dispRatio = if (base.dailyDisplacementKm > 0) latest.dailyDisplacementKm / base.dailyDisplacementKm else 1.0f
                    val activeRatio = if (base.dailyStepCount > 0) stepRatio else dispRatio
                    val moveBadge = when {
                        activeRatio < 0.6f -> "Pace Slowed"
                        activeRatio > 1.4f -> "Active Flow"
                        else -> "Steady Flow"
                    }
                    val moveBadgeColor = when {
                        activeRatio < 0.6f -> AlertWarning
                        else -> MaterialTheme.colorScheme.primary
                    }
                    val moveDesc = when {
                        activeRatio < 0.6f -> "Your physical movement is quieter today. Consider taking a short, gentle walk to refresh your body and mind."
                        activeRatio > 1.4f -> "You've been highly active today! Excellent job channeling your physical energy and staying in flow."
                        else -> "Your steps and physical mobility are matching your typical baseline patterns."
                    }
                    QualitativeInsightCard(
                        title = "Physical Mobility",
                        icon = Icons.Default.DirectionsRun,
                        badgeText = moveBadge,
                        badgeColor = moveBadgeColor,
                        description = moveDesc,
                        onClick = {
                            activeDetailSector = "Movement"
                            activeDetailIcon = Icons.Default.DirectionsRun
                            showMetricsDrawer = false
                        }
                    )

                    // Social Card
                    val isProxyActive = com.example.mhealth.services.MHealthNotificationListenerService.isServiceEnabled(context)
                    val callDiff = latest.callsPerDay - base.callsPerDay
                    val socialBadge = when {
                        callDiff < -2.0f -> "Social Pause"
                        else -> "Connected Flow"
                    }
                    val socialBadgeColor = when {
                        callDiff < -2.0f -> AlertWarning
                        else -> MaterialTheme.colorScheme.primary
                    }
                    val socialDesc = when {
                        callDiff < -2.0f -> "We noticed a quiet stretch in your communications. Reaching out to a close friend or family member for a brief chat can offer a comforting boost."
                        else -> "Your social connection rhythm and phone conversations are consistent with your usual baseline."
                    }
                    val trackingNote = if (!isProxyActive)
                        "⚠ Notification access is off — relational data uses dialer app launch signals as a proxy and will be 0 until the notification listener is enabled in Settings."
                    else if (latest.callsPerDay == 0f && base.callsPerDay == 0f)
                        "📡 Tracking active via dialer launch proxy. Relational metrics will populate as you use your phone for calls."
                    else null
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        QualitativeInsightCard(
                            title = "Relational Frequency",
                            icon = Icons.Default.Call,
                            badgeText = socialBadge,
                            badgeColor = socialBadgeColor,
                            description = socialDesc,
                            onClick = {
                                activeDetailSector = "Social"
                                activeDetailIcon = Icons.Default.Call
                                showMetricsDrawer = false
                            }
                        )
                        if (trackingNote != null) {
                            Text(
                                text = trackingNote,
                                fontSize = 10.sp,
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(0.65f),
                                lineHeight = 13.sp,
                                modifier = Modifier.padding(horizontal = 4.dp)
                            )
                        }
                    }

                    // Screen Card
                    val screenDiff = latest.screenTimeHours - base.screenTimeHours
                    val screenBadge = when {
                        screenDiff > 2.0f -> "Screen Elevated"
                        screenDiff < -2.0f -> "Screen Reduced"
                        else -> "Within Norms"
                    }
                    val screenBadgeColor = when {
                        Math.abs(screenDiff) > 2.0f -> AlertWarning
                        else -> MaterialTheme.colorScheme.primary
                    }
                    val screenDesc = when {
                        screenDiff > 2.0f -> "Your screen interaction is higher than usual today. Taking a few intentional digital-free breaks can help reduce eye strain and clear your mind."
                        screenDiff < -2.0f -> "Your screen time is beautifully low today! Enjoying this digital space is a wonderful way to reconnect with your surroundings."
                        else -> "Your daily screen interactions, unlocks, and app session pacing are steady."
                    }
                    QualitativeInsightCard(
                        title = "Digital Interaction Dynamics",
                        icon = Icons.Default.Smartphone,
                        badgeText = screenBadge,
                        badgeColor = screenBadgeColor,
                        description = screenDesc,
                        onClick = {
                            activeDetailSector = "Screen"
                            activeDetailIcon = Icons.Default.Smartphone
                            showMetricsDrawer = false
                        }
                    )

                    // Interaction Tempo Card
                    val isAccessibilityActive = com.example.mhealth.services.MHealthAccessibilityService.isServiceEnabled(context)
                    val tempoRatio = if (base.keystrokeSpeed > 0) latest.keystrokeSpeed / base.keystrokeSpeed else 1.0f
                    val tempoBadge = when {
                        tempoRatio < 0.8f -> "Measured Cadence"
                        tempoRatio > 1.25f -> "Swift Cadence"
                        else -> "Steady Cadence"
                    }
                    val tempoBadgeColor = when {
                        tempoRatio > 1.25f -> AlertWarning
                        else -> MaterialTheme.colorScheme.primary
                    }
                    val tempoDesc = when {
                        tempoRatio < 0.8f -> "Your typing pace is more deliberate and measured. Taking extra time to write can indicate a quiet, thoughtful state of mind."
                        tempoRatio > 1.25f -> "Your key interactions show a swift cadence. A faster typing and scrolling tempo suggests high energy or active processing."
                        else -> "Your writing speed and reading scroll pace are flowing in harmony with your baseline."
                    }
                    val tempoNote = if (!isAccessibilityActive)
                        "⚠ Interaction Dynamics permission is off. Enable it under Settings → System Permissions to track typing speed, backspace ratio, and scroll velocity."
                    else if (latest.keystrokeSpeed == 0f)
                        "📡 Accessibility service active. Typing and scroll metrics will populate as you use your keyboard throughout the day."
                    else null
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        QualitativeInsightCard(
                            title = "Interaction Tempo & Cadence",
                            icon = Icons.Default.Keyboard,
                            badgeText = tempoBadge,
                            badgeColor = tempoBadgeColor,
                            description = tempoDesc,
                            onClick = {
                                activeDetailSector = "Interaction Tempo"
                                activeDetailIcon = Icons.Default.Keyboard
                                showMetricsDrawer = false
                            }
                        )
                        if (tempoNote != null) {
                            Text(
                                text = tempoNote,
                                fontSize = 10.sp,
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(0.65f),
                                lineHeight = 13.sp,
                                modifier = Modifier.padding(horizontal = 4.dp)
                            )
                        }
                    }

                    // Location Mobility Card
                    val entropyDiff = latest.locationEntropy - base.locationEntropy
                    val locationBadge = when {
                        entropyDiff < -0.3f -> "Mobility Confined"
                        entropyDiff > 0.3f -> "Expansive Travel"
                        else -> "Steady Range"
                    }
                    val locationBadgeColor = when {
                        entropyDiff < -0.3f -> AlertWarning
                        else -> MaterialTheme.colorScheme.primary
                    }
                    val locationDesc = when {
                        entropyDiff < -0.3f -> "Your geographic variance is lower today, indicating you stayed in familiar or confined locations. Taking a small excursion can break routine monotony."
                        entropyDiff > 0.3f -> "You explored new or wider areas today! Expanding your spatial range can have positive effects on mental flexibility."
                        else -> "Your geographic exploration and travel patterns are consistent with your baseline."
                    }
                    QualitativeInsightCard(
                        title = "Location Mobility & Variance",
                        icon = Icons.Default.Explore,
                        badgeText = locationBadge,
                        badgeColor = locationBadgeColor,
                        description = locationDesc,
                        onClick = {
                            activeDetailSector = "Location Mobility"
                            activeDetailIcon = Icons.Default.Explore
                            showMetricsDrawer = false
                        }
                    )

                    Spacer(modifier = Modifier.height(24.dp))
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
    description: String,
    onClick: (() -> Unit)? = null
) {
    Card(
        modifier = Modifier.fillMaxWidth().then(
            if (onClick != null) Modifier.clickable { onClick() } else Modifier
        ),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
    ) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.weight(1f).padding(end = 8.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .size(32.dp)
                            .clip(CircleShape)
                            .background(MaterialTheme.colorScheme.primary.copy(0.1f)),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(icon, null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(16.dp))
                    }
                    Spacer(Modifier.width(10.dp))
                    Text(title, fontWeight = FontWeight.Bold, fontSize = 14.sp, fontFamily = Fredoka)
                }
                
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(6.dp))
                        .background(badgeColor.copy(alpha = 0.12f))
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                ) {
                    Text(
                        text = badgeText,
                        fontSize = 10.sp,
                        color = badgeColor,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        maxLines = 1
                    )
                }
            }

            Text(
                text = description,
                fontSize = 12.sp,
                lineHeight = 17.sp,
                color = MaterialTheme.colorScheme.onBackground.copy(0.7f)
            )
        }
    }
}

@Composable
fun QualitativeTrendChart(features: List<PersonalityVector>, baseline: PersonalityVector? = null) {
    val primary = MaterialTheme.colorScheme.primary
    val surfaceVariant = MaterialTheme.colorScheme.onSurfaceVariant
    val reversed = features.take(7).reversed()

    // Compute composite rhythm adherence scores (0-100, 100 = baseline match)
    val scores = remember(reversed, baseline) {
        reversed.map { day ->
            if (baseline == null) 50f
            else {
                val deviations = listOf(
                    safeDev(day.sleepDurationHours, baseline.sleepDurationHours, 1.5f),
                    safeDev(day.dailyStepCount, baseline.dailyStepCount, baseline.dailyStepCount.coerceAtLeast(500f)),
                    safeDev(day.callsPerDay, baseline.callsPerDay, baseline.callsPerDay.coerceAtLeast(3f)),
                    safeDev(day.screenTimeHours, baseline.screenTimeHours, baseline.screenTimeHours.coerceAtLeast(1f))
                )
                val avgDev = deviations.average().toFloat().coerceIn(0f, 2f)
                ((1f - avgDev / 2f) * 100f).coerceIn(0f, 100f)
            }
        }
    }

    // Day-of-week labels
    val dayLabels = remember(reversed) {
        val cal = Calendar.getInstance()
        cal.add(Calendar.DAY_OF_YEAR, -(reversed.size - 1))
        reversed.map {
            val label = SimpleDateFormat("EEE", Locale.getDefault()).format(cal.time)
            cal.add(Calendar.DAY_OF_YEAR, 1)
            label.take(1)
        }
    }

    Column {
        Canvas(Modifier.fillMaxWidth().height(100.dp)) {
            if (scores.size < 2) return@Canvas

            val paddingTop = 8.dp.toPx()
            val paddingBottom = 4.dp.toPx()
            val chartHeight = size.height - paddingTop - paddingBottom
            val spacing = size.width / (scores.size - 1)

            // Baseline reference line at 70%
            val baselineY = size.height - paddingBottom - (0.7f * chartHeight)
            drawLine(
                color = surfaceVariant.copy(0.3f),
                start = Offset(0f, baselineY),
                end = Offset(size.width, baselineY),
                strokeWidth = 1.dp.toPx(),
                pathEffect = PathEffect.dashPathEffect(floatArrayOf(8.dp.toPx(), 6.dp.toPx()))
            )

            // Gradient fill path
            val fillPath = Path()
            val linePath = Path()
            scores.forEachIndexed { idx, score ->
                val x = idx * spacing
                val y = size.height - paddingBottom - ((score / 100f) * chartHeight)
                if (idx == 0) {
                    fillPath.moveTo(x, y)
                    linePath.moveTo(x, y)
                } else {
                    fillPath.lineTo(x, y)
                    linePath.lineTo(x, y)
                }
            }
            // Close fill path
            val fillClosePath = Path().apply {
                addPath(fillPath)
                lineTo((scores.size - 1) * spacing, size.height - paddingBottom)
                lineTo(0f, size.height - paddingBottom)
                close()
            }
            drawPath(
                path = fillClosePath,
                brush = Brush.verticalGradient(
                    listOf(primary.copy(0.25f), primary.copy(0.02f)),
                    startY = paddingTop,
                    endY = size.height - paddingBottom
                )
            )
            drawPath(
                path = linePath,
                brush = Brush.horizontalGradient(listOf(primary.copy(0.6f), primary)),
                style = Stroke(width = 2.5.dp.toPx(), cap = StrokeCap.Round)
            )

            // Dots
            scores.forEachIndexed { idx, score ->
                val x = idx * spacing
                val y = size.height - paddingBottom - ((score / 100f) * chartHeight)
                drawCircle(color = primary, radius = 3.5.dp.toPx(), center = Offset(x, y))
            }
        }

        // Day-of-week labels row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            dayLabels.forEach { label ->
                Text(
                    text = label,
                    fontSize = 10.sp,
                    color = surfaceVariant,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.weight(1f)
                )
            }
        }
    }
}

private fun safeDev(current: Float, base: Float, scale: Float): Float {
    if (scale <= 0f) return 0f
    return (abs(current - base) / scale).coerceAtMost(2.0f)
}

// =============================================================================
// Per-Sector Detail Screen (T7)
@Composable
fun SectorDetailScreen(
    sectorName: String,
    sectorIcon: ImageVector,
    features: List<PersonalityVector>,
    baselineEntities: List<com.example.mhealth.logic.db.BaselineEntity>,
    checkinHistory: List<org.json.JSONObject>,
    onBack: () -> Unit
) {
    val primary = MaterialTheme.colorScheme.primary
    val surfaceVariant = MaterialTheme.colorScheme.onSurfaceVariant
    val context = LocalContext.current
    var timeRange by remember { mutableIntStateOf(7) }

    val featureKeys = remember(sectorName) {
        when (sectorName) {
            "Sleep" -> listOf("sleepDurationHours" to "Sleep Duration (h)", "wakeTimeHour" to "Wake Time", "sleepTimeHour" to "Bedtime")
            "Movement" -> listOf("dailyStepCount" to "Steps", "dailyDisplacementKm" to "Distance (km)", "activeMinutes" to "Active Minutes")
            "Social" -> listOf("callsPerDay" to "Calls/Day", "uniqueContacts" to "Unique Contacts", "conversationFrequency" to "Conversations")
            "Screen" -> listOf("screenTimeHours" to "Screen Time (h)", "unlockCount" to "Unlocks", "appLaunchCount" to "App Launches")
            "Daylight" -> listOf("daylightExposureMinutes" to "Daylight (min)")
            "Charging" -> listOf("chargeRegularity" to "Charge Regularity", "chargeDurationHours" to "Charge Hours")
            "Interaction Tempo" -> listOf("keystrokeSpeed" to "Typing Speed (chars/s)", "backspaceRatio" to "Backspace Ratio", "scrollVelocity" to "Scroll Velocity (px/s)")
            "Location Mobility" -> listOf("locationEntropy" to "Location Entropy", "homeTimeRatio" to "Time at Home", "dailyDisplacementKm" to "Distance (km)")
            else -> emptyList()
        }
    }

    val db = remember { com.example.mhealth.logic.db.MHealthDatabase.getInstance(context.applicationContext) }
    val dailyFeaturesList by produceState<List<com.example.mhealth.logic.db.DailyFeaturesEntity>>(emptyList(), db, timeRange) {
        val userId = DataRepository.userProfile.value?.email ?: "patient@lumen.health"
        value = db.dailyFeaturesDao().getLatestN(userId, timeRange).reversed()
    }
    val baseMap = baselineEntities.associate { it.featureName to it.baselineValue }

    LazyColumn(
        modifier = Modifier.fillMaxSize().background(Color.Transparent),
        contentPadding = PaddingValues(start = 24.dp, top = 24.dp, end = 24.dp, bottom = 96.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            Row(verticalAlignment = Alignment.CenterVertically) {
                IconButton(onClick = onBack) {
                    Icon(Icons.Default.ArrowBack, "Back", tint = MaterialTheme.colorScheme.onBackground)
                }
                Spacer(Modifier.width(8.dp))
                Icon(sectorIcon, null, tint = primary, modifier = Modifier.size(24.dp))
                Spacer(Modifier.width(8.dp))
                Text(sectorName, fontSize = 22.sp, fontWeight = FontWeight.ExtraBold, fontFamily = Fredoka)
            }
        }

        // Time range selector
        item {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                listOf(7 to "7d", 14 to "14d", 30 to "30d").forEach { (days, label) ->
                    val selected = timeRange == days
                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = if (selected) primary else MaterialTheme.colorScheme.surface,
                        border = BorderStroke(1.dp, if (selected) primary else MaterialTheme.colorScheme.outline.copy(0.15f)),
                        modifier = Modifier.clickable { timeRange = days }
                    ) {
                        Text(
                            text = label, fontSize = 12.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka,
                            color = if (selected) Color.Black else surfaceVariant,
                            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
                        )
                    }
                }
            }
        }

        // Charts for each feature in the sector
        featureKeys.forEach { (key, label) ->
            item {
                val baseLine = baseMap[key]

                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(label, fontSize = 14.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                        if (baseLine != null) {
                            Text("Baseline: ${formatValue(baseLine, key)}", fontSize = 10.sp, color = surfaceVariant, modifier = Modifier.padding(top = 2.dp))
                        }
                        
                        Spacer(Modifier.height(12.dp))
                        FeatureLineChart(dailyFeatures = dailyFeaturesList, key = key, baseline = baseLine, color = primary)
                        
                        if (dailyFeaturesList.isNotEmpty()) {
                            Spacer(Modifier.height(12.dp))
                            HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.08f))
                            Spacer(Modifier.height(8.dp))
                            
                            Text("Daily Logs:", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = surfaceVariant)
                            Spacer(Modifier.height(6.dp))
                            
                            androidx.compose.foundation.lazy.LazyRow(
                                horizontalArrangement = Arrangement.spacedBy(10.dp),
                                contentPadding = PaddingValues(horizontal = 2.dp)
                            ) {
                                items(dailyFeaturesList.reversed()) { feat ->
                                    val dayVal = getFeatureValueFromEntity(feat, key)
                                    val parsedDate = try {
                                        val date = SimpleDateFormat("yyyy-MM-dd", Locale.US).parse(feat.date)
                                        if (date != null) SimpleDateFormat("EEE, MMM d", Locale.US).format(date) else feat.date
                                    } catch (e: Exception) { feat.date }
                                    
                                    Column(
                                        horizontalAlignment = Alignment.CenterHorizontally,
                                        modifier = Modifier
                                            .clip(RoundedCornerShape(8.dp))
                                            .background(MaterialTheme.colorScheme.onSurface.copy(0.03f))
                                            .padding(horizontal = 10.dp, vertical = 6.dp)
                                    ) {
                                        Text(parsedDate, fontSize = 9.sp, color = surfaceVariant.copy(0.8f))
                                        Text(formatValue(dayVal, key), fontSize = 12.sp, fontWeight = FontWeight.Bold, color = primary, modifier = Modifier.padding(top = 2.dp))
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


fun getFeatureValue(vec: PersonalityVector, key: String): Float {
    return when (key) {
        "screenTimeHours" -> vec.screenTimeHours
        "unlockCount" -> vec.unlockCount
        "appLaunchCount" -> vec.appLaunchCount
        "callsPerDay" -> vec.callsPerDay
        "uniqueContacts" -> vec.uniqueContacts
        "conversationFrequency" -> vec.conversationFrequency
        "dailyStepCount" -> vec.dailyStepCount
        "dailyDisplacementKm" -> vec.dailyDisplacementKm
        "activeMinutes" -> vec.activeMinutes
        "sleepDurationHours" -> vec.sleepDurationHours
        "wakeTimeHour" -> vec.wakeTimeHour
        "sleepTimeHour" -> vec.sleepTimeHour
        "daylightExposureMinutes" -> vec.daylightExposureMinutes
        "chargeRegularity" -> vec.chargeRegularity
        "chargeDurationHours" -> vec.chargeDurationHours
        "locationEntropy" -> vec.locationEntropy
        "homeTimeRatio" -> vec.homeTimeRatio
        else -> 0f
    }
}

fun formatValue(value: Float, key: String): String {
    return when (key) {
        "screenTimeHours" -> "%.1f h".format(value)
        "sleepDurationHours", "chargeDurationHours" -> "%.1f h".format(value)
        "callDurationMinutes", "activeMinutes", "daylightExposureMinutes" -> "%.0f m".format(value)
        "dailyStepCount" -> "%.0f".format(value)
        "dailyDisplacementKm" -> "%.1f km".format(value)
        "chargeRegularity" -> "%.0f%%".format(value * 100f)
        "locationEntropy" -> "%.2f".format(value)
        "homeTimeRatio" -> "%.0f%%".format(value * 100f)
        "wakeTimeHour", "sleepTimeHour" -> {
            val hour = value.toInt()
            val min = ((value - hour) * 60f).roundToInt().coerceIn(0, 59)
            "%02d:%02d".format(hour % 24, min)
        }
        "unlockCount", "appLaunchCount", "callsPerDay", "uniqueContacts" -> "%.0f".format(value)
        else -> "%.1f".format(value)
    }
}

@OptIn(androidx.compose.ui.text.ExperimentalTextApi::class)
@Composable
fun FeatureLineChart(
    dailyFeatures: List<com.example.mhealth.logic.db.DailyFeaturesEntity>,
    key: String,
    baseline: Float?,
    color: Color
) {
    val textMeasurer = rememberTextMeasurer()
    val surfaceVariant = MaterialTheme.colorScheme.onSurfaceVariant
    val outlineColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.2f)

    if (dailyFeatures.size < 2) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(100.dp),
            contentAlignment = Alignment.Center
        ) {
            Text("Need more data points to plot", fontSize = 11.sp, color = surfaceVariant)
        }
        return
    }

    val values = remember(dailyFeatures, key) {
        dailyFeatures.map { getFeatureValueFromEntity(it, key) }
    }

    val dates = remember(dailyFeatures) {
        dailyFeatures.map { feat ->
            try {
                val d = SimpleDateFormat("yyyy-MM-dd", Locale.US).parse(feat.date)
                if (d != null) SimpleDateFormat("d/M", Locale.US).format(d) else feat.date
            } catch (e: Exception) { feat.date }
        }
    }

    Canvas(
        modifier = Modifier
            .fillMaxWidth()
            .height(140.dp)
            .padding(vertical = 4.dp)
    ) {
        val labelWidth = 50.dp.toPx()
        val bottomOffset = 24.dp.toPx()
        val chartWidth = size.width - labelWidth
        val chartHeight = size.height - bottomOffset

        if (chartWidth <= 0f || chartHeight <= 0f) return@Canvas

        val maxV = (values.maxOrNull() ?: 1f).coerceAtLeast(baseline ?: 1f)
        val minV = (values.minOrNull() ?: 0f).coerceAtMost(baseline ?: 0f)
        val range = (maxV - minV).coerceAtLeast(0.1f)

        // 1. Draw horizontal grid lines & Y labels (min, mid, max)
        val gridLines = listOf(0f, 0.5f, 1f)
        gridLines.forEach { ratio ->
            val y = ratio * chartHeight
            val valueAtY = maxV - ratio * range

            // Grid line
            drawLine(
                color = outlineColor,
                start = Offset(labelWidth, y),
                end = Offset(size.width, y),
                strokeWidth = 1.dp.toPx(),
                pathEffect = PathEffect.dashPathEffect(floatArrayOf(4.dp.toPx(), 4.dp.toPx()))
            )

            // Y label
            val textLayoutResult = textMeasurer.measure(
                text = formatValue(valueAtY, key),
                style = androidx.compose.ui.text.TextStyle(
                    fontSize = 8.sp,
                    color = surfaceVariant.copy(0.6f),
                    fontFamily = FontFamily.SansSerif
                )
            )
            drawText(
                textLayoutResult = textLayoutResult,
                topLeft = Offset(4.dp.toPx(), y - textLayoutResult.size.height / 2f)
            )
        }

        // 2. Draw baseline if available
        if (baseline != null) {
            val bY = chartHeight - ((baseline - minV) / range) * chartHeight
            drawLine(
                color = surfaceVariant.copy(0.4f),
                start = Offset(labelWidth, bY),
                end = Offset(size.width, bY),
                strokeWidth = 1.5.dp.toPx(),
                pathEffect = PathEffect.dashPathEffect(floatArrayOf(6.dp.toPx(), 4.dp.toPx()))
            )
            val baseTextLayout = textMeasurer.measure(
                text = "Baseline: ${formatValue(baseline, key)}",
                style = androidx.compose.ui.text.TextStyle(
                    fontSize = 8.sp,
                    color = surfaceVariant.copy(0.7f),
                    fontWeight = FontWeight.Bold
                )
            )
            drawText(
                textLayoutResult = baseTextLayout,
                topLeft = Offset(size.width - baseTextLayout.size.width - 4.dp.toPx(), bY - baseTextLayout.size.height - 2.dp.toPx())
            )
        }

        // 3. Draw line chart path
        val xSpacing = chartWidth / (values.size - 1).coerceAtLeast(1)
        val points = values.mapIndexed { idx, v ->
            val x = labelWidth + idx * xSpacing
            val y = chartHeight - ((v - minV) / range) * chartHeight
            Offset(x, y)
        }

        val fillPath = Path().apply {
            moveTo(points.first().x, chartHeight)
            points.forEach { lineTo(it.x, it.y) }
            lineTo(points.last().x, chartHeight)
            close()
        }
        drawPath(
            path = fillPath,
            brush = Brush.verticalGradient(
                colors = listOf(color.copy(0.18f), color.copy(0.01f)),
                startY = 0f,
                endY = chartHeight
            )
        )

        val linePath = Path().apply {
            moveTo(points.first().x, points.first().y)
            points.forEach { lineTo(it.x, it.y) }
        }
        drawPath(
            path = linePath,
            color = color,
            style = Stroke(width = 2.dp.toPx(), cap = StrokeCap.Round)
        )

        // Draw dots and X-axis labels
        points.forEachIndexed { idx, pt ->
            drawCircle(
                color = color,
                radius = 3.dp.toPx(),
                center = pt
            )

            // X-axis label (draw every 2nd or 3rd to avoid overlap on long history, or all if small)
            val step = if (values.size > 14) 3 else if (values.size > 7) 2 else 1
            if (idx % step == 0) {
                val dateLabel = dates.getOrNull(idx) ?: ""
                val labelLayout = textMeasurer.measure(
                    text = dateLabel,
                    style = androidx.compose.ui.text.TextStyle(
                        fontSize = 8.sp,
                        color = surfaceVariant.copy(0.7f),
                        fontWeight = FontWeight.Bold
                    )
                )
                drawText(
                    textLayoutResult = labelLayout,
                    topLeft = Offset(pt.x - labelLayout.size.width / 2f, chartHeight + 6.dp.toPx())
                )
            }
        }
    }
}

// =============================================================================
// Journal Entry Card (T6 - shows check-in + note inline)
// =============================================================================
@Composable
fun JournalEntryCard(entry: org.json.JSONObject) {
    val date = entry.optString("date", "")
    val mood = entry.optInt("mood", 3)
    val energy = entry.optInt("energy", 3)
    val sleep = entry.optInt("sleep", 3)
    val anxiety = entry.optInt("anxiety", 3)
    val note = entry.optString("note", "")

    val parsedDate = remember(date) {
        try {
            val d = SimpleDateFormat("yyyy-MM-dd", Locale.US).parse(date)
            if (d != null) SimpleDateFormat("EEEE, MMM d, yyyy", Locale.US).format(d) else date
        } catch (e: Exception) { date }
    }

    val moodStr = when (mood) {
        1 -> "😞 Down"
        2 -> "😕 Uneasy"
        3 -> "😐 Neutral"
        4 -> "🙂 Good"
        else -> "😊 Excellent"
    }

    val energyStr = when (energy) {
        1 -> "Low Energy"
        2 -> "Mod. Low"
        3 -> "Moderate"
        4 -> "High"
        else -> "Very High"
    }

    val sleepStr = when (sleep) {
        1 -> "Poor Sleep"
        2 -> "Mod. Sleep"
        3 -> "Good Sleep"
        4 -> "Great Sleep"
        else -> "Excellent"
    }

    val anxietyStr = when (anxiety) {
        1 -> "Severe Stress"
        2 -> "High Stress"
        3 -> "Mod. Stress"
        4 -> "Mild Stress"
        else -> "Calm / Relaxed"
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.08f))
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            // Header with date and overall mood
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = parsedDate,
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
                
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(8.dp))
                        .background(MaterialTheme.colorScheme.primary.copy(0.1f))
                        .padding(horizontal = 10.dp, vertical = 4.dp)
                ) {
                    Text(
                        text = moodStr,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary,
                        fontFamily = Fredoka
                    )
                }
            }
            
            Spacer(Modifier.height(12.dp))
            
            // Grid of parameters
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                val pillModifier = Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(8.dp))
                    .background(MaterialTheme.colorScheme.onSurface.copy(alpha = 0.04f))
                    .padding(vertical = 8.dp)
                
                Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = pillModifier) {
                    Text("Energy", fontSize = 9.sp, color = MaterialTheme.colorScheme.onSurfaceVariant, fontWeight = FontWeight.Bold)
                    Text(energyStr, fontSize = 10.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onSurface, modifier = Modifier.padding(top = 2.dp))
                }
                Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = pillModifier) {
                    Text("Sleep", fontSize = 9.sp, color = MaterialTheme.colorScheme.onSurfaceVariant, fontWeight = FontWeight.Bold)
                    Text(sleepStr, fontSize = 10.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onSurface, modifier = Modifier.padding(top = 2.dp))
                }
                Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = pillModifier) {
                    Text("Calmness", fontSize = 9.sp, color = MaterialTheme.colorScheme.onSurfaceVariant, fontWeight = FontWeight.Bold)
                    Text(anxietyStr, fontSize = 10.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onSurface, modifier = Modifier.padding(top = 2.dp))
                }
            }
            
            if (note.isNotBlank()) {
                Spacer(Modifier.height(14.dp))
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(12.dp))
                        .background(MaterialTheme.colorScheme.primary.copy(0.03f))
                        .border(1.dp, MaterialTheme.colorScheme.primary.copy(0.08f), RoundedCornerShape(12.dp))
                        .padding(12.dp)
                ) {
                    Row(verticalAlignment = Alignment.Top) {
                        Text(
                            text = "✍️",
                            fontSize = 16.sp,
                            modifier = Modifier.padding(end = 8.dp)
                        )
                        Text(
                            text = note,
                            fontSize = 12.sp,
                            fontStyle = androidx.compose.ui.text.font.FontStyle.Normal,
                            color = MaterialTheme.colorScheme.onBackground.copy(0.8f),
                            lineHeight = 18.sp
                        )
                    }
                }
            }
        }
    }
}

fun getFeatureValueFromEntity(feat: com.example.mhealth.logic.db.DailyFeaturesEntity, key: String): Float {
    return when (key) {
        "screenTimeHours" -> feat.screenTimeHours
        "unlockCount" -> feat.unlockCount
        "appLaunchCount" -> feat.appLaunchCount
        "callsPerDay" -> feat.callsPerDay
        "uniqueContacts" -> feat.uniqueContacts
        "conversationFrequency" -> feat.conversationFrequency
        "dailyStepCount" -> feat.dailyStepCount
        "dailyDisplacementKm" -> feat.dailyDisplacementKm
        "activeMinutes" -> feat.activeMinutes
        "sleepDurationHours" -> feat.sleepDurationHours
        "wakeTimeHour" -> feat.wakeTimeHour
        "sleepTimeHour" -> feat.sleepTimeHour
        "daylightExposureMinutes" -> feat.daylightExposureMinutes
        "chargeRegularity" -> feat.chargeRegularity
        "chargeDurationHours" -> feat.chargeDurationHours
        "keystrokeSpeed" -> feat.keystrokeSpeed
        "backspaceRatio" -> feat.backspaceRatio
        "scrollVelocity" -> feat.scrollVelocity
        "locationEntropy" -> feat.locationEntropy
        "homeTimeRatio" -> feat.homeTimeRatio
        else -> 0f
    }
}

@OptIn(androidx.compose.ui.text.ExperimentalTextApi::class)
@Composable
fun RhythmConsistencyChart(
    features: List<PersonalityVector>,
    baseline: PersonalityVector?,
    timeRange: Int
) {
    val textMeasurer = rememberTextMeasurer()
    val surfaceVariant = MaterialTheme.colorScheme.onSurfaceVariant
    val outlineColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.2f)
    val primary = MaterialTheme.colorScheme.primary

    val reversed = remember(features, timeRange) { features.take(timeRange).reversed() }

    if (reversed.size < 2) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(140.dp),
            contentAlignment = Alignment.Center
        ) {
            Text("Need more data points to plot consistency", fontSize = 12.sp, color = surfaceVariant)
        }
        return
    }

    val scores = remember(reversed, baseline) {
        reversed.map { day ->
            if (baseline == null) 50f
            else {
                val deviations = listOf(
                    safeDev(day.sleepDurationHours, baseline.sleepDurationHours, 1.5f),
                    safeDev(day.dailyStepCount, baseline.dailyStepCount, baseline.dailyStepCount.coerceAtLeast(500f)),
                    safeDev(day.callsPerDay, baseline.callsPerDay, baseline.callsPerDay.coerceAtLeast(3f)),
                    safeDev(day.screenTimeHours, baseline.screenTimeHours, baseline.screenTimeHours.coerceAtLeast(1f))
                )
                val avgDev = deviations.average().toFloat().coerceIn(0f, 2f)
                ((1f - avgDev / 2f) * 100f).coerceIn(0f, 100f)
            }
        }
    }

    // Day labels
    val dayLabels = remember(reversed) {
        val cal = Calendar.getInstance()
        cal.add(Calendar.DAY_OF_YEAR, -(reversed.size - 1))
        reversed.map {
            val label = SimpleDateFormat("EEE", Locale.getDefault()).format(cal.time)
            cal.add(Calendar.DAY_OF_YEAR, 1)
            label
        }
    }

    Canvas(
        modifier = Modifier
            .fillMaxWidth()
            .height(160.dp)
            .padding(vertical = 8.dp)
    ) {
        val labelWidth = 50.dp.toPx()
        val bottomOffset = 24.dp.toPx()
        val chartWidth = size.width - labelWidth
        val chartHeight = size.height - bottomOffset

        if (chartWidth <= 0f || chartHeight <= 0f) return@Canvas

        // 1. Draw horizontal grid lines and Y-axis labels (0%, 25%, 50%, 75%, 100%)
        val gridLines = listOf(0f, 0.25f, 0.5f, 0.75f, 1f)
        gridLines.forEach { ratio ->
            val y = ratio * chartHeight
            val valueAtY = (1f - ratio) * 100f

            // Draw grid line
            drawLine(
                color = outlineColor,
                start = Offset(labelWidth, y),
                end = Offset(size.width, y),
                strokeWidth = 1.dp.toPx(),
                pathEffect = PathEffect.dashPathEffect(floatArrayOf(4.dp.toPx(), 4.dp.toPx()))
            )

            // Draw Y-axis text label
            val textLayoutResult = textMeasurer.measure(
                text = String.format("%.0f%%", valueAtY),
                style = androidx.compose.ui.text.TextStyle(
                    fontSize = 9.sp,
                    color = surfaceVariant.copy(0.7f),
                    fontFamily = FontFamily.SansSerif
                )
            )
            drawText(
                textLayoutResult = textLayoutResult,
                topLeft = Offset(4.dp.toPx(), y - textLayoutResult.size.height / 2f)
            )
        }

        // 2. Draw Baseline/Threshold reference line (70% as standard threshold)
        val thresholdY = 0.3f * chartHeight // 70% from bottom
        drawLine(
            color = Color(0xFF81C784).copy(0.6f),
            start = Offset(labelWidth, thresholdY),
            end = Offset(size.width, thresholdY),
            strokeWidth = 1.5.dp.toPx(),
            pathEffect = PathEffect.dashPathEffect(floatArrayOf(6.dp.toPx(), 4.dp.toPx()))
        )
        val thresholdLabel = textMeasurer.measure(
            text = "Target: 70%",
            style = androidx.compose.ui.text.TextStyle(
                fontSize = 8.sp,
                color = Color(0xFF81C784).copy(0.8f),
                fontWeight = FontWeight.Bold
            )
        )
        drawText(
            textLayoutResult = thresholdLabel,
            topLeft = Offset(size.width - thresholdLabel.size.width - 4.dp.toPx(), thresholdY - thresholdLabel.size.height - 2.dp.toPx())
        )

        // 3. Plot data points and draw lines
        val xSpacing = chartWidth / (scores.size - 1).coerceAtLeast(1)
        val points = scores.mapIndexed { idx, score ->
            val x = labelWidth + idx * xSpacing
            val y = chartHeight - (score / 100f) * chartHeight
            Offset(x, y)
        }

        // Draw gradient area
        val fillPath = Path().apply {
            moveTo(points.first().x, chartHeight)
            points.forEach { lineTo(it.x, it.y) }
            lineTo(points.last().x, chartHeight)
            close()
        }
        drawPath(
            path = fillPath,
            brush = Brush.verticalGradient(
                colors = listOf(primary.copy(0.2f), primary.copy(0.01f)),
                startY = 0f,
                endY = chartHeight
            )
        )

        // Draw line path
        val linePath = Path().apply {
            moveTo(points.first().x, points.first().y)
            points.forEach { lineTo(it.x, it.y) }
        }
        drawPath(
            path = linePath,
            color = primary,
            style = Stroke(width = 2.5.dp.toPx(), cap = StrokeCap.Round)
        )

        // Draw circles & X labels
        points.forEachIndexed { idx, pt ->
            drawCircle(
                color = primary,
                radius = 3.5.dp.toPx(),
                center = pt
            )
            drawCircle(
                color = Color.White,
                radius = 1.5.dp.toPx(),
                center = pt
            )

            // Draw X-axis label
            val step = when {
                timeRange > 20 -> 5
                timeRange > 10 -> 2
                else -> 1
            }
            val dayLabel = dayLabels.getOrNull(idx) ?: ""
            if (dayLabel.isNotEmpty() && (idx % step == 0 || idx == scores.size - 1)) {
                val labelLayout = textMeasurer.measure(
                    text = dayLabel.take(3), // Limit to 3 chars ("Mon")
                    style = androidx.compose.ui.text.TextStyle(
                        fontSize = 9.sp,
                        fontWeight = FontWeight.Bold,
                        color = surfaceVariant.copy(alpha = 0.8f)
                    )
                )
                drawText(
                    textLayoutResult = labelLayout,
                    topLeft = Offset(pt.x - labelLayout.size.width / 2f, chartHeight + 6.dp.toPx())
                )
            }
        }
    }
}

@Composable
fun RhythmConsistencyGauge(score: Float) {
    val primary = MaterialTheme.colorScheme.primary
    val outlineColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.15f)
    
    val coherenceText = when {
        score >= 80f -> "Highly Coherent"
        score >= 60f -> "Stable"
        else -> "Adapting Pacing"
    }
    
    val animatedScore by animateFloatAsState(
        targetValue = score,
        animationSpec = tween(durationMillis = 1000, easing = FastOutSlowInEasing),
        label = "GaugeScore"
    )

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier.fillMaxWidth().padding(vertical = 16.dp)
    ) {
        Box(contentAlignment = Alignment.Center, modifier = Modifier.size(120.dp)) {
            // Background track circle
            Canvas(modifier = Modifier.fillMaxSize()) {
                drawArc(
                    color = outlineColor,
                    startAngle = 135f,
                    sweepAngle = 270f,
                    useCenter = false,
                    alpha = 0.5f,
                    style = Stroke(width = 12.dp.toPx(), cap = StrokeCap.Round)
                )
                drawArc(
                    color = outlineColor,
                    startAngle = 135f,
                    sweepAngle = 270f,
                    useCenter = false,
                    alpha = 1.0f,
                    style = Stroke(width = 8.dp.toPx(), cap = StrokeCap.Round)
                )
            }
            // Active arc with glow layers
            Canvas(modifier = Modifier.fillMaxSize()) {
                val sweep = 270f * (animatedScore / 100f)
                // Base thick glow
                drawArc(
                    color = primary,
                    startAngle = 135f,
                    sweepAngle = sweep,
                    useCenter = false,
                    alpha = 0.15f,
                    style = Stroke(width = 18.dp.toPx(), cap = StrokeCap.Round)
                )
                // Middle bloom glow
                drawArc(
                    color = primary,
                    startAngle = 135f,
                    sweepAngle = sweep,
                    useCenter = false,
                    alpha = 0.4f,
                    style = Stroke(width = 12.dp.toPx(), cap = StrokeCap.Round)
                )
                // Crisp sharp core
                drawArc(
                    color = primary,
                    startAngle = 135f,
                    sweepAngle = sweep,
                    useCenter = false,
                    alpha = 1.0f,
                    style = Stroke(width = 6.dp.toPx(), cap = StrokeCap.Round)
                )
            }
            // Score text
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    text = "${animatedScore.toInt()}%",
                    fontSize = 28.sp,
                    fontWeight = FontWeight.ExtraBold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Text(
                    text = coherenceText,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = primary
                )
            }
        }
    }
}

@OptIn(androidx.compose.ui.text.ExperimentalTextApi::class)
@Composable
fun BehavioralFingerprintRadar(latest: PersonalityVector, base: PersonalityVector) {
    val primary = MaterialTheme.colorScheme.primary
    val outlineColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.15f)
    val textMeasurer = rememberTextMeasurer()
    val onSurface = MaterialTheme.colorScheme.onSurface
    
    val restVal = (1.0f - Math.abs(latest.sleepDurationHours - base.sleepDurationHours) / 3f).coerceIn(0.1f, 1f)
    val mobVal = (if (base.dailyStepCount > 0) latest.dailyStepCount / base.dailyStepCount else 1.0f).coerceIn(0.1f, 2.0f) / 2.0f
    val socVal = (if (base.callsPerDay > 0) latest.callsPerDay / base.callsPerDay else 1.0f).coerceIn(0.1f, 2.0f) / 2.0f
    val digVal = (1.0f - Math.abs(latest.screenTimeHours - base.screenTimeHours) / 4f).coerceIn(0.1f, 1f)
    val dayVal = (if (base.daylightExposureMinutes > 0) latest.daylightExposureMinutes / base.daylightExposureMinutes else 1.0f).coerceIn(0.1f, 2.0f) / 2.0f
    val cadVal = (if (base.keystrokeSpeed > 0) latest.keystrokeSpeed / base.keystrokeSpeed else 1.0f).coerceIn(0.1f, 2.0f) / 2.0f

    val values = listOf(restVal, mobVal, socVal, digVal, dayVal, cadVal)
    val labels = listOf("Rest", "Mobility", "Social", "Digital", "Daylight", "Cadence")

    Canvas(
        modifier = Modifier
            .fillMaxWidth()
            .height(200.dp)
            .padding(24.dp)
    ) {
        val centerX = size.width / 2f
        val centerY = size.height / 2f
        val maxRadius = (Math.min(size.width, size.height) / 2f) - 16.dp.toPx()
        
        if (maxRadius <= 0f) return@Canvas
        
        val gridLevels = listOf(0.33f, 0.66f, 1.0f)
        gridLevels.forEach { level ->
            val path = Path()
            for (i in 0 until 6) {
                val angleRad = Math.toRadians(i * 60.0 - 90.0)
                val r = maxRadius * level
                val x = centerX + r * Math.cos(angleRad).toFloat()
                val y = centerY + r * Math.sin(angleRad).toFloat()
                if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            path.close()
            drawPath(path = path, color = outlineColor, style = Stroke(width = 1.dp.toPx()))
        }

        for (i in 0 until 6) {
            val angleRad = Math.toRadians(i * 60.0 - 90.0)
            val outerX = centerX + maxRadius * Math.cos(angleRad).toFloat()
            val outerY = centerY + maxRadius * Math.sin(angleRad).toFloat()
            
            drawLine(
                color = outlineColor,
                start = Offset(centerX, centerY),
                end = Offset(outerX, outerY),
                strokeWidth = 1.dp.toPx()
            )
            
            val label = labels[i]
            val labelLayout = textMeasurer.measure(
                text = label,
                style = androidx.compose.ui.text.TextStyle(
                    fontFamily = Fredoka,
                    fontSize = 10.sp,
                    color = onSurface.copy(0.6f)
                )
            )
            val textX = outerX + (10.dp.toPx() * Math.cos(angleRad).toFloat()) - (labelLayout.size.width / 2f)
            val textY = outerY + (10.dp.toPx() * Math.sin(angleRad).toFloat()) - (labelLayout.size.height / 2f)
            
            drawText(labelLayout, topLeft = Offset(textX, textY))
        }

        val dataPath = Path()
        for (i in 0 until 6) {
            val angleRad = Math.toRadians(i * 60.0 - 90.0)
            val r = maxRadius * values[i]
            val x = centerX + r * Math.cos(angleRad).toFloat()
            val y = centerY + r * Math.sin(angleRad).toFloat()
            if (i == 0) dataPath.moveTo(x, y) else dataPath.lineTo(x, y)
        }
        dataPath.close()
        
        drawPath(path = dataPath, color = primary.copy(alpha = 0.2f))
        drawPath(path = dataPath, color = primary, style = Stroke(width = 2.dp.toPx()))
    }
}

fun com.example.mhealth.logic.db.DailyFeaturesEntity.toModelVector() = PersonalityVector(
    screenTimeHours = screenTimeHours,
    unlockCount = unlockCount,
    appLaunchCount = appLaunchCount,
    notificationsToday = notificationsToday,
    socialAppRatio = socialAppRatio,
    callsPerDay = callsPerDay,
    callDurationMinutes = callDurationMinutes,
    uniqueContacts = uniqueContacts,
    conversationFrequency = conversationFrequency,
    dailyDisplacementKm = dailyDisplacementKm,
    locationEntropy = locationEntropy,
    homeTimeRatio = homeTimeRatio,
    wakeTimeHour = wakeTimeHour,
    sleepTimeHour = sleepTimeHour,
    sleepDurationHours = sleepDurationHours,
    dailyStepCount = dailyStepCount,
    activeMinutes = activeMinutes,
    keystrokeSpeed = keystrokeSpeed,
    backspaceRatio = backspaceRatio,
    scrollVelocity = scrollVelocity,
    daylightExposureMinutes = daylightExposureMinutes,
    chargeRegularity = chargeRegularity,
    chargeDurationHours = chargeDurationHours
)

fun saveNoteForDate(prefs: SharedPreferences, dateStr: String, note: String) {
    val historyStr = prefs.getString("daily_checkin_history", "[]") ?: "[]"
    try {
        val array = org.json.JSONArray(historyStr)
        val list = mutableListOf<org.json.JSONObject>()
        var foundDate = false
        
        for (i in 0 until array.length()) {
            val obj = array.getJSONObject(i)
            if (obj.getString("date") == dateStr) {
                if (note.isNotBlank()) obj.put("note", note) else obj.remove("note")
                foundDate = true
            }
            list.add(obj)
        }
        
        if (!foundDate && note.isNotBlank()) {
            val newObj = org.json.JSONObject().apply {
                put("date", dateStr)
                put("mood", 3)
                put("energy", 3)
                put("sleep", 3)
                put("anxiety", 3)
                put("note", note)
            }
            list.add(newObj)
        }
        
        val newArray = org.json.JSONArray()
        list.forEach { newArray.put(it) }
        prefs.edit().putString("daily_checkin_history", newArray.toString()).apply()
    } catch (e: Exception) {
        e.printStackTrace()
    }
}

@Composable
fun UnifiedTimelineCard(
    dateStr: String,
    checkinEntry: org.json.JSONObject?,
    analysisResult: com.example.mhealth.logic.db.AnalysisResultEntity?,
    onAnnotate: () -> Unit
) {
    val parsedDate = remember(dateStr) {
        try {
            val d = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.US).parse(dateStr)
            if (d != null) java.text.SimpleDateFormat("EEEE, MMM d, yyyy", java.util.Locale.US).format(d) else dateStr
        } catch (e: Exception) { dateStr }
    }
    
    val hasCheckin = checkinEntry != null
    val note = checkinEntry?.optString("note", "") ?: ""
    val hasNote = note.isNotBlank()
    
    val anomalyDetected = analysisResult?.anomalyDetected ?: false
    
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(
            1.dp, 
            if (anomalyDetected) AlertWarning.copy(0.4f) else MaterialTheme.colorScheme.outline.copy(0.08f)
        )
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = parsedDate,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                    
                    if (hasCheckin) {
                        val mood = checkinEntry!!.optInt("mood", 3)
                        val moodStr = when (mood) {
                            1 -> "😞 Down"
                            2 -> "😕 Uneasy"
                            3 -> "😐 Neutral"
                            4 -> "🙂 Good"
                            else -> "😊 Excellent"
                        }
                        Text(
                            text = "Mood: $moodStr",
                            fontSize = 11.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(top = 2.dp)
                        )
                    } else {
                        Text(
                            text = "No log entry",
                            fontSize = 11.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(0.6f),
                            modifier = Modifier.padding(top = 2.dp)
                        )
                    }
                }
                
                Row(verticalAlignment = Alignment.CenterVertically) {
                    if (anomalyDetected) {
                        Surface(
                            shape = RoundedCornerShape(8.dp),
                            color = AlertWarning.copy(0.12f),
                            border = BorderStroke(1.dp, AlertWarning.copy(0.3f)),
                            modifier = Modifier.padding(end = 8.dp)
                        ) {
                            Text(
                                text = "Rhythm Shift",
                                fontSize = 10.sp,
                                fontWeight = FontWeight.Bold,
                                color = AlertWarning,
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                            )
                        }
                    }
                    
                    IconButton(
                        onClick = onAnnotate,
                        modifier = Modifier.size(32.dp)
                    ) {
                        Icon(
                            imageVector = if (hasNote) Icons.Default.EditNote else Icons.Default.AddComment,
                            contentDescription = "Annotate",
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(20.dp)
                        )
                    }
                }
            }
            
            if (hasNote) {
                Spacer(Modifier.height(10.dp))
                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    color = MaterialTheme.colorScheme.surfaceVariant.copy(0.2f),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.04f))
                ) {
                    Text(
                        text = note,
                        fontSize = 12.sp,
                        color = MaterialTheme.colorScheme.onBackground.copy(0.85f),
                        lineHeight = 17.sp,
                        fontFamily = Fredoka,
                        modifier = Modifier.padding(12.dp)
                    )
                }
            } else if (anomalyDetected) {
                Spacer(Modifier.height(8.dp))
                Text(
                    text = "A change in your daily rhythms was detected today. Tap the comment icon to add a context note (e.g., went on a trip, caught a cold).",
                    fontSize = 11.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(0.7f),
                    fontFamily = Fredoka,
                    lineHeight = 15.sp,
                    style = androidx.compose.ui.text.TextStyle(fontStyle = androidx.compose.ui.text.font.FontStyle.Italic)
                )
            }
        }
    }
}

@Composable
fun RhythmDetailScreen(
    features: List<PersonalityVector>,
    baseline: PersonalityVector?,
    checkinHistory: List<org.json.JSONObject>,
    onBack: () -> Unit
) {
    val primary = MaterialTheme.colorScheme.primary
    val surfaceVariant = MaterialTheme.colorScheme.onSurfaceVariant
    val context = LocalContext.current
    var timeRange by remember { mutableIntStateOf(7) }
    val prefs = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }
    
    var checkinRefreshTrigger by remember { mutableStateOf(0) }
    val reactiveCheckinHistory = remember(prefs, checkinRefreshTrigger) { getCheckinHistoryList(prefs) }

    val db = remember { com.example.mhealth.logic.db.MHealthDatabase.getInstance(context.applicationContext) }
    
    val dailyFeaturesList by produceState<List<com.example.mhealth.logic.db.DailyFeaturesEntity>>(emptyList(), db, timeRange) {
        val userId = DataRepository.userProfile.value?.email ?: "patient@lumen.health"
        value = db.dailyFeaturesDao().getLatestN(userId, timeRange).reversed()
    }
    
    val analysisResultsList by produceState<List<com.example.mhealth.logic.db.AnalysisResultEntity>>(emptyList(), db, timeRange) {
        val userId = DataRepository.userProfile.value?.email ?: "patient@lumen.health"
        value = db.analysisResultDao().getLatestN(userId, timeRange)
    }

    val modelVectors = remember(dailyFeaturesList) {
        dailyFeaturesList.map { it.toModelVector() }
    }

    val scores = remember(modelVectors, baseline) {
        modelVectors.map { day ->
            if (baseline == null) 50f
            else {
                val deviations = listOf(
                    safeDev(day.sleepDurationHours, baseline.sleepDurationHours, 1.5f),
                    safeDev(day.dailyStepCount, baseline.dailyStepCount, baseline.dailyStepCount.coerceAtLeast(500f)),
                    safeDev(day.callsPerDay, baseline.callsPerDay, baseline.callsPerDay.coerceAtLeast(3f)),
                    safeDev(day.screenTimeHours, baseline.screenTimeHours, baseline.screenTimeHours.coerceAtLeast(1f))
                )
                val avgDev = deviations.average().toFloat().coerceIn(0f, 2f)
                ((1f - avgDev / 2f) * 100f).coerceIn(0f, 100f)
            }
        }
    }

    val latestScore = remember(scores) { scores.lastOrNull() ?: 100f }

    val datesList = remember(timeRange) {
        val list = mutableListOf<String>()
        val cal = java.util.Calendar.getInstance()
        for (i in 0 until timeRange) {
            list.add(java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.US).format(cal.time))
            cal.add(java.util.Calendar.DAY_OF_YEAR, -1)
        }
        list
    }

    var editingDate by remember { mutableStateOf<String?>(null) }
    var editingNoteText by remember { mutableStateOf("") }

    if (editingDate != null) {
        val dateStr = editingDate!!
        val formattedDate = remember(dateStr) {
            try {
                val d = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.US).parse(dateStr)
                if (d != null) java.text.SimpleDateFormat("EEEE, MMMM d, yyyy", java.util.Locale.US).format(d) else dateStr
            } catch (e: Exception) { dateStr }
        }
        
        Dialog(onDismissRequest = { editingDate = null }) {
            Surface(
                shape = RoundedCornerShape(24.dp),
                color = MaterialTheme.colorScheme.surface,
                modifier = Modifier.fillMaxWidth().padding(16.dp),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.15f))
            ) {
                Column(modifier = Modifier.padding(24.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    Text(
                        text = "Reflect on $formattedDate",
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                    
                    OutlinedTextField(
                        value = editingNoteText,
                        onValueChange = { editingNoteText = it },
                        modifier = Modifier.fillMaxWidth().height(120.dp),
                        placeholder = { Text("What happened today? (e.g. went on a trip, caught a cold, stressful workday)", fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant.copy(0.5f)) },
                        shape = RoundedCornerShape(12.dp),
                        textStyle = androidx.compose.ui.text.TextStyle(fontFamily = Fredoka, fontSize = 14.sp)
                    )
                    
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.End,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        TextButton(onClick = { editingDate = null }) {
                            Text("Cancel", fontFamily = Fredoka, fontWeight = FontWeight.Bold)
                        }
                        Spacer(Modifier.width(8.dp))
                        Button(
                            onClick = {
                                saveNoteForDate(prefs, dateStr, editingNoteText.trim())
                                checkinRefreshTrigger += 1
                                editingDate = null
                            },
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Text("Save", fontFamily = Fredoka, fontWeight = FontWeight.Bold)
                        }
                    }
                }
            }
        }
    }

    LazyColumn(
        modifier = Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background),
        contentPadding = PaddingValues(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Header
        item {
            Row(verticalAlignment = Alignment.CenterVertically) {
                IconButton(onClick = onBack) {
                    Icon(Icons.Default.ArrowBack, "Back", tint = MaterialTheme.colorScheme.onBackground)
                }
                Spacer(Modifier.width(8.dp))
                Icon(Icons.Default.Timeline, null, tint = primary, modifier = Modifier.size(24.dp))
                Spacer(Modifier.width(8.dp))
                Text("Overall Rhythm", fontSize = 22.sp, fontWeight = FontWeight.ExtraBold, fontFamily = Fredoka)
            }
        }

        // Time range selector
        item {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                listOf(7 to "7d", 14 to "14d", 30 to "30d").forEach { (days, label) ->
                    val selected = timeRange == days
                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = if (selected) primary else MaterialTheme.colorScheme.surface,
                        border = BorderStroke(1.dp, if (selected) primary else MaterialTheme.colorScheme.outline.copy(0.15f)),
                        modifier = Modifier.clickable { timeRange = days }
                    ) {
                        Text(
                            text = label, fontSize = 12.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka,
                            color = if (selected) Color.Black else surfaceVariant,
                            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
                        )
                    }
                }
            }
        }

        // Rhythm Consistency Gauge Card (T39)
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
            ) {
                Column(modifier = Modifier.padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "Rhythm Consistency Score",
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground,
                        modifier = Modifier.align(Alignment.Start)
                    )
                    Text(
                        text = "Your consistency score is compiled by comparing sleep timing, physical mobility, key tempo, and communication boundaries against your personal baseline.",
                        fontSize = 11.sp,
                        color = surfaceVariant,
                        modifier = Modifier.padding(top = 2.dp, bottom = 12.dp).align(Alignment.Start)
                    )
                    
                    RhythmConsistencyGauge(score = latestScore)
                }
            }
        }

        // 6-Axis Behavioral Fingerprint Radar Chart (T40)
        if (modelVectors.isNotEmpty() && baseline != null) {
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(20.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            text = "Behavioral Fingerprint",
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.onBackground
                        )
                        Text(
                            text = "A multidimensional comparison mapping your current activity coordinates directly to your locked baseline signature.",
                            fontSize = 11.sp,
                            color = surfaceVariant,
                            modifier = Modifier.padding(top = 2.dp, bottom = 12.dp)
                        )
                        BehavioralFingerprintRadar(latest = modelVectors.last(), base = baseline)
                    }
                }
            }
        }

        // Rhythm Consistency Trend Chart
        if (modelVectors.size >= 2) {
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(20.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            text = "Routine Consistency Trend",
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.onBackground
                        )
                        Spacer(Modifier.height(12.dp))
                        RhythmConsistencyChart(features = modelVectors, baseline = baseline, timeRange = timeRange)
                    }
                }
            }
        }

        // Unified Journal History & Note Event Annotations (T38)
        item {
            Text(
                text = "Journal History & Notes",
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                modifier = Modifier.padding(top = 8.dp)
            )
        }

        if (datesList.isEmpty()) {
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
                ) {
                    Text(
                        text = "No history found for the selected time range.",
                        fontSize = 12.sp,
                        color = surfaceVariant,
                        modifier = Modifier.padding(16.dp),
                        textAlign = TextAlign.Center
                    )
                }
            }
        } else {
            datesList.forEach { dateStr ->
                val checkinEntry = reactiveCheckinHistory.firstOrNull { it.optString("date") == dateStr }
                val analysisResult = analysisResultsList.firstOrNull { it.date == dateStr }
                
                item {
                    UnifiedTimelineCard(
                        dateStr = dateStr,
                        checkinEntry = checkinEntry,
                        analysisResult = analysisResult,
                        onAnnotate = {
                            editingDate = dateStr
                            editingNoteText = checkinEntry?.optString("note", "") ?: ""
                        }
                    )
                }
            }
        }
    }
}

// =============================================================================
// Mood × Behavior Correlation Card (T9)
// =============================================================================
@Composable
fun MoodBehaviorCorrelationCard(
    checkinHistory: List<org.json.JSONObject>,
    features: List<PersonalityVector>
) {
    if (checkinHistory.size < 5 || features.size < 5) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(20.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
        ) {
            Row(modifier = Modifier.padding(18.dp), verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Default.Assessment,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(24.dp)
                )
                Spacer(Modifier.width(16.dp))
                Column {
                    Text(
                        text = "Behavioral Correlations",
                        fontSize = 15.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                    Text(
                        text = "Log at least 5 daily check-ins to unlock personalized correlations showing how your sleep, steps, and screen time influence your daily mood.",
                        fontSize = 12.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        lineHeight = 17.sp,
                        modifier = Modifier.padding(top = 4.dp)
                    )
                }
            }
        }
        return
    }

    val moodEntries = checkinHistory.takeLast(14)
    val lowMoodDays = moodEntries.filter { it.optInt("mood", 3) <= 2 }
    val highMoodDays = moodEntries.filter { it.optInt("mood", 3) >= 4 }

    val lowDates = lowMoodDays.map { it.optString("date") }.toSet()
    val highDates = highMoodDays.map { it.optString("date") }.toSet()

    // Helper to extract feature vectors corresponding to a set of dates
    fun getFeaturesForDates(dates: Set<String>): List<PersonalityVector> {
        return features.takeLast(14).filterIndexed { idx, _ ->
            val cal = Calendar.getInstance()
            cal.add(Calendar.DAY_OF_YEAR, -(13 - idx))
            val dateStr = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(cal.time)
            dateStr in dates
        }
    }

    val lowMoodFeatures = getFeaturesForDates(lowDates)
    val highMoodFeatures = getFeaturesForDates(highDates)

    val avgSteps = features.takeLast(14).map { it.dailyStepCount }.average().toFloat()
    val avgSleep = features.takeLast(14).map { it.sleepDurationHours }.average().toFloat()
    val avgScreen = features.takeLast(14).map { it.screenTimeHours }.average().toFloat()

    val correlationCards = mutableListOf<@Composable () -> Unit>()

    // 1. Physical vs Mood
    if (lowMoodFeatures.isNotEmpty() || highMoodFeatures.isNotEmpty()) {
        val highSteps = if (highMoodFeatures.isNotEmpty()) highMoodFeatures.map { it.dailyStepCount }.average().toFloat() else avgSteps
        val lowSteps = if (lowMoodFeatures.isNotEmpty()) lowMoodFeatures.map { it.dailyStepCount }.average().toFloat() else avgSteps
        
        if (abs(highSteps - lowSteps) > 500f) {
            correlationCards.add {
                CorrelationItemCard(
                    icon = Icons.Default.DirectionsRun,
                    title = "Steps & Mood Resonance",
                    message = "On higher mood days, you averaged ${highSteps.roundToInt()} steps, compared to ${lowSteps.roundToInt()} steps on lower mood days.",
                    tintColor = TealAccent
                )
            }
        }
    }

    // 2. Sleep vs Mood
    if (lowMoodFeatures.isNotEmpty() || highMoodFeatures.isNotEmpty()) {
        val highSleep = if (highMoodFeatures.isNotEmpty()) highMoodFeatures.map { it.sleepDurationHours }.average().toFloat() else avgSleep
        val lowSleep = if (lowMoodFeatures.isNotEmpty()) lowMoodFeatures.map { it.sleepDurationHours }.average().toFloat() else avgSleep

        if (abs(highSleep - lowSleep) > 0.5f) {
            correlationCards.add {
                CorrelationItemCard(
                    icon = Icons.Default.Bedtime,
                    title = "Sleep Recovery & Mood",
                    message = "You secured ${String.format("%.1f", highSleep)} hours of sleep before high mood days vs ${String.format("%.1f", lowSleep)} hours on lower mood cycles.",
                    tintColor = MaterialTheme.colorScheme.primary
                )
            }
        }
    }

    // 3. Digital vs Mood
    if (lowMoodFeatures.isNotEmpty() || highMoodFeatures.isNotEmpty()) {
        val highScreen = if (highMoodFeatures.isNotEmpty()) highMoodFeatures.map { it.screenTimeHours }.average().toFloat() else avgScreen
        val lowScreen = if (lowMoodFeatures.isNotEmpty()) lowMoodFeatures.map { it.screenTimeHours }.average().toFloat() else avgScreen

        if (abs(highScreen - lowScreen) > 0.5f) {
            correlationCards.add {
                CorrelationItemCard(
                    icon = Icons.Default.PhoneAndroid,
                    title = "Screen Time & Sentiment",
                    message = "On low mood days, you spent ${String.format("%.1f", lowScreen)} hours on screen compared to ${String.format("%.1f", highScreen)} hours on higher mood days.",
                    tintColor = AlertRose
                )
            }
        }
    }

    if (correlationCards.isEmpty()) {
        // Fallback: Default card showing they are highly balanced
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(20.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
        ) {
            Row(modifier = Modifier.padding(18.dp), verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Default.TrendingUp,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(24.dp)
                )
                Spacer(Modifier.width(16.dp))
                Column {
                    Text(
                        text = "Behavioral Equilibrium",
                        fontSize = 15.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                    Text(
                        text = "Your daily sleep, steps, and screen duration remain closely balanced between high and low mood check-ins. Keep logging to track fine-grained trends.",
                        fontSize = 12.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        lineHeight = 17.sp,
                        modifier = Modifier.padding(top = 4.dp)
                    )
                }
            }
        }
    } else {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            correlationCards.forEach { card ->
                card()
            }
        }
    }
}

@Composable
fun CorrelationItemCard(icon: ImageVector, title: String, message: String, tintColor: Color) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
    ) {
        Row(
            modifier = Modifier.padding(18.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(44.dp)
                    .clip(CircleShape)
                    .background(tintColor.copy(0.1f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = tintColor,
                    modifier = Modifier.size(20.dp)
                )
            }
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = title,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Text(
                    text = message,
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    lineHeight = 16.sp,
                    modifier = Modifier.padding(top = 4.dp)
                )
            }
        }
    }
}

// =============================================================================
// Personal Milestones Card (T10)
// =============================================================================
class MilestoneItem(val icon: ImageVector, val tint: Color, val message: String)

@Composable
fun MilestoneCard(prefs: SharedPreferences, features: List<PersonalityVector>) {
    val streak = remember(prefs) { getActiveStreak(prefs) }
    val primaryColor = MaterialTheme.colorScheme.primary
    val milestones = remember(features, streak, primaryColor) {
        val list = mutableListOf<MilestoneItem>()
        if (streak >= 7) {
            list.add(MilestoneItem(Icons.Default.Whatshot, AlertRose, "${streak}-day check-in streak! Incredible consistency."))
        } else if (streak >= 3) {
            list.add(MilestoneItem(Icons.Default.Whatshot, AlertRose, "${streak}-day check-in streak — keep it going!"))
        }

        if (features.size >= 7) {
            val recentSleep = features.take(7).map { it.sleepDurationHours }
            val sleepStd = kotlin.math.sqrt(recentSleep.map { (it - recentSleep.average()).let { d -> d * d } }.average()).toFloat()
            if (sleepStd < 0.5f) {
                list.add(MilestoneItem(Icons.Default.Bedtime, primaryColor, "Consistent Sleeper — your bedtime varied by less than 30 min this week!"))
            }

            val recentSteps = features.take(7).map { it.dailyStepCount }
            val bestDay = recentSteps.maxOrNull() ?: 0f
            if (bestDay > 8000f) {
                list.add(MilestoneItem(Icons.Default.DirectionsRun, TealAccent, "Peak day: ${bestDay.roundToInt()} steps!"))
            }
        }
        list
    }

    if (milestones.isEmpty()) return

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primary.copy(0.06f)),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(0.15f))
    ) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("Milestones", fontSize = 14.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka, color = MaterialTheme.colorScheme.primary)
            milestones.forEach { m ->
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(
                        imageVector = m.icon,
                        contentDescription = null,
                        tint = m.tint,
                        modifier = Modifier.size(16.dp)
                    )
                    Text(m.message, fontSize = 12.sp, color = MaterialTheme.colorScheme.onBackground.copy(0.8f), lineHeight = 17.sp)
                }
            }
        }
    }
}

// Helper: parse check-in history from SharedPreferences
fun getCheckinHistoryList(prefs: SharedPreferences): List<org.json.JSONObject> {
    val historyStr = prefs.getString("daily_checkin_history", "[]") ?: "[]"
    return try {
        val array = org.json.JSONArray(historyStr)
        (0 until array.length()).map { array.getJSONObject(it) }
    } catch (e: Exception) { emptyList() }
}

// =============================================================================
// Check In Screen Composable
// =============================================================================
@Composable
fun CheckInScreen() {
    val context = LocalContext.current
    val prefs = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }
    
    var journalText by remember { mutableStateOf("") }
    val todayStr = remember { SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date()) }
    
    var checkinRefreshTrigger by remember { mutableIntStateOf(0) }
    val history = remember(prefs, checkinRefreshTrigger) { getCheckinHistoryList(prefs) }
    
    val journalEntries = remember(history) {
        history.filter { it.optString("note").isNotBlank() }
    }
    
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentPadding = PaddingValues(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Header
        item {
            Column(modifier = Modifier.fillMaxWidth()) {
                Text(
                    text = "Lumen.",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = BrandingFont,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.padding(bottom = 6.dp)
                )
                Text(
                    text = "Journal Pad",
                    fontSize = 24.sp,
                    fontWeight = FontWeight.ExtraBold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Text(
                    text = "A quiet space to write and reflect on your days",
                    fontSize = 13.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 4.dp)
                )
            }
        }
        
        // Pad Card
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    val todayEntry = remember(journalEntries) { journalEntries.find { it.optString("date") == todayStr } }
                    val alreadySubmitted = todayEntry != null

                    Text(
                        text = if (alreadySubmitted) "Today's Reflection Saved" else "Write something...",
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = if (alreadySubmitted) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onBackground
                    )
                    
                    if (alreadySubmitted) {
                        Text(
                            text = "You've already written a reflection for today. Take time to pause, breathe, and return tomorrow.",
                            fontSize = 12.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    
                    OutlinedTextField(
                        value = if (alreadySubmitted) (todayEntry?.optString("note") ?: "") else journalText,
                        onValueChange = { if (!alreadySubmitted && it.length <= 500) journalText = it },
                        enabled = !alreadySubmitted,
                        placeholder = { Text("What's on your mind today? Write freely...", fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant.copy(0.5f)) },
                        modifier = Modifier.fillMaxWidth().height(150.dp),
                        shape = RoundedCornerShape(12.dp),
                        textStyle = androidx.compose.ui.text.TextStyle(fontFamily = Fredoka, fontSize = 14.sp),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = MaterialTheme.colorScheme.primary,
                            unfocusedBorderColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.3f),
                            disabledBorderColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.15f),
                            disabledTextColor = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    )
                    
                    if (!alreadySubmitted) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "${journalText.length}/500",
                                fontSize = 10.sp,
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(0.5f)
                            )
                            
                            Button(
                                onClick = {
                                    if (journalText.isNotBlank()) {
                                        // Save with a default middle value for sliders (since we routed sliders contextually)
                                        saveCheckinToHistory(prefs, 3, 3, 3, 3, journalText.trim())
                                        journalText = ""
                                        checkinRefreshTrigger += 1
                                        Toast.makeText(context, "Journal entry saved!", Toast.LENGTH_SHORT).show()
                                    }
                                },
                                enabled = journalText.isNotBlank(),
                                modifier = Modifier.height(38.dp),
                                shape = RoundedCornerShape(10.dp),
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = MaterialTheme.colorScheme.primary,
                                    disabledContainerColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.3f),
                                    contentColor = Color.Black,
                                    disabledContentColor = Color.Black.copy(alpha = 0.3f)
                                )
                            ) {
                                Text("Save Entry", fontWeight = FontWeight.Bold, fontFamily = Fredoka, fontSize = 12.sp)
                            }
                        }
                    }
                }
            }
        }
        
        // History Header
        if (journalEntries.isNotEmpty()) {
            item {
                Text(
                    text = "Past Entries",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.padding(top = 8.dp)
                )
            }
            
            items(journalEntries) { entry ->
                val dateStr = entry.optString("date")
                val note = entry.optString("note")
                val formattedDate = remember(dateStr) {
                    try {
                        val d = SimpleDateFormat("yyyy-MM-dd", Locale.US).parse(dateStr)
                        if (d != null) SimpleDateFormat("EEEE, MMMM d, yyyy").format(d) else dateStr
                    } catch (e: Exception) { dateStr }
                }
                
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.5f)),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
                ) {
                    Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(
                            text = formattedDate,
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.primary,
                            fontFamily = Fredoka
                        )
                        Text(
                            text = note,
                            fontSize = 13.sp,
                            lineHeight = 18.sp,
                            color = MaterialTheme.colorScheme.onBackground
                        )
                    }
                }
            }
        }
    }
}

// =============================================================================
// History Screen Composable
// =============================================================================
@Composable
fun HistoryScreen() {
    val context = LocalContext.current
    val prefs = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }
    val history = remember(prefs) { getCheckinHistoryList(prefs).reversed() }

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Transparent),
        contentPadding = PaddingValues(start = 24.dp, top = 24.dp, end = 24.dp, bottom = 96.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Header
        item {
            Column(modifier = Modifier.fillMaxWidth()) {
                Text(
                    text = "Lumen.",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = BrandingFont,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.padding(bottom = 6.dp)
                )
                Text(
                    text = "Rhythm History",
                    fontSize = 24.sp,
                    fontWeight = FontWeight.ExtraBold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Text(
                    text = "Your journey of self-reflection and balance over time",
                    fontSize = 13.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 4.dp)
                )
            }
        }

        if (history.isEmpty()) {
            item {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 40.dp),
                    shape = RoundedCornerShape(24.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.5f)),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
                ) {
                    Column(
                        modifier = Modifier.padding(32.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .size(64.dp)
                                .clip(CircleShape)
                                .background(MaterialTheme.colorScheme.primary.copy(0.1f)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = Icons.Default.History,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.size(32.dp)
                            )
                        }

                        Text(
                            text = "No history recorded yet",
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.onBackground
                        )

                        Text(
                            text = "Complete a daily check-in or write a journal entry to start building your timeline of self-reflections.",
                            fontSize = 12.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            textAlign = TextAlign.Center,
                            lineHeight = 18.sp
                        )
                    }
                }
            }
        } else {
            items(history) { entry ->
                val dateStr = entry.optString("date")
                val mood = entry.optInt("mood", 3)
                val anxiety = entry.optInt("anxiety", 3)
                val note = entry.optString("note", "")

                val formattedDate = remember(dateStr) {
                    try {
                        val d = SimpleDateFormat("yyyy-MM-dd", Locale.US).parse(dateStr)
                        if (d != null) SimpleDateFormat("EEEE, MMMM d, yyyy").format(d) else dateStr
                    } catch (e: Exception) { dateStr }
                }

                val moodLabel = listOf("Very Low", "Low", "Neutral", "Good", "Great").getOrElse(mood - 1) { "Neutral" }
                val moodEmoji = listOf("😞", "🙁", "😐", "🙂", "😄").getOrElse(mood - 1) { "😐" }

                val anxietyLabel = listOf("Tense", "Anxious", "Neutral", "Calm", "Peaceful").getOrElse(anxiety - 1) { "Neutral" }
                val anxietyEmoji = listOf("😫", "😟", "😐", "😌", "🧘").getOrElse(anxiety - 1) { "😐" }

                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(20.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.5f)),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
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
                            Text(
                                text = formattedDate,
                                fontSize = 13.sp,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.primary,
                                fontFamily = Fredoka
                            )
                        }

                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Surface(
                                shape = RoundedCornerShape(8.dp),
                                color = MaterialTheme.colorScheme.primary.copy(alpha = 0.08f),
                                border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.15f))
                            ) {
                                Row(
                                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                                ) {
                                    Text(moodEmoji, fontSize = 12.sp)
                                    Text("Mood: $moodLabel", fontSize = 11.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka, color = MaterialTheme.colorScheme.primary)
                                }
                            }

                            Surface(
                                shape = RoundedCornerShape(8.dp),
                                color = MaterialTheme.colorScheme.secondary.copy(alpha = 0.08f),
                                border = BorderStroke(1.dp, MaterialTheme.colorScheme.secondary.copy(alpha = 0.15f))
                            ) {
                                Row(
                                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                                ) {
                                    Text(anxietyEmoji, fontSize = 12.sp)
                                    Text("Stress: $anxietyLabel", fontSize = 11.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka, color = MaterialTheme.colorScheme.secondary)
                                }
                            }
                        }

                        if (note.isNotBlank()) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(top = 4.dp),
                                horizontalArrangement = Arrangement.spacedBy(10.dp)
                            ) {
                                Box(
                                    modifier = Modifier
                                        .width(3.dp)
                                        .height(IntrinsicSize.Max)
                                        .background(MaterialTheme.colorScheme.primary)
                                )
                                Text(
                                    text = note,
                                    fontSize = 13.sp,
                                    lineHeight = 18.sp,
                                    color = MaterialTheme.colorScheme.onBackground,
                                    modifier = Modifier.weight(1f)
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// Settings Screen Composable
// =============================================================================
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
    
    var masterTracking by remember { mutableStateOf(prefs.getBoolean("master_tracking_enabled", true)) }
    var locationTracking by remember { mutableStateOf(prefs.getBoolean("location_tracking_enabled", true)) }
    var communicationLogs by remember { mutableStateOf(prefs.getBoolean("communication_logs_enabled", true)) }
    
    var dailyReminders by remember { mutableStateOf(prefs.getBoolean("daily_reminders_enabled", true)) }
    var monthlyReminders by remember { mutableStateOf(prefs.getBoolean("monthly_reminders_enabled", true)) }
    var autoBackupEnabled by remember { mutableStateOf(prefs.getBoolean("auto_backup_enabled", true)) }
    var weeklySummaryEnabled by remember { mutableStateOf(prefs.getBoolean("weekly_summary_notifications_enabled", true)) }

    var showPrivacyDialog by remember { mutableStateOf(false) }

    // Reactive Permission States
    var isNotificationAccessGranted by remember {
        mutableStateOf(com.example.mhealth.services.MHealthNotificationListenerService.isServiceEnabled(context))
    }
    var isLocationPermissionGranted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED)
    }
    var isBackgroundLocationGranted by remember {
        mutableStateOf(
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_BACKGROUND_LOCATION) == PackageManager.PERMISSION_GRANTED
            } else {
                true
            }
        )
    }
    var isAccessibilityGranted by remember {
        mutableStateOf(com.example.mhealth.services.MHealthAccessibilityService.isServiceEnabled(context))
    }
    var showLocationDisclosure by remember { mutableStateOf(false) }
    var showAccessibilityDisclosure by remember { mutableStateOf(false) }
    val isReminderDismissed = prefs.getBoolean("home_permissions_reminder_dismissed", false)

    val lifecycleOwner = androidx.compose.ui.platform.LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val observer = androidx.lifecycle.LifecycleEventObserver { _, event ->
            if (event == androidx.lifecycle.Lifecycle.Event.ON_RESUME) {
                isNotificationAccessGranted = com.example.mhealth.services.MHealthNotificationListenerService.isServiceEnabled(context)
                isLocationPermissionGranted = ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
                isBackgroundLocationGranted = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_BACKGROUND_LOCATION) == PackageManager.PERMISSION_GRANTED
                } else {
                    true
                }
                isAccessibilityGranted = com.example.mhealth.services.MHealthAccessibilityService.isServiceEnabled(context)
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
        contentPadding = PaddingValues(24.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp)
    ) {
        item {
            Column(modifier = Modifier.fillMaxWidth()) {
                Text(
                    text = "Lumen.",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = BrandingFont,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.padding(bottom = 6.dp)
                )
                Text(
                    text = "Profile & Settings",
                    fontSize = 24.sp,
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

        item {
            InfoCard("System Permissions", headerColor = MaterialTheme.colorScheme.primary) {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    PermissionSettingRow(
                        title = "GPS Location Permission",
                        subtitle = "Required to track daily movement",
                        isGranted = isLocationPermissionGranted,
                        isReminderDismissed = isReminderDismissed,
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
                        isReminderDismissed = isReminderDismissed,
                        onClick = {
                            context.startActivity(Intent("android.settings.ACTION_NOTIFICATION_LISTENER_SETTINGS"))
                        }
                    )
                    HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.1f))
                    PermissionSettingRow(
                        title = "Digital Psychomotor Dynamics",
                        subtitle = "Required for typing speed & scroll velocity",
                        isGranted = isAccessibilityGranted,
                        isReminderDismissed = isReminderDismissed,
                        onClick = {
                            showAccessibilityDisclosure = true
                        }
                    )

                }
            }
        }

        item {
            InfoCard("Data Collection Toggles", headerColor = MaterialTheme.colorScheme.primary) {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    ToggleRow(
                        title = "Master Tracking",
                        subtitle = "Enable all sensor logs passively in the background",
                        checked = masterTracking,
                        color = MaterialTheme.colorScheme.primary,
                        onToggle = {
                            masterTracking = it
                            prefs.edit().putBoolean("master_tracking_enabled", it).apply()
                            Toast.makeText(context, if (it) "Background tracking resumed" else "Background tracking paused", Toast.LENGTH_SHORT).show()
                        }
                    )
                    HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.1f))
                    ToggleRow(
                        title = "Location GPS Tracking",
                        subtitle = "Saves spatial displacement and location variety entropy",
                        checked = locationTracking,
                        color = MaterialTheme.colorScheme.primary,
                        onToggle = {
                            locationTracking = it
                            prefs.edit().putBoolean("location_tracking_enabled", it).apply()
                        }
                    )
                    HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.1f))
                    ToggleRow(
                        title = "Communication Logs",
                        subtitle = "Passively log outbound call length and contact counts",
                        checked = communicationLogs,
                        color = MaterialTheme.colorScheme.primary,
                        onToggle = {
                            communicationLogs = it
                            prefs.edit().putBoolean("communication_logs_enabled", it).apply()
                        }
                    )
                }
            }
        }

        item {
            InfoCard("Home Location GPS Anchor", headerColor = MaterialTheme.colorScheme.primary) {
                Column(modifier = Modifier.fillMaxWidth()) {
                    if (homeLocation != null) {
                        val (lat, lon) = checkNotNull(homeLocation)
                        val locationLabel = remember(lat, lon) { geocodeLocationName(context, lat, lon) }
                        val isAuto = remember(prefs, homeLocation) { prefs.getBoolean("home_location_set_automatically", false) }
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.Home, null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(18.dp))
                            Spacer(Modifier.width(8.dp))
                            Text(
                                text = if (isAuto) "✓ Home set automatically: $locationLabel" else "✓ Home set: $locationLabel",
                                fontSize = 13.sp, color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Medium
                            )
                        }
                    } else {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.weight(1f)) {
                                Icon(Icons.Default.LocationOff, null, tint = MaterialTheme.colorScheme.error, modifier = Modifier.size(18.dp))
                                Spacer(Modifier.width(8.dp))
                                Text("Home coordinate anchor is not set yet.", fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                            Spacer(Modifier.width(8.dp))
                            val badgeColor = if (isReminderDismissed) AlertRose else AlertWarning
                            val badgeText = if (isReminderDismissed) "Action Required - Not Set" else "Not Set"
                            Surface(
                                shape = RoundedCornerShape(6.dp),
                                color = badgeColor.copy(alpha = 0.12f),
                                border = BorderStroke(1.dp, badgeColor)
                            ) {
                                Text(
                                    text = badgeText,
                                    fontSize = 10.sp,
                                    fontWeight = FontWeight.Bold,
                                    color = badgeColor,
                                    fontFamily = Fredoka,
                                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                                )
                            }
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
                        colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        if (homeCapturing) {
                            CircularProgressIndicator(Modifier.size(18.dp), color = Color.Black, strokeWidth = 2.dp)
                            Spacer(Modifier.width(8.dp))
                        }
                        Text(if (homeCapturing) "Getting GPS fix..." else "📌 Reset Current Location as Home", color = Color.Black, fontSize = 13.sp, fontFamily = Fredoka, fontWeight = FontWeight.Bold)
                    }
                }
            }
        }

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
                                "This will permanently delete ALL collected telemetry data — daily features, app sessions, notification logs, behavioral baselines, and analysis history.",
                                fontSize = 13.sp, lineHeight = 18.sp, color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            Spacer(Modifier.height(10.dp))
                            Text(
                                "✅ Your data will be automatically backed up as a JSON file in your Lumen files directory before deletion.",
                                fontSize = 12.sp, lineHeight = 16.sp, color = MaterialTheme.colorScheme.primary,
                                fontWeight = FontWeight.Medium
                            )
                            Spacer(Modifier.height(6.dp))
                            Text(
                                "Both tracking modes will restart from Day 1 setup in sync.",
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

        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
            ) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text("Data Management", fontWeight = FontWeight.Bold, fontSize = 14.sp, fontFamily = Fredoka)
                    Text(
                        "Lumen runs 100% on-device. Export your reports locally or share with your doctor, import existing backups, or clear your history.",
                        fontSize = 12.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        lineHeight = 17.sp
                    )

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
                                .height(48.dp),
                            shape = RoundedCornerShape(10.dp)
                        ) {
                            Icon(Icons.Default.Download, null, tint = Color.Black, modifier = Modifier.size(18.dp))
                            Spacer(Modifier.width(8.dp))
                            Text("Export", color = Color.Black, fontWeight = FontWeight.Bold, fontSize = 12.sp, fontFamily = Fredoka)
                        }

                        Button(
                            onClick = {
                                importLauncher.launch(arrayOf("application/json"))
                            },
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
                            modifier = Modifier
                                .weight(1f)
                                .height(48.dp),
                            shape = RoundedCornerShape(10.dp)
                        ) {
                            Icon(Icons.Default.Upload, null, tint = MaterialTheme.colorScheme.onPrimaryContainer, modifier = Modifier.size(18.dp))
                            Spacer(Modifier.width(8.dp))
                            Text("Import", color = MaterialTheme.colorScheme.onPrimaryContainer, fontWeight = FontWeight.Bold, fontSize = 12.sp, fontFamily = Fredoka)
                        }
                    }
                }
            }
        }

        item {
            InfoCard("Notifications", headerColor = MaterialTheme.colorScheme.primary) {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    ToggleRow(
                        title = "Daily Check-in Reminders",
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
                }
            }
        }

        val countryName = profile?.country ?: "N/A"
        val helplines = getHelplinesByCountry(countryName)
        
        item {
            InfoCard("Support & Crisis Resources", headerColor = MaterialTheme.colorScheme.primary) {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text(
                        text = "Resources for ${if (countryName != "N/A") countryName else "Worldwide"}:",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    
                    helplines.forEach { helpline ->
                        Card(
                            onClick = {
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
                        .clickable {
                            try {
                                val feedbackIntent = Intent(Intent.ACTION_SENDTO).apply {
                                    data = Uri.parse("mailto:")
                                    putExtra(Intent.EXTRA_EMAIL, arrayOf("support@lumenapp.health"))
                                    putExtra(Intent.EXTRA_SUBJECT, "Lumen App Feedback")
                                    putExtra(Intent.EXTRA_TEXT, "Hi Lumen team,\n\nFeedback / Bug report:\n\n")
                                }
                                context.startActivity(feedbackIntent)
                            } catch (e: Exception) {
                                Toast.makeText(context, "No email app found. Please email support@lumenapp.health directly.", Toast.LENGTH_LONG).show()
                            }
                        }
                        .padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.Email, null, tint = MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.width(12.dp))
                        Column {
                            Text("Share Feedback", fontWeight = FontWeight.Bold, fontSize = 14.sp, fontFamily = Fredoka)
                            Text("Report a bug or suggest a feature", fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                    Icon(Icons.Default.ChevronRight, null, tint = MaterialTheme.colorScheme.primary)
                }
            }
        }

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
                        .clickable { showPrivacyDialog = true }
                        .padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.Security, null, tint = MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.width(12.dp))
                        Text("Privacy Policy & Terms", fontWeight = FontWeight.Bold, fontSize = 14.sp, fontFamily = Fredoka)
                    }
                    Icon(Icons.Default.ChevronRight, null, tint = MaterialTheme.colorScheme.primary)
                }
            }
        }

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
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error.copy(0.12f), contentColor = MaterialTheme.colorScheme.error),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.error.copy(0.3f)),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(48.dp),
                shape = RoundedCornerShape(10.dp)
            ) {
                Text("Wipe & Reset Local Databases", fontWeight = FontWeight.SemiBold, fontFamily = Fredoka)
            }
        }
    }

    if (showThemeDialog) {
        Dialog(onDismissRequest = { showThemeDialog = false }) {
            Card(
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
            ) {
                Column(
                    modifier = Modifier.padding(24.dp),
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Text("Select Theme Appearance", fontWeight = FontWeight.Bold, fontSize = 16.sp, fontFamily = Fredoka)
                    
                    listOf("dark" to "Dark Mode", "light" to "Light Mode", "system" to "System Default").forEach { (key, label) ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable {
                                    themeMode = key
                                    prefs.edit().putString("app_theme_mode", key).apply()
                                    showThemeDialog = false
                                }
                                .padding(vertical = 8.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            RadioButton(
                                selected = themeMode == key,
                                onClick = {
                                    themeMode = key
                                    prefs.edit().putString("app_theme_mode", key).apply()
                                    showThemeDialog = false
                                }
                            )
                            Spacer(Modifier.width(8.dp))
                            Text(label, fontSize = 14.sp)
                        }
                    }
                }
            }
        }
    }

    if (showPrivacyDialog) {
        Dialog(onDismissRequest = { showPrivacyDialog = false }) {
            Card(
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                modifier = Modifier.heightIn(max = 400.dp)
            ) {
                Column(
                    modifier = Modifier.padding(24.dp),
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Text("Privacy Policy & Terms", fontWeight = FontWeight.Bold, fontSize = 16.sp, fontFamily = Fredoka)
                    
                    Column(
                        modifier = Modifier
                            .weight(1f)
                            .verticalScroll(rememberScrollState())
                    ) {
                        Text(
                            text = "Lumen is designed to prioritize patient privacy. All passive sensor telemetry is processed entirely locally on this physical device. Your screen usage, GPS locations, communication records, and check-in history are never uploaded to any remote servers.\n\nYou have absolute ownership of your data. You can export your encrypted baseline backup files at any time to share with a physician, or permanently delete all local databases via the Safety Reset option.",
                            fontSize = 13.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            lineHeight = 18.sp
                        )
                    }
                    
                    Button(
                        onClick = { showPrivacyDialog = false },
                        colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Close", color = Color.Black, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                    }
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

    if (showAccessibilityDisclosure) {
        AccessibilityDisclosureDialog(
            onDismiss = { showAccessibilityDisclosure = false },
            onConfirm = {
                showAccessibilityDisclosure = false
                val intent = Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
                context.startActivity(intent)
            }
        )
    }
}

@Composable
fun PermissionSettingRow(
    title: String,
    subtitle: String,
    isGranted: Boolean,
    isReminderDismissed: Boolean,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(title, fontWeight = FontWeight.Bold, fontSize = 13.5.sp, fontFamily = Fredoka)
            Text(subtitle, fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Spacer(Modifier.width(8.dp))
        
        val badgeText = when {
            isGranted -> "Enabled"
            isReminderDismissed -> "Action Required - Disabled"
            else -> "Disabled"
        }
        val badgeColor = when {
            isGranted -> TealAccent
            isReminderDismissed -> AlertRose
            else -> AlertWarning
        }
        
        Surface(
            shape = RoundedCornerShape(6.dp),
            color = badgeColor.copy(alpha = 0.12f),
            border = BorderStroke(1.dp, badgeColor)
        ) {
            Text(
                text = badgeText,
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold,
                color = badgeColor,
                fontFamily = Fredoka,
                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
            )
        }
    }
}

// =============================================================================
// Onboarding Wizard Composable
// =============================================================================
@Composable
fun OnboardingWizard(onComplete: () -> Unit) {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    
    var step by remember { mutableIntStateOf(1) } // 1: Splash, 2: Demographics, 3: Routines, 4: Lifestyle, 5: Clinical, 6: Home GPS, 7: Permissions, 8: PHQ-9, 9: GAD-7, 10: Stressors, 11: Finalize
    
    // Step 2 State (Demographics)
    var name by remember { mutableStateOf("") }
    var gender by remember { mutableStateOf("") }
    var age by remember { mutableStateOf("") }
    var profession by remember { mutableStateOf("") }
    var country by remember { mutableStateOf("") }
    var livingSituation by remember { mutableStateOf("") }
    var isStudent by remember { mutableStateOf(false) }
    var showErrors by remember { mutableStateOf(false) }

    // Step 3 State (Routines)
    var typicalWake by remember { mutableFloatStateOf(7.0f) }
    var typicalSleep by remember { mutableFloatStateOf(23.0f) }
    var commuteMinutes by remember { mutableFloatStateOf(30.0f) }
    var routineConsistency by remember { mutableStateOf("") }

    // Step 4 State (Lifestyle Sliders, 1-5 scale)
    var screenReliance by remember { mutableFloatStateOf(3.0f) }
    var communicationActivity by remember { mutableFloatStateOf(3.0f) }
    var physicalMovement by remember { mutableFloatStateOf(3.0f) }
    var sleepHygiene by remember { mutableFloatStateOf(3.0f) }
    var moodReflection by remember { mutableFloatStateOf(3.0f) }
    var checkinLikelihood by remember { mutableFloatStateOf(3.0f) }
    var travelRegularity by remember { mutableFloatStateOf(3.0f) }
    var socialEngagement by remember { mutableFloatStateOf(3.0f) }
    var chargingConsistency by remember { mutableFloatStateOf(3.0f) }
    var appUsagePredictability by remember { mutableFloatStateOf(3.0f) }

    // Step 5 State (Clinical Status)
    var hasChronicCondition by remember { mutableStateOf(false) }
    var inTherapy by remember { mutableStateOf(false) }
    var physicalHealthRating by remember { mutableFloatStateOf(7.0f) }

    // Step 6 State (Home Location Capture)
    var homeCapturing by remember { mutableStateOf(false) }
    var homeSet by remember { mutableStateOf(DataRepository.homeLocation.value != null) }
    var showLocationDisclosureStep6 by remember { mutableStateOf(false) }

    // Step 7 State (System Permissions)
    var isNotificationAccessGranted by remember {
        mutableStateOf(com.example.mhealth.services.MHealthNotificationListenerService.isServiceEnabled(ctx))
    }
    var isLocationPermissionGranted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(ctx, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED)
    }
    var isBackgroundLocationGranted by remember {
        mutableStateOf(
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                ContextCompat.checkSelfPermission(ctx, Manifest.permission.ACCESS_BACKGROUND_LOCATION) == PackageManager.PERMISSION_GRANTED
            } else {
                true
            }
        )
    }
    var isTelemetryGranted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(ctx, Manifest.permission.READ_CONTACTS) == PackageManager.PERMISSION_GRANTED &&
            ContextCompat.checkSelfPermission(ctx, Manifest.permission.READ_CALENDAR) == PackageManager.PERMISSION_GRANTED
        )
    }
    var isAccessibilityGranted by remember {
        mutableStateOf(com.example.mhealth.services.MHealthAccessibilityService.isServiceEnabled(ctx))
    }
    var isUsageStatsGranted by remember {
        mutableStateOf(hasUsageStatsPermission(ctx))
    }
    var showLocationDisclosure by remember { mutableStateOf(false) }
    var showTelemetryDisclosure by remember { mutableStateOf(false) }
    var showAccessibilityDisclosure by remember { mutableStateOf(false) }

    // Refresh permission statuses when returning from OS settings
    val lifecycleOwner = androidx.compose.ui.platform.LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val observer = androidx.lifecycle.LifecycleEventObserver { _, event ->
            if (event == androidx.lifecycle.Lifecycle.Event.ON_RESUME) {
                isNotificationAccessGranted = com.example.mhealth.services.MHealthNotificationListenerService.isServiceEnabled(ctx)
                isLocationPermissionGranted = ContextCompat.checkSelfPermission(ctx, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
                isBackgroundLocationGranted = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    ContextCompat.checkSelfPermission(ctx, Manifest.permission.ACCESS_BACKGROUND_LOCATION) == PackageManager.PERMISSION_GRANTED
                } else {
                    true
                }
                isTelemetryGranted = ContextCompat.checkSelfPermission(ctx, Manifest.permission.READ_CONTACTS) == PackageManager.PERMISSION_GRANTED &&
                                     ContextCompat.checkSelfPermission(ctx, Manifest.permission.READ_CALENDAR) == PackageManager.PERMISSION_GRANTED
                isAccessibilityGranted = com.example.mhealth.services.MHealthAccessibilityService.isServiceEnabled(ctx)
                isUsageStatsGranted = hasUsageStatsPermission(ctx)
                homeSet = DataRepository.homeLocation.value != null
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }

    val telemetryLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        isTelemetryGranted = results[Manifest.permission.READ_CONTACTS] == true &&
                             results[Manifest.permission.READ_CALENDAR] == true
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

    val phq9Answers = remember { mutableStateListOf(*Array(2) { -1 }) }
    val phq9Questions = listOf(
        "Little interest or pleasure in doing things.",
        "Feeling down, depressed, or hopeless."
    )

    val gad7Answers = remember { mutableStateListOf(*Array(2) { -1 }) }
    val gad7Questions = listOf(
        "Feeling nervous, anxious, or on edge.",
        "Not being able to stop or control worrying."
    )

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

    fun formatTimeFloat(valFloat: Float): String {
        val totalMinutes = (valFloat * 60).roundToInt()
        val hours = totalMinutes / 60
        val minutes = totalMinutes % 60
        return "%02d:%02d".format(hours, minutes)
    }

    Column(
        Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .navigationBarsPadding()
    ) {
        Box(
            Modifier
                .fillMaxWidth()
                .background(
                    Brush.verticalGradient(
                        listOf(
                            MaterialTheme.colorScheme.surface,
                            MaterialTheme.colorScheme.surface.copy(0.6f)
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
                        2 -> "Tell us a bit about yourself"
                        3 -> "Your Daily Rhythms"
                        4 -> "Lifestyle Habits"
                        5 -> "Health Context"
                        6 -> "Set your home"
                        7 -> "System Permissions"
                        8 -> "Personal Well-being Survey"
                        9 -> "Daily Calmness & Reflection Checklist"
                        10 -> "Have you been through anything big lately?"
                        else -> "Setup Completed"
                    },
                    fontSize = 24.sp,
                    fontWeight = FontWeight.ExtraBold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Text(
                    text = when (step) {
                        1 -> "A quiet companion for your wellness"
                        2 -> "These details remain entirely private on this device"
                        3 -> "Helps Lumen understand your target routines"
                        4 -> "Understand your typical habits to personalize your wellness profile"
                        5 -> "Provides lifestyle baseline context to customize thresholds"
                        6 -> "Used locally to evaluate daily time spent at home"
                        7 -> "Lumen runs passively offline and requires permissions to collect telemetry"
                        8 -> "Answer honestly to help establish your personal well-being baseline"
                        9 -> "Helps Lumen personalize its guidance to support you"
                        10 -> "Identifies transient life events that might mimic indicators"
                        else -> "Lumen will quietly monitor in the background."
                    },
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 4.dp)
                )
            }
        }

        Box(Modifier.weight(1f)) {
            when (step) {
                1 -> {
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
                            "Calming Reflection & Biomarkers",
                            fontSize = 20.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.onBackground
                        )
                        Spacer(Modifier.height(12.dp))
                        Text(
                            "Lumen operates 100% locally and offline. It gathers passive behavioral telemetry—such as sleep patterns, physical movement, typing speeds, and social frequency—to construct a personalized habit profile and assist your wellness journey.",
                            fontSize = 13.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
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
                            Text("Get Started", color = Color.Black, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                        }
                    }
                }
                2 -> {
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
                                            onClick = {
                                                profession = opt
                                                isStudent = (opt == "Student")
                                                expanded = false
                                            }
                                        )
                                    }
                                }
                            }
                        }
                        item {
                            var expandedLiving by remember { mutableStateOf(false) }
                            Box(Modifier.fillMaxWidth()) {
                                OutlinedTextField(
                                    value = livingSituation, onValueChange = {},
                                    label = { Text("Living Situation") },
                                    readOnly = true,
                                    isError = showErrors && livingSituation.isBlank(),
                                    trailingIcon = { IconButton(onClick = { expandedLiving = !expandedLiving }) { Icon(Icons.Default.ArrowDropDown, null) } },
                                    modifier = Modifier.fillMaxWidth(),
                                    shape = RoundedCornerShape(10.dp)
                                )
                                Box(
                                    Modifier
                                        .matchParentSize()
                                        .clickable { expandedLiving = !expandedLiving })
                                DropdownMenu(expanded = expandedLiving, onDismissRequest = { expandedLiving = false }) {
                                    listOf("Alone", "With Family", "Roommates", "Hostel").forEach { opt ->
                                        DropdownMenuItem(
                                            text = { Text(opt) },
                                            onClick = { livingSituation = opt; expandedLiving = false }
                                        )
                                    }
                                }
                            }
                        }
                        item {
                            var expandedCountry by remember { mutableStateOf(false) }
                            var countrySearch by remember { mutableStateOf("") }
                            val allCountries = listOf(
                                "United States", "United Kingdom", "Canada", "Australia", 
                                "India", "Germany", "France", "Spain", "Italy", 
                                "Japan", "Brazil", "Mexico", "South Africa", "Other"
                            )
                            val filteredCountries = allCountries.filter { it.contains(countrySearch, ignoreCase = true) }
                            
                            Box(Modifier.fillMaxWidth()) {
                                OutlinedTextField(
                                    value = countrySearch, 
                                    onValueChange = { 
                                        countrySearch = it
                                        expandedCountry = true
                                        country = allCountries.find { c -> c.equals(it, ignoreCase=true) } ?: ""
                                    },
                                    label = { Text("Country") },
                                    isError = showErrors && country.isBlank(),
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .onFocusChanged { if (it.isFocused) expandedCountry = true },
                                    shape = RoundedCornerShape(10.dp)
                                )
                                DropdownMenu(
                                    expanded = expandedCountry && filteredCountries.isNotEmpty(),
                                    onDismissRequest = { expandedCountry = false },
                                    modifier = Modifier.heightIn(max = 240.dp)
                                ) {
                                    filteredCountries.forEach { c ->
                                        DropdownMenuItem(
                                            text = { Text(c) },
                                            onClick = { 
                                                countrySearch = c
                                                country = c
                                                expandedCountry = false
                                            }
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
                3 -> {
                    LazyColumn(
                        Modifier
                            .fillMaxSize()
                            .padding(24.dp),
                        verticalArrangement = Arrangement.spacedBy(20.dp)
                    ) {
                        item {
                            Text("Routines help Lumen construct a reference circadian rhythm context.", fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }

                        item {
                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(12.dp),
                                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(0.2f)),
                                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
                            ) {
                                Column(Modifier.padding(16.dp)) {
                                    Text("Typical Sleep Time: ${formatTimeFloat(typicalSleep)}", fontWeight = FontWeight.Bold, fontSize = 14.sp, fontFamily = Fredoka)
                                    Slider(
                                        value = typicalSleep,
                                        onValueChange = { typicalSleep = it },
                                        valueRange = 0f..24f,
                                        steps = 47,
                                        colors = SliderDefaults.colors(thumbColor = MaterialTheme.colorScheme.primary)
                                    )
                                }
                            }
                        }

                        item {
                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(12.dp),
                                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(0.2f)),
                                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
                            ) {
                                Column(Modifier.padding(16.dp)) {
                                    Text("Typical Daily Commute: ${commuteMinutes.toInt()} minutes", fontWeight = FontWeight.Bold, fontSize = 14.sp, fontFamily = Fredoka)
                                    Slider(
                                        value = commuteMinutes,
                                        onValueChange = { commuteMinutes = it },
                                        valueRange = 0f..120f,
                                        steps = 23,
                                        colors = SliderDefaults.colors(thumbColor = MaterialTheme.colorScheme.primary)
                                    )
                                }
                            }
                        }

                        item {
                            var expandedConsistency by remember { mutableStateOf(false) }
                            Box(Modifier.fillMaxWidth()) {
                                OutlinedTextField(
                                    value = routineConsistency, onValueChange = {},
                                    label = { Text("Routine Consistency") },
                                    readOnly = true,
                                    isError = showErrors && routineConsistency.isBlank(),
                                    trailingIcon = { IconButton(onClick = { expandedConsistency = !expandedConsistency }) { Icon(Icons.Default.ArrowDropDown, null) } },
                                    modifier = Modifier.fillMaxWidth(),
                                    shape = RoundedCornerShape(10.dp)
                                )
                                Box(
                                    Modifier
                                        .matchParentSize()
                                        .clickable { expandedConsistency = !expandedConsistency })
                                DropdownMenu(expanded = expandedConsistency, onDismissRequest = { expandedConsistency = false }) {
                                    listOf("Rigid", "Flexible", "Variable").forEach { opt ->
                                        DropdownMenuItem(
                                            text = { Text(opt) },
                                            onClick = { routineConsistency = opt; expandedConsistency = false }
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
                4 -> {
                    LazyColumn(
                        Modifier
                            .fillMaxSize()
                            .padding(24.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        item {
                            Text("Rate your typical phone usage and activities to personalize your wellness profile.", fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }

                        item {
                            LifestyleSlider(
                                title = "Screen Reliance",
                                description = "How reliant are you on your phone screen daily?",
                                value = screenReliance,
                                onValueChange = { screenReliance = it }
                            )
                        }

                        item {
                            LifestyleSlider(
                                title = "Communication Activity",
                                description = "How active are you in daily calls and messages?",
                                value = communicationActivity,
                                onValueChange = { communicationActivity = it }
                            )
                        }

                        item {
                            LifestyleSlider(
                                title = "Physical Movement",
                                description = "How active is your daily physical movement?",
                                value = physicalMovement,
                                onValueChange = { physicalMovement = it }
                            )
                        }

                        item {
                            LifestyleSlider(
                                title = "Sleep Hygiene",
                                description = "How consistent is your sleep routine?",
                                value = sleepHygiene,
                                onValueChange = { sleepHygiene = it }
                            )
                        }

                        item {
                            LifestyleSlider(
                                title = "Digital-Mood Reflection",
                                description = "How strongly does your mood reflect your screen use?",
                                value = moodReflection,
                                onValueChange = { moodReflection = it }
                            )
                        }

                        item {
                            LifestyleSlider(
                                title = "Wellness Check-in Likelihood",
                                description = "How likely are you to complete daily check-ins?",
                                value = checkinLikelihood,
                                onValueChange = { checkinLikelihood = it }
                            )
                        }

                        item {
                            LifestyleSlider(
                                title = "Location/Travel Regularity",
                                description = "How predictable is your daily travel routine?",
                                value = travelRegularity,
                                onValueChange = { travelRegularity = it }
                            )
                        }

                        item {
                            LifestyleSlider(
                                title = "Social Engagement",
                                description = "How regular are your social interactions?",
                                value = socialEngagement,
                                onValueChange = { socialEngagement = it }
                            )
                        }

                        item {
                            LifestyleSlider(
                                title = "Charging Habits",
                                description = "How consistent is your phone charging routine?",
                                value = chargingConsistency,
                                onValueChange = { chargingConsistency = it }
                            )
                        }

                        item {
                            LifestyleSlider(
                                title = "App Usage Patterns",
                                description = "How predictable are the apps you use daily?",
                                value = appUsagePredictability,
                                onValueChange = { appUsagePredictability = it }
                            )
                        }
                    }
                }
                5 -> {
                    LazyColumn(
                        Modifier
                            .fillMaxSize()
                            .padding(24.dp),
                        verticalArrangement = Arrangement.spacedBy(20.dp)
                    ) {
                        item {
                            Text("Provide wellness context to customize tracking thresholds.", fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }

                        item {
                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(16.dp),
                                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
                            ) {
                                Column(Modifier.padding(16.dp)) {
                                    OnboardingSwitchRow(
                                        title = "Diagnosed Condition",
                                        subtitle = "I have a diagnosed/chronic mental health condition",
                                        checked = hasChronicCondition,
                                        onCheckedChange = { hasChronicCondition = it }
                                    )
                                    HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(0.1f), modifier = Modifier.padding(vertical = 8.dp))
                                    OnboardingSwitchRow(
                                        title = "In Therapy",
                                        subtitle = "I am currently in professional therapy",
                                        checked = inTherapy,
                                        onCheckedChange = { inTherapy = it }
                                    )
                                }
                            }
                        }

                        item {
                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(12.dp),
                                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(0.2f)),
                                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
                            ) {
                                Column(Modifier.padding(16.dp)) {
                                    Text("General Physical Health: ${physicalHealthRating.toInt()}/10", fontWeight = FontWeight.Bold, fontSize = 14.sp, fontFamily = Fredoka)
                                    Slider(
                                        value = physicalHealthRating,
                                        onValueChange = { physicalHealthRating = it },
                                        valueRange = 1f..10f,
                                        steps = 8,
                                        colors = SliderDefaults.colors(thumbColor = MaterialTheme.colorScheme.primary)
                                    )
                                }
                            }
                        }
                    }
                }
                6 -> {
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

                        if (!isLocationPermissionGranted) {
                            Button(
                                onClick = {
                                    showLocationDisclosureStep6 = true
                                },
                                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.secondary),
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(50.dp),
                                shape = RoundedCornerShape(12.dp)
                            ) {
                                Text("Grant GPS Location Permission", color = Color.White, fontFamily = Fredoka)
                            }
                            if (showLocationDisclosureStep6) {
                                LocationDisclosureDialog(
                                    onDismiss = { showLocationDisclosureStep6 = false },
                                    onConfirm = {
                                        showLocationDisclosureStep6 = false
                                        locPermissionLauncher.launch(
                                            arrayOf(
                                                Manifest.permission.ACCESS_FINE_LOCATION,
                                                Manifest.permission.ACCESS_COARSE_LOCATION
                                            )
                                        )
                                    }
                                )
                            }
                        } else {
                            if (homeSet) {
                                val loc = DataRepository.homeLocation.value
                                Card(
                                    shape = RoundedCornerShape(8.dp),
                                    colors = CardDefaults.cardColors(containerColor = TealAccent.copy(0.12f))
                                ) {
                                    Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                                        Icon(Icons.Default.CheckCircle, null, tint = TealAccent, modifier = Modifier.size(20.dp))
                                        Spacer(Modifier.width(8.dp))
                                        Text(
                                            "Coordinates Anchored: %.4f, %.4f".format(loc?.first ?: 0.0, loc?.second ?: 0.0),
                                            fontSize = 12.sp,
                                            fontWeight = FontWeight.SemiBold,
                                            color = TealAccent
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
                            // Removed manual coordinates button and input fields
                        }
                    }
                }
                7 -> {
                    Box(Modifier.fillMaxSize()) {
                        LazyColumn(
                            Modifier
                                .fillMaxSize()
                                .padding(24.dp),
                            verticalArrangement = Arrangement.spacedBy(16.dp)
                        ) {
                            item {
                                Text(
                                    text = "Lumen needs access to system permissions to passively monitor telemetry. All data is processed 100% locally.",
                                    fontSize = 13.sp,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }

                            item {
                                PermissionStatusCard(
                                    title = "App Usage Telemetry",
                                    description = "Required to track screen time, app launch patterns, and unlock counts.",
                                    isGranted = isUsageStatsGranted,
                                    onClick = {
                                        ctx.startActivity(Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS))
                                    }
                                )
                            }

                            item {
                                PermissionStatusCard(
                                    title = "Notification Listener Access",
                                    description = "Required to track notification rates and music playtimes.",
                                    isGranted = isNotificationAccessGranted,
                                    onClick = {
                                        ctx.startActivity(Intent("android.settings.ACTION_NOTIFICATION_LISTENER_SETTINGS"))
                                    }
                                )
                            }

                            item {
                                PermissionStatusCard(
                                    title = "GPS Location & Background Tracking",
                                    description = "Required to analyze spatial entropy, home ratio, and daily displacement.",
                                    isGranted = isLocationPermissionGranted && isBackgroundLocationGranted,
                                    onClick = {
                                        showLocationDisclosure = true
                                    }
                                )
                            }

                            item {
                                PermissionStatusCard(
                                    title = "Behavioral Rhythms Telemetry",
                                    description = "Required to analyze contact interactions, calendar events, and physical steps.",
                                    isGranted = isTelemetryGranted,
                                    onClick = {
                                        showTelemetryDisclosure = true
                                    }
                                )
                            }

                            item {
                                PermissionStatusCard(
                                    title = "Digital Psychomotor Dynamics",
                                    description = "Required to analyze typing speeds, backspace ratios, and scroll velocity.",
                                    isGranted = isAccessibilityGranted,
                                    onClick = {
                                        showAccessibilityDisclosure = true
                                    }
                                )
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

                        if (showTelemetryDisclosure) {
                            TelemetryDisclosureDialog(
                                onDismiss = { showTelemetryDisclosure = false },
                                onConfirm = {
                                    showTelemetryDisclosure = false
                                    telemetryLauncher.launch(
                                        arrayOf(
                                            Manifest.permission.READ_CONTACTS,
                                            Manifest.permission.READ_CALENDAR,
                                            Manifest.permission.ACTIVITY_RECOGNITION
                                        )
                                    )
                                }
                            )
                        }

                        if (showAccessibilityDisclosure) {
                            AccessibilityDisclosureDialog(
                                onDismiss = { showAccessibilityDisclosure = false },
                                onConfirm = {
                                    showAccessibilityDisclosure = false
                                    val intent = Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
                                    ctx.startActivity(intent)
                                }
                            )
                        }
                    }
                }
                8 -> {
                    ScreenerWizard(
                        questions = phq9Questions,
                        answers = phq9Answers,
                        options = optionsList,
                        onCompleted = { step = 9 }
                    )
                }
                9 -> {
                    ScreenerWizard(
                        questions = gad7Questions,
                        answers = gad7Answers,
                        options = optionsList,
                        onCompleted = { step = 10 }
                    )
                }
                10 -> {
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
                    }
                }
                11 -> {
                    val checkinEnabled by DataRepository.checkinNotificationsEnabled.collectAsState()
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
                                .background(TealAccent.copy(0.12f)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(Icons.Default.Check, null, tint = TealAccent, modifier = Modifier.size(40.dp))
                        }
                        
                        Text(
                            "Lumen is Ready",
                            fontWeight = FontWeight.Bold,
                            fontSize = 20.sp,
                            fontFamily = Fredoka,
                            textAlign = TextAlign.Center
                        )

                        Card(
                            Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(16.dp),
                            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(0.3f))
                        ) {
                            Text(
                                text = "Your baseline preferences have been established. Lumen is now ready to begin local, private tracking to assist your wellness journey. We will check in periodically to help you stay in tune with your routines.",
                                fontSize = 13.sp,
                                modifier = Modifier.padding(16.dp),
                                lineHeight = 18.sp,
                                color = MaterialTheme.colorScheme.onBackground.copy(0.8f)
                            )
                        }

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
                                colors = SwitchDefaults.colors(checkedThumbColor = Color.Black)
                            )
                        }

                        Spacer(Modifier.weight(1f))

                        Button(
                            onClick = {
                                val localPref = ctx.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE)
                                localPref.edit().apply {
                                    putBoolean("first_login_complete", true)
                                    
                                    // Save demographics
                                    putString("user_name", name)
                                    putString("user_gender", gender)
                                    putInt("user_age", age.toIntOrNull() ?: 25)
                                    putString("user_profession", profession)
                                    putString("user_country", country)
                                    putString("user_living_situation", livingSituation)
                                    putBoolean("user_is_student", isStudent)
                                    
                                    // Save routines
                                    putFloat("user_typical_wake", typicalWake)
                                    putFloat("user_typical_sleep", typicalSleep)
                                    putInt("user_commute_minutes", commuteMinutes.toInt())
                                    putString("user_routine_consistency", routineConsistency.lowercase())
                                    
                                    // Save lifestyle sliders
                                    putInt("user_lifestyle_screen", screenReliance.toInt())
                                    putInt("user_lifestyle_communication", communicationActivity.toInt())
                                    putInt("user_lifestyle_movement", physicalMovement.toInt())
                                    putInt("user_lifestyle_sleep", sleepHygiene.toInt())
                                    putInt("user_lifestyle_behavioral", moodReflection.toInt())
                                    putInt("user_lifestyle_engagement", checkinLikelihood.toInt())
                                    putInt("user_lifestyle_travel", travelRegularity.toInt())
                                    putInt("user_lifestyle_social", socialEngagement.toInt())
                                    putInt("user_lifestyle_charging", chargingConsistency.toInt())
                                    putInt("user_lifestyle_app_usage", appUsagePredictability.toInt())
                                    
                                    // Save clinical status
                                    putBoolean("user_has_chronic_condition", hasChronicCondition)
                                    putBoolean("user_in_therapy", inTherapy)
                                    putInt("user_physical_health_rating", physicalHealthRating.toInt())
                                }.apply()

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
                                    withContext(Dispatchers.Main) {
                                        DataRepository.initWithDb(ctx, "patient@lumen.health")
                                        DataRepository.completeOnboarding()
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
                            Text("Start My Journey", color = Color.Black, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                        }
                    }
                }
            }
        }

        // Navigation row at bottom of content (if not splash, screener or finalize)
        if (step in 2..7 || step == 10) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(24.dp),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Button(
                    onClick = { if (step > 1) step-- },
                    enabled = step > 1,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.outline.copy(0.1f),
                        contentColor = MaterialTheme.colorScheme.onBackground
                    ),
                    shape = RoundedCornerShape(10.dp),
                    modifier = Modifier.height(48.dp)
                ) {
                    Text("Back", fontFamily = Fredoka, fontWeight = FontWeight.ExtraBold)
                }

                Button(
                    onClick = {
                        when (step) {
                            2 -> {
                                if (name.isBlank() || gender.isBlank() || age.isBlank() || profession.isBlank() || country.isBlank() || livingSituation.isBlank()) {
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
                            }
                            3 -> {
                                if (routineConsistency.isBlank()) {
                                    showErrors = true
                                    Toast.makeText(ctx, "Please select routine consistency", Toast.LENGTH_SHORT).show()
                                } else {
                                    step = 4
                                }
                            }
                            4 -> step = 5
                            5 -> step = 6
                            6 -> step = 7
                            7 -> step = 8
                            10 -> {
                                // Calculate Calibration Scores
                                val totalPhq = (phq9Answers.sum() * 9f / 2f).roundToInt()
                                val totalGad = (gad7Answers.sum() * 7f / 2f).roundToInt()
                                val totalEvents = selectedStressors.filter { it.value && it.key < stressors.size - 1 }.size

                                DataRepository.saveScreenerScores(totalPhq, totalGad, totalEvents)
                                
                                val localPref = ctx.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE)
                                localPref.edit().apply {
                                    putInt("screener_phq9", totalPhq)
                                    putInt("screener_gad7", totalGad)
                                    putInt("screener_life_events", totalEvents)
                                }.apply()

                                step = 11
                            }
                        }
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                    shape = RoundedCornerShape(10.dp),
                    modifier = Modifier
                        .width(140.dp)
                        .height(48.dp)
                ) {
                    Text(
                        text = if (step == 10) "Complete" else "Next",
                        fontFamily = Fredoka,
                        color = Color.Black,
                        fontWeight = FontWeight.ExtraBold
                    )
                }
            }
        }
    }
}

@Composable
fun LifestyleSlider(
    title: String,
    description: String,
    value: Float,
    onValueChange: (Float) -> Unit
) {
    val labels = listOf("Not Very Much", "Slightly", "Somewhat", "A Lot", "Very Much")
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(0.2f)),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
    ) {
        Column(Modifier.padding(16.dp)) {
            Text(title, fontWeight = FontWeight.Bold, fontSize = 14.sp, fontFamily = Fredoka)
            Text(description, fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 2.dp))
            Spacer(Modifier.height(8.dp))
            Slider(
                value = value,
                onValueChange = onValueChange,
                valueRange = 1f..5f,
                steps = 3,
                colors = SliderDefaults.colors(
                    activeTrackColor = MaterialTheme.colorScheme.primary,
                    inactiveTrackColor = MaterialTheme.colorScheme.outline.copy(0.2f),
                    thumbColor = MaterialTheme.colorScheme.primary,
                    activeTickColor = Color.Transparent,
                    inactiveTickColor = Color.Transparent
                ),
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(Modifier.height(2.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                labels.forEachIndexed { index, label ->
                    val isSelected = index + 1 == value.toInt()
                    Text(
                        text = label,
                        fontSize = 8.5.sp,
                        fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                        color = if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onBackground.copy(0.4f),
                        textAlign = TextAlign.Center,
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }
    }
}

@Composable
fun OnboardingSwitchRow(
    title: String,
    subtitle: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(title, fontWeight = FontWeight.Bold, fontSize = 14.sp, fontFamily = Fredoka)
            Text(subtitle, fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Switch(
            checked = checked,
            onCheckedChange = onCheckedChange,
            colors = SwitchDefaults.colors(checkedThumbColor = Color.Black)
        )
    }
}

@Composable
fun PermissionStatusCard(
    title: String,
    description: String,
    isGranted: Boolean,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(title, fontWeight = FontWeight.Bold, fontSize = 14.sp, fontFamily = Fredoka)
                Text(description, fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Spacer(Modifier.width(12.dp))
            Surface(
                shape = RoundedCornerShape(8.dp),
                color = if (isGranted) TealAccent.copy(0.12f) else AlertRose.copy(0.12f),
                border = BorderStroke(1.dp, if (isGranted) TealAccent else AlertRose)
            ) {
                Text(
                    text = if (isGranted) "Granted" else "Tap to Enable",
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    color = if (isGranted) TealAccent else AlertRose,
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
                )
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
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Question ${qIndex + 1} of ${questions.size}",
                fontSize = 12.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
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

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.05f)),
            border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.1f))
        ) {
            Text(
                text = questions[qIndex],
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                modifier = Modifier.padding(20.dp),
                lineHeight = 22.sp,
                color = MaterialTheme.colorScheme.onBackground
            )
        }

        Text(
            text = "Over the last 2 weeks, how often have you been bothered by this?",
            fontSize = 12.sp,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(bottom = 8.dp)
        )

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
                            RoundedCornerShape(12.dp)
                        )
                        .background(if (selected) MaterialTheme.colorScheme.primary.copy(0.05f) else Color.Transparent)
                        .padding(horizontal = 16.dp, vertical = 14.dp)
                ) {
                    RadioButton(
                        selected = selected,
                        onClick = { answers[qIndex] = score },
                        colors = RadioButtonDefaults.colors(selectedColor = MaterialTheme.colorScheme.primary)
                    )
                    Spacer(Modifier.width(8.dp))
                    Text(
                        text = text,
                        fontSize = 13.5.sp,
                        color = MaterialTheme.colorScheme.onBackground.copy(0.85f)
                    )
                }
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Button(
                onClick = { if (qIndex > 0) qIndex-- },
                enabled = qIndex > 0,
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.outline.copy(0.1f),
                    contentColor = MaterialTheme.colorScheme.onBackground
                ),
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
                Text(
                    text = if (qIndex == questions.size - 1) "Complete" else "Next",
                    fontFamily = Fredoka,
                    color = Color.Black,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}

// =============================================================================
// Helper Methods & Logic Layer
// =============================================================================
@Composable
fun StaggeredFadeIn(
    index: Int,
    content: @Composable () -> Unit
) {
    var visible by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) {
        val delayTime = (index * 30).coerceAtMost(150).toLong()
        delay(delayTime)
        visible = true
    }
    
    val alpha by animateFloatAsState(
        targetValue = if (visible) 1f else 0f,
        animationSpec = tween(durationMillis = 300),
        label = "FadeAlpha"
    )
    val translationY by animateFloatAsState(
        targetValue = if (visible) 0f else 40f,
        animationSpec = tween(durationMillis = 300, easing = EaseOut),
        label = "FadeTranslationY"
    )
    
    Box(
        modifier = Modifier.graphicsLayer {
            this.alpha = alpha
            this.translationY = translationY
        }
    ) {
        content()
    }
}
@Composable
fun ToggleRow(title: String, subtitle: String, checked: Boolean, color: Color, onToggle: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(title, fontSize = 13.5.sp, fontWeight = FontWeight.Medium, color = MaterialTheme.colorScheme.onBackground)
            Text(subtitle, fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Switch(
            checked = checked, onCheckedChange = onToggle,
            colors = SwitchDefaults.colors(checkedThumbColor = Color.Black, checkedTrackColor = color)
        )
    }
}

@Composable
fun InfoCard(
    title: String,
    headerColor: Color = TealAccent,
    modifier: Modifier = Modifier,
    content: @Composable ColumnScope.() -> Unit
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
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
                    color = MaterialTheme.colorScheme.onBackground,
                    fontFamily = Fredoka
                )
            }
            content()
        }
    }
}

fun getGreeting(): String {
    val hour = Calendar.getInstance().get(Calendar.HOUR_OF_DAY)
    return when (hour) {
        in 5..11 -> "Good Morning"
        in 12..16 -> "Good Afternoon"
        else -> "Good Evening"
    }
}

fun geocodeLocationName(context: Context, lat: Double, lon: Double): String {
    return try {
        val geocoder = Geocoder(context, Locale.getDefault())
        @Suppress("DEPRECATION")
        val addresses = geocoder.getFromLocation(lat, lon, 1)
        if (!addresses.isNullOrEmpty()) {
            val addr = addresses[0]
            val parts = listOfNotNull(addr.subLocality, addr.locality, addr.adminArea)
            if (parts.isNotEmpty()) parts.joinToString(", ") else "📍 Home location set"
        } else "📍 Home location set"
    } catch (e: Exception) {
        "📍 Home location set"
    }
}

fun generateBehavioralSummary(
    isBuilding: Boolean,
    score: Float,
    weeklyFeatures: List<PersonalityVector>,
    baseline: List<com.example.mhealth.logic.db.BaselineEntity>,
    isDnaReady: Boolean
): String {
    if (isBuilding || score < 0f || !isDnaReady || weeklyFeatures.isEmpty()) {
        return "Lumen is calibrating your daily rhythms. Continue your normal routines while we establish your baseline."
    }
    if (baseline.isEmpty() || weeklyFeatures.size < 2) {
        return "Gently mapping your routines. Your daily rhythm story will appear here as more telemetry is registered."
    }

    val baseMap = baseline.associate { it.featureName to (it.baselineValue to it.stdDeviation) }
    val recent = weeklyFeatures.take(7)

    // Gather deviations
    val deviations = mutableListOf<String>()
    var sleepShift = 0f
    var screenShift = 0f
    var stepShift = 0f
    var socialShift = 0f
    var spatialShift = 0f
    var daylightShift = 0f

    // 1. Sleep
    val sleepBase = baseMap["sleepDurationHours"]
    if (sleepBase != null && sleepBase.second > 0f) {
        val avgSleep = recent.map { it.sleepDurationHours }.average().toFloat()
        val diff = avgSleep - sleepBase.first
        val zScore = diff / sleepBase.second.coerceAtLeast(0.5f)
        if (abs(zScore) > 1.0f) {
            sleepShift = zScore
            deviations.add(if (diff < 0) "shorter sleep windows (-${String.format("%.1f", abs(diff))}h)" else "longer rest periods (+${String.format("%.1f", abs(diff))}h)")
        }
    }

    // 2. Screen
    val screenBase = baseMap["screenTimeHours"]
    if (screenBase != null && screenBase.second > 0f) {
        val avgScreen = recent.map { it.screenTimeHours }.average().toFloat()
        val diff = avgScreen - screenBase.first
        val zScore = diff / screenBase.second.coerceAtLeast(0.5f)
        if (abs(zScore) > 1.0f) {
            screenShift = zScore
            deviations.add(if (diff > 0) "elevated screen engagement (+${String.format("%.1f", abs(diff))}h)" else "reduced screen time (-${String.format("%.1f", abs(diff))}h)")
        }
    }

    // 3. Activity (Steps)
    val stepsBase = baseMap["dailyStepCount"]
    if (stepsBase != null && stepsBase.second > 0f) {
        val avgSteps = recent.map { it.dailyStepCount }.average().toFloat()
        val diff = avgSteps - stepsBase.first
        val zScore = diff / stepsBase.second.coerceAtLeast(500f)
        if (abs(zScore) > 1.0f) {
            stepShift = zScore
            deviations.add(if (diff > 0) "increased physical movement" else "decreased physical steps")
        }
    }

    // 4. Social (Calls & Social Ratio)
    val callsBase = baseMap["callsPerDay"]
    if (callsBase != null && callsBase.second > 0f) {
        val avgCalls = recent.map { it.callsPerDay }.average().toFloat()
        val diff = avgCalls - callsBase.first
        val zScore = diff / callsBase.second.coerceAtLeast(0.5f)
        if (abs(zScore) > 1.0f) {
            socialShift = zScore
            deviations.add(if (diff > 0) "frequent social contact" else "reduced social interactions")
        }
    }

    // 5. Spatial (Entropy & Displacement)
    val entBase = baseMap["locationEntropy"]
    if (entBase != null && entBase.second > 0f) {
        val avgEnt = recent.map { it.locationEntropy }.average().toFloat()
        val diff = avgEnt - entBase.first
        val zScore = diff / entBase.second.coerceAtLeast(0.1f)
        if (abs(zScore) > 1.0f) {
            spatialShift = zScore
            deviations.add(if (diff > 0) "greater environmental variety" else "staying in familiar locations")
        }
    }

    // 6. Daylight
    val daylightBase = baseMap["daylightExposureMinutes"]
    if (daylightBase != null && daylightBase.second > 0f) {
        val avgDaylight = recent.map { it.daylightExposureMinutes }.average().toFloat()
        val diff = avgDaylight - daylightBase.first
        val zScore = diff / daylightBase.second.coerceAtLeast(10f)
        if (abs(zScore) > 1.0f) {
            daylightShift = zScore
            deviations.add(if (diff > 0) "increased outdoor light exposure" else "low daylight exposure")
        }
    }

    if (deviations.isEmpty()) {
        return "Your daily routines are flowing in beautiful alignment. You're maintaining a steady balance across screen time, activity, and sleep. Keep nurturing this steady rhythm! ✨"
    }

    // Build story based on multi-dimensional shifts
    val intro = "Lumen has observed a few subtle shifts in your behavioral rhythm this week, characterized by " + 
        when (deviations.size) {
            1 -> deviations[0]
            2 -> "${deviations[0]} and ${deviations[1]}"
            else -> deviations.dropLast(1).joinToString(", ") + ", and " + deviations.last()
        } + "."

    val analysisText = java.lang.StringBuilder(intro)

    // Multi-dimensional cohesive analysis
    if (screenShift > 1.0f && sleepShift < -1.0f) {
        analysisText.append(" Late-night digital engagement is correlating with reduced rest. Unplugging earlier could help restore sleep consistency.")
    } else if (stepShift < -1.0f && spatialShift < -1.0f) {
        analysisText.append(" A quiet physical flow matches a preference for staying indoors. A short walk in a new setting might help refresh your outlook.")
    } else if (socialShift < -1.0f && stepShift < -1.0f) {
        analysisText.append(" A quieter social rhythm is paired with lower physical energy. Gentle self-care and a brief contact with a loved one could boost resilience.")
    } else if (daylightShift < -1.0f && screenShift > 1.0f) {
        analysisText.append(" Low outdoor light and elevated screen use suggest a indoor-heavy cycle. Stepping outside for 10 minutes can reset your body clock.")
    } else {
        // Fallback single-dimension highlights
        if (sleepShift < -1.0f) {
            analysisText.append(" Shorter sleep durations indicate a need for recovery. Prioritizing rest tonight could bring back balance.")
        } else if (screenShift > 1.0f) {
            analysisText.append(" Higher digital engagement suggests softer screen boundaries. Setting small offline windows could clear mental clutter.")
        } else if (stepShift < -1.0f) {
            analysisText.append(" Physical movement is quiet compared to your baseline. A gentle stretch or quick walk can renew your physical energy.")
        } else if (socialShift < -1.0f) {
            analysisText.append(" Social interaction has dipped. Connecting with someone close, even briefly, can provide comforting emotional grounding.")
        }
    }

    return analysisText.toString()
}

fun getActiveStreak(prefs: SharedPreferences): Int {
    val currentStreak = prefs.getInt("checkin_streak_current", 0)
    val lastDateStr = prefs.getString("checkin_streak_last_date", "") ?: ""
    if (lastDateStr.isEmpty()) return 0
    
    try {
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
        
        return if (diffDays <= 1) currentStreak else 0
    } catch (e: Exception) {
        return 0
    }
}

fun recordDailyCheckin(prefs: SharedPreferences, mood: Int, energy: Int, sleep: Int, anxiety: Int) {
    val todayStr = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date())
    val lastDateStr = prefs.getString("checkin_streak_last_date", "") ?: ""
    
    val currentStreak = prefs.getInt("checkin_streak_current", 0)
    val yesterdayStr = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(
        Calendar.getInstance().apply { add(Calendar.DAY_OF_YEAR, -1) }.time
    )
    
    val newStreak = when (lastDateStr) {
        todayStr -> currentStreak
        yesterdayStr -> currentStreak + 1
        else -> 1
    }
    
    prefs.edit().apply {
        putString("daily_checkin_date_last", todayStr)
        putInt("daily_checkin_mood", mood)
        putInt("daily_checkin_energy", energy)
        putInt("daily_checkin_sleep", sleep)
        putInt("daily_checkin_anxiety", anxiety)
        putInt("checkin_streak_current", newStreak)
        putString("checkin_streak_last_date", todayStr)
    }.apply()
}

fun saveCheckinToHistory(prefs: SharedPreferences, mood: Int, energy: Int, sleep: Int, anxiety: Int, note: String = "") {
    val todayStr = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date())
    val historyStr = prefs.getString("daily_checkin_history", "[]") ?: "[]"
    
    try {
        val array = org.json.JSONArray(historyStr)
        val list = mutableListOf<org.json.JSONObject>()
        var foundToday = false
        
        for (i in 0 until array.length()) {
            val obj = array.getJSONObject(i)
            if (obj.getString("date") == todayStr) {
                obj.put("mood", mood)
                obj.put("energy", energy)
                obj.put("sleep", sleep)
                obj.put("anxiety", anxiety)
                if (note.isNotBlank()) obj.put("note", note) else obj.remove("note")
                foundToday = true
            }
            list.add(obj)
        }
        
        if (!foundToday) {
            val newObj = org.json.JSONObject().apply {
                put("date", todayStr)
                put("mood", mood)
                put("energy", energy)
                put("sleep", sleep)
                put("anxiety", anxiety)
                if (note.isNotBlank()) put("note", note)
            }
            list.add(newObj)
        }
        
        // No cap — keep unlimited history for user reflection
        val newArray = org.json.JSONArray()
        list.forEach { newArray.put(it) }
        
        prefs.edit().putString("daily_checkin_history", newArray.toString()).apply()
    } catch (e: Exception) {
        e.printStackTrace()
    }
}

fun getWeeklyCheckinAverageMood(prefs: SharedPreferences): Float {
    val historyStr = prefs.getString("daily_checkin_history", "[]") ?: "[]"
    try {
        val array = org.json.JSONArray(historyStr)
        if (array.length() == 0) return 0f
        var total = 0
        for (i in 0 until array.length()) {
            total += array.getJSONObject(i).getInt("mood")
        }
        return total.toFloat() / array.length()
    } catch (e: Exception) {
        return 0f
    }
}

fun getMonthlyCooldownDays(prefs: SharedPreferences): Int {
    val lastDateStr = prefs.getString("monthly_checkin_last_date", "") ?: ""
    if (lastDateStr.isEmpty()) return 0
    
    try {
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
        
        return (30 - diffDays).toInt().coerceAtLeast(0)
    } catch (e: Exception) {
        return 0
    }
}

data class Helpline(
    val name: String,
    val number: String,
    val availability: String,
    val type: HelplineType
)

enum class HelplineType { PHONE, WEB }

fun getHelplinesByCountry(country: String): List<Helpline> {
    val cleanCountry = country.trim().lowercase(Locale.ROOT)
    return when {
        cleanCountry.contains("india") -> listOf(
            Helpline("iCall India", "9152987821", "Mon-Sat: 10 AM - 8 PM", HelplineType.PHONE),
            Helpline("Vandrevala Foundation", "1860-2662-345", "24/7 Availability", HelplineType.PHONE),
            Helpline("NIMHANS Helpline", "080-46110007", "24/7 Availability", HelplineType.PHONE)
        )
        cleanCountry.contains("united states") || cleanCountry.contains("us") || cleanCountry.contains("u.s.") || cleanCountry.contains("america") -> listOf(
            Helpline("Suicide & Crisis Lifeline", "988", "24/7 Availability", HelplineType.PHONE),
            Helpline("NAMI HelpLine", "1-800-950-6264", "Mon-Fri: 10 AM - 10 PM EST", HelplineType.PHONE),
            Helpline("Crisis Text Line", "Text HOME to 741741", "24/7 SMS Support", HelplineType.PHONE)
        )
        cleanCountry.contains("united kingdom") || cleanCountry.contains("uk") || cleanCountry.contains("u.k.") || cleanCountry.contains("britain") || cleanCountry.contains("england") -> listOf(
            Helpline("Samaritans UK", "116 123", "24/7 Availability", HelplineType.PHONE),
            Helpline("Mind Helpline", "0300 123 3393", "Mon-Fri: 9 AM - 6 PM", HelplineType.PHONE),
            Helpline("NHS Mental Health", "111", "24/7 Availability", HelplineType.PHONE)
        )
        cleanCountry.contains("canada") -> listOf(
            Helpline("Suicide Crisis Helpline", "988", "24/7 Availability", HelplineType.PHONE),
            Helpline("Talk Suicide Canada", "1-833-456-4566", "24/7 Availability", HelplineType.PHONE),
            Helpline("Kids Help Phone", "1-800-668-6868", "24/7 Youth Support", HelplineType.PHONE)
        )
        cleanCountry.contains("australia") -> listOf(
            Helpline("Lifeline Australia", "13 11 14", "24/7 Availability", HelplineType.PHONE),
            Helpline("Beyond Blue", "1300 22 4636", "24/7 Availability", HelplineType.PHONE),
            Helpline("Kids Helpline", "1800 55 1800", "24/7 Youth Support", HelplineType.PHONE)
        )
        else -> listOf(
            Helpline("IASP Crisis Centres", "https://www.iasp.info/resources/Crisis_Centres/", "Find local help worldwide", HelplineType.WEB),
            Helpline("Befrienders Worldwide", "https://www.befrienders.org/", "Find crisis support in your country", HelplineType.WEB)
        )
    }
}

private fun exportDataToUri(context: Context, uri: android.net.Uri) {
    if (context !is ComponentActivity) return
    val activity = context as ComponentActivity
    activity.lifecycleScope.launch(Dispatchers.IO) {
        try {
            val db = MHealthDatabase.getInstance(context)
            val userId = DataRepository.userProfile.value?.email ?: "local_patient@lumen.health"
            
            val dailyHistory = db.dailyFeaturesDao().getAllFeatures(userId)
            val baselineRows = db.baselineDao().getBaseline(userId)
            val analysisReports = db.analysisResultDao().getAll(userId)
            val profile = db.userProfileDao().getProfile(userId)
            
            val masterJson = org.json.JSONObject()
            
            masterJson.put("profile", org.json.JSONObject().apply {
                put("userId", userId)
                put("baselineReady", profile?.baselineReady ?: false)
                put("onboardingDate", profile?.onboardingDate ?: "")
                put("currentStatus", profile?.currentStatus ?: "Collecting")
            })

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
                    put("conversationFrequency", day.conversationFrequency)
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
                    put("conversationFrequency", liveVector.conversationFrequency)
                })
                
                todayObj.put("location_snapshots", DataRepository.locationSnapshots.value.joinToString(";") { "${it.lat},${it.lon},${it.timeMs}" })
                todayObj.put("charge_hours", DataRepository.accumulatedChargeHours.value)
                todayObj.put("bg_audio_ms", DataRepository.accumulatedBgAudioMs.value)
                todayObj.put("step_baseline", DataRepository.stepBaseline.value ?: -1f)
                
                masterJson.put("today_live", todayObj)
            }

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

            // Add DNA Profile if present
            val dnaProfileJson = DataRepository.s1ProfileJson.value
            if (dnaProfileJson != null) {
                try {
                    masterJson.put("dna_profile", org.json.JSONObject(dnaProfileJson))
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }

            // Add Onboarding Calibration Data
            try {
                val localPref = context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE)
                masterJson.put("onboarding_calibration", org.json.JSONObject().apply {
                    put("phq9_score", localPref.getInt("screener_phq9", 0))
                    put("gad7_score", localPref.getInt("screener_gad7", 0))
                    put("life_events", localPref.getInt("screener_life_events", 0))
                    put("demographics", org.json.JSONObject().apply {
                        put("name", localPref.getString("user_name", ""))
                        put("gender", localPref.getString("user_gender", ""))
                        put("age", localPref.getInt("user_age", 25))
                        put("profession", localPref.getString("user_profession", ""))
                        put("country", localPref.getString("user_country", ""))
                        put("living_situation", localPref.getString("user_living_situation", ""))
                        put("is_student", localPref.getBoolean("user_is_student", false))
                    })
                    put("lifestyle_sliders", org.json.JSONObject().apply {
                        put("screen", localPref.getInt("user_lifestyle_screen", 3))
                        put("communication", localPref.getInt("user_lifestyle_communication", 3))
                        put("movement", localPref.getInt("user_lifestyle_movement", 3))
                        put("sleep", localPref.getInt("user_lifestyle_sleep", 3))
                        put("behavioral", localPref.getInt("user_lifestyle_behavioral", 3))
                        put("engagement", localPref.getInt("user_lifestyle_engagement", 3))
                        put("travel", localPref.getInt("user_lifestyle_travel", 3))
                        put("social", localPref.getInt("user_lifestyle_social", 3))
                        put("charging", localPref.getInt("user_lifestyle_charging", 3))
                        put("app_usage", localPref.getInt("user_lifestyle_app_usage", 3))
                    })
                })
            } catch (e: Exception) {
                e.printStackTrace()
            }

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
    val activity = context as ComponentActivity
    Toast.makeText(context, "Importing backup...", Toast.LENGTH_SHORT).show()

    activity.lifecycleScope.launch(Dispatchers.IO) {
        try {
            val contentResolver = context.contentResolver
            val inputStream = contentResolver.openInputStream(uri) ?: throw Exception("Cannot open file")
            val jsonString = inputStream.bufferedReader().use { it.readText() }
            val masterJson = org.json.JSONObject(jsonString)
            
            val db = MHealthDatabase.getInstance(context)
            
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

            if (masterJson.has("dna_profile")) {
                val dnaObj = masterJson.getJSONObject("dna_profile")
                val now = System.currentTimeMillis()
                db.personDnaDao().insert(com.example.mhealth.logic.db.PersonDnaEntity(
                    person_id = userId,
                    dna_json = dnaObj.toString(),
                    created_at = now,
                    last_updated = now
                ))
            }

            if (masterJson.has("onboarding_calibration")) {
                val calObj = masterJson.getJSONObject("onboarding_calibration")
                val localPref = context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE)
                localPref.edit().apply {
                    putInt("screener_phq9", calObj.optInt("phq9_score", 0))
                    putInt("screener_gad7", calObj.optInt("gad7_score", 0))
                    putInt("screener_life_events", calObj.optInt("life_events", 0))
                    
                    if (calObj.has("demographics")) {
                        val demo = calObj.getJSONObject("demographics")
                        putString("user_name", demo.optString("name", ""))
                        putString("user_gender", demo.optString("gender", ""))
                        putInt("user_age", demo.optInt("age", 25))
                        putString("user_profession", demo.optString("profession", ""))
                        putString("user_country", demo.optString("country", ""))
                        putString("user_living_situation", demo.optString("living_situation", ""))
                        putBoolean("user_is_student", demo.optBoolean("is_student", false))
                    }
                    if (calObj.has("lifestyle_sliders")) {
                        val sliders = calObj.getJSONObject("lifestyle_sliders")
                        putInt("user_lifestyle_screen", sliders.optInt("screen", 3))
                        putInt("user_lifestyle_communication", sliders.optInt("communication", 3))
                        putInt("user_lifestyle_movement", sliders.optInt("movement", 3))
                        putInt("user_lifestyle_sleep", sliders.optInt("sleep", 3))
                        putInt("user_lifestyle_behavioral", sliders.optInt("behavioral", 3))
                        putInt("user_lifestyle_engagement", sliders.optInt("engagement", 3))
                        putInt("user_lifestyle_travel", sliders.optInt("travel", 3))
                        putInt("user_lifestyle_social", sliders.optInt("social", 3))
                        putInt("user_lifestyle_charging", sliders.optInt("charging", 3))
                        putInt("user_lifestyle_app_usage", sliders.optInt("app_usage", 3))
                    }
                }.apply()
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

@Composable
fun AccessibilityDisclosureDialog(
    onDismiss: () -> Unit,
    onConfirm: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        icon = {
            Icon(
                imageVector = Icons.Default.Info,
                contentDescription = null,
                tint = TealAccent,
                modifier = Modifier.size(36.dp)
            )
        },
        title = {
            Text(
                text = "Consent for Accessibility Service",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                color = MaterialTheme.colorScheme.onSurface,
                textAlign = TextAlign.Center
            )
        },
        text = {
            Column(
                verticalArrangement = Arrangement.spacedBy(10.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
            ) {
                Text(
                    text = "Lumen utilizes Android's Accessibility Services API to monitor digital psychomotor dynamics in the background. This service acts as a secure, event-only observer.",
                    fontSize = 13.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    lineHeight = 18.sp
                )
                Text(
                    text = "What we monitor and why:",
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontFamily = Fredoka
                )
                Text(
                    text = "• Keystroke speed (characters per second) — to analyze cognitive speed and motor changes.\n" +
                           "• Backspace ratio (frequency of corrections) — to detect motor planning variation.\n" +
                           "• Scroll dynamics (velocity and direction) — to evaluate psychomotor agitation or retardation.",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    lineHeight = 17.sp
                )
                Text(
                    text = "Privacy Assurances:",
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontFamily = Fredoka
                )
                Text(
                    text = "• Lumen does NOT read, capture, or store text inputs, passwords, message content, or sensitive personal data.\n" +
                           "• We set canRetrieveWindowContent = false to programmatically guarantee privacy.\n" +
                           "• 100% Offline: All metrics are computed locally on this device. No telemetry data is transmitted to the cloud or third parties.",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    lineHeight = 17.sp
                )
                Text(
                    text = "Consent Action:",
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontFamily = Fredoka
                )
                Text(
                    text = "To consent, click 'Agree'. You will be redirected to the Android system settings. Go to Installed Apps, tap 'Lumen. Interaction Dynamics', and toggle the switch to enable it. You can disable this service at any time.",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    lineHeight = 17.sp
                )
            }
        },
        confirmButton = {
            Button(
                onClick = onConfirm,
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
            ) {
                Text("Agree", color = Color.Black, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Decline", color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Medium, fontFamily = Fredoka)
            }
        },
        shape = RoundedCornerShape(24.dp),
        containerColor = MaterialTheme.colorScheme.surface
    )
}

@Composable
fun LocationDisclosureDialog(
    onDismiss: () -> Unit,
    onConfirm: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        icon = {
            Icon(
                imageVector = Icons.Default.LocationOn,
                contentDescription = null,
                tint = TealAccent,
                modifier = Modifier.size(36.dp)
            )
        },
        title = {
            Text(
                text = "Consent for Location Tracking",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                color = MaterialTheme.colorScheme.onSurface,
                textAlign = TextAlign.Center
            )
        },
        text = {
            Column(
                verticalArrangement = Arrangement.spacedBy(10.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
            ) {
                Text(
                    text = "Lumen collects location data, including background location, to enable movement pattern tracking, spatial stability baseline estimation, daily displacement calculation, and location entropy mapping even when the app is closed or not in use.",
                    fontSize = 13.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    lineHeight = 18.sp
                )
                Text(
                    text = "How we use location:",
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontFamily = Fredoka
                )
                Text(
                    text = "• Displacement Distance: To calculate how far you travel daily to differentiate active vs. homebound states.\n" +
                           "• Location Entropy: To measure the variety of places you visit to detect behavioral changes.\n" +
                           "• Home Time Ratio: To calculate the portion of the day spent at home, which is a major indicator of behavioral routines.",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    lineHeight = 17.sp
                )
                Text(
                    text = "Why Background Access is Needed:",
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontFamily = Fredoka
                )
                Text(
                    text = "• Telemetry must be collected continuously in the background to compute daily metrics accurately.\n" +
                           "• Disrupted background tracking results in incomplete data, compromising the accuracy of routine anomaly assessments.",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    lineHeight = 17.sp
                )
                Text(
                    text = "Privacy Assurances:",
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontFamily = Fredoka
                )
                Text(
                    text = "• 100% Offline: GPS coordinates are processed entirely on-device and mapped to a general ~110m grid. Your actual coordinates are never uploaded to any server or shared with third parties.",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    lineHeight = 17.sp
                )
                Text(
                    text = "Consent Action:",
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontFamily = Fredoka
                )
                Text(
                    text = "To consent, click 'Agree'. You will first grant foreground location, and then be directed to system settings. Under Location permissions, select 'Allow all the time' to enable background tracking.",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    lineHeight = 17.sp
                )
            }
        },
        confirmButton = {
            Button(
                onClick = onConfirm,
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
            ) {
                Text("Agree", color = Color.Black, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Decline", color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Medium, fontFamily = Fredoka)
            }
        },
        shape = RoundedCornerShape(24.dp),
        containerColor = MaterialTheme.colorScheme.surface
    )
}

@Composable
fun TelemetryDisclosureDialog(
    onDismiss: () -> Unit,
    onConfirm: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        icon = {
            Icon(
                imageVector = Icons.Default.Favorite,
                contentDescription = null,
                tint = TealAccent,
                modifier = Modifier.size(36.dp)
            )
        },
        title = {
            Text(
                text = "Consent for Digital Rhythms Telemetry",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                color = MaterialTheme.colorScheme.onSurface,
                textAlign = TextAlign.Center
            )
        },
        text = {
            Column(
                verticalArrangement = Arrangement.spacedBy(10.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
            ) {
                Text(
                    text = "Lumen analyzes aggregated daily activity patterns to calibrate your behavioral wellness baseline.",
                    fontSize = 13.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    lineHeight = 18.sp
                )
                Text(
                    text = "What we analyze:",
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontFamily = Fredoka
                )
                Text(
                    text = "• Communication Dynamics (Contacts): Monitors daily contact interactions to measure social connectivity.\n" +
                           "• Calendar Engagement: Checks the count of meetings/events to monitor lifestyle structure.\n" +
                           "• Physical Activity (Activity Recognition): Tracks steps and active time to detect psychomotor agitation.\n" +
                           "• Creative Expression (Media files count): Counts gallery changes as an indicator of engagement.",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    lineHeight = 17.sp
                )
                Text(
                    text = "Privacy Assurances:",
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontFamily = Fredoka
                )
                Text(
                    text = "• We do NOT read, store, or transmit call audio, message text, contact names, phone numbers, or calendar contents.\n" +
                           "• Only daily aggregated counts (integers/durations) are calculated.\n" +
                           "• 100% Offline: No personal data ever leaves your phone.",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    lineHeight = 17.sp
                )
                Text(
                    text = "Consent Action:",
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontFamily = Fredoka
                )
                Text(
                    text = "Click 'Agree' to proceed to the system permission requests for Call Logs, Contacts, Calendar, Activity, and Storage.",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    lineHeight = 17.sp
                )
            }
        },
        confirmButton = {
            Button(
                onClick = onConfirm,
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
            ) {
                Text("Agree", color = Color.Black, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Decline", color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Medium, fontFamily = Fredoka)
            }
        },
        shape = RoundedCornerShape(24.dp),
        containerColor = MaterialTheme.colorScheme.surface
    )
}

private fun exportDataAsJson(context: Context, filePrefix: String = "mhealth_backup_before_reset_") {
    if (context !is ComponentActivity) return
    val activity = context as ComponentActivity
    activity.lifecycleScope.launch(Dispatchers.IO) {
        try {
            val userId = DataRepository.userProfile.value?.email ?: "local_patient@lumen.health"
            val backupDataStr = com.example.mhealth.logic.JsonConverter.buildBackupJson(context, userId)
            val file = java.io.File(context.getExternalFilesDir(null), "$filePrefix${System.currentTimeMillis()}.json")
            file.writeText(backupDataStr)
            withContext(Dispatchers.Main) {
                Toast.makeText(context, "Backup auto-saved to: ${file.name}", Toast.LENGTH_LONG).show()
            }
        } catch (e: Exception) {
            Log.e("MHealth", "Auto-backup failed: ${e.message}")
        }
    }
}

// =============================================================================
// Habit Quest & Anonymized Research Sharing Composables
// =============================================================================

@Composable
fun HabitQuestsSection() {
    val context = LocalContext.current
    val prefs = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }
    
    // Sequence to trigger recomposition when settings are updated in dialog
    var configSeq by remember { mutableStateOf(0) }
    var showCustomizeDialog by remember { mutableStateOf(false) }

    val sunsetEnabled = remember(configSeq) { prefs.getBoolean("habit_digital_sunset_enabled", false) }
    val sunsetTarget = remember(configSeq) { prefs.getInt("habit_digital_sunset_target", 30) }
    val sunsetStreak = remember(configSeq) { prefs.getInt("habit_digital_sunset_streak", 0) }

    val circadianEnabled = remember(configSeq) { prefs.getBoolean("habit_circadian_anchor_enabled", false) }
    val circadianTarget = remember(configSeq) { prefs.getFloat("habit_circadian_anchor_target", 23f) }
    val circadianStreak = remember(configSeq) { prefs.getInt("habit_circadian_anchor_streak", 0) }

    val movementEnabled = remember(configSeq) { prefs.getBoolean("habit_movement_boost_enabled", false) }
    val movementTarget = remember(configSeq) { prefs.getInt("habit_movement_boost_target", 6000) }
    val movementStreak = remember(configSeq) { prefs.getInt("habit_movement_boost_streak", 0) }

    val focusEnabled = remember(configSeq) { prefs.getBoolean("habit_focus_mode_enabled", false) }
    val focusTarget = remember(configSeq) { prefs.getFloat("habit_focus_mode_target", 0.20f) }
    val focusStreak = remember(configSeq) { prefs.getInt("habit_focus_mode_streak", 0) }

    val liveVector by DataRepository.latestVector.collectAsState()
    val stepsToday = liveVector?.dailyStepCount ?: 0f
    val socialRatioToday = liveVector?.socialAppRatio ?: 0f

    val sunsetUsage by produceState(0f, configSeq) {
        withContext(Dispatchers.IO) {
            value = com.example.mhealth.logic.DataCollector(context).getScreenTimeAfter9PMToday()
        }
    }

    if (showCustomizeDialog) {
        ManageHabitsDialog(
            onDismiss = {
                showCustomizeDialog = false
                configSeq++
            }
        )
    }

    val anyHabit = sunsetEnabled || circadianEnabled || movementEnabled || focusEnabled

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(24.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
    ) {
        Column(modifier = Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.EmojiEvents,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(Modifier.width(8.dp))
                    Text(
                        text = "Active Habit Quests",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                }
                TextButton(
                    onClick = { showCustomizeDialog = true },
                    contentPadding = PaddingValues(horizontal = 12.dp, vertical = 0.dp),
                    modifier = Modifier.height(28.dp)
                ) {
                    Text("Customize", fontSize = 12.sp, fontFamily = Fredoka, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary)
                }
            }

            if (!anyHabit) {
                Column(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 12.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        text = "No active habit quests.",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Medium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Text(
                        text = "Choose a micro-habit target to anchor your circadian, movement, or screen habits.",
                        fontSize = 11.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(0.7f),
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(horizontal = 16.dp)
                    )
                }
            } else {
                Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
                    if (sunsetEnabled) {
                        HabitProgressRow(
                            title = "Digital Sunset",
                            subtitle = "Keep screen time after 9 PM under ${sunsetTarget}m",
                            currentValue = sunsetUsage,
                            targetValue = sunsetTarget.toFloat(),
                            streak = sunsetStreak,
                            isLowerBetter = true,
                            unit = "min"
                        )
                    }
                    if (circadianEnabled) {
                        val displayHour = if (circadianTarget > 12f) (circadianTarget - 12f).toInt() else circadianTarget.toInt()
                        val amPm = if (circadianTarget >= 12f && circadianTarget < 24f) "PM" else "AM"
                        HabitProgressRow(
                            title = "Circadian Anchor",
                            subtitle = "Sleep before ${displayHour}:00 $amPm",
                            currentValue = liveVector?.sleepTimeHour ?: -1f,
                            targetValue = circadianTarget,
                            streak = circadianStreak,
                            isBedtime = true,
                            unit = ""
                        )
                    }
                    if (movementEnabled) {
                        HabitProgressRow(
                            title = "Movement Boost",
                            subtitle = "Walk at least ${movementTarget} steps today",
                            currentValue = stepsToday,
                            targetValue = movementTarget.toFloat(),
                            streak = movementStreak,
                            isLowerBetter = false,
                            unit = "steps"
                        )
                    }
                    if (focusEnabled) {
                        val pctToday = (socialRatioToday * 100).roundToInt()
                        val pctTarget = (focusTarget * 100).roundToInt()
                        HabitProgressRow(
                            title = "Focus Mode Ratio",
                            subtitle = "Social app screen ratio under ${pctTarget}%",
                            currentValue = socialRatioToday * 100f,
                            targetValue = focusTarget * 100f,
                            streak = focusStreak,
                            isLowerBetter = true,
                            unit = "%"
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun HabitProgressRow(
    title: String,
    subtitle: String,
    currentValue: Float,
    targetValue: Float,
    streak: Int,
    isLowerBetter: Boolean = false,
    isBedtime: Boolean = false,
    unit: String = ""
) {
    val progress = when {
        isBedtime -> {
            if (currentValue < 0f) 0.5f
            else {
                val diff = targetValue - currentValue
                if (diff >= 0f) 1.0f else 0.0f
            }
        }
        else -> {
            if (targetValue == 0f) 0f
            else (currentValue / targetValue).coerceIn(0f, 1f)
        }
    }

    val isMet = when {
        isBedtime -> {
            if (currentValue < 0f) false
            else {
                val hour = currentValue
                val target = targetValue
                if (target >= 12f) {
                    hour >= target || hour < 5f
                } else {
                    hour <= target && hour >= 0f
                }
            }
        }
        isLowerBetter -> currentValue <= targetValue
        else -> currentValue >= targetValue
    }

    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(title, fontWeight = FontWeight.Bold, fontSize = 13.sp, fontFamily = Fredoka)
                Text(subtitle, fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                if (streak > 0) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Star,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(12.dp)
                        )
                        Text(
                            text = "$streak",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.primary,
                            fontFamily = Fredoka
                        )
                    }
                }
                val statusText = if (isBedtime && currentValue < 0f) "Pending Sleep" else if (isMet) "On Track" else if (isLowerBetter) "Over Target" else "Pending"
                val statusColor = if (isBedtime && currentValue < 0f) MaterialTheme.colorScheme.onSurfaceVariant.copy(0.6f) else if (isMet) TealAccent else MaterialTheme.colorScheme.error
                Text(
                    text = statusText,
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold,
                    color = statusColor,
                    fontFamily = Fredoka,
                    modifier = Modifier
                        .background(statusColor.copy(0.12f), RoundedCornerShape(4.dp))
                        .padding(horizontal = 6.dp, vertical = 2.dp)
                )
            }
        }

        if (!isBedtime || currentValue >= 0f) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(6.dp)
                    .clip(CircleShape)
                    .background(MaterialTheme.colorScheme.outline.copy(0.15f))
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxHeight()
                        .fillMaxWidth(progress)
                        .clip(CircleShape)
                        .background(if (isMet) TealAccent else MaterialTheme.colorScheme.primary)
                )
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End
            ) {
                val currentStr = if (unit == "%") "${currentValue.roundToInt()}" else String.format(Locale.US, "%.0f", currentValue)
                val targetStr = if (unit == "%") "${targetValue.roundToInt()}" else String.format(Locale.US, "%.0f", targetValue)
                Text(
                    text = "$currentStr / $targetStr $unit",
                    fontSize = 10.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(0.7f)
                )
            }
        }
    }
}

@Composable
fun ManageHabitsDialog(onDismiss: () -> Unit) {
    val context = LocalContext.current
    val prefs = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }

    var sunsetEnabled by remember { mutableStateOf(prefs.getBoolean("habit_digital_sunset_enabled", false)) }
    var sunsetTarget by remember { mutableStateOf(prefs.getInt("habit_digital_sunset_target", 30).toFloat()) }

    var circadianEnabled by remember { mutableStateOf(prefs.getBoolean("habit_circadian_anchor_enabled", false)) }
    var circadianTarget by remember { mutableStateOf(prefs.getFloat("habit_circadian_anchor_target", 23f)) }

    var movementEnabled by remember { mutableStateOf(prefs.getBoolean("habit_movement_boost_enabled", false)) }
    var movementTarget by remember { mutableStateOf(prefs.getInt("habit_movement_boost_target", 6000).toFloat()) }

    var focusEnabled by remember { mutableStateOf(prefs.getBoolean("habit_focus_mode_enabled", false)) }
    var focusTarget by remember { mutableStateOf(prefs.getFloat("habit_focus_mode_target", 0.20f)) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text("Customize Habits", fontWeight = FontWeight.Bold, fontSize = 18.sp, fontFamily = Fredoka)
        },
        text = {
            LazyColumn(
                verticalArrangement = Arrangement.spacedBy(16.dp),
                modifier = Modifier.fillMaxWidth().heightIn(max = 400.dp)
            ) {
                item {
                    Text("Select which wellness targets to anchor and verify daily.", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }

                // Digital Sunset
                item {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text("Digital Sunset Screen Time", fontWeight = FontWeight.Bold, fontSize = 13.sp)
                                Text("Limit screen time after 9:00 PM", fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                            Switch(checked = sunsetEnabled, onCheckedChange = { sunsetEnabled = it })
                        }
                        if (sunsetEnabled) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Slider(
                                    value = sunsetTarget,
                                    onValueChange = { sunsetTarget = it },
                                    valueRange = 10f..120f,
                                    steps = 21,
                                    modifier = Modifier.weight(1f)
                                )
                                Spacer(Modifier.width(8.dp))
                                Text("${sunsetTarget.roundToInt()}m", fontSize = 12.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                            }
                        }
                    }
                }

                // Bedtime Anchor
                item {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text("Circadian Bedtime Anchor", fontWeight = FontWeight.Bold, fontSize = 13.sp)
                                Text("Keep your sleep hour before a target", fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                            Switch(checked = circadianEnabled, onCheckedChange = { circadianEnabled = it })
                        }
                        if (circadianEnabled) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Slider(
                                    value = circadianTarget,
                                    onValueChange = { circadianTarget = it },
                                    valueRange = 20f..26f, // 8 PM to 2 AM
                                    steps = 12,
                                    modifier = Modifier.weight(1f)
                                )
                                Spacer(Modifier.width(8.dp))
                                val displayHour = if (circadianTarget > 24f) (circadianTarget - 24f).roundToInt() else if (circadianTarget > 12f) (circadianTarget - 12f).roundToInt() else circadianTarget.roundToInt()
                                val amPm = if (circadianTarget >= 12f && circadianTarget < 24f) "PM" else "AM"
                                Text("${displayHour}:00 $amPm", fontSize = 12.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                            }
                        }
                    }
                }

                // Movement Boost
                item {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text("Movement Steps Boost", fontWeight = FontWeight.Bold, fontSize = 13.sp)
                                Text("Minimum daily steps count target", fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                            Switch(checked = movementEnabled, onCheckedChange = { movementEnabled = it })
                        }
                        if (movementEnabled) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Slider(
                                    value = movementTarget,
                                    onValueChange = { movementTarget = it },
                                    valueRange = 2000f..15000f,
                                    steps = 26,
                                    modifier = Modifier.weight(1f)
                                )
                                Spacer(Modifier.width(8.dp))
                                Text("${movementTarget.roundToInt()}", fontSize = 12.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                            }
                        }
                    }
                }

                // Focus Mode Ratio
                item {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text("Focus Social App Ratio", fontWeight = FontWeight.Bold, fontSize = 13.sp)
                                Text("Keep social apps usage under ratio", fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                            Switch(checked = focusEnabled, onCheckedChange = { focusEnabled = it })
                        }
                        if (focusEnabled) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Slider(
                                    value = focusTarget * 100f,
                                    onValueChange = { focusTarget = it / 100f },
                                    valueRange = 5f..50f,
                                    steps = 9,
                                    modifier = Modifier.weight(1f)
                                )
                                Spacer(Modifier.width(8.dp))
                                Text("${(focusTarget * 100).roundToInt()}%", fontSize = 12.sp, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                            }
                        }
                    }
                }
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    prefs.edit().apply {
                        putBoolean("habit_digital_sunset_enabled", sunsetEnabled)
                        putInt("habit_digital_sunset_target", sunsetTarget.roundToInt())
                        
                        putBoolean("habit_circadian_anchor_enabled", circadianEnabled)
                        putFloat("habit_circadian_anchor_target", circadianTarget)
                        
                        putBoolean("habit_movement_boost_enabled", movementEnabled)
                        putInt("habit_movement_boost_target", movementTarget.roundToInt())
                        
                        putBoolean("habit_focus_mode_enabled", focusEnabled)
                        putFloat("habit_focus_mode_target", focusTarget)
                    }.apply()
                    Toast.makeText(context, "Habit targets updated successfully!", Toast.LENGTH_SHORT).show()
                    onDismiss()
                },
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
            ) {
                Text("Save", color = Color.Black, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel", color = MaterialTheme.colorScheme.primary, fontFamily = Fredoka)
            }
        },
        shape = RoundedCornerShape(24.dp),
        containerColor = MaterialTheme.colorScheme.surface
    )
}

@Composable
fun WeeklyDigestDialog(
    weeklyFeatures: List<PersonalityVector>,
    baseline: PersonalityVector,
    onDismiss: () -> Unit
) {
    val context = LocalContext.current
    val prefs = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }

    // Find the Sunday of the week
    val weekStartStr = remember {
        val cal = Calendar.getInstance()
        cal.set(Calendar.DAY_OF_WEEK, Calendar.SUNDAY)
        SimpleDateFormat("yyyy-MM-dd", Locale.US).format(cal.time)
    }

    val reflectionKey = "weekly_reflection_$weekStartStr"
    var reflectionText by remember { mutableStateOf(prefs.getString(reflectionKey, "") ?: "") }

    // Compute stats
    val avgSleep = remember(weeklyFeatures) { weeklyFeatures.map { it.sleepDurationHours }.average().toFloat() }
    val baseSleep = baseline.sleepDurationHours
    val sleepPct = if (baseSleep > 0f) (avgSleep / baseSleep * 100).roundToInt() else 100

    val avgSteps = remember(weeklyFeatures) { weeklyFeatures.map { it.dailyStepCount }.average().toFloat() }
    val baseSteps = baseline.dailyStepCount
    val stepPct = if (baseSteps > 0f) (avgSteps / baseSteps * 100).roundToInt() else 100

    val avgScreen = remember(weeklyFeatures) { weeklyFeatures.map { it.screenTimeHours }.average().toFloat() }
    val baseScreen = baseline.screenTimeHours
    val screenPct = if (baseScreen > 0f) (avgScreen / baseScreen * 100).roundToInt() else 100

    val avgSocial = remember(weeklyFeatures) { weeklyFeatures.map { it.socialAppRatio }.average().toFloat() }
    val baseSocial = baseline.socialAppRatio
    val socialPct = if (baseSocial > 0f) (avgSocial / baseSocial * 100).roundToInt() else 100

    // Compute consistency score list for daily consistency sparkline
    val dailyScores = remember(weeklyFeatures, baseline) {
        weeklyFeatures.map { day ->
            val deviations = listOf(
                safeDev(day.sleepDurationHours, baseline.sleepDurationHours, 1.5f),
                safeDev(day.dailyStepCount, baseline.dailyStepCount, baseline.dailyStepCount.coerceAtLeast(500f)),
                safeDev(day.callsPerDay, baseline.callsPerDay, baseline.callsPerDay.coerceAtLeast(3f)),
                safeDev(day.screenTimeHours, baseline.screenTimeHours, baseline.screenTimeHours.coerceAtLeast(1f))
            )
            val avgDev = deviations.average().toFloat().coerceIn(0f, 2f)
            ((1f - avgDev / 2f) * 100f).coerceIn(0f, 100f)
        }
    }
    val avgWeeklyConsistency = remember(dailyScores) {
        if (dailyScores.isNotEmpty()) dailyScores.average().toFloat() else 100f
    }

    val dayLabels = remember(weeklyFeatures) {
        val cal = Calendar.getInstance()
        cal.add(Calendar.DAY_OF_YEAR, -(weeklyFeatures.size - 1))
        weeklyFeatures.map {
            val label = SimpleDateFormat("EEE", Locale.getDefault()).format(cal.time)
            cal.add(Calendar.DAY_OF_YEAR, 1)
            label
        }
    }

    // Load active habit configurations for habit quest completions rate
    val sunsetTarget = remember { prefs.getInt("habit_digital_sunset_target", 30) }
    val circadianTarget = remember { prefs.getFloat("habit_circadian_anchor_target", 23f) }
    val movementTarget = remember { prefs.getInt("habit_movement_boost_target", 6000) }
    val focusTarget = remember { prefs.getFloat("habit_focus_mode_target", 0.20f) }

    val sunsetEnabled = remember { prefs.getBoolean("habit_digital_sunset_enabled", false) }
    val circadianEnabled = remember { prefs.getBoolean("habit_circadian_anchor_enabled", false) }
    val movementEnabled = remember { prefs.getBoolean("habit_movement_boost_enabled", false) }
    val focusEnabled = remember { prefs.getBoolean("habit_focus_mode_enabled", false) }

    var movementHits = 0
    var circadianHits = 0
    var sunsetHits = 0
    var focusHits = 0
    weeklyFeatures.forEach { day ->
        if (day.dailyStepCount >= movementTarget) movementHits++
        if (day.sleepTimeHour > 0f && day.sleepTimeHour <= circadianTarget) {
            circadianHits++
            sunsetHits++
        }
        if (day.socialAppRatio <= focusTarget) focusHits++
    }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false, decorFitsSystemWindows = false)
    ) {
        Surface(
            modifier = Modifier.fillMaxSize(),
            color = MaterialTheme.colorScheme.background
        ) {
            val digestNavPad = rememberNavBarPadding()
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .statusBarsPadding()
                    .padding(start = 24.dp, end = 24.dp, top = 24.dp, bottom = digestNavPad + 24.dp)
            ) {
                // Header
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text(
                            text = "Weekly Digest",
                            fontSize = 24.sp,
                            fontWeight = FontWeight.ExtraBold,
                            fontFamily = Fredoka,
                            color = MaterialTheme.colorScheme.onBackground
                        )
                        Text(
                            text = "Sunday Summary report card",
                            fontSize = 12.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    IconButton(
                        onClick = onDismiss,
                        modifier = Modifier
                            .background(MaterialTheme.colorScheme.surfaceVariant.copy(0.5f), CircleShape)
                            .size(36.dp)
                    ) {
                        Icon(Icons.Default.Close, null, modifier = Modifier.size(18.dp))
                    }
                }

                Spacer(Modifier.height(20.dp))

                LazyColumn(
                    verticalArrangement = Arrangement.spacedBy(16.dp),
                    modifier = Modifier.weight(1f)
                ) {
                    // Gauge & Summary Card
                    item {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(20.dp),
                            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
                        ) {
                            Row(
                                modifier = Modifier.padding(16.dp),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(16.dp)
                            ) {
                                Box(
                                    contentAlignment = Alignment.Center,
                                    modifier = Modifier.size(80.dp)
                                ) {
                                    CircularProgressIndicator(
                                        progress = avgWeeklyConsistency / 100f,
                                        strokeWidth = 8.dp,
                                        color = MaterialTheme.colorScheme.primary,
                                        trackColor = MaterialTheme.colorScheme.surfaceVariant,
                                        modifier = Modifier.fillMaxSize()
                                    )
                                    Text(
                                        text = "${avgWeeklyConsistency.roundToInt()}%",
                                        fontSize = 16.sp,
                                        fontWeight = FontWeight.ExtraBold,
                                        color = MaterialTheme.colorScheme.primary,
                                        fontFamily = Fredoka
                                    )
                                }
                                Column(modifier = Modifier.weight(1f)) {
                                    Text(
                                        text = when {
                                            avgWeeklyConsistency >= 85f -> "Excellent Consistency"
                                            avgWeeklyConsistency >= 70f -> "Healthy Rhythm Flow"
                                            else -> "Rhythm Disruption Alert"
                                        },
                                        fontSize = 15.sp,
                                        fontWeight = FontWeight.Bold,
                                        fontFamily = Fredoka,
                                        color = MaterialTheme.colorScheme.onBackground
                                    )
                                    Spacer(Modifier.height(2.dp))
                                    Text(
                                        text = "Your aggregate lifestyle consistency average was ${avgWeeklyConsistency.roundToInt()}% for the past 7 days.",
                                        fontSize = 11.sp,
                                        lineHeight = 15.sp,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                            }
                        }
                    }

                    // Mini Daily Consistency Sparkline
                    if (dailyScores.isNotEmpty()) {
                        item {
                            MiniConsistencyBarChart(scores = dailyScores, labels = dayLabels)
                        }
                    }

                    // Active Habit Quests Completion rate
                    item {
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
                                Text(
                                    text = "Habit Quest Achievements",
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 14.sp,
                                    fontFamily = Fredoka,
                                    color = MaterialTheme.colorScheme.onBackground
                                )
                                val hasAnyHabit = sunsetEnabled || circadianEnabled || movementEnabled || focusEnabled
                                if (!hasAnyHabit) {
                                    Text(
                                        text = "No active habit quests configured. Customize them on the Home Screen to track your weekly completion rates.",
                                        fontSize = 11.sp,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                } else {
                                    if (sunsetEnabled) {
                                        WeeklyHabitStatusRow("Digital Sunset", sunsetHits, weeklyFeatures.size)
                                    }
                                    if (circadianEnabled) {
                                        WeeklyHabitStatusRow("Circadian Anchor", circadianHits, weeklyFeatures.size)
                                    }
                                    if (movementEnabled) {
                                        WeeklyHabitStatusRow("Movement Boost", movementHits, weeklyFeatures.size)
                                    }
                                    if (focusEnabled) {
                                        WeeklyHabitStatusRow("Focus Ratio", focusHits, weeklyFeatures.size)
                                    }
                                }
                            }
                        }
                    }

                    // Multi-axis score vs baseline
                    item {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(16.dp),
                            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
                        ) {
                            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                                Text("Telemetry Comparison vs Baseline", fontWeight = FontWeight.Bold, fontSize = 14.sp, fontFamily = Fredoka)
                                
                                WeeklyMetricRow("Sleep Hours", "${String.format("%.1f", avgSleep)} hrs / ${String.format("%.1f", baseSleep)} hrs", sleepPct >= 90)
                                WeeklyMetricRow("Movement Steps", "${avgSteps.roundToInt()} / ${baseSteps.roundToInt()} steps", stepPct >= 90)
                                WeeklyMetricRow("Screen Duration", "${String.format("%.1f", avgScreen)} hrs / ${String.format("%.1f", baseScreen)} hrs", screenPct <= 110)
                                WeeklyMetricRow("Social App Ratio", "${(avgSocial*100).roundToInt()}% / ${(baseSocial*100).roundToInt()}%", socialPct >= 90)
                            }
                        }
                    }

                    // Highlights & Watch Items combined
                    item {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(16.dp),
                            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primary.copy(0.02f)),
                            border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(0.08f))
                        ) {
                            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                                // Highlight Section
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Icon(
                                        imageVector = Icons.Default.EmojiEvents,
                                        contentDescription = null,
                                        tint = MaterialTheme.colorScheme.primary,
                                        modifier = Modifier.size(16.dp)
                                    )
                                    Spacer(Modifier.width(8.dp))
                                    Text("Weekly Highlight", fontWeight = FontWeight.Bold, fontSize = 13.sp, fontFamily = Fredoka, color = MaterialTheme.colorScheme.primary)
                                }
                                val highlightStr = when {
                                    stepPct > 110 -> "Your steps count was exceptionally strong this week, fueling your physical energy and circadian resilience."
                                    sleepPct > 105 -> "You secured deep, restorative sleep windows, providing ample recovery for mind and body."
                                    screenPct < 90 -> "You successfully reclaimed quiet offline spaces, significantly reducing digital eye strain and mental fatigue."
                                    else -> "You maintained a highly balanced, predictable lifestyle rhythm throughout the entire week. Fantastic consistency!"
                                }
                                Text(highlightStr, fontSize = 11.sp, lineHeight = 16.sp, color = MaterialTheme.colorScheme.onBackground.copy(0.8f))
                                
                                Divider(color = MaterialTheme.colorScheme.outline.copy(0.1f))

                                // Watch Section
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Icon(
                                        imageVector = Icons.Default.Warning,
                                        contentDescription = null,
                                        tint = AlertRose,
                                        modifier = Modifier.size(16.dp)
                                    )
                                    Spacer(Modifier.width(8.dp))
                                    Text("Rhythm Watch Item", fontWeight = FontWeight.Bold, fontSize = 13.sp, fontFamily = Fredoka, color = AlertRose)
                                }
                                val watchStr = when {
                                    screenPct > 115 -> "Screen time is elevated compared to your baseline. Introducing screen-free gaps in the afternoon could restore focus."
                                    sleepPct < 85 -> "Your sleep window is shorter than usual this week. Prioritize a regular, early bedtime to help recharge your body clock."
                                    stepPct < 80 -> "Physical steps are lower than baseline. A gentle daily 15-minute walk can help anchor your energy levels."
                                    else -> "No major circadian drifts or digital spikes detected. Continue checking in daily to maintain this healthy flow."
                                }
                                Text(watchStr, fontSize = 11.sp, lineHeight = 16.sp, color = MaterialTheme.colorScheme.onBackground.copy(0.8f))
                            }
                        }
                    }

                    // Qualitative notes reflection
                    item {
                        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text(
                                text = "Qualitative Reflection",
                                fontWeight = FontWeight.Bold,
                                fontSize = 14.sp,
                                fontFamily = Fredoka,
                                color = MaterialTheme.colorScheme.onBackground
                            )
                            OutlinedTextField(
                                value = reflectionText,
                                onValueChange = {
                                    reflectionText = it
                                    prefs.edit().putString(reflectionKey, it).apply()
                                },
                                placeholder = { Text("Write a brief qualitative note about your week (e.g. stress levels, sleep environment changes)...", fontSize = 12.sp) },
                                modifier = Modifier.fillMaxWidth().height(120.dp),
                                shape = RoundedCornerShape(12.dp)
                            )
                        }
                    }
                }

                Spacer(Modifier.height(16.dp))

                Button(
                    onClick = onDismiss,
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(10.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
                ) {
                    Text("Close Summary", color = Color.Black, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                }
            }
        }
    }
}

@Composable
fun WeeklyHabitStatusRow(title: String, hits: Int, total: Int) {
    val progress = if (total > 0) hits.toFloat() / total else 0f
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(title, fontSize = 12.sp, fontWeight = FontWeight.Medium, color = MaterialTheme.colorScheme.onBackground)
            Text("$hits/$total Days (${(progress*100).roundToInt()}%)", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary)
        }
        LinearProgressIndicator(
            progress = progress,
            modifier = Modifier.fillMaxWidth().height(6.dp).clip(RoundedCornerShape(3.dp)),
            color = MaterialTheme.colorScheme.primary,
            trackColor = MaterialTheme.colorScheme.surfaceVariant
        )
    }
}

@Composable
fun MiniConsistencyBarChart(
    scores: List<Float>,
    labels: List<String>,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                text = "Daily Rhythm Consistency Trends",
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                color = MaterialTheme.colorScheme.primary
            )
            
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(100.dp)
            ) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val count = scores.size
                    if (count > 0) {
                        val spacing = size.width / (count + 1)
                        val maxScore = 100f
                        scores.forEachIndexed { idx, score ->
                            val x = spacing * (idx + 1)
                            val barHeight = (score / maxScore) * (size.height - 20.dp.toPx())
                            val barWidth = 16.dp.toPx()
                            val top = size.height - 20.dp.toPx() - barHeight
                            
                            // Draw bar gradient
                            drawRoundRect(
                                brush = Brush.verticalGradient(
                                    colors = listOf(
                                        TealAccent,
                                        TealAccent.copy(alpha = 0.3f)
                                    )
                                ),
                                topLeft = Offset(x - barWidth / 2, top),
                                size = androidx.compose.ui.geometry.Size(barWidth, barHeight),
                                cornerRadius = androidx.compose.ui.geometry.CornerRadius(4.dp.toPx(), 4.dp.toPx())
                            )
                            
                            // Draw score text above bar
                            drawContext.canvas.nativeCanvas.apply {
                                val paint = android.graphics.Paint().apply {
                                    color = android.graphics.Color.WHITE
                                    textSize = 10.sp.toPx()
                                    textAlign = android.graphics.Paint.Align.CENTER
                                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                                }
                                drawText(
                                    "${score.roundToInt()}%",
                                    x,
                                    top - 4.dp.toPx(),
                                    paint
                                )
                            }
                        }
                    }
                }
            }
            
            // X-axis Labels Row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                labels.forEach { label ->
                    Text(
                        text = label,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.width(32.dp),
                        textAlign = TextAlign.Center
                    )
                }
            }
        }
    }
}

@Composable
fun WeeklyMetricRow(label: String, value: String, isPositive: Boolean) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(label, fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(value, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
            Text(if (isPositive) "✓" else "!", color = if (isPositive) TealAccent else MaterialTheme.colorScheme.error, fontWeight = FontWeight.Bold, fontSize = 12.sp)
        }
    }
}

@Composable
fun ResearchContributionDialog(onDismiss: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val prefs = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Default.Science,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(20.dp)
                )
                Spacer(Modifier.width(8.dp))
                Text("Support Mental Health Research", fontWeight = FontWeight.Bold, fontSize = 16.sp, fontFamily = Fredoka)
            }
        },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text(
                    "You can anonymously contribute your daily aggregated rhythm trends to help build open-source mental health models. The shared dataset strictly observes privacy guarantees:",
                    fontSize = 12.sp, lineHeight = 17.sp, color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Lock,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(16.dp)
                    )
                    Text(
                        "Privacy Protections:",
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )
                }
                Text(
                    "• Zero PII: Your name, coordinates, phone logs, and app lists are stripped entirely.\n" +
                    "• Timeline Blinding: Actual dates and timestamps are removed; daily records are indexed as Day 1, Day 2, etc.\n" +
                    "• Noise Perturbation: Small differential noise is injected into step counts (+/- 150) and screen times (+/- 10 min) to prevent tracing back to individuals.",
                    fontSize = 11.sp, lineHeight = 16.sp, color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Medium
                )
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    val formUrl = "https://docs.google.com/forms/d/e/1FAIpQLScuBGMbL17yUOdADwgrFvHj2EfMcvPLC3fOBlqmJV8PhxUuuQ/viewform?usp=sharing"
                    try {
                        val intent = Intent(Intent.ACTION_VIEW, android.net.Uri.parse(formUrl))
                        context.startActivity(intent)
                        prefs.edit().putBoolean("research_share_completed", true).apply()
                    } catch (e: Exception) {
                        Log.e("MHealth", "Failed to open form: ${e.message}")
                    }
                    onDismiss()
                },
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
            ) {
                Text("Anonymize & Share", color = Color.Black, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Decline", color = MaterialTheme.colorScheme.primary, fontFamily = Fredoka)
            }
        },
        shape = RoundedCornerShape(24.dp),
        containerColor = MaterialTheme.colorScheme.surface
    )
}

@Composable
fun WindDownCompanionCard() {
    val context = LocalContext.current
    val prefs = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }
    var configSeq by remember { mutableStateOf(0) }
    var showActiveOverlay by remember { mutableStateOf(false) }

    val enabled = remember(configSeq) { prefs.getBoolean("wind_down_enabled", false) }

    // T-D: Adaptive circadian baseline — derive personal sleep target from historical checkin data
    // instead of forcing a hard-coded 11 PM on all users.
    val sleepTarget = remember(configSeq) {
        val anchorOverride = prefs.getFloat("habit_circadian_anchor_target", -1f)
        if (anchorOverride > 0f) {
            // User has explicitly set a bedtime anchor in Habit Goals — respect that
            anchorOverride
        } else {
            // Derive average sleep-start time from checkin history (last 14 days)
            val historyStr = prefs.getString("checkin_history", null)
            if (!historyStr.isNullOrBlank()) {
                try {
                    val arr = org.json.JSONArray(historyStr)
                    val sleepTimes = mutableListOf<Float>()
                    for (i in 0 until minOf(arr.length(), 14)) {
                        val obj = arr.optJSONObject(i)
                        val st = obj?.optDouble("sleep_time", -1.0)?.toFloat() ?: -1f
                        if (st > 0f) sleepTimes.add(st)
                    }
                    if (sleepTimes.size >= 3) {
                        sleepTimes.average().toFloat() // personal baseline
                    } else 23f // not enough data yet — default 11 PM
                } catch (_: Exception) { 23f }
            } else 23f
        }
    }
    val isAdaptiveTarget = remember(configSeq) { prefs.getFloat("habit_circadian_anchor_target", -1f) < 0f }

    // Check if bedtime goal met last night
    val lastNightMet = remember {
        val lastSleepTime = prefs.getFloat("last_recorded_sleep_time_hour", -1f)
        lastSleepTime > 0f && lastSleepTime <= sleepTarget
    }

    if (showActiveOverlay) {
        WindDownOverlay(
            sleepTarget = sleepTarget,
            onDismiss = { showActiveOverlay = false }
        )
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(24.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.Bedtime,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(Modifier.width(8.dp))
                    Text(
                        text = "Wind-Down Companion",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                }
                
                Switch(
                    checked = enabled,
                    onCheckedChange = {
                        prefs.edit().putBoolean("wind_down_enabled", it).apply()
                        configSeq++
                    },
                    colors = SwitchDefaults.colors(
                        checkedThumbColor = Color.Black,
                        checkedTrackColor = MaterialTheme.colorScheme.primary
                    )
                )
            }

            val targetHour = sleepTarget.toInt()
            val targetMin = ((sleepTarget - targetHour) * 60).roundToInt()
            val targetStr = String.format("%02d:%02d %s", 
                if (targetHour > 12) targetHour - 12 else if (targetHour == 0) 12 else targetHour,
                targetMin,
                if (targetHour >= 12) "PM" else "AM"
            )

            Text(
                text = if (isAdaptiveTarget)
                    "Your personal baseline: $targetStr. Lumen calculated this from your recent sleep patterns — it adapts as your routine evolves."
                else
                    "Target bedtime: $targetStr. Your companion will help you unplug and wind down 30 minutes before sleep.",
                fontSize = 12.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                lineHeight = 17.sp
            )

            if (lastNightMet) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(MaterialTheme.colorScheme.primary.copy(0.08f))
                        .padding(horizontal = 12.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Default.Check,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(16.dp)
                    )
                    Spacer(Modifier.width(8.dp))
                    Text(
                        text = "Bedtime anchor met last night!",
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary,
                        fontFamily = Fredoka
                    )
                }
            }

            Button(
                onClick = { showActiveOverlay = true },
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(10.dp),
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary.copy(0.12f))
            ) {
                Text(
                    text = "Activate Wind-Down Mode",
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    fontSize = 13.sp
                )
            }
        }
    }
}

@Composable
fun WindDownOverlay(
    sleepTarget: Float,
    onDismiss: () -> Unit
) {
    val context = LocalContext.current
    val prefs = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }
    
    // Save target/actual notes
    var reflectionText by remember { mutableStateOf("") }
    var didLogReflection by remember { mutableStateOf(false) }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false, decorFitsSystemWindows = false)
    ) {
        Surface(
            modifier = Modifier.fillMaxSize(),
            color = Color(0xFF030712) // Extremely deep space dark blue
        ) {
            val windDownNavPad = rememberNavBarPadding()
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .statusBarsPadding()
                    .padding(start = 24.dp, end = 24.dp, top = 24.dp, bottom = windDownNavPad + 24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                // Header
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Wind-Down Mode",
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = Color(0xFF93C5FD) // Light ice blue
                    )
                    IconButton(
                        onClick = onDismiss,
                        modifier = Modifier
                            .background(Color.White.copy(0.08f), CircleShape)
                            .size(36.dp)
                    ) {
                        Icon(Icons.Default.Close, null, tint = Color.White, modifier = Modifier.size(18.dp))
                    }
                }

                Spacer(Modifier.height(24.dp))

                LazyColumn(
                    modifier = Modifier.weight(1f),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(24.dp)
                ) {
                    // Pulsing Lotus Breathing Widget
                    item {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(24.dp),
                            colors = CardDefaults.cardColors(containerColor = Color(0xFF111827)),
                            border = BorderStroke(1.dp, Color(0xFF1F2937))
                        ) {
                            Column(
                                modifier = Modifier.padding(20.dp),
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.spacedBy(16.dp)
                            ) {
                                Text(
                                    text = "Anchoring Breath",
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 15.sp,
                                    fontFamily = Fredoka,
                                    color = Color.White
                                )
                                Text(
                                    text = "Inhale deep through your nose, hold, and release slowly.",
                                    fontSize = 12.sp,
                                    textAlign = TextAlign.Center,
                                    color = Color(0xFF9CA3AF)
                                )
                                Box(
                                    modifier = Modifier
                                        .height(140.dp)
                                        .fillMaxWidth(),
                                    contentAlignment = Alignment.Center
                                ) {
                                    CalmLotusPulse()
                                }
                            }
                        }
                    }

                    // Journal Sleep Reflection
                    item {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(24.dp),
                            colors = CardDefaults.cardColors(containerColor = Color(0xFF111827)),
                            border = BorderStroke(1.dp, Color(0xFF1F2937))
                        ) {
                            Column(
                                modifier = Modifier.padding(20.dp),
                                verticalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                Text(
                                    text = "Clear Your Mind",
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 15.sp,
                                    fontFamily = Fredoka,
                                    color = Color.White
                                )
                                Text(
                                    text = "Write down any lingering tasks, worries, or thoughts keeping your mind active.",
                                    fontSize = 12.sp,
                                    color = Color(0xFF9CA3AF),
                                    lineHeight = 16.sp
                                )
                                
                                if (!didLogReflection) {
                                    OutlinedTextField(
                                        value = reflectionText,
                                        onValueChange = { reflectionText = it },
                                        placeholder = { Text("Release your thoughts here...", color = Color(0xFF4B5563), fontSize = 12.sp) },
                                        modifier = Modifier.fillMaxWidth().height(100.dp),
                                        shape = RoundedCornerShape(12.dp),
                                        colors = OutlinedTextFieldDefaults.colors(
                                            focusedTextColor = Color.White,
                                            unfocusedTextColor = Color.White,
                                            focusedBorderColor = Color(0xFF3B82F6),
                                            unfocusedBorderColor = Color(0xFF374151)
                                        )
                                    )
                                    Button(
                                        onClick = {
                                            if (reflectionText.isNotBlank()) {
                                                val todayStr = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date())
                                                prefs.edit().putString("wind_down_reflection_$todayStr", reflectionText).apply()
                                                didLogReflection = true
                                            }
                                        },
                                        modifier = Modifier.fillMaxWidth(),
                                        shape = RoundedCornerShape(10.dp),
                                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1E3A8A))
                                    ) {
                                        Text("Log Reflection", color = Color.White, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                                    }
                                } else {
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .clip(RoundedCornerShape(8.dp))
                                            .background(Color(0xFF065F46))
                                            .padding(12.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Text("✔", color = Color(0xFF34D399), fontWeight = FontWeight.Bold)
                                        Spacer(Modifier.width(8.dp))
                                        Text("Your thoughts are recorded and cleared. Rest easy.", color = Color(0xFFD1FAE5), fontSize = 12.sp)
                                    }
                                }
                            }
                        }
                    }

                    // Sleep tips/Science
                    item {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(24.dp),
                            colors = CardDefaults.cardColors(containerColor = Color(0xFF111827).copy(0.4f)),
                            border = BorderStroke(1.dp, Color(0xFF1F2937).copy(0.5f))
                        ) {
                            Column(
                                modifier = Modifier.padding(16.dp),
                                verticalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                Text(
                                    text = "Circadian Pro-Tip",
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 13.sp,
                                    color = Color(0xFF60A5FA),
                                    fontFamily = Fredoka
                                )
                                Text(
                                    text = "Your screen emits blue wavelengths that trick your brain into thinking it is daytime. Turn down brightness or put the screen away now to enable melatonin release.",
                                    fontSize = 11.sp,
                                    lineHeight = 15.sp,
                                    color = Color(0xFF9CA3AF)
                                )
                            }
                        }
                    }
                }

                Spacer(Modifier.height(16.dp))

                Button(
                    onClick = {
                        // Log bedtime event
                        val todayStr = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date())
                        val nowHour = Calendar.getInstance().get(Calendar.HOUR_OF_DAY) + (Calendar.getInstance().get(Calendar.MINUTE) / 60f)
                        prefs.edit().putFloat("last_recorded_sleep_time_hour", nowHour).apply()
                        onDismiss()
                    },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(10.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF3B82F6))
                ) {
                    Text("Set Sleep Anchor & Lock Device", color = Color.White, fontWeight = FontWeight.Bold, fontFamily = Fredoka)
                }
            }
        }
    }
}

@Composable
fun DigitalDetoxCard() {
    val context = LocalContext.current
    val prefs = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }
    var configSeq by remember { mutableStateOf(0) }
    var showActiveOverlay by remember { mutableStateOf(false) }

    val streak = remember(configSeq) { prefs.getInt("detox_streak", 0) }
    val isInterrupted = remember(configSeq) { prefs.getBoolean("detox_interrupted", false) }
    var selectedDuration by remember { mutableStateOf(15) } // default 15 minutes

    if (showActiveOverlay) {
        DigitalDetoxTimerOverlay(
            durationMinutes = selectedDuration,
            onDismiss = {
                showActiveOverlay = false
                configSeq++
            }
        )
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(24.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.PhoneAndroid,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(Modifier.width(8.dp))
                    Text(
                        text = "Digital Detox Timer",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                }
                
                if (streak > 0) {
                    Row(
                        modifier = Modifier
                            .background(MaterialTheme.colorScheme.primary.copy(0.1f), RoundedCornerShape(8.dp))
                            .padding(horizontal = 8.dp, vertical = 2.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = Icons.Default.Star,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(12.dp)
                        )
                        Spacer(Modifier.width(4.dp))
                        Text(
                            text = "$streak Streak",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.primary,
                            fontFamily = Fredoka
                        )
                    }
                }
            }

            Text(
                text = "Unplug from all screens and notifications. Lumen will track if you stay away from your phone during the timer.",
                fontSize = 12.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                lineHeight = 17.sp
            )

            if (isInterrupted) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(AlertRose.copy(0.1f))
                        .padding(horizontal = 12.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("⚠️", fontSize = 14.sp)
                        Spacer(Modifier.width(8.dp))
                        Text(
                            text = "Last detox was interrupted. Try again to rebuild your streak!",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            color = AlertRose,
                            fontFamily = Fredoka
                        )
                    }
                    IconButton(
                        onClick = {
                            prefs.edit().putBoolean("detox_interrupted", false).apply()
                            configSeq++
                        },
                        modifier = Modifier.size(24.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Close,
                            contentDescription = "Dismiss",
                            tint = AlertRose,
                            modifier = Modifier.size(14.dp)
                        )
                    }
                }
            }

            // Duration selector
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                listOf(15, 30, 45, 60).forEach { mins ->
                    val isSel = selectedDuration == mins
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .clip(RoundedCornerShape(8.dp))
                            .background(if (isSel) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant.copy(0.4f))
                            .clickable { selectedDuration = mins }
                            .padding(vertical = 8.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = "${mins}m",
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold,
                            color = if (isSel) Color.Black else MaterialTheme.colorScheme.onSurfaceVariant,
                            fontFamily = Fredoka
                        )
                    }
                }
            }

            Button(
                onClick = {
                    prefs.edit().putBoolean("detox_interrupted", false).apply()
                    showActiveOverlay = true
                },
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(10.dp),
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
            ) {
                Text(
                    text = "Start Digital Detox",
                    color = Color.Black,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    fontSize = 13.sp
                )
            }
        }
    }
}

@Composable
fun DigitalDetoxTimerOverlay(
    durationMinutes: Int,
    onDismiss: () -> Unit
) {
    val context = LocalContext.current
    val prefs = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }
    
    var timeRemainingMs by remember { mutableStateOf(durationMinutes * 60 * 1000L) }
    var detoxFinished by remember { mutableStateOf(false) }
    var isCancelled by remember { mutableStateOf(false) }

    // Start tracking in SharedPreferences
    LaunchedEffect(Unit) {
        prefs.edit()
            .putBoolean("detox_active", true)
            .putLong("detox_end_timestamp", System.currentTimeMillis() + timeRemainingMs)
            .apply()
    }

    // Register dynamic broadcast receiver for screen unlock / use
    DisposableEffect(context) {
        val receiver = object : android.content.BroadcastReceiver() {
            override fun onReceive(ctx: Context, intent: android.content.Intent) {
                if (intent.action == android.content.Intent.ACTION_USER_PRESENT || intent.action == android.content.Intent.ACTION_SCREEN_ON) {
                    val active = prefs.getBoolean("detox_active", false)
                    if (active) {
                        prefs.edit()
                            .putBoolean("detox_active", false)
                            .putBoolean("detox_interrupted", true)
                            .putInt("detox_streak", 0)
                            .apply()
                        
                        showDetoxNotification(ctx, "Detox Interrupted ⚠️", "You unlocked your phone! Reconnect with your environment.")
                        isCancelled = true
                    }
                }
            }
        }
        val filter = android.content.IntentFilter().apply {
            addAction(android.content.Intent.ACTION_USER_PRESENT)
            addAction(android.content.Intent.ACTION_SCREEN_ON)
        }
        
        androidx.core.content.ContextCompat.registerReceiver(
            context,
            receiver,
            filter,
            androidx.core.content.ContextCompat.RECEIVER_EXPORTED
        )
        
        onDispose {
            context.unregisterReceiver(receiver)
        }
    }

    // Countdown Timer Loop
    LaunchedEffect(isCancelled, detoxFinished) {
        if (!isCancelled && !detoxFinished) {
            while (timeRemainingMs > 0) {
                delay(1000L)
                timeRemainingMs -= 1000L
                val active = prefs.getBoolean("detox_active", false)
                val interrupted = prefs.getBoolean("detox_interrupted", false)
                if (!active || interrupted) {
                    isCancelled = true
                    break
                }
            }
            if (timeRemainingMs <= 0 && !isCancelled) {
                prefs.edit()
                    .putBoolean("detox_active", false)
                    .putBoolean("detox_interrupted", false)
                    .putInt("detox_streak", prefs.getInt("detox_streak", 0) + 1)
                    .apply()
                showDetoxNotification(context, "Detox Completed! 🎉", "Great job! You stayed away for $durationMinutes minutes.")
                detoxFinished = true
            }
        }
    }

    Dialog(
        onDismissRequest = {
            // Dismissing the dialog counts as interrupting unless it's finished
            if (!detoxFinished) {
                prefs.edit()
                    .putBoolean("detox_active", false)
                    .putBoolean("detox_interrupted", true)
                    .putInt("detox_streak", 0)
                    .apply()
            }
            onDismiss()
        },
        properties = DialogProperties(usePlatformDefaultWidth = false, decorFitsSystemWindows = false)
    ) {
        Surface(
            modifier = Modifier.fillMaxSize(),
            color = Color(0xFF0F172A) // Sleek slate dark background
        ) {
            val detoxNavPad = rememberNavBarPadding()
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .statusBarsPadding()
                    .padding(start = 24.dp, end = 24.dp, top = 24.dp, bottom = detoxNavPad + 24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.SpaceBetween
            ) {
                // Header
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.End
                ) {
                    IconButton(
                        onClick = {
                            if (!detoxFinished) {
                                prefs.edit()
                                    .putBoolean("detox_active", false)
                                    .putBoolean("detox_interrupted", true)
                                    .putInt("detox_streak", 0)
                                    .apply()
                            }
                            onDismiss()
                        },
                        modifier = Modifier
                            .background(Color.White.copy(0.08f), CircleShape)
                            .size(36.dp)
                    ) {
                        Icon(Icons.Default.Close, null, tint = Color.White, modifier = Modifier.size(18.dp))
                    }
                }

                // Center Content
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(24.dp),
                    modifier = Modifier.weight(1f, fill = false)
                ) {
                    Text(
                        text = if (detoxFinished) "Detox Completed!" else if (isCancelled) "Detox Interrupted" else "Digital Detox",
                        fontSize = 28.sp,
                        fontWeight = FontWeight.ExtraBold,
                        fontFamily = Fredoka,
                        color = if (detoxFinished) Color(0xFF10B981) else if (isCancelled) AlertRose else Color.White
                    )

                    Spacer(Modifier.height(16.dp))

                    // Timer Dial
                    Box(
                        contentAlignment = Alignment.Center,
                        modifier = Modifier.size(200.dp)
                    ) {
                        val progress = if (detoxFinished) 1f else if (isCancelled) 0f else timeRemainingMs.toFloat() / (durationMinutes * 60 * 1000f)
                        CircularProgressIndicator(
                            progress = progress,
                            strokeWidth = 12.dp,
                            color = if (detoxFinished) Color(0xFF10B981) else if (isCancelled) AlertRose else MaterialTheme.colorScheme.primary,
                            trackColor = Color.White.copy(0.06f),
                            modifier = Modifier.fillMaxSize()
                        )

                        val minutes = (timeRemainingMs / 1000L) / 60
                        val seconds = (timeRemainingMs / 1000L) % 60
                        Text(
                            text = if (detoxFinished) "✨" else if (isCancelled) "❌" else String.format("%02d:%02d", minutes, seconds),
                            fontSize = 36.sp,
                            fontWeight = FontWeight.ExtraBold,
                            color = Color.White,
                            fontFamily = Fredoka
                        )
                    }

                    Spacer(Modifier.height(16.dp))

                    Text(
                        text = if (detoxFinished) "Fantastic work! You have successfully reclaimed screen-free time to align your biological rhythms."
                               else if (isCancelled) "The digital detox was broken. Turn off screen and place your device face down next time."
                               else "Lock your phone and put it down. Do not turn on the screen or open any apps.",
                        fontSize = 14.sp,
                        color = Color.White.copy(0.7f),
                        textAlign = TextAlign.Center,
                        lineHeight = 20.sp,
                        modifier = Modifier.padding(horizontal = 16.dp)
                    )
                }

                // Footer Action
                Button(
                    onClick = {
                        if (!detoxFinished && !isCancelled) {
                            prefs.edit()
                                .putBoolean("detox_active", false)
                                .putBoolean("detox_interrupted", true)
                                .putInt("detox_streak", 0)
                                .apply()
                        }
                        onDismiss()
                    },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(10.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (detoxFinished) Color(0xFF10B981) else Color.White.copy(0.12f)
                    )
                ) {
                    Text(
                        text = if (detoxFinished) "Go to Dashboard" else "End Session",
                        color = if (detoxFinished) Color.Black else Color.White,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka
                    )
                }
            }
        }
    }
}

fun showDetoxNotification(context: Context, title: String, text: String) {
    val channelId = "lumen_detox_channel"
    val manager = context.getSystemService(Context.NOTIFICATION_SERVICE) as android.app.NotificationManager
    if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
        val channel = android.app.NotificationChannel(
            channelId,
            "Digital Detox",
            android.app.NotificationManager.IMPORTANCE_DEFAULT
        )
        manager.createNotificationChannel(channel)
    }
    val builder = androidx.core.app.NotificationCompat.Builder(context, channelId)
        .setSmallIcon(android.R.drawable.ic_lock_idle_lock)
        .setContentTitle(title)
        .setContentText(text)
        .setPriority(androidx.core.app.NotificationCompat.PRIORITY_DEFAULT)
        .setAutoCancel(true)
    manager.notify(99, builder.build())
}



