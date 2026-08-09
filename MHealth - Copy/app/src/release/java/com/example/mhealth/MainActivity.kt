package com.example.mhealth

import com.example.mhealth.ui.screens.*
import com.example.mhealth.ui.components.*
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
val Fredoka = FontFamily(
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
            surface = Color(0xFFEDF2F0),
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
    ACTIVITIES("Activities", Icons.Default.CheckCircle),
    INSIGHTS("Insights", Icons.Default.Timeline),
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

    val isDark = when (themeMode) {
        "light" -> false
        "system" -> isSystemInDarkTheme()
        else -> true
    }

    LumenTheme(themeMode = themeMode) {
        Surface(
            modifier = Modifier.fillMaxSize(),
            color = MaterialTheme.colorScheme.background
        ) {
            when (appState) {
                LumenNavState.ONBOARDING -> OnboardingWizard(onComplete = {
                    prefs.edit().putBoolean("first_login_complete", true).apply()
                    appState = LumenNavState.DASHBOARD
                })
                LumenNavState.DASHBOARD -> DashboardScreen()
            }
        }
    }
}

@Composable
fun DashboardScreen() {
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
        bottomBar = {
            NavigationBar(
                containerColor = MaterialTheme.colorScheme.surface,
                tonalElevation = 0.dp,
                modifier = Modifier.navigationBarsPadding()
            ) {
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
                    LumenDest.HOME -> HomeScreen(
                        onNavigateToInsights = { selectedTab = LumenDest.INSIGHTS },
                        onNavigateToActivities = { selectedTab = LumenDest.ACTIVITIES },
                        onNavigateToCheckIn = { selectedTab = LumenDest.CHECKIN }
                    )
                    LumenDest.ACTIVITIES -> ActivitiesScreen()
                    LumenDest.INSIGHTS -> InsightsScreen()
                    LumenDest.CHECKIN -> CheckInScreen()
                    LumenDest.SETTINGS -> SettingsScreen()
                }
            }
        }
    }
}

// =============================================================================
// Extracted Screens Delegate (com.example.mhealth.ui.screens.*)
// HomeScreen, ActivitiesScreen, InsightsScreen, CheckInScreen, SettingsScreen
// =============================================================================

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
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.12f))
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

fun exportDataToUri(context: Context, uri: android.net.Uri) {
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

fun importBackupDataFromJson(context: Context, uri: android.net.Uri) {
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

fun exportDataAsJson(context: Context, filePrefix: String = "mhealth_backup_before_reset_") {
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
                    Text("🛡️", fontSize = 18.sp)
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
                    Text("🔥 $streak", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary, fontFamily = Fredoka)
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

fun safeDev(current: Float, base: Float, scale: Float): Float {
    if (base <= 0f) return 0f
    val dev = kotlin.math.abs(current - base) / scale
    return dev.coerceIn(0f, 2.0f)
}

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
                                    Text("🎉", fontSize = 16.sp)
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
                                    Text("⚠️", fontSize = 16.sp)
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
                Text("🔬", fontSize = 20.sp)
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
                Text(
                    "🔒 Privacy Protections:\n" +
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
                    Text("🌙", fontSize = 18.sp)
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
                    Text("🎉", fontSize = 14.sp)
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
                    Text("📵", fontSize = 18.sp)
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
                        Text("🔥", fontSize = 12.sp)
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
            .putBoolean("detox_interrupted", false)
            .putLong("detox_end_timestamp", System.currentTimeMillis() + timeRemainingMs)
            .apply()
    }

    // Countdown Timer Loop
    LaunchedEffect(isCancelled, detoxFinished) {
        if (!isCancelled && !detoxFinished) {
            while (timeRemainingMs > 0) {
                delay(1000L)
                timeRemainingMs -= 1000L
            }
            if (timeRemainingMs <= 0 && !isCancelled) {
                prefs.edit()
                    .putBoolean("detox_active", false)
                    .putBoolean("detox_interrupted", false)
                    .putInt("detox_streak", prefs.getInt("detox_streak", 0) + 1)
                    .apply()
                showDetoxNotification(context, "Digital Detox Completed!", "Great job! You completed your $durationMinutes minute detox.")
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



