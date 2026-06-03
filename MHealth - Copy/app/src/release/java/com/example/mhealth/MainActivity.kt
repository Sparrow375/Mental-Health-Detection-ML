package com.example.mhealth

import android.Manifest
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
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
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
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.graphics.toArgb
import androidx.compose.material.icons.automirrored.filled.ShowChart

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        
        // Synchronously initialize the local data repository
        DataRepository.init(applicationContext)
        
        setContent {
            LumenAppShell()
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
fun MainLumenDashboard() {
    var selectedTab by remember { mutableStateOf(LumenDest.HOME) }
    val context = LocalContext.current
    
    val perms = buildList {
        addAll(listOf(
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
                    LumenDest.HOME -> HomeScreen(onNavigateToCheckIn = { selectedTab = LumenDest.CHECKIN })
                    LumenDest.INSIGHTS -> InsightsScreen()
                    LumenDest.CHECKIN -> CheckInScreen()
                    LumenDest.SETTINGS -> SettingsScreen()
                }
            }
        }
    }
}

// =============================================================================
// Home Screen Composable
// =============================================================================
@Composable
fun HomeScreen(onNavigateToCheckIn: () -> Unit) {
    val context = LocalContext.current
    val userProfile by DataRepository.userProfile.collectAsState()
    val latestResult by DataRepository.latestAnalysisResult.collectAsState()
    val provisional by DataRepository.provisionalAnalysis.collectAsState()
    val isBuilding by DataRepository.isBuildingBaseline.collectAsState()
    
    val activeResult = provisional ?: latestResult
    val score = activeResult?.effectiveScore ?: -1f
    
    val name = (userProfile?.name ?: "").trim()
    val greeting = getGreeting()
    
    val statusText = when {
        isBuilding -> "Getting to know your rhythms..."
        score < 0f -> "Getting to know your rhythms..."
        score < 0.25f -> "Your routines feel steady this week."
        score < 0.38f -> "Your rhythms look mostly balanced."
        score < 0.50f -> "A few patterns feel slightly off this week."
        else -> "Some changes detected. A check-in might help."
    }
    
    val prefs = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }
    val activeStreak = remember(prefs) { getActiveStreak(prefs) }
    
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
    var showLocationDisclosure by remember { mutableStateOf(false) }
    val homeLocationState by DataRepository.homeLocation.collectAsState()
    val isHomeSet = homeLocationState != null

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

    val hasMissingPermissions = !isNotificationAccessGranted || !isLocationPermissionGranted || !isBackgroundLocationGranted || !isHomeSet

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        horizontalAlignment = Alignment.CenterHorizontally,
        contentPadding = PaddingValues(24.dp),
        verticalArrangement = Arrangement.spacedBy(24.dp)
    ) {
        item {
            Column(
                modifier = Modifier.fillMaxWidth(),
                horizontalAlignment = Alignment.Start
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.padding(bottom = 6.dp)
                ) {
                    Text(
                        text = "L U M E N",
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.primary,
                        letterSpacing = 2.sp
                    )
                    Spacer(Modifier.width(2.dp))
                    Box(
                        modifier = Modifier
                            .size(4.dp)
                            .clip(CircleShape)
                            .background(MaterialTheme.colorScheme.primary)
                    )
                }
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

        if (!isReminderDismissed && hasMissingPermissions) {
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
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 16.dp),
                contentAlignment = Alignment.Center
            ) {
                CalmLotusPulse()
            }
        }
        
        item {
            Text(
                text = statusText,
                fontSize = 18.sp,
                fontWeight = FontWeight.Medium,
                fontFamily = Fredoka,
                color = MaterialTheme.colorScheme.onBackground,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(horizontal = 16.dp)
            )
        }
        
        item {
            StaggeredFadeIn(index = 1) {
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
                                Text("🔥", fontSize = 24.sp)
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
        
        item {
            StaggeredFadeIn(index = 2) {
                Card(
                    onClick = onNavigateToCheckIn,
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(20.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primary.copy(0.08f)),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(0.15f))
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
                                    .background(MaterialTheme.colorScheme.primary.copy(0.15f)),
                                contentAlignment = Alignment.Center
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Favorite,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.primary,
                                    modifier = Modifier.size(20.dp)
                                )
                            }
                            Spacer(Modifier.width(16.dp))
                            Column {
                                Text(
                                    text = "How are you feeling today?",
                                    fontSize = 15.sp,
                                    fontWeight = FontWeight.Bold,
                                    fontFamily = Fredoka,
                                    color = MaterialTheme.colorScheme.primary
                                )
                                Text(
                                    text = "Take 30 seconds to log your current state.",
                                    fontSize = 12.sp,
                                    color = MaterialTheme.colorScheme.onBackground.copy(0.7f)
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

// =============================================================================
// Insights Screen Composable
// =============================================================================
@Composable
fun InsightsScreen() {
    val weeklyFeatures by DataRepository.weeklyFeatureHistory.collectAsState()
    val baseline by DataRepository.baseline.collectAsState()
    val isDnaReady by DataRepository.isDnaBaselineReady.collectAsState()
    val context = LocalContext.current
    
    val prefs = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }
    val showInsights = weeklyFeatures.size >= 2
    
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentPadding = PaddingValues(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            Column(modifier = Modifier.fillMaxWidth()) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.padding(bottom = 6.dp)
                ) {
                    Text(
                        text = "L U M E N",
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.primary,
                        letterSpacing = 2.sp
                    )
                    Spacer(Modifier.width(2.dp))
                    Box(
                        modifier = Modifier
                            .size(4.dp)
                            .clip(CircleShape)
                            .background(MaterialTheme.colorScheme.primary)
                    )
                }
                Text(
                    text = "Your Rhythms",
                    fontSize = 24.sp,
                    fontWeight = FontWeight.ExtraBold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Text(
                    text = "A gentle look at how your week has been flowing",
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
                                Text("🔬", fontSize = 24.sp)
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
                // 1. Mood Trend Card
                val avgMood = getWeeklyCheckinAverageMood(prefs)
                if (avgMood > 0f) {
                    item {
                        StaggeredFadeIn(index = 0) {
                            val moodMsg = when {
                                avgMood >= 4.0f -> "Your mood has been consistently positive this week."
                                avgMood >= 3.0f -> "Your mood has been mostly stable this week."
                                else -> "Your mood has been on the lower side this week. Remember to take things slow."
                            }
                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(16.dp),
                                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primary.copy(0.05f)),
                                border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(0.1f))
                            ) {
                                Row(
                                    modifier = Modifier.padding(16.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Text("🌟", fontSize = 24.sp)
                                    Spacer(Modifier.width(16.dp))
                                    Column {
                                        Text(
                                            text = "Weekly Mood: %.1f / 5".format(avgMood),
                                            fontSize = 15.sp,
                                            fontWeight = FontWeight.Bold,
                                            fontFamily = Fredoka,
                                            color = MaterialTheme.colorScheme.primary
                                        )
                                        Text(
                                            text = moodMsg,
                                            fontSize = 12.sp,
                                            color = MaterialTheme.colorScheme.onBackground.copy(0.7f),
                                            modifier = Modifier.padding(top = 2.dp)
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
                
                // 2. Sparkline Trend Chart
                item {
                    StaggeredFadeIn(index = 1) {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(16.dp),
                            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
                        ) {
                            Column(modifier = Modifier.padding(16.dp)) {
                                Text(
                                    text = "Rhythm Consistency",
                                    fontSize = 14.sp,
                                    fontWeight = FontWeight.Bold,
                                    fontFamily = Fredoka,
                                    color = MaterialTheme.colorScheme.onBackground
                                )
                                Text(
                                    text = "A rolling depiction of routine adherence. A stable, flat trend highlights healthy routine consistency.",
                                    fontSize = 11.5.sp,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    modifier = Modifier.padding(top = 4.dp, bottom = 12.dp)
                                )
                                
                                Box(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .height(100.dp)
                                ) {
                                    QualitativeTrendChart(weeklyFeatures)
                                }
                                
                                Spacer(Modifier.height(8.dp))
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Text("Previous 7 Days", fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    Text("Today", fontSize = 10.sp, color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Bold)
                                }
                            }
                        }
                    }
                }
                
                // 3. 4 Qualitative Insight Cards
                
                // Sleep Card
                item {
                    val sleepDiff = latest.sleepDurationHours - base.sleepDurationHours
                    val badgeText = when {
                        sleepDiff > 1.5f -> "Elongated"
                        sleepDiff < -1.5f -> "Contracted"
                        else -> "Balanced"
                    }
                    val badgeColor = when {
                        Math.abs(sleepDiff) > 1.5f -> MaterialTheme.colorScheme.error
                        else -> MaterialTheme.colorScheme.primary
                    }
                    val desc = when {
                        sleepDiff > 1.5f -> "Your sleep duration is significantly longer than your typical baseline. This might represent vegetative hypersomnia or a withdrawal state."
                        sleepDiff < -1.5f -> "Your sleep cycle is compressed. Restricting rest or waking too early can decay cognitive resilience."
                        else -> "Sleep durations and DND silent gaps are stable, demonstrating strong circular time consistency."
                    }
                    
                    StaggeredFadeIn(index = 2) {
                        QualitativeInsightCard(
                            title = "Sleep & Circadian Alignment",
                            icon = Icons.Default.NightsStay,
                            badgeText = badgeText,
                            badgeColor = badgeColor,
                            description = desc
                        )
                    }
                }
                
                // Movement Card
                item {
                    val stepRatio = if (base.dailyStepCount > 0) latest.dailyStepCount / base.dailyStepCount else 1.0f
                    val dispRatio = if (base.dailyDisplacementKm > 0) latest.dailyDisplacementKm / base.dailyDisplacementKm else 1.0f
                    val activeRatio = if (base.dailyStepCount > 0) stepRatio else dispRatio
                    
                    val badgeText = when {
                        activeRatio < 0.6f -> "Reduced"
                        activeRatio > 1.4f -> "Elevated"
                        else -> "Stable"
                    }
                    val badgeColor = when {
                        activeRatio < 0.6f -> MaterialTheme.colorScheme.error
                        activeRatio > 1.4f -> AlertWarning
                        else -> MaterialTheme.colorScheme.primary
                    }
                    val desc = when {
                        activeRatio < 0.6f -> "Physical activity is notably lower than your locked routine. Consider introducing small active windows to promote circulation."
                        activeRatio > 1.4f -> "Physical activity is highly elevated. Active pacing or energetic exercise has been registered."
                        else -> "Mobility levels and physical tracking features match your standard behavior metrics."
                    }
                    
                    StaggeredFadeIn(index = 3) {
                        QualitativeInsightCard(
                            title = "Physical Mobility",
                            icon = Icons.Default.DirectionsRun,
                            badgeText = badgeText,
                            badgeColor = badgeColor,
                            description = desc
                        )
                    }
                }
                
                // Social Card
                item {
                    val callDiff = latest.callsPerDay - base.callsPerDay
                    val badgeText = when {
                        callDiff < -2.0f -> "Low Engagement"
                        else -> "Stable Outbound"
                    }
                    val badgeColor = when {
                        callDiff < -2.0f -> MaterialTheme.colorScheme.error
                        else -> MaterialTheme.colorScheme.primary
                    }
                    val desc = when {
                        callDiff < -2.0f -> "We observed a significant retraction in calls and contact frequencies. Maintaining active connection guards against emotional withdrawal."
                        else -> "Call logs, unique contacts, and conversation frequency variables remain steady."
                    }
                    
                    StaggeredFadeIn(index = 4) {
                        QualitativeInsightCard(
                            title = "Relational Frequency",
                            icon = Icons.Default.Call,
                            badgeText = badgeText,
                            badgeColor = badgeColor,
                            description = desc
                        )
                    }
                }
                
                // Screen Card
                item {
                    val screenDiff = latest.screenTimeHours - base.screenTimeHours
                    val badgeText = when {
                        screenDiff > 2.0f -> "Increased Screen"
                        screenDiff < -2.0f -> "Reduced Screen"
                        else -> "Within Norms"
                    }
                    val badgeColor = when {
                        Math.abs(screenDiff) > 2.0f -> AlertWarning
                        else -> MaterialTheme.colorScheme.primary
                    }
                    val desc = when {
                        screenDiff > 2.0f -> "Digital interaction is elevated. Extended evening engagement or quick unlock pickup bursts can indicate restlessness."
                        screenDiff < -2.0f -> "Screen interactions have contracted. This highlights lower digital dependence or reduced social apps ratio."
                        else -> "Daily screen time hours, lock counts, and app session metrics are steady."
                    }
                    
                    StaggeredFadeIn(index = 5) {
                        QualitativeInsightCard(
                            title = "Digital Interaction Dynamics",
                            icon = Icons.Default.Smartphone,
                            badgeText = badgeText,
                            badgeColor = badgeColor,
                            description = desc
                        )
                    }
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
        modifier = Modifier.fillMaxWidth(),
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
// Check In Screen Composable
// =============================================================================
@Composable
fun CheckInScreen() {
    val context = LocalContext.current
    var subTab by remember { mutableStateOf(0) }
    val prefs = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 24.dp, vertical = 12.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "L U M E N",
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.primary,
                    letterSpacing = 2.sp
                )
                Spacer(Modifier.width(2.dp))
                Box(
                    modifier = Modifier
                        .size(4.dp)
                        .clip(CircleShape)
                        .background(MaterialTheme.colorScheme.primary)
                )
            }
        }
        TabRow(
            selectedTabIndex = subTab,
            containerColor = MaterialTheme.colorScheme.surface,
            contentColor = MaterialTheme.colorScheme.primary,
            indicator = { tabPositions ->
                TabRowDefaults.SecondaryIndicator(
                    modifier = Modifier.tabIndicatorOffset(tabPositions[subTab]),
                    color = MaterialTheme.colorScheme.primary
                )
            }
        ) {
            Tab(
                selected = subTab == 0,
                onClick = { subTab = 0 },
                text = { Text("Daily Check-in", fontWeight = FontWeight.ExtraBold, fontFamily = Fredoka) }
            )
            Tab(
                selected = subTab == 1,
                onClick = { subTab = 1 },
                text = { Text("Monthly Check-in", fontWeight = FontWeight.ExtraBold, fontFamily = Fredoka) }
            )
        }
        
        Box(modifier = Modifier.weight(1f)) {
            when (subTab) {
                0 -> DailyCheckinTab(prefs)
                1 -> MonthlyCheckinTab(prefs)
            }
        }
    }
}

@Composable
fun DailyCheckinTab(prefs: SharedPreferences) {
    val todayStr = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date())
    var lastCheckinDate by remember { mutableStateOf(prefs.getString("daily_checkin_date_last", "") ?: "") }
    
    val alreadyCheckedIn = lastCheckinDate == todayStr
    
    if (alreadyCheckedIn) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Box(
                modifier = Modifier
                    .size(80.dp)
                    .clip(CircleShape)
                    .background(MaterialTheme.colorScheme.primary.copy(0.1f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Default.CheckCircle,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(48.dp)
                )
            }
            Spacer(Modifier.height(24.dp))
            Text(
                text = "You've checked in today!",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                color = MaterialTheme.colorScheme.onBackground
            )
            Spacer(Modifier.height(8.dp))
            Text(
                text = "Thank you for taking time to log your routines. See you tomorrow!",
                fontSize = 13.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(horizontal = 24.dp)
            )
        }
    } else {
        var mood by remember { mutableIntStateOf(3) }
        var energy by remember { mutableIntStateOf(3) }
        var sleep by remember { mutableIntStateOf(3) }
        var anxiety by remember { mutableIntStateOf(3) }
        
        val scrollState = rememberScrollState()
        
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(scrollState)
                .padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Text(
                text = "Daily Reflection",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Fredoka,
                color = MaterialTheme.colorScheme.onBackground
            )
            Text(
                text = "Tune in to your body and mind. How are you feeling right now?",
                fontSize = 13.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            
            Spacer(Modifier.height(8.dp))
            
            PremiumCheckinSlider(
                question = "How has your mood been lately?",
                value = mood,
                labels = listOf("Very Low", "Low", "Neutral", "Good", "Great"),
                onValueChange = { mood = it }
            )
            
            PremiumCheckinSlider(
                question = "How is your energy level?",
                value = energy,
                labels = listOf("Exhausted", "Low", "Neutral", "Active", "Energized"),
                onValueChange = { energy = it }
            )
            
            PremiumCheckinSlider(
                question = "How well did you sleep last night?",
                value = sleep,
                labels = listOf("Terrible", "Poor", "Neutral", "Good", "Excellent"),
                onValueChange = { sleep = it }
            )
            
            PremiumCheckinSlider(
                question = "How anxious have you been feeling?",
                value = anxiety,
                labels = listOf("Tense", "Anxious", "Neutral", "Calm", "Very Peaceful"),
                onValueChange = { anxiety = it }
            )
            
            Spacer(Modifier.height(16.dp))
            
            Button(
                onClick = {
                    recordDailyCheckin(prefs, mood, energy, sleep, anxiety)
                    saveCheckinToHistory(prefs, mood, energy, sleep, anxiety)
                    lastCheckinDate = todayStr
                },
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(52.dp),
                shape = RoundedCornerShape(12.dp)
            ) {
                Text(
                    text = "Save Check-in",
                    color = Color.Black,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    fontSize = 15.sp
                )
            }
        }
    }
}

@Composable
fun PremiumCheckinSlider(
    question: String,
    value: Int,
    labels: List<String>,
    onValueChange: (Int) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(0.1f))
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = question,
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onBackground,
                fontFamily = Fredoka
            )
            Spacer(Modifier.height(16.dp))
            
            Slider(
                value = value.toFloat(),
                onValueChange = { onValueChange(it.roundToInt()) },
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
            
            Spacer(Modifier.height(8.dp))
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                labels.forEachIndexed { index, label ->
                    val isSelected = index + 1 == value
                    Text(
                        text = label,
                        fontSize = 11.sp,
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
fun MonthlyCheckinTab(prefs: SharedPreferences) {
    var activeWizard by remember { mutableStateOf(false) }
    val cooldownDays = remember(activeWizard) { getMonthlyCooldownDays(prefs) }
    
    val phq9Answers = remember { mutableStateListOf(*Array(9) { -1 }) }
    val gad7Answers = remember { mutableStateListOf(*Array(7) { -1 }) }
    
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
    val optionsList = listOf("Not at all", "Several days", "More than half the days", "Nearly every day")
    
    var wizardStep by remember { mutableIntStateOf(1) }
    
    if (activeWizard) {
        if (wizardStep == 1) {
            ScreenerWizard(
                questions = phq9Questions,
                answers = phq9Answers,
                options = optionsList,
                onCompleted = { wizardStep = 2 }
            )
        } else {
            ScreenerWizard(
                questions = gad7Questions,
                answers = gad7Answers,
                options = optionsList,
                onCompleted = {
                    val totalPhq = phq9Answers.sum()
                    val totalGad = gad7Answers.sum()
                    val todayStr = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date())
                    
                    DataRepository.saveScreenerScores(totalPhq, totalGad, DataRepository.recentLifeEventsCount.value)
                    prefs.edit().putString("monthly_checkin_last_date", todayStr).apply()
                    
                    activeWizard = false
                    wizardStep = 1
                }
            )
        }
    } else {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (cooldownDays > 0) {
                Box(
                    modifier = Modifier
                        .size(100.dp)
                        .clip(CircleShape)
                        .background(MaterialTheme.colorScheme.primary.copy(0.08f)),
                    contentAlignment = Alignment.Center
                ) {
                    Text("⏳", fontSize = 40.sp)
                }
                Spacer(Modifier.height(24.dp))
                Text(
                    text = "Assessment Cooldown",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    text = "Lumen only requests a detailed wellness check-in every 30 days.\nYour next check-in will be available in $cooldownDays days.",
                    fontSize = 13.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = TextAlign.Center,
                    lineHeight = 18.sp
                )
            } else {
                Box(
                    modifier = Modifier
                        .size(100.dp)
                        .clip(CircleShape)
                        .background(MaterialTheme.colorScheme.primary.copy(0.1f)),
                    contentAlignment = Alignment.Center
                ) {
                    Text("📋", fontSize = 40.sp)
                }
                Spacer(Modifier.height(24.dp))
                Text(
                    text = "Monthly Reflection Check-in",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = Fredoka,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    text = "A detailed wellness assessment to help Lumen recalibrate its tracking sensitivity and align with your baseline trends.",
                    fontSize = 13.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.padding(horizontal = 16.dp),
                    lineHeight = 18.sp
                )
                Spacer(Modifier.height(32.dp))
                Button(
                    onClick = { activeWizard = true },
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(52.dp),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Text(
                        text = "Start Assessment",
                        color = Color.Black,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        fontSize = 15.sp
                    )
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
    var homeCapturing by remember { mutableStateOf(false) }

    val prefs = remember(context) { context.getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE) }
    
    var themeMode by remember { mutableStateOf(prefs.getString("app_theme_mode", "dark") ?: "dark") }
    var showThemeDialog by remember { mutableStateOf(false) }
    
    var masterTracking by remember { mutableStateOf(prefs.getBoolean("master_tracking_enabled", true)) }
    var locationTracking by remember { mutableStateOf(prefs.getBoolean("location_tracking_enabled", true)) }
    var communicationLogs by remember { mutableStateOf(prefs.getBoolean("communication_logs_enabled", true)) }
    
    var dailyReminders by remember { mutableStateOf(prefs.getBoolean("daily_reminders_enabled", true)) }
    var monthlyReminders by remember { mutableStateOf(prefs.getBoolean("monthly_reminders_enabled", true)) }

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
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.padding(bottom = 6.dp)
                ) {
                    Text(
                        text = "L U M E N",
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = Fredoka,
                        color = MaterialTheme.colorScheme.primary,
                        letterSpacing = 2.sp
                    )
                    Spacer(Modifier.width(2.dp))
                    Box(
                        modifier = Modifier
                            .size(4.dp)
                            .clip(CircleShape)
                            .background(MaterialTheme.colorScheme.primary)
                    )
                }
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
                                locPermissionLauncher.launch(
                                    arrayOf(
                                        Manifest.permission.ACCESS_FINE_LOCATION,
                                        Manifest.permission.ACCESS_COARSE_LOCATION
                                    )
                                )
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
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.Home, null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(18.dp))
                            Spacer(Modifier.width(8.dp))
                            Text(
                                text = "✓ Home GPS set: %.4f, %.4f".format(lat, lon),
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
                        4 -> "Lifestyle Calibration"
                        5 -> "Health Context"
                        6 -> "Set your home"
                        7 -> "System Permissions"
                        8 -> "Personal Well-being Survey"
                        9 -> "Daily Calmness & Reflection Checklist"
                        10 -> "Have you been through anything big lately?"
                        else -> "Calibration Completed"
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
                        4 -> "Calibrate telemetry weights to match your personal baseline profile"
                        5 -> "Provides lifestyle baseline context to customize thresholds"
                        6 -> "Used locally to evaluate daily time spent at home"
                        7 -> "Lumen runs passively offline and requires permissions to collect telemetry"
                        8 -> "Answer honestly to help establish your personal well-being baseline"
                        9 -> "Helps Lumen calibrate tracking sensitivities to keep you safe"
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
                            "Lumen operates 100% locally and offline. It gathers passive behavioral telemetry—such as sleep patterns, physical movement, typing speeds, and social frequency—to construct a personalized Digital DNA and assist your wellness journey.",
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
                            Text("Rate your typical phone usage and activities from 1 to 5 to calibrate anomaly model weights.", fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
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
                                    locPermissionLauncher.launch(
                                        arrayOf(
                                            Manifest.permission.ACCESS_FINE_LOCATION,
                                            Manifest.permission.ACCESS_COARSE_LOCATION
                                        )
                                    )
                                },
                                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.secondary),
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(50.dp),
                                shape = RoundedCornerShape(12.dp)
                            ) {
                                Text("Grant GPS Location Permission", color = Color.White, fontFamily = Fredoka)
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
                        }
                    }
                }
                7 -> {
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

                        if (showLocationDisclosure) {
                            item {
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

                        if (showTelemetryDisclosure) {
                            item {
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
                        }

                        if (showAccessibilityDisclosure) {
                            item {
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
                                val totalPhq = phq9Answers.sum()
                                val totalGad = gad7Answers.sum()
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
                        text = if (step == 10) "Calibrate" else "Next",
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
            Row(verticalAlignment = Alignment.CenterVertically) {
                Slider(
                    value = value,
                    onValueChange = onValueChange,
                    valueRange = 1f..5f,
                    steps = 3,
                    colors = SliderDefaults.colors(thumbColor = MaterialTheme.colorScheme.primary),
                    modifier = Modifier.weight(1f)
                )
                Spacer(Modifier.width(12.dp))
                Text("${value.toInt()}/5", fontWeight = FontWeight.Bold, fontSize = 14.sp, color = MaterialTheme.colorScheme.primary)
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
        delay((index * 100).toLong())
        visible = true
    }
    
    val alpha by animateFloatAsState(
        targetValue = if (visible) 1f else 0f,
        animationSpec = tween(durationMillis = 600),
        label = "FadeAlpha"
    )
    val translationY by animateFloatAsState(
        targetValue = if (visible) 0f else 40f,
        animationSpec = tween(durationMillis = 600, easing = EaseOut),
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

fun saveCheckinToHistory(prefs: SharedPreferences, mood: Int, energy: Int, sleep: Int, anxiety: Int) {
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
            }
            list.add(newObj)
        }
        
        val trimmedList = if (list.size > 7) list.takeLast(7) else list
        val newArray = org.json.JSONArray()
        trimmedList.forEach { newArray.put(it) }
        
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
                    text = "Lumen requires access to location services, including background location, to monitor your movement behaviors and establish spatial stability baselines.",
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

