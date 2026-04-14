package com.example.mhealth.logic

import android.Manifest
import android.annotation.SuppressLint
import android.app.ActivityManager
import android.app.usage.NetworkStatsManager
import android.app.usage.StorageStatsManager
import android.app.usage.UsageEvents
import android.app.usage.UsageStatsManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.net.NetworkCapabilities
import android.os.BatteryManager
import android.os.Build
import android.os.Environment
import android.os.Looper
import android.os.Process
import android.os.StatFs
import android.os.storage.StorageManager
import android.provider.CalendarContract
import android.provider.CallLog
import android.provider.ContactsContract
import android.provider.MediaStore
import android.provider.Telephony
import android.util.Log
import com.example.mhealth.logic.db.AppSessionEntity
import com.example.mhealth.logic.db.MHealthDatabase
import com.example.mhealth.logic.db.NotificationEventEntity
import com.example.mhealth.models.LatLonPoint
import com.example.mhealth.models.PersonalityVector
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult as GmsLocationResult
import com.google.android.gms.tasks.Tasks
import java.util.Calendar
import java.util.Locale
import java.util.concurrent.TimeUnit
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.util.UUID

/**
 * DataCollector — All metrics sourced from the same Android APIs that
 * Digital Wellbeing uses internally. No hardcoded values anywhere.
 *
 * KEY DESIGN: Screen time is computed by replaying raw UsageEvents with
 * MOVE_TO_FOREGROUND / MOVE_TO_BACKGROUND pairs, not from the stale
 * queryUsageStats(INTERVAL_DAILY) aggregate (which only updates once/day).
 *
 * This is the correct, real-time method — identical to how Digital Wellbeing
 * computes the per-app screen time shown in Settings → Digital Wellbeing.
 */
class DataCollector(private val context: Context) : SensorEventListener {

    private val TAG = "MHealth.DataCollector"
    private val sensorManager = checkNotNull(context.getSystemService(SensorManager::class.java)) { "SensorManager not available" }

    private var locationCallback: LocationCallback? = null

    // Cumulative steps since device boot — we take a daily delta
    private var rawStepsSinceBoot = 0f

    // ── Adaptive GPS System ───────────────────────────────────────────────────
    // State machine that adjusts polling interval based on activity
    private val gpsStateManager = GpsStateManager(context)

    // ── Level 2 Behavioral DNA: Session Tracking ──────────────────────────────
    private val sessionScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val dateFormat = java.text.SimpleDateFormat("yyyy-MM-dd", Locale.US)
    private val recentNotificationTimes = mutableMapOf<String, Long>() // pkg → last notification epoch_ms

    /** Tracks which sessions have been persisted to avoid duplicates within a day window. */
    private val persistedSessionKeys = mutableSetOf<String>()

    init {
        DataRepository.init(context)
        // Step counter now handled by GpsStateManager for adaptive tracking
    }

    // =========================================================================
    //  Public API — called every 15 min by MonitoringService
    // =========================================================================

    fun collectSnapshot(
        locationSnapshots: List<LatLonPoint>,
        overrideStartMs: Long? = null,
        overrideEndMs: Long? = null
    ): PersonalityVector {
        val now = overrideEndMs ?: System.currentTimeMillis()
        val startOfDay = overrideStartMs ?: getStartOfDayMs()

        // Step delta — register baseline once per day, then subtract
        // Note: For historical snapshots, dailySteps might be less accurate as rawStepsSinceBoot is live,
        // but we prioritize screen time and app usage for baseline.
        DataRepository.setStepBaseline(rawStepsSinceBoot)
        val dailySteps = (rawStepsSinceBoot - (DataRepository.stepBaseline.value ?: rawStepsSinceBoot))
            .coerceAtLeast(0f)

        // === Core: parse raw UsageEvents (same source as Digital Wellbeing) ===
        val events = parseUsageEvents(startOfDay, now)
        val sleepData = calculateSleepProxy(startOfDay, now)

        val locationData  = calculateLocationMetrics(locationSnapshots, startOfDay, now)
        val comms         = collectCommunicationStats(startOfDay)
        val batteryInfo   = getBatteryInfo()
        val systemInfo    = getSystemInfo(startOfDay, now)
        val calEvents     = countCalendarEvents(startOfDay, now)
        val mediaCount    = countMediaAdded(startOfDay, now)
        val appInstalls   = countAppInstalls(startOfDay, now)
        val contacts      = countUniqueContactsToday(startOfDay)  // fix: was starred contacts
        val downloads     = countDownloads(startOfDay, now)
        val storageGB     = getStorageUsedGB()
        val appUninstalls = countAppUninstalls()
        val upiLaunches   = countUpiLaunches(events.appLaunches)
        val totalApps     = countTotalApps()

        // Notification count natively parsed from UsageEvents (Type 12)
        val notifCount = events.notificationCount.toFloat()

        Log.i(TAG, "Snapshot OK [Range: $startOfDay to $now] — screen:%.1fh unlocks:${events.unlockCount} launches:${events.launchCount} notifs:$notifCount steps:$dailySteps"
            .format(events.screenTimeMs / 3_600_000.0))

        // Background audio: use AudioManager-based accumulation from MonitoringService ticks
        val bgAudioHours = DataRepository.accumulatedBgAudioMs.value / 3_600_000f
        return PersonalityVector(
            // Digital Wellbeing primary metrics
            screenTimeHours      = events.screenTimeMs / 3_600_000f,
            unlockCount          = events.unlockCount.toFloat(),
            appLaunchCount       = events.launchCount.toFloat(),
            notificationsToday   = notifCount,
            socialAppRatio       = events.socialRatio,

            // Communication
            callsPerDay          = comms.callCount.toFloat(),
            callDurationMinutes  = comms.callDurationMinutes,
            uniqueContacts       = contacts.toFloat(),
            // conversationFrequency = avg daily events per unique contact (not duplicate of callsPerDay)
            conversationFrequency= if (contacts > 0) comms.callCount.toFloat() / contacts else comms.callCount.toFloat(),

            // Location & movement
            dailyDisplacementKm  = locationData.displacementKm,
            locationEntropy      = locationData.entropy,
            homeTimeRatio        = locationData.homeRatio,
            placesVisited        = locationData.placesCount.toFloat(),

            // Sleep proxy (from longest gap heuristics)
            wakeTimeHour         = sleepData.wakeTimeHour,
            sleepTimeHour        = sleepData.sleepTimeHour,
            sleepDurationHours   = sleepData.sleepDurationHours,
            darkDurationHours    = estimateDark(events.screenOffMs),

            // System
            chargeDurationHours  = DataRepository.accumulatedChargeHours.value,
            memoryUsagePercent   = systemInfo.memoryPercent,
            networkWifiMB        = systemInfo.wifiMB,
            networkMobileMB      = systemInfo.mobileMB,
            mediaCountToday      = mediaCount.toFloat(),
            appInstallsToday     = appInstalls.toFloat(),
            calendarEventsToday  = calEvents.toFloat(),

            // New expanded features
            downloadsToday       = downloads.toFloat(),
            storageUsedGB        = storageGB,
            appUninstallsToday   = appUninstalls.toFloat(),
            upiTransactionsToday = upiLaunches.toFloat(),
            totalAppsCount       = totalApps.toFloat(),
            backgroundAudioHours = bgAudioHours,

            dailySteps           = dailySteps,

            appBreakdown         = events.appMinutes,
            notificationBreakdown = events.notificationBreakdown,
            appLaunchesBreakdown = events.appLaunches
        )
    }

    /** Start passive continuous location tracking with adaptive intervals. */
    @SuppressLint("MissingPermission")
    fun startContinuousLocationTracking() {
        if (context.checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED &&
            context.checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            Log.w(TAG, "Location permission not granted — skipping continuous GPS capture")
            return
        }

        try {
            val fusedClient = LocationServices.getFusedLocationProviderClient(context)

            // Re-use existing callback to prevent multiple registrations
            if (locationCallback != null) return

            // ADAPTIVE GPS: Use state machine to determine polling interval
            // STATIONARY: 30 min, WALKING: 5 min, VEHICLE: 2 min
            val initialIntervalMs = gpsStateManager.getCurrentIntervalMs()
            val accuracyThreshold = gpsStateManager.getCurrentAccuracyThreshold()

            val locationRequest = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, initialIntervalMs)
                .setMinUpdateDistanceMeters(30f)  // Still require 30m movement for callback
                .setWaitForAccurateLocation(true)  // Wait for GPS fix, don't use cell/WiFi
                .build()

            locationCallback = object : LocationCallback() {
                override fun onLocationResult(result: GmsLocationResult) {
                    val loc = result.locations.lastOrNull() ?: return
                    val ageMs = System.currentTimeMillis() - loc.time
                    val currentThreshold = gpsStateManager.getCurrentAccuracyThreshold()
                    val freshnessOk = ageMs <= (15 * 60 * 1000L)

                    // State-aware accuracy filter
                    if (freshnessOk && loc.accuracy <= currentThreshold) {
                        val point = LatLonPoint(loc.latitude, loc.longitude, System.currentTimeMillis(), loc.accuracy, loc.speed)
                        DataRepository.addLocationSnapshot(point)
                        gpsStateManager.onGpsFixReceived(point)  // Update state machine

                        Log.i(TAG, "GPS fix (${gpsStateManager.getCurrentIntervalMs() / 60_000}min): %.5f, %.5f (acc: %.1fm, spd: %.1fkm/h)"
                            .format(loc.latitude, loc.longitude, loc.accuracy, loc.speed * 3.6f))
                    } else {
                        Log.w(TAG, "GPS rejected: acc=${loc.accuracy}m (need ≤${currentThreshold}m), age=${ageMs / 1000}s")
                    }
                }
            }

            locationCallback?.let { fusedClient.requestLocationUpdates(locationRequest, it, Looper.getMainLooper()) }
            Log.i(TAG, "Adaptive GPS started: STATIONARY=${gpsStateManager.getCurrentIntervalMs() / 60_000}min interval, acc≤${accuracyThreshold}m")
        } catch (e: SecurityException) {
            Log.e(TAG, "SecurityException accessing location: ${e.message}")
        } catch (e: Exception) {
            Log.w(TAG, "GPS tracking start error: ${e.message}")
        }
    }

    fun stopContinuousLocationTracking() {
        try {
            locationCallback?.let {
                LocationServices.getFusedLocationProviderClient(context).removeLocationUpdates(it)
                locationCallback = null
                gpsStateManager.unregister()
                Log.i(TAG, "Adaptive GPS tracking stopped.")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Error stopping location tracking: ${e.message}")
        }
    }

    /**
     * ADAPTIVE GPS: Proactive one-shot GPS poll called every monitoring tick.
     * Uses state-aware accuracy threshold - stationary needs tighter filter,
     * vehicle needs faster capture.
     */
    @SuppressLint("MissingPermission")
    fun captureProactiveLocationSnapshot() {
        if (context.checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED &&
            context.checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            return
        }
        try {
            val fusedClient = LocationServices.getFusedLocationProviderClient(context)
            val accuracyThreshold = gpsStateManager.getCurrentAccuracyThreshold()

            fusedClient.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, null)
                .addOnSuccessListener { loc ->
                    if (loc != null) {
                        val ageMs = System.currentTimeMillis() - loc.time
                        // State-aware accuracy filter (STATIONARY=200m, WALKING=100m, VEHICLE=50m)
                        if (ageMs <= 20 * 60 * 1000L && loc.accuracy <= accuracyThreshold) {
                            val point = LatLonPoint(loc.latitude, loc.longitude, System.currentTimeMillis(), loc.accuracy, loc.speed)
                            DataRepository.addLocationSnapshot(point)
                            gpsStateManager.onGpsFixReceived(point)

                            Log.i(TAG, "Proactive GPS (${gpsStateManager.getCurrentIntervalMs() / 60_000}min): %.5f, %.5f (acc: %.1fm, spd: %.1fkm/h)"
                                .format(loc.latitude, loc.longitude, loc.accuracy, loc.speed * 3.6f))
                        } else {
                            Log.w(TAG, "Proactive GPS rejected: age=${ageMs/1000}s acc=${loc.accuracy}m (need ≤${accuracyThreshold}m)")
                        }
                    }
                }
                .addOnFailureListener { e ->
                    Log.w(TAG, "Proactive GPS fix failed: ${e.message}")
                }
        } catch (e: Exception) {
            Log.w(TAG, "captureProactiveLocationSnapshot error: ${e.message}")
        }
    }

    /** Capture a one-shot GPS fix and save it as the user's HOME location. */
    @SuppressLint("MissingPermission")
    fun captureHomeLocation(onResult: (success: Boolean) -> Unit) {
        if (context.checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED &&
            context.checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            Log.w(TAG, "captureHomeLocation: location permission not granted")
            onResult(false)
            return
        }
        try {
            val fusedClient = LocationServices.getFusedLocationProviderClient(context)
            // Home location needs high accuracy - wait for good GPS fix
            fusedClient.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, null)
                .addOnSuccessListener { loc ->
                    if (loc != null && loc.accuracy <= 100f) {
                        DataRepository.setHomeLocation(loc.latitude, loc.longitude)
                        Log.i(TAG, "Home location saved: %.5f, %.5f (acc: %.1fm)".format(loc.latitude, loc.longitude, loc.accuracy))
                        onResult(true)
                    } else if (loc != null) {
                        Log.w(TAG, "Home location fix too inaccurate: %.1fm (need ≤100m)".format(loc.accuracy))
                        onResult(false)
                    } else {
                        Log.w(TAG, "captureHomeLocation: no location fix obtained")
                        onResult(false)
                    }
                }
                .addOnFailureListener { e ->
                    Log.e(TAG, "captureHomeLocation error: ${e.message}")
                    onResult(false)
                }
        } catch (e: Exception) {
            Log.e(TAG, "captureHomeLocation exception: ${e.message}")
            onResult(false)
        }
    }

    // =========================================================================
    //  Digital Wellbeing equivalent: parse raw UsageEvents
    //
    //  Android's Digital Wellbeing app reads UsageEvents directly from
    //  UsageStatsManager.queryEvents() and replays FOREGROUND/BACKGROUND
    //  pairs to compute live per-app durations. We do exactly the same.
    //
    //  Event types used:
    //    MOVE_TO_FOREGROUND     = 1  → app becomes visible
    //    MOVE_TO_BACKGROUND     = 2  → app goes behind
    //    SCREEN_INTERACTIVE     = 15 → screen turns on
    //    SCREEN_NON_INTERACTIVE = 16 → screen turns off
    //    KEYGUARD_HIDDEN        = 18 → screen unlocked (= 1 unlock)
    //    DEVICE_SHUTDOWN        = 26 → device shutdown
    // =========================================================================
    private data class EventsResult(
        val screenTimeMs: Long,
        val unlockCount: Int,
        val launchCount: Int,
        val socialRatio: Float,
        val screenOffMs: Long,             // total time screen was off
        val backgroundAudioMs: Long,       // duration intentional audio played in bg
        val appMinutes: Map<String, Long>, // package → foreground minutes
        val appLaunches: Map<String, Int>, // package → launch count
        val notificationCount: Int,        // total notification interruptions
        val notificationBreakdown: Map<String, Int> // package → notification count
    )

    // calculateSleepProxy is defined lower in this file (production version with Core Sleep filter)



    private val EXCLUDED_PACKAGES = setOf(
        "android",
        "com.android.systemui",
        "com.google.android.gms",
        // Stock / AOSP launchers
        "com.android.launcher",
        "com.android.launcher2",
        "com.android.launcher3",
        "com.google.android.apps.nexuslauncher",
        // Samsung
        "com.sec.android.app.launcher",
        // MIUI / Xiaomi
        "com.miui.home",
        // OnePlus / OxygenOS
        "net.oneplus.launcher",
        // Realme
        "com.realme.launcher",
        // Oppo / ColorOS
        "com.oppo.launcher",
        // Vivo / FuntouchOS
        "com.vivo.launcher",
        // Huawei
        "com.huawei.android.launcher",
        // Nokia
        "com.evenwell.powersaving.g3",
        // Sony
        "com.sonyericsson.home",
        // LG
        "com.lge.launcher3",
        context.packageName
    )

    /**
     * MUSIC-ONLY audio apps — used by MonitoringService to decide whether the
     * currently active MediaSession should count as background audio.
     *
     * Strategy (3-layer):
     *   1. Exact package match against this list (covers side-loaded / regional apps).
     *   2. Package name keyword scan ("music", "player", "fm", "radio", "audio").
     *   3. OS ApplicationInfo.CATEGORY_AUDIO flag (Android 8+).
     */
    val MUSIC_APP_PACKAGES = setOf(
        // Global / mainstream
        "com.spotify.music",
        "com.google.android.apps.youtube.music",
        "com.amazon.mp3",
        "com.soundcloud.android",
        "com.apple.android.music",             // Apple Music Android
        "com.deezer.android",
        "com.tidal.music",
        "com.pandora.android",
        "com.lastfm.android",
        // India-specific
        "com.gaana",
        "in.hungama.music",
        "com.jio.media.jiobeats",
        "com.wynk.music",
        "in.sharpcollective.app",              // JioSaavn
        // Open-source / alternative
        "com.maxmpz.audioplayer",              // Poweramp
        "org.outertune.app",
        "com.kabouzeid.gramophone",             // Gramophone
        "code.name.monkey.retromusic",          // RetroMusicPlayer
        "com.ichi2.anki",                      // Anki (excluded by keyword anyway)
        "com.mp3player.musicplayer.free",
        "com.bxtech.music",                    // BlackHole
        "io.github.muntashirakon.Music",        // Auxio
        "com.yandex.music",
        "com.naver.music.phone",               // Naver VIBE
        "com.melon.android",
        "com.bugs.android.player",             // Bugs Music
        // Podcast / radio that should count
        "tunein.player",
        "com.anchor.android",
        "com.audible.application",
        "com.acast.podcast",
        "com.podbean.app.podcast"
    )

    /** Returns true if the given package should be counted as intentional music/audio listening. */
    fun isMusicApp(pkg: String): Boolean {
        if (pkg.isBlank()) return false
        val lower = pkg.lowercase()
        // 1. Exact match
        if (lower in MUSIC_APP_PACKAGES) return true
        // 2. Keyword heuristic — catches regional / sideloaded players
        if (lower.contains("music") || lower.contains("player") ||
            lower.contains(".fm") || lower.contains("radio") ||
            lower.contains("audio") || lower.contains("podcast") ||
            lower.contains("spotify") || lower.contains("gaana") ||
            lower.contains("saavn") || lower.contains("hungama")) return true
        // 3. OS category (Android 8+)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            return try {
                val info = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    context.packageManager.getApplicationInfo(pkg, PackageManager.ApplicationInfoFlags.of(0L))
                } else {
                    context.packageManager.getApplicationInfo(pkg, 0)
                }
                info.category == ApplicationInfo.CATEGORY_AUDIO
            } catch (e: Exception) { false }
        }
        return false
    }

    private fun parseUsageEvents(startMs: Long, endMs: Long): EventsResult {
        val usm = checkNotNull(context.getSystemService(UsageStatsManager::class.java)) { "UsageStatsManager not available" }

        // Pure raw-event iteration — 100% boundary accurate (same as Digital Wellbeing)
        val events = usm.queryEvents(startMs, endMs)
        val event  = UsageEvents.Event()

        // FG/BG session tracking
        val appFgStart    = mutableMapOf<String, Long>() // pkg → foreground start timestamp
        val finalAppMs    = mutableMapOf<String, Long>() // pkg → total foreground ms
        var totalScreenMs = 0L

        val appLaunches = mutableMapOf<String, Int>()
        val lastBgAt    = mutableMapOf<String, Long>()

        var unlocks     = 0
        var launches    = 0
        var screenOffAt = startMs   // track screen-off accumulation
        var totalOffMs  = 0L
        var screenIsOn  = true      // assume screen on at query start

        var notifications    = 0
        val appNotifications = mutableMapOf<String, Int>()

        while (events.hasNextEvent()) {
            events.getNextEvent(event)
            val ts  = event.timeStamp.coerceIn(startMs, endMs)
            val pkg = event.packageName ?: ""

            when (event.eventType) {
                // ── App comes to foreground (1) ────────────────────────────────
                UsageEvents.Event.MOVE_TO_FOREGROUND -> {
                    if (!isExcluded(pkg)) {
                        appFgStart[pkg] = ts
                        val lastBg = lastBgAt[pkg] ?: 0L
                        if (lastBg == 0L || (ts - lastBg > 1500L)) {
                            launches++
                            appLaunches[pkg] = (appLaunches[pkg] ?: 0) + 1
                        }
                    }
                }

                // ── App goes to background (2) ─────────────────────────────────
                UsageEvents.Event.MOVE_TO_BACKGROUND -> {
                    if (!isExcluded(pkg)) {
                        val fgStart = appFgStart.remove(pkg)
                        if (fgStart != null && fgStart <= ts) {
                            val duration = ts - fgStart
                            finalAppMs[pkg] = (finalAppMs[pkg] ?: 0L) + duration
                            totalScreenMs  += duration
                        }
                        lastBgAt[pkg] = ts
                    }
                }

                // ── Screen Unlock (18 = KEYGUARD_HIDDEN) ───────────────────────
                18 -> unlocks++

                // ── Screen Turned ON (15 = SCREEN_INTERACTIVE) ─────────────────
                15 -> {
                    if (!screenIsOn) {
                        val gap = ts - screenOffAt
                        totalOffMs += gap
                        screenIsOn = true
                    }
                }

                // ── Screen Turned OFF (16 = SCREEN_NON_INTERACTIVE) ────────────
                16 -> {
                    if (screenIsOn) {
                        screenOffAt = ts
                        screenIsOn  = false
                    }
                }

                // ── Notification Interruption (12 = NOTIFICATION_INTERRUPTION) ──
                12 -> {
                    if (!isExcluded(pkg)) {
                        notifications++
                        appNotifications[pkg] = (appNotifications[pkg] ?: 0) + 1
                    }
                }
            }
        }

        // Calculate social interaction time
        val socialInteractionMs = finalAppMs.filterKeys { pkg ->
            isSocialApp(pkg)
        }.values.sum()

        val minutes = finalAppMs.mapValues { it.value / 60_000L }.filter { it.value > 0 }

        return EventsResult(
            screenTimeMs      = totalScreenMs,
            unlockCount       = unlocks,
            launchCount       = launches,
            socialRatio       = if (totalScreenMs > 0) socialInteractionMs.toFloat() / totalScreenMs else 0f,
            screenOffMs       = totalOffMs,
            backgroundAudioMs = 0L,
            appMinutes        = minutes,
            appLaunches       = appLaunches,
            notificationCount = notifications,
            notificationBreakdown = appNotifications
        )
    }

    private val pm = context.packageManager
    private val labelCache = mutableMapOf<String, String>()
    private val isExcludedCache = mutableMapOf<String, Boolean>()

    private fun isExcluded(pkg: String): Boolean {
        if (pkg.isBlank()) return true
        
        // Final result cache - avoids all string logic and label lookups after first hit
        isExcludedCache[pkg]?.let { return it }

        val lower = pkg.lowercase()

        // 1. Exact matches (fastest) — captures standard system pkgs and known launchers
        val result = if (lower == "android" || lower in EXCLUDED_PACKAGES) {
            true
        } else if (lower.contains("launcher") ||
            lower.contains("systemui") ||
            lower.startsWith("com.android.providers") ||
            lower.startsWith("com.android.server") ||
            lower.startsWith("com.android.permission") ||
            lower.contains("quicksearchbox")) {
            // 2. Keyword/Prefix matches (fast)
            true
        } else {
            // 3. Label-based Catch-all: fetches the display name (e.g. "Home", "System Launcher")
            // Uses a cache to avoid expensive PackageManager calls in the event loop.
            val label = labelCache[pkg] ?: try {
                val info = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    pm.getApplicationInfo(pkg, PackageManager.ApplicationInfoFlags.of(0L))
                } else {
                    pm.getApplicationInfo(pkg, 0)
                }
                val l = pm.getApplicationLabel(info).toString().lowercase(Locale.ROOT).trim()
                labelCache[pkg] = l
                l
            } catch (e: Exception) { "" }

            label == "home" || label == "launcher" || label == "system launcher" || label == "system home"
        }

        isExcludedCache[pkg] = result
        return result
    }

    /**
     * Identifies social media and communication apps for behavioral modeling.
     * Combines official Android categories with common package name patterns.
     */
    private fun isSocialApp(pkg: String): Boolean {
        return try {
            val appInfo = pm.getApplicationInfo(pkg, 0)
            val isSocialCat = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                appInfo.category == ApplicationInfo.CATEGORY_SOCIAL
            } else false
            
            val lower = pkg.lowercase()
            val isSocialName = lower.contains("facebook") || lower.contains("instagram") ||
                               lower.contains("whatsapp") || lower.contains("twitter") ||
                               lower.contains("snapchat") || lower.contains("tiktok") ||
                               lower.contains("messenger") || lower.contains("reddit") ||
                               lower.contains("telegram") || lower.contains("discord") ||
                               lower.contains("linkedin") || lower.contains("tinder")
                               
            isSocialCat || isSocialName
        } catch (e: Exception) {
            false
        }
    }

    // =========================================================================
    //  Sleep estimation — event-pair based, Core Sleep window 2AM–10AM
    // =========================================================================

    /** Convert epoch-ms → hour-of-day float  (e.g. 23:47 → 23.78) */
    private fun msToHour(ms: Long): Float {
        if (ms == 0L || ms == Long.MAX_VALUE || ms == Long.MIN_VALUE) return 0f
        val cal = Calendar.getInstance()
        cal.timeInMillis = ms
        return cal.get(Calendar.HOUR_OF_DAY) + cal.get(Calendar.MINUTE) / 60f
    }

    /**
     * Dark duration = total screen-off minutes today.
     * Uses the raw screenOffMs already computed by parseUsageEvents — no heuristics.
     */
    private fun estimateDark(screenOffMs: Long): Float {
        return (screenOffMs / 3_600_000f).coerceIn(0f, 24f)
    }

    data class SleepResult(val sleepTimeHour: Float, val wakeTimeHour: Float, val sleepDurationHours: Float)

    /**
     * 3-Signal Fusion Sleep Proxy (Unlock + DND + Screen Gap)
     * Rolling 24-hour window natively supports shift workers and naps.
     */
    private fun calculateSleepProxy(todayStartMs: Long, nowMs: Long): SleepResult {
        val usm = context.getSystemService(UsageStatsManager::class.java)
            ?: return SleepResult(0f, 0f, 0f)

        // ── Window: Fixed 18-hour overnight window (6 PM yesterday → 12 PM today relative to 'nowMs') ──
        // This excludes daytime inactivity (e.g. phone on desk at work) from ever
        // being mistaken for the primary sleep episode. Night-shift workers are
        // intentionally excluded from this heuristic for now.
        val windowCal = Calendar.getInstance()
        windowCal.timeInMillis = nowMs
        // End = 'today' 12:00 PM, or now if we haven't reached noon yet (keeps it live)
        windowCal.set(Calendar.HOUR_OF_DAY, 12)
        windowCal.set(Calendar.MINUTE, 0)
        windowCal.set(Calendar.SECOND, 0)
        windowCal.set(Calendar.MILLISECOND, 0)
        val windowEndMs = minOf(windowCal.timeInMillis, nowMs)
        
        // Start = 'yesterday' 6:00 PM
        windowCal.add(Calendar.DAY_OF_YEAR, -1)
        windowCal.set(Calendar.HOUR_OF_DAY, 18)
        windowCal.set(Calendar.MINUTE, 0)
        windowCal.set(Calendar.SECOND, 0)
        windowCal.set(Calendar.MILLISECOND, 0)
        val windowStartMs = windowCal.timeInMillis

        val events = usm.queryEvents(windowStartMs, windowEndMs)
        val ev     = UsageEvents.Event()

        // ── Step 1: Build screen sessions from raw events ─────────────────────
        data class ScreenSession(val onMs: Long, val offMs: Long)
        val sessions = mutableListOf<ScreenSession>()

        var screenOnAt  = windowStartMs
        var screenIsOn  = true  // assume screen is on at start

        while (events.hasNextEvent()) {
            events.getNextEvent(ev)
            when (ev.eventType) {
                15 -> { // SCREEN_INTERACTIVE — screen turned on
                    if (!screenIsOn) { screenOnAt = ev.timeStamp; screenIsOn = true }
                }
                16 -> { // SCREEN_NON_INTERACTIVE — screen turned off
                    if (screenIsOn) {
                        sessions.add(ScreenSession(screenOnAt, ev.timeStamp))
                        screenIsOn = false
                    }
                }
            }
        }
        // If screen is still on at end of window
        if (screenIsOn) sessions.add(ScreenSession(screenOnAt, windowEndMs))

        // ── Step 2: Identify screen-off gaps ──────────────────────────────────
        data class Gap(val startMs: Long, val endMs: Long)
        val gaps = mutableListOf<Gap>()

        if (sessions.isEmpty()) {
            gaps.add(Gap(windowStartMs, windowEndMs))
        } else {
            if (sessions[0].onMs > windowStartMs) {
                gaps.add(Gap(windowStartMs, sessions[0].onMs))
            }
            for (i in 0 until sessions.size - 1) {
                gaps.add(Gap(sessions[i].offMs, sessions[i + 1].onMs))
            }
            gaps.add(Gap(sessions.last().offMs, windowEndMs))
        }

        // ── Step 3: Merge gaps separated by short 5-minute micro-wakes ────────
        val MICRO_WAKE_MS = 5 * 60_000L  // 5 minutes
        val mergedGaps = mutableListOf<Gap>()
        var current = gaps.firstOrNull() ?: return SleepResult(0f, 0f, 0f)

        for (i in 1 until gaps.size) {
            val bridgeOnMs = gaps[i].startMs - current.endMs
            if (bridgeOnMs <= MICRO_WAKE_MS) {
                current = Gap(current.startMs, gaps[i].endMs) // Merge
            } else {
                mergedGaps.add(current)
                current = gaps[i]
            }
        }
        mergedGaps.add(current)

        // ── Step 4: Pick the longest gap (primary sleep episode) ──────────────
        val bestGap = mergedGaps.maxByOrNull { it.endMs - it.startMs }
            ?: return SleepResult(0f, 0f, 0f)

        var sleepTs = bestGap.startMs
        var wakeTs  = bestGap.endMs

        // ── Step 5: DND Fusion (Action Interruption Filter) ───────────────────
        val dndOnMs = DataRepository.dndOnMs.value
        val dndOffMs = DataRepository.dndOffMs.value
        val FORTY_FIVE_MINS = 45L * 60_000L

        // Fuse sleep time if DND was enabled today
        if (dndOnMs > 0 && kotlin.math.abs(dndOnMs - sleepTs) <= FORTY_FIVE_MINS) {
            sleepTs = (sleepTs + dndOnMs) / 2
        } else if (dndOnMs > 0 && dndOnMs > sleepTs && dndOnMs < wakeTs) {
            // Conservative: user was definitely asleep when BOTH matched
            sleepTs = maxOf(sleepTs, dndOnMs)
        }

        // Fuse wake time if DND was disabled today
        if (dndOffMs > 0 && kotlin.math.abs(dndOffMs - wakeTs) <= FORTY_FIVE_MINS) {
            wakeTs = (wakeTs + dndOffMs) / 2
        } else if (dndOffMs > 0 && dndOffMs < wakeTs && dndOffMs > sleepTs) {
            // Wake up as soon as EITHER triggered
            wakeTs = minOf(wakeTs, dndOffMs)
        }

        val durationHrs = ((wakeTs - sleepTs).toFloat() / 3_600_000f).coerceIn(0f, 16f)

        return SleepResult(
            sleepTimeHour      = msToHour(sleepTs),
            wakeTimeHour       = msToHour(wakeTs),
            sleepDurationHours = durationHrs
        )
    }

    // =========================================================================
    //  Location metrics — Haversine polyline from GPS snapshots
    // =========================================================================

    private data class LocationResult(
        val displacementKm: Float, val entropy: Float,
        val homeRatio: Float, val placesCount: Int
    )

    private fun calculateLocationMetrics(
        snaps: List<LatLonPoint>,
        startOfDayMs: Long,
        nowMs: Long
    ): LocationResult {
        // We use the full snaps array (which reset at midnight anyway) for true polyline.
        if (snaps.size < 2) return LocationResult(0f, 0f, 1f, 1)

        // ── Displacement: cell-transition + speed filter ────────────────────────────────
        // 1. Cell deduplication eliminates GPS hover/drift (two fixes in same 11m cell = 0 distance).
        // 2. Speed filter: if computed speed between two cells exceeds 6 km/h the phone is
        //    almost certainly in a vehicle. That segment is logged but NOT added to distance.
        //    6 km/h = 1.667 m/s (fast walk ≈ 5.5 km/h; anything above 6 is vehicle territory).
        val WALK_SPEED_MS = 6.0 / 3.6   // m/s

        // ADAPTIVE GPS: State-aware accuracy filter
        // STATIONARY=200m, WALKING=100m, VEHICLE=50m - tighter than before for better quality
        val accuracyThreshold = gpsStateManager.getCurrentAccuracyThreshold()
        val filteredSnaps = snaps.filter { it.accuracy <= accuracyThreshold }
        if (filteredSnaps.size < 2) return LocationResult(0f, 0f, 1f, 1)

        // Sort chronologically so delta-time between consecutive fixes is always positive.
        val sortedSnaps = filteredSnaps.sortedBy { it.timeMs }

        var distKm = 0.0
        var lastCell: String? = null
        var lastCellLat = 0.0
        var lastCellLon = 0.0
        var lastCellTimeMs = 0L

        for (snap in sortedSnaps) {
            val cell = "${"%.4f".format(snap.lat)},${"%.4f".format(snap.lon)}"
            if (lastCell == null) {
                // Anchor first cell — no distance to add yet
                lastCell = cell; lastCellLat = snap.lat; lastCellLon = snap.lon; lastCellTimeMs = snap.timeMs
            } else if (cell != lastCell) {
                // New cell reached — compute haversine and add to total.
                // Speed is computed for logcat ONLY (walk vs vehicle label).
                // ALL movement counts — walking, driving, metro, everything.
                val segmentKm    = haversine(lastCellLat, lastCellLon, snap.lat, snap.lon)
                val timeDeltaSec = (snap.timeMs - lastCellTimeMs) / 1000.0
                val speedMs      = if (timeDeltaSec > 0.0) (segmentKm * 1000.0) / timeDeltaSec else 0.0
                val modeTag      = if (speedMs > WALK_SPEED_MS) "vehicle" else "walk"

                distKm += segmentKm   // always count — total displacement includes all transport
                Log.d(TAG, "GPS segment ($modeTag): %.3fkm at %.1fkm/h".format(segmentKm, speedMs * 3.6))

                lastCell = cell; lastCellLat = snap.lat; lastCellLon = snap.lon; lastCellTimeMs = snap.timeMs
            }
        }

        // ── Time-weighted entropy (same logic as Home Time) ────────────────────
        // Instead of counting GPS pings per cell, we measure the actual wall-clock
        // time (ms) spent at each grid cell using consecutive ping timestamps.
        // Each gap is capped at 12h to prevent bridging overnight absences.
        val ENTROPY_BRIDGE_CAP_MS = 12L * 3600_000L
        val cellTimeMs = mutableMapOf<String, Long>()
        val todaySnapsForEntropy = filteredSnaps
            .filter { it.timeMs in startOfDayMs..nowMs }
            .sortedBy { it.timeMs }
        for (i in 0 until todaySnapsForEntropy.size - 1) {
            val s = todaySnapsForEntropy[i]
            // FIX: %.4f to match displacement cell resolution
            val key = "${"%.4f".format(s.lat)},${"%.4f".format(s.lon)}"
            val gap = (todaySnapsForEntropy[i + 1].timeMs - s.timeMs).coerceIn(0L, ENTROPY_BRIDGE_CAP_MS)
            cellTimeMs[key] = (cellTimeMs[key] ?: 0L) + gap
        }
        // Bridge last snap → now (keeps entropy live even if GPS goes quiet at home)
        if (todaySnapsForEntropy.isNotEmpty()) {
            val last = todaySnapsForEntropy.last()
            val key = "${"%.4f".format(last.lat)},${"%.4f".format(last.lon)}"
            val gap = (nowMs - last.timeMs).coerceIn(0L, ENTROPY_BRIDGE_CAP_MS)
            cellTimeMs[key] = (cellTimeMs[key] ?: 0L) + gap
        }
        val totalTimeMs = cellTimeMs.values.sum().toDouble()
        val entropy = if (totalTimeMs > 0.0) {
            cellTimeMs.values.sumOf { t ->
                val p = t / totalTimeMs
                -p * ln(p)
            }.toFloat()
        } else 0f

        // ── homeTimeRatio ──────────────────────────────────────────────────────
        // Formula: wall-clock ms within 500m of home / 86_400_000 (24h)
        //
        // KEY FIXES vs original:
        //  A. Budget-phone fix: homeSnaps uses accuracy ≤ 800m (not 200m).
        //     Many budget phones report 250-400m accuracy — the 200m filter was
        //     rejecting EVERY fix on those devices, giving homeRatio = 0 all day.
        //     For home-detection we only need ~500m precision, so 800m is safe.
        //  B. Overnight bridge fix: midnight → first-snap was previously anchored
        //     on today's first snap. If the patient left home at 07:00 and the
        //     first fix is 07:05 OUTSIDE home, all 7 hours of sleep were missed.
        //     Now we use yesterday's last GPS fix (saved just before midnight reset)
        //     as the true overnight anchor.
        //  C. Zero-snap fallback: if no GPS fixes pass even the 800m filter today,
        //     use yesterday's last location to estimate if the patient is still home.
        val homeLat = DataRepository.getHomeLatitude()
        val homeLon = DataRepository.getHomeLongitude()

        val homeRatio: Float
        if (homeLat != null && homeLon != null) {
            val BRIDGE_CAP_MS = 12L * 3600_000L
            val DAY_MS        = 24L * 3600_000L

            fun isNearHome(s: LatLonPoint) = haversine(s.lat, s.lon, homeLat, homeLon) < 0.5

            // FIX A: Relaxed accuracy filter (800m) for home-detection only.
            // Displacement still uses the strict 200m filter — unaffected.
            val homeSnaps = snaps
                .filter { it.timeMs in startOfDayMs..nowMs && it.accuracy <= 800f }
                .sortedBy { it.timeMs }

            // FIX B: Yesterday's last known fix — the true overnight anchor.
            val lastNight = DataRepository.getLastLocationBeforeMidnight()

            var homeTimeMs = 0L

            if (homeSnaps.isEmpty()) {
                // FIX C: Zero GPS day. If last night's fix was near home, the
                // patient probably hasn't gone anywhere. Conservatively count
                // midnight → now (capped at 12h) as home time.
                if (lastNight != null && isNearHome(lastNight)) {
                    homeTimeMs = minOf(nowMs - startOfDayMs, BRIDGE_CAP_MS)
                    Log.d(TAG, "homeTimeRatio: no GPS today, using last-night anchor (near home)")
                }
            } else {
                // Step 1: midnight → first snap of today.
                // Use lastNight's location as anchor if available; otherwise fall
                // back to today's first snap (old behaviour, less accurate).
                val firstSnap = homeSnaps.first()
                val midnightNearHome = when {
                    lastNight != null -> isNearHome(lastNight)   // accurate
                    else              -> isNearHome(firstSnap)   // fallback
                }
                if (midnightNearHome) {
                    homeTimeMs += minOf(firstSnap.timeMs - startOfDayMs, BRIDGE_CAP_MS)
                }

                // Step 2: consecutive snap pairs
                for (i in 0 until homeSnaps.size - 1) {
                    if (isNearHome(homeSnaps[i])) {
                        val gap = homeSnaps[i + 1].timeMs - homeSnaps[i].timeMs
                        homeTimeMs += minOf(gap, BRIDGE_CAP_MS)
                    }
                }

                // Step 3: last snap → now (real-time update)
                val lastSnap = homeSnaps.last()
                if (isNearHome(lastSnap)) {
                    homeTimeMs += minOf(nowMs - lastSnap.timeMs, BRIDGE_CAP_MS)
                }
            }

            val elapsedDayMs = (nowMs - startOfDayMs).coerceAtLeast(1L)
            homeRatio = (homeTimeMs.toFloat() / elapsedDayMs).coerceIn(0f, 1f)
        } else {
            // Home location not set — fall back to cell with most time spent
            val maxCellMs = cellTimeMs.maxByOrNull { it.value }?.value?.toDouble() ?: 0.0
            homeRatio = if (totalTimeMs > 0.0) (maxCellMs / totalTimeMs).toFloat().coerceIn(0f, 1f) else 0f
        }

        // placesVisited = distinct %.4f grid cells (~11m) where actual time was logged.
        // Finer resolution means two rooms in the same building are counted separately.
        val placesVisited = if (cellTimeMs.isNotEmpty()) cellTimeMs.size else (if (lastCell != null) 1 else 0)
        return LocationResult(distKm.toFloat(), entropy, homeRatio, placesVisited)
    }

    private fun haversine(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val R = 6371.0
        val dLat = Math.toRadians(lat2 - lat1)
        val dLon = Math.toRadians(lon2 - lon1)
        val a = sin(dLat/2).pow(2) +
                cos(Math.toRadians(lat1)) * cos(Math.toRadians(lat2)) * sin(dLon/2).pow(2)
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))
    }

    // =========================================================================
    //  Communication — ContentResolver (call log + SMS)
    // =========================================================================

    private data class CommStats(val callCount: Int, val callDurationMinutes: Float)

    private fun collectCommunicationStats(since: Long): CommStats {
        var calls = 0
        var totalDurationSeconds = 0f
        try {
            context.contentResolver.query(
                CallLog.Calls.CONTENT_URI, arrayOf(CallLog.Calls.NUMBER, CallLog.Calls.DURATION),
                "${CallLog.Calls.DATE} >= ?", arrayOf(since.toString()), null
            )?.use { cursor ->
                calls = cursor.count
                val durIndex = cursor.getColumnIndex(CallLog.Calls.DURATION)
                if (durIndex != -1) {
                    while (cursor.moveToNext()) {
                        totalDurationSeconds += cursor.getLong(durIndex).toFloat()
                    }
                }
            }
        } catch (e: Exception) { Log.e(TAG, "Comm error: ${e.message}") }
        return CommStats(calls, totalDurationSeconds / 60f)
    }

    private fun countUniqueContactsToday(startOfDay: Long): Int {
        val uniqueNumbers = mutableSetOf<String>()
        try {
            context.contentResolver.query(
                CallLog.Calls.CONTENT_URI,
                arrayOf(CallLog.Calls.NUMBER),
                "${CallLog.Calls.DATE} >= ?",
                arrayOf(startOfDay.toString()), null
            )?.use { cursor ->
                val numIndex = cursor.getColumnIndex(CallLog.Calls.NUMBER)
                while (cursor.moveToNext()) {
                    val num = cursor.getString(numIndex)?.replace("\\s".toRegex(), "") ?: continue
                    if (num.isNotBlank()) uniqueNumbers.add(num)
                }
            }
        } catch (e: Exception) { Log.e(TAG, "UniqueContacts error: ${e.message}") }
        return uniqueNumbers.size
    }

    // Kept for backward compatibility — counts starred (favourite) contacts overall
    private fun countStarredContacts(): Int = try {
        context.contentResolver.query(
            ContactsContract.Contacts.CONTENT_URI,
            arrayOf(ContactsContract.Contacts._ID),
            "${ContactsContract.Contacts.STARRED} = 1", null, null
        )?.use { it.count } ?: 0
    } catch (e: Exception) { 0 }

    // =========================================================================
    //  Battery — BatteryManager broadcast (same source as Settings)
    // =========================================================================

    data class BatteryResult(val level: Float, val isCharging: Boolean)

    fun getBatteryInfo(): BatteryResult {
        val intent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val lvl    = intent?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale  = intent?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
        val pct    = if (scale > 0) lvl * 100f / scale else 0f
        val status = intent?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
        val isCharging = status == BatteryManager.BATTERY_STATUS_CHARGING ||
                         status == BatteryManager.BATTERY_STATUS_FULL
        return BatteryResult(pct, isCharging)
    }

    // =========================================================================
    //  System — Storage (StatFs) + Memory (ActivityManager) + Network (NetworkStatsManager)
    // =========================================================================

    private data class SystemResult(
        val storagePercent: Float, val memoryPercent: Float,
        val wifiMB: Float, val mobileMB: Float
    )

    private fun getSystemInfo(startMs: Long, endMs: Long): SystemResult {
        // Storage — same as Settings → Storage
        val stat  = StatFs(Environment.getDataDirectory().path)
        val total = stat.blockCountLong * stat.blockSizeLong
        val avail = stat.availableBlocksLong * stat.blockSizeLong
        val storagePct = if (total > 0) (total - avail) * 100f / total else 0f

        // Memory — same as Settings → Memory
        val am = checkNotNull(context.getSystemService(ActivityManager::class.java)) { "ActivityManager not available" }
        val mem  = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        val memPct = if (mem.totalMem > 0) (mem.totalMem - mem.availMem) * 100f / mem.totalMem else 0f

        // Network — same as Settings → Network & Internet → Data usage
        var wifiMB = 0f; var mobileMB = 0f
        try {
            val nsm = checkNotNull(context.getSystemService(NetworkStatsManager::class.java)) { "NetworkStatsManager not available" }
            nsm.querySummaryForDevice(NetworkCapabilities.TRANSPORT_WIFI, null, startMs, endMs)
                .let { wifiMB = (it.rxBytes + it.txBytes) / (1024f * 1024f) }
            nsm.querySummaryForDevice(NetworkCapabilities.TRANSPORT_CELLULAR, null, startMs, endMs)
                .let { mobileMB = (it.rxBytes + it.txBytes) / (1024f * 1024f) }
        } catch (e: Exception) { Log.e(TAG, "Network stats: ${e.message}") }

        return SystemResult(storagePct, memPct, wifiMB, mobileMB)
    }

    private fun countMediaAdded(since: Long, untilMs: Long): Int = try {
        context.contentResolver.query(
            MediaStore.Files.getContentUri("external"),
            arrayOf(MediaStore.Files.FileColumns._ID),
            "${MediaStore.Files.FileColumns.DATE_ADDED} >= ? AND ${MediaStore.Files.FileColumns.DATE_ADDED} <= ?",
            arrayOf((since / 1000).toString(), (untilMs / 1000).toString()), null
        )?.use { it.count } ?: 0
    } catch (e: Exception) { 0 }

    private fun countAppInstalls(since: Long, untilMs: Long): Int = try {
        val packages = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            context.packageManager.getInstalledPackages(PackageManager.PackageInfoFlags.of(0L))
        } else {
            context.packageManager.getInstalledPackages(0)
        }
        packages.count { it.firstInstallTime in since..untilMs }
    } catch (e: Exception) { 0 }

    private fun countCalendarEvents(startMs: Long, endMs: Long): Int = try {
        context.contentResolver.query(
            CalendarContract.Events.CONTENT_URI,
            arrayOf(CalendarContract.Events._ID),
            "${CalendarContract.Events.DTSTART} >= ? AND ${CalendarContract.Events.DTSTART} <= ?",
            arrayOf(startMs.toString(), endMs.toString()), null
        )?.use { it.count } ?: 0
    } catch (e: Exception) { 0 }

    // =========================================================================
    //  Step counter — TYPE_STEP_COUNTER sensor
    //  Raw value is cumulative since boot; we subtract the morning baseline
    //  captured in DataRepository.setStepBaseline() to get today's delta.
    // =========================================================================
    override fun onSensorChanged(event: SensorEvent?) {
        if (event?.sensor?.type == Sensor.TYPE_STEP_COUNTER) {
            rawStepsSinceBoot = event.values[0]
        }
    }
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    // =========================================================================
    //  Helpers
    // =========================================================================
    fun getStartOfDayMs(): Long {
        return Calendar.getInstance().apply {
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }.timeInMillis
    }

    // ── New feature collectors ─────────────────────────────────────────────────

    private fun countDownloads(since: Long, untilMs: Long): Int = try {
        // Scientifically bypass MediaStore index lags by checking the physical filesystem layer natively:
        val dir = android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_DOWNLOADS)
        var count = 0
        dir.listFiles()?.forEach { file ->
            if (file.isFile && file.lastModified() in since..untilMs) count++
        }
        count
    } catch (e: Exception) { 0 }

    private fun getStorageUsedGB(): Float {
        return try {
            val stat  = StatFs(Environment.getDataDirectory().path)
            val total = stat.blockCountLong * stat.blockSizeLong
            val avail = stat.availableBlocksLong * stat.blockSizeLong
            ((total - avail) / (1024f * 1024f * 1024f))
        } catch (e: Exception) { 0f }
    }

    private fun countAppUninstalls(): Int = try {
        val prefs = context.getSharedPreferences("mhealth_prefs", Context.MODE_PRIVATE)
        val currentCount = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            context.packageManager.getInstalledPackages(PackageManager.PackageInfoFlags.of(0L)).size
        } else {
            context.packageManager.getInstalledPackages(0).size
        }
        val prevKey = "prev_pkg_count"
        val storedCount = prefs.getInt(prevKey, currentCount)
        prefs.edit().putInt(prevKey, currentCount).apply()
        val removed = (storedCount - currentCount).coerceAtLeast(0)
        removed
    } catch (e: Exception) { 0 }

    private val UPI_PACKAGES = listOf(
        "com.google.android.apps.nbu.paisa.user",   // Google Pay
        "net.one97.paytm",                           // Paytm
        "com.phonepe.app",                           // PhonePe
        "in.amazon.mShop.android.shopping",          // Amazon Pay
        "com.mobikwik_new",                          // MobiKwik
        "com.freecharge.android",                    // FreeCharge
        "com.myairtelapp",                           // Airtel Thanks
        "com.boi_mobile",                            // Bank of India Mobile
        "com.sbi.upi",                               // SBI Pay / YONO
        "com.axis.mobile",                           // Axis Mobile
        "com.csam.icici.bank.imobile",               // ICICI iMobile
        "com.hdfcbank.payzapp",                      // HDFC PayZapp
        "com.dreamplug.androidapp"                   // CRED
    )

    private fun countUpiLaunches(appLaunches: Map<String, Int>): Int =
        appLaunches.filterKeys { pkg -> UPI_PACKAGES.any { pkg.startsWith(it) } }.values.sum()

    private fun countTotalApps(): Int {
        return try {
            val packages = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                context.packageManager.getInstalledPackages(PackageManager.PackageInfoFlags.of(0L))
            } else {
                context.packageManager.getInstalledPackages(0)
            }
            packages.count { pkg ->
                val isSystemApp = (pkg.applicationInfo?.flags ?: 0) and ApplicationInfo.FLAG_SYSTEM != 0
                !isSystemApp
            }
        } catch (e: Exception) {
            0
        }
    }

    // ── Category-based storage breakdown ──────────────────────────────────────
    // Returns map of category label → storage used in GB for installed apps
    fun getStorageByCategory(): Map<String, Float> {
        val result = mutableMapOf<String, Float>()
        val pm = context.packageManager

        val packages = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            pm.getInstalledPackages(PackageManager.PackageInfoFlags.of(0L))
        } else {
            pm.getInstalledPackages(0)
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val ssm = context.getSystemService(StorageStatsManager::class.java)
            val storageUuid = StorageManager.UUID_DEFAULT

            for (pkg in packages) {
                // Efficiently skip system-level packages so OS Bloat doesn't skew metric
                val isSystemApp = (pkg.applicationInfo?.flags ?: 0) and ApplicationInfo.FLAG_SYSTEM != 0
                if (isSystemApp) continue

                var usedBytes = 0L
                if (ssm != null) {
                    try {
                        val stats = ssm.queryStatsForPackage(storageUuid, pkg.packageName, android.os.Process.myUserHandle())
                        usedBytes = stats.appBytes + stats.cacheBytes + stats.dataBytes
                    } catch (e: Exception) {
                        try { usedBytes = java.io.File(pkg.applicationInfo?.sourceDir ?: "").length() } catch (_: Exception) {}
                    }
                } else {
                    try { usedBytes = java.io.File(pkg.applicationInfo?.sourceDir ?: "").length() } catch (_: Exception) {}
                }

                if (usedBytes < 1_000_000L) continue // skip < 1MB
                val label = getCategoryLabel(pkg.applicationInfo, pm)
                // Scientifically exact to OS Settings: use decimal GB (1_000_000_000f) rather than binary GiB
                result[label] = (result[label] ?: 0f) + (usedBytes / 1_000_000_000f)
            }
        }
        return result.toList().sortedByDescending { it.second }.take(6).toMap()
    }

    // ── Category-based app installs today ─────────────────────────────────────
    // Returns map of category label → count of apps installed today
    fun getAppInstallsByCategory(since: Long): Map<String, Int> {
        val result = mutableMapOf<String, Int>()
        val pm = context.packageManager
        val packages = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            pm.getInstalledPackages(PackageManager.PackageInfoFlags.of(0L))
        } else {
            pm.getInstalledPackages(0)
        }
        for (pkg in packages) {
            if (pkg.firstInstallTime < since) continue
            val label = getCategoryLabel(pkg.applicationInfo, pm)
            result[label] = (result[label] ?: 0) + 1
        }
        return result
    }

    // ── Total installed apps by category (all Play Store apps, not just today) ─
    // Returns map of category label → total count of installed apps in that category
    fun getAllAppsByCategory(): Map<String, Int> {
        val result = mutableMapOf<String, Int>()
        val pm = context.packageManager
        val packages = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            pm.getInstalledPackages(PackageManager.PackageInfoFlags.of(0L))
        } else {
            pm.getInstalledPackages(0)
        }
        for (pkg in packages) {
            // Only count user-installed apps (exclude system apps)
            val isSystemApp = (pkg.applicationInfo?.flags ?: 0) and ApplicationInfo.FLAG_SYSTEM != 0
            if (isSystemApp) continue
            val label = getCategoryLabel(pkg.applicationInfo, pm)
            result[label] = (result[label] ?: 0) + 1
        }
        // Sort descending by count
        return result.toList().sortedByDescending { it.second }.toMap()
    }

    // =========================================================================
    //  Level 2 Behavioral DNA — Session Logging
    // =========================================================================

    /**
     * Record notification time for trigger detection.
     * Called when a notification is received for a package.
     */
    fun recordNotificationTime(pkg: String, timestampMs: Long = System.currentTimeMillis()) {
        recentNotificationTimes[pkg] = timestampMs
    }

    /**
     * Parse UsageEvents and log per-session data to app_sessions Room table.
     * Called alongside collectSnapshot() to build the session history needed for DNA.
     *
     * Trigger detection:
     *   NOTIFICATION = session opened within 10s of notification from same package
     *   SELF = all other cases
     */
    fun logSessionsFromEvents(startMs: Long, endMs: Long) {
        val usm = try {
            context.getSystemService(UsageStatsManager::class.java) ?: return
        } catch (e: Exception) { return }

        val events = usm.queryEvents(startMs, endMs)
        val event = UsageEvents.Event()

        val db = MHealthDatabase.getInstance(context)
        val sessionDao = db.appSessionDao()

        // Track FG start times per package
        val appFgStart = mutableMapOf<String, Long>()
        val appInteractionCount = mutableMapOf<String, Int>()

        val sessionsToInsert = mutableListOf<AppSessionEntity>()
        val notifEventsToInsert = mutableListOf<NotificationEventEntity>()

        while (events.hasNextEvent()) {
            events.getNextEvent(event)
            val ts = event.timeStamp.coerceIn(startMs, endMs)
            val pkg = event.packageName ?: ""

            when (event.eventType) {
                UsageEvents.Event.MOVE_TO_FOREGROUND -> {
                    if (!isExcluded(pkg)) {
                        appFgStart[pkg] = ts
                        appInteractionCount[pkg] = 1
                    }
                }
                UsageEvents.Event.MOVE_TO_BACKGROUND -> {
                    if (!isExcluded(pkg)) {
                        val fgStart = appFgStart.remove(pkg)
                        val interactions = appInteractionCount.remove(pkg) ?: 1
                        if (fgStart != null && fgStart <= ts) {
                            // Determine trigger: check both local map and DataRepository (NLS-sourced)
                            val lastNotifTime = maxOf(
                                recentNotificationTimes[pkg] ?: 0L,
                                DataRepository.getRecentNotificationTime(pkg)
                            )
                            val trigger = if (lastNotifTime > 0 && kotlin.math.abs(fgStart - lastNotifTime) <= 10_000L) {
                                "NOTIFICATION"
                            } else {
                                "SELF"
                            }
                            // Clear used notification time to prevent re-detection
                            if (trigger == "NOTIFICATION") {
                                DataRepository.clearRecentNotificationTime(pkg)
                            }
                            val dateStr = synchronized(dateFormat) {
                                dateFormat.format(ts)
                            }
                            sessionsToInsert.add(
                                AppSessionEntity(
                                    session_id = UUID.randomUUID().toString(),
                                    app_package = pkg,
                                    open_timestamp = fgStart,
                                    close_timestamp = ts,
                                    trigger = trigger,
                                    interaction_count = interactions,
                                    date = dateStr
                                )
                            )
                            // Log TAP notification event when session was triggered by notification
                            if (trigger == "NOTIFICATION") {
                                val latencyMs = fgStart - lastNotifTime
                                notifEventsToInsert.add(
                                    NotificationEventEntity(
                                        event_id = UUID.randomUUID().toString(),
                                        app_package = pkg,
                                        arrival_timestamp = lastNotifTime,
                                        action = "TAP",
                                        tap_latency_min = latencyMs / 60_000f,
                                        date = dateStr
                                    )
                                )
                            }
                        }
                    }
                }
                // Count user interactions within a session (launches, config changes)
                23 -> { // USER_INTERACTION event type
                    if (!isExcluded(pkg) && appFgStart.containsKey(pkg)) {
                        appInteractionCount[pkg] = (appInteractionCount[pkg] ?: 1) + 1
                    }
                }
                // Track notification times for trigger detection (both local + NLS via DataRepository)
                12 -> { // NOTIFICATION_INTERRUPTION
                    if (!isExcluded(pkg)) {
                        recentNotificationTimes[pkg] = ts
                        // Also push to shared DataRepository so NLS and DataCollector are in sync
                        DataRepository.setRecentNotificationTime(pkg, ts)
                    }
                }
            }
        }

        // Insert all sessions asynchronously
        if (sessionsToInsert.isNotEmpty()) {
            sessionScope.launch {
                try {
                    sessionDao.insertAll(sessionsToInsert)
                    Log.d(TAG, "Logged ${sessionsToInsert.size} app sessions to Room")
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to log sessions: ${e.message}")
                }
            }
        }

        // Insert notification TAP events asynchronously
        if (notifEventsToInsert.isNotEmpty()) {
            sessionScope.launch {
                try {
                    val notifDao = db.notificationEventDao()
                    notifDao.insertAll(notifEventsToInsert)
                    Log.d(TAG, "Logged ${notifEventsToInsert.size} notification TAP events to Room")
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to log notification events: ${e.message}")
                }
            }
        }
    }

    private fun getCategoryLabel(info: ApplicationInfo?,
                                  pm: PackageManager): String {
        if (info == null) return "Other"
        val name = info.packageName?.lowercase() ?: ""
        // Name-based heuristics (Catches side-loaded apps or mismatched manifest OS categories)
        return when {
            name.contains("game") || name.contains("pubg") || name.contains("clash") 
                || name.contains("candy") || name.contains("chess") || name.contains("roblox")
                || name.contains("minecraft") || name.contains("callofduty") || name.contains("freefire")
                || name.contains("mihoyo") || name.contains("riotgames") || name.contains("ludo") -> "Games"
            name.contains("pay") || name.contains("upi") || name.contains("bank")
                || name.contains("wallet") || name.contains("phonepe")
                || name.contains("paytm") -> "Finance"
            name.contains("instagram") || name.contains("whatsapp")
                || name.contains("twitter") || name.contains("facebook")
                || name.contains("snapchat") || name.contains("tiktok")
                || name.contains("telegram") || name.contains("discord") -> "Social"
            name.contains("youtube") || name.contains("netflix")
                || name.contains("spotify") || name.contains("hotstar")
                || name.contains("prime") || name.contains("music") -> "Media"
            name.contains("camera") || name.contains("photo")
                || name.contains("gallery") || name.contains("editor") -> "Photos"
            name.contains("health") || name.contains("fitness")
                || name.contains("steps") || name.contains("workout") -> "Health"
            else -> if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                when (info.category) {
                    ApplicationInfo.CATEGORY_GAME -> "Games"
                    ApplicationInfo.CATEGORY_SOCIAL -> "Social"
                    ApplicationInfo.CATEGORY_PRODUCTIVITY -> "Productivity"
                    ApplicationInfo.CATEGORY_MAPS -> "Maps"
                    ApplicationInfo.CATEGORY_NEWS -> "News"
                    ApplicationInfo.CATEGORY_AUDIO -> "Media"
                    ApplicationInfo.CATEGORY_VIDEO -> "Media"
                    ApplicationInfo.CATEGORY_IMAGE -> "Photos"
                    else -> "Other"
                }
            } else "Other"
        }
    }
}
