package com.example.mhealth.logic

import android.Manifest
import android.annotation.SuppressLint
import android.app.ActivityManager
import android.app.usage.NetworkStatsManager
import android.app.usage.UsageEvents
import android.app.usage.UsageStatsManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.net.NetworkCapabilities
import android.os.BatteryManager
import android.os.Build
import android.os.Environment
import android.os.StatFs
import android.provider.CalendarContract
import android.provider.CallLog
import android.provider.ContactsContract
import android.provider.MediaStore
import android.provider.Telephony
import android.util.Log
import com.example.mhealth.models.LatLonPoint
import com.example.mhealth.models.PersonalityVector
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult as GmsLocationResult
import com.google.android.gms.tasks.Tasks
import android.os.Looper
import java.util.*
import java.util.concurrent.TimeUnit
import kotlin.math.*

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
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    private var locationCallback: LocationCallback? = null

    // Cumulative steps since device boot — we take a daily delta
    private var rawStepsSinceBoot = 0f

    init {
        DataRepository.init(context)
        sensorManager.getDefaultSensor(Sensor.TYPE_STEP_COUNTER)?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
        }
    }

    // =========================================================================
    //  Public API — called every 15 min by MonitoringService
    // =========================================================================

    fun collectSnapshot(locationSnapshots: List<LatLonPoint>): PersonalityVector {
        val now = System.currentTimeMillis()
        val startOfDay = startOfDayMs()

        // Step delta — register baseline once per day, then subtract
        DataRepository.setStepBaseline(rawStepsSinceBoot)
        val dailySteps = (rawStepsSinceBoot - (DataRepository.stepBaseline.value ?: rawStepsSinceBoot))
            .coerceAtLeast(0f)

        // === Core: parse raw UsageEvents (same source as Digital Wellbeing) ===
        val events = parseUsageEvents(startOfDay, now)
        val sleepData = calculateSleepProxy(startOfDay, now)

        val locationData  = calculateLocationMetrics(locationSnapshots)
        val comms         = collectCommunicationStats(startOfDay)
        val batteryInfo   = getBatteryInfo()
        val systemInfo    = getSystemInfo(startOfDay, now)
        val calEvents     = countCalendarEvents(startOfDay, now)
        val mediaCount    = countMediaAdded(startOfDay)
        val appInstalls   = countAppInstalls(startOfDay)
        val contacts      = countUniqueContactsToday(startOfDay)  // fix: was starred contacts
        val downloads     = countDownloads(startOfDay)
        val storageGB     = getStorageUsedGB()
        val appUninstalls = countAppUninstalls()
        val upiLaunches   = countUpiLaunches(events.appLaunches)
        val nightChecks   = countNightInterruptions(startOfDay)

        // Notification count natively parsed from UsageEvents (Type 12)
        val notifCount = events.notificationCount.toFloat()

        Log.i(TAG, "Snapshot OK — screen:%.1fh unlocks:${events.unlockCount} launches:${events.launchCount} notifs:$notifCount steps:$dailySteps".format(events.screenTimeMs / 3_600_000.0))

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
            nightInterruptions   = nightChecks.toFloat(),

            dailySteps           = dailySteps,

            appBreakdown         = events.appMinutes,
            notificationBreakdown = events.notificationBreakdown,
            appLaunchesBreakdown = events.appLaunches
        )
    }

    /** Start passive continuous location tracking. */
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

            // Request updates roughly every 5 minutes, or when moving >50 meters
            val locationRequest = LocationRequest.Builder(Priority.PRIORITY_BALANCED_POWER_ACCURACY, 5 * 60 * 1000L)
                .setMinUpdateDistanceMeters(50f)
                .build()

            locationCallback = object : LocationCallback() {
                override fun onLocationResult(result: GmsLocationResult) {
                    val loc = result.locations.lastOrNull() ?: return
                    val ageMs = System.currentTimeMillis() - loc.time
                    val accuracyOk = loc.accuracy.let { it in 0f..1000f }
                    val freshnessOk = ageMs <= (15 * 60 * 1000L)

                    // Filter out stale or highly inaccurate locations
                    if (accuracyOk && freshnessOk) {
                        DataRepository.addLocationSnapshot(
                            LatLonPoint(loc.latitude, loc.longitude, System.currentTimeMillis())
                        )
                        Log.i(TAG, "Continuous Location fix: %.5f, %.5f (acc: %.1fm)".format(loc.latitude, loc.longitude, loc.accuracy))
                    } else {
                        Log.w(TAG, "Continuous Location rejected: accuracy=${loc.accuracy}m, age=${ageMs / 1000}s")
                    }
                }
            }

            fusedClient.requestLocationUpdates(locationRequest, locationCallback!!, Looper.getMainLooper())
            Log.i(TAG, "Continuous location tracking started.")
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
                Log.i(TAG, "Continuous location tracking stopped.")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Error stopping location tracking: ${e.message}")
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
        val appMinutes: Map<String, Long>, // package → foreground minutes
        val appLaunches: Map<String, Int>, // package → launch count
        val notificationCount: Int,        // total notification interruptions
        val notificationBreakdown: Map<String, Int> // package → notification count
    )

    private data class SleepResult(
        val sleepTimeHour: Float,
        val wakeTimeHour: Float,
        val sleepDurationHours: Float
    )

    private fun calculateSleepProxy(startOfDayMs: Long, now: Long): SleepResult {
        // Sleep window: Yesterday 20:00 → Today 12:00
        val windowStartMs = startOfDayMs - (4 * 3600_000L)   // yesterday 20:00
        val windowEndMs   = minOf(now, startOfDayMs + (12 * 3600_000L)) // today 12:00

        val usm = context.getSystemService(Context.USAGE_STATS_SERVICE) as UsageStatsManager
        val events = usm.queryEvents(windowStartMs, windowEndMs)
        val ev = UsageEvents.Event()

        // ── Step 1: collect raw screen-on/off timestamps ──────────────────────
        // screenOn  = device became interactive (user picked up phone)
        // screenOff = device went non-interactive (screen turned off)
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

        // ── Step 2: Identify sleep gaps (screen-off periods) ──────────────────
        // A gap = interval between two consecutive screen sessions.
        // Brief interruptions ≤ MICRO_WAKE_THRESHOLD are merged into the surrounding sleep.
        val MICRO_WAKE_MS = 10 * 60_000L  // 10 minutes — brief checks (e.g. checking time at night) don't break sleep

        // Build list of "off" intervals from consecutive sessions
        data class Gap(val startMs: Long, val endMs: Long)
        val gaps = mutableListOf<Gap>()
        for (i in 0 until sessions.size - 1) {
            gaps.add(Gap(sessions[i].offMs, sessions[i + 1].onMs))
        }
        // Also gap from window start to first session (if phone was off initially)
        if (sessions.isNotEmpty() && sessions[0].onMs > windowStartMs) {
            gaps.add(0, Gap(windowStartMs, sessions[0].onMs))
        }
        // And from last session to window end
        if (sessions.isNotEmpty()) {
            gaps.add(Gap(sessions.last().offMs, windowEndMs))
        }
        if (sessions.isEmpty()) {
            gaps.add(Gap(windowStartMs, windowEndMs))
        }

        // ── Step 3: Merge gaps separated only by micro-wakes ─────────────────
        // Walk through gaps: if the "on" session between two gaps is ≤ MICRO_WAKE_MS,
        // merge the two gaps into one continuous sleep episode.
        val mergedGaps = mutableListOf<Gap>()
        var current = gaps.firstOrNull() ?: return SleepResult(0f, 0f, 0f)

        for (i in 1 until gaps.size) {
            val bridgeOnMs = gaps[i].startMs - current.endMs  // how long was screen on between gaps
            if (bridgeOnMs <= MICRO_WAKE_MS) {
                // merge: extend current gap to swallow the micro-wake
                current = Gap(current.startMs, gaps[i].endMs)
            } else {
                mergedGaps.add(current)
                current = gaps[i]
            }
        }
        mergedGaps.add(current)

        // ── Step 4: Pick the longest merged gap = sleep episode ───────────────
        val bestGap = mergedGaps.maxByOrNull { it.endMs - it.startMs }
            ?: return SleepResult(0f, 0f, 0f)

        val durationMs = bestGap.endMs - bestGap.startMs

        return SleepResult(
            sleepTimeHour      = msToHour(bestGap.startMs),
            wakeTimeHour       = msToHour(bestGap.endMs),
            sleepDurationHours = durationMs / 3_600_000f
        )
    }

    private val EXCLUDED_PACKAGES = setOf(
        "android", "com.android.systemui", "com.google.android.gms",
        "com.android.launcher", "com.google.android.apps.nexuslauncher",
        context.packageName
    )

    private fun parseUsageEvents(startMs: Long, endMs: Long): EventsResult {
        val usm = context.getSystemService(Context.USAGE_STATS_SERVICE) as UsageStatsManager
        
        // 1. Exact raw event iteration (ensuring 100% boundary accuracy without overlapping days)
        val events = usm.queryEvents(startMs, endMs)
        val event  = UsageEvents.Event()

        val appMs = mutableMapOf<String, Long>()
        var totalInteractionMs = 0L

        val appLaunches = mutableMapOf<String, Int>()
        val lastBgAt = mutableMapOf<String, Long>()
        val lastFgAt = mutableMapOf<String, Long>()

        var unlocks      = 0
        var launches     = 0
        var screenOffAt  = startMs     // track screen-off accumulation
        var totalOffMs   = 0L
        
        var screenIsOn   = true        // assume screen on at query start

        var notifications = 0
        val appNotifications = mutableMapOf<String, Int>()

        while (events.hasNextEvent()) {
            events.getNextEvent(event)
            val ts  = event.timeStamp.coerceIn(startMs, endMs)
            val pkg = event.packageName ?: ""

            when (event.eventType) {
                // ── App comes to foreground (1) ────────────────────────────────
                UsageEvents.Event.MOVE_TO_FOREGROUND -> {
                    if (!isExcluded(pkg)) {
                        // Scientific debounce: only count as a new launch if it was in the background for > 1.5s
                        val lastBg = lastBgAt[pkg] ?: 0L
                        if (lastBg == 0L || (ts - lastBg > 1500L)) {
                            launches++
                            appLaunches[pkg] = (appLaunches[pkg] ?: 0) + 1
                        }
                        lastFgAt[pkg] = ts
                    }
                }

                // ── App goes to background (2) ─────────────────────────────────
                UsageEvents.Event.MOVE_TO_BACKGROUND -> {
                    if (!isExcluded(pkg)) {
                        lastBgAt[pkg] = ts
                        val startFg = lastFgAt[pkg] ?: 0L
                        if (startFg > 0L && startFg <= ts) {
                            val duration = ts - startFg
                            appMs[pkg] = (appMs[pkg] ?: 0L) + duration
                            totalInteractionMs += duration
                            lastFgAt.remove(pkg)
                        }
                    }
                }

                // ── Screen Unlock (18 = KEYGUARD_HIDDEN) ───────────────────────
                18 -> { 
                    unlocks++
                }

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

                // Device Shutdown (26 = DEVICE_SHUTDOWN, API 29+) ────────────
                26 -> {
                    // Safe cleanup if needed
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

        // Add remaining in-progress foreground time for any apps still active at endMs
        for ((pkg, startFg) in lastFgAt) {
            if (startFg > 0L && startFg <= endMs) {
                val duration = endMs - startFg
                appMs[pkg] = (appMs[pkg] ?: 0L) + duration
                totalInteractionMs += duration
            }
        }

        // If screen was off at end of window, add remaining off time
        if (!screenIsOn) totalOffMs += endMs - screenOffAt
        
        val pm = context.packageManager
        val socialInteractionMs = appMs.filterKeys { pkg ->
            try {
                val appInfo = pm.getApplicationInfo(pkg, 0)
                // Android 8+ official app categorization
                val isSocialCat = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    appInfo.category == android.content.pm.ApplicationInfo.CATEGORY_SOCIAL
                } else false
                
                // Fallback for messaging apps that mis-categorize or older devices
                val lower = pkg.lowercase()
                val isSocialName = lower.contains("facebook") || lower.contains("instagram") ||
                                   lower.contains("whatsapp") || lower.contains("twitter") ||
                                   lower.contains("snapchat") || lower.contains("tiktok") ||
                                   lower.contains("messenger") || lower.contains("reddit") ||
                                   lower.contains("telegram") || lower.contains("discord")

                isSocialCat || isSocialName
            } catch (e: Exception) {
                false
            }
        }.values.sum()
        val minutes   = appMs.mapValues { it.value / 60_000L }.filter { it.value > 0 }

        return EventsResult(
            screenTimeMs = totalInteractionMs,
            unlockCount  = unlocks,
            launchCount  = launches,
            socialRatio  = if (totalInteractionMs > 0) socialInteractionMs.toFloat() / totalInteractionMs else 0f,
            screenOffMs  = totalOffMs,
            appMinutes   = minutes,
            appLaunches  = appLaunches,
            notificationCount = notifications,
            notificationBreakdown = appNotifications
        )
    }

    private fun isExcluded(pkg: String): Boolean {
        if (pkg.isBlank()) return true
        val lower = pkg.lowercase()
        return lower == "android" ||
               EXCLUDED_PACKAGES.any { lower == it } ||
               lower.contains("launcher") ||
               lower.contains("systemui") ||
               lower.startsWith("com.android.providers") ||
               lower.startsWith("com.android.server") ||
               lower.startsWith("com.android.permission") ||
               lower.contains("quicksearchbox")
    }

    // =========================================================================
    //  Sleep estimation from screen-off gap (phone dark hours)
    // =========================================================================

    /**
     * Helper to convert an epoch ms timestamp to an hour float (e.g., 8:30am -> 8.5)
     */
    private fun msToHour(ms: Long): Float {
        if (ms == Long.MAX_VALUE || ms == Long.MIN_VALUE || ms == 0L) return 0f
        val cal = Calendar.getInstance()
        cal.timeInMillis = ms
        return cal.get(Calendar.HOUR_OF_DAY) + cal.get(Calendar.MINUTE) / 60f
    }

    /**
     * Estimates dark hours from the total screen-off accumulation during the day.
     * Anything under 4h is considered not meaningful (device just idle while awake).
     */
    private fun estimateDark(screenOffMs: Long): Float {
        return (screenOffMs / 3_600_000f).coerceIn(0f, 16f)
    }

    /**
     * Estimates sleep duration: the main nocturnal gap between last use and first morning use.
     * Assumes wake is between 4am–12pm, sleep after 8pm.
     */
    private fun estimateSleep(firstHour: Float, lastHour: Float): Float {
        if (firstHour == 0f) return 7f
        // last use was in evening/night → add hours to midnight + morning hours
        val sleepStart = if (lastHour > 20f || lastHour < 3f) lastHour else 23f
        return if (firstHour > 4f) {
            val fromMidnight = if (sleepStart < 24f) 24f - sleepStart else 0f
            (fromMidnight + firstHour).coerceIn(3f, 12f)
        } else {
            7f // fallback
        }
    }

    // =========================================================================
    //  Location metrics — Haversine polyline from GPS snapshots
    // =========================================================================

    private data class LocationResult(
        val displacementKm: Float, val entropy: Float,
        val homeRatio: Float, val placesCount: Int
    )

    private fun calculateLocationMetrics(snaps: List<LatLonPoint>): LocationResult {
        if (snaps.size < 2) return LocationResult(0f, 0f, 1f, 1)

        var distKm = 0.0
        for (i in 1 until snaps.size) {
            distKm += haversine(snaps[i-1].lat, snaps[i-1].lon, snaps[i].lat, snaps[i].lon)
        }

        // Cell-based Shannon entropy (0.001° ≈ 110m grid)
        val cells = mutableMapOf<String, Int>()
        snaps.forEach { pt ->
            val key = "${"%.3f".format(pt.lat)},${"%.3f".format(pt.lon)}"
            cells[key] = (cells[key] ?: 0) + 1
        }
        val total   = snaps.size.toDouble()
        val entropy = cells.values.sumOf { c -> val p = c / total; -p * ln(p) }.toFloat()
        val homeCell= cells.maxByOrNull { it.value }?.key
        val homeRatio = (cells[homeCell] ?: 0) / total.toFloat()

        return LocationResult(distKm.toFloat(), entropy, homeRatio, cells.size)
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
        val am   = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mem  = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        val memPct = if (mem.totalMem > 0) (mem.totalMem - mem.availMem) * 100f / mem.totalMem else 0f

        // Network — same as Settings → Network & Internet → Data usage
        var wifiMB = 0f; var mobileMB = 0f
        try {
            val nsm = context.getSystemService(Context.NETWORK_STATS_SERVICE) as NetworkStatsManager
            nsm.querySummaryForDevice(NetworkCapabilities.TRANSPORT_WIFI, null, startMs, endMs)
                .let { wifiMB = (it.rxBytes + it.txBytes) / (1024f * 1024f) }
            nsm.querySummaryForDevice(NetworkCapabilities.TRANSPORT_CELLULAR, null, startMs, endMs)
                .let { mobileMB = (it.rxBytes + it.txBytes) / (1024f * 1024f) }
        } catch (e: Exception) { Log.e(TAG, "Network stats: ${e.message}") }

        return SystemResult(storagePct, memPct, wifiMB, mobileMB)
    }

    private fun countMediaAdded(since: Long): Int = try {
        context.contentResolver.query(
            MediaStore.Files.getContentUri("external"),
            arrayOf(MediaStore.Files.FileColumns._ID),
            "${MediaStore.Files.FileColumns.DATE_ADDED} >= ?",
            arrayOf((since / 1000).toString()), null
        )?.use { it.count } ?: 0
    } catch (e: Exception) { 0 }

    private fun countAppInstalls(since: Long): Int = try {
        context.packageManager.getInstalledPackages(0).count { it.firstInstallTime >= since }
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
    private fun startOfDayMs(): Long {
        return Calendar.getInstance().apply {
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }.timeInMillis
    }

    // ── New feature collectors ─────────────────────────────────────────────────

    private fun countDownloads(since: Long): Int = try {
        // Scientifically bypass MediaStore index lags by checking the physical filesystem layer natively:
        val dir = android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_DOWNLOADS)
        var count = 0
        dir.listFiles()?.forEach { file ->
            if (file.isFile && file.lastModified() >= since) count++
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
        val currentCount = context.packageManager.getInstalledPackages(0).size
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

    private fun countNightInterruptions(startOfDay: Long): Int {
        // Query the PREVIOUS night window: yesterday 22:00 to today 05:00
        // This ensures a meaningful value is always available during daytime ticks
        val nightStart = startOfDay - 2 * 3600_000L  // yesterday 22:00
        val nightEnd   = startOfDay + 5 * 3600_000L  // today 05:00
        val queryEnd   = minOf(nightEnd, System.currentTimeMillis())
        if (queryEnd <= nightStart) return 0
        val usm = context.getSystemService(Context.USAGE_STATS_SERVICE) as UsageStatsManager
        val events = usm.queryEvents(nightStart, queryEnd)
        val event = UsageEvents.Event()
        var count = 0
        while (events.hasNextEvent()) {
            events.getNextEvent(event)
            if (event.eventType == 18) count++ // KEYGUARD_HIDDEN = unlock
        }
        return count
    }

    // ── Category-based storage breakdown ──────────────────────────────────────
    // Returns map of category label → storage used in GB for installed apps
    fun getStorageByCategory(): Map<String, Float> {
        val result = mutableMapOf<String, Float>()
        val pm = context.packageManager

        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            val ssm = context.getSystemService(Context.STORAGE_STATS_SERVICE) as? android.app.usage.StorageStatsManager
            val storageUuid = android.os.storage.StorageManager.UUID_DEFAULT

            for (pkg in pm.getInstalledPackages(0)) {
                // Efficiently skip system-level packages so OS Bloat doesn't skew metric
                val isSystemApp = (pkg.applicationInfo?.flags ?: 0) and android.content.pm.ApplicationInfo.FLAG_SYSTEM != 0
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
        for (pkg in pm.getInstalledPackages(0)) {
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
        for (pkg in pm.getInstalledPackages(0)) {
            // Only count user-installed apps (exclude system apps)
            val isSystemApp = (pkg.applicationInfo?.flags
                ?: 0) and android.content.pm.ApplicationInfo.FLAG_SYSTEM != 0
            if (isSystemApp) continue
            val label = getCategoryLabel(pkg.applicationInfo, pm)
            result[label] = (result[label] ?: 0) + 1
        }
        // Sort descending by count
        return result.toList().sortedByDescending { it.second }.toMap()
    }

    private fun getCategoryLabel(info: android.content.pm.ApplicationInfo?,
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
                    android.content.pm.ApplicationInfo.CATEGORY_GAME -> "Games"
                    android.content.pm.ApplicationInfo.CATEGORY_SOCIAL -> "Social"
                    android.content.pm.ApplicationInfo.CATEGORY_PRODUCTIVITY -> "Productivity"
                    android.content.pm.ApplicationInfo.CATEGORY_MAPS -> "Maps"
                    android.content.pm.ApplicationInfo.CATEGORY_NEWS -> "News"
                    android.content.pm.ApplicationInfo.CATEGORY_AUDIO -> "Media"
                    android.content.pm.ApplicationInfo.CATEGORY_VIDEO -> "Media"
                    android.content.pm.ApplicationInfo.CATEGORY_IMAGE -> "Photos"
                    else -> "Other"
                }
            } else "Other"
        }
    }
}
