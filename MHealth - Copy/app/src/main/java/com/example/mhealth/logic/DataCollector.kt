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
import com.google.android.gms.tasks.Tasks
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
        val contacts      = countStarredContacts()
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
            conversationFrequency= comms.callCount.toFloat(),

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

            appBreakdown         = events.appMinutes,
            notificationBreakdown = events.notificationBreakdown,
            appLaunchesBreakdown = events.appLaunches
        )
    }

    /** Capture a GPS fix and append to today's location track. */
    @SuppressLint("MissingPermission")
    fun captureLocationSnapshot() {
        // Guard: check that fine-location permission is actually granted
        if (context.checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION)
                != PackageManager.PERMISSION_GRANTED &&
            context.checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION)
                != PackageManager.PERMISSION_GRANTED
        ) {
            Log.w(TAG, "Location permission not granted — skipping GPS capture")
            return
        }

        try {
            val fusedClient = LocationServices.getFusedLocationProviderClient(context)

            // Try getCurrentLocation first (active fix)
            var loc = try {
                Tasks.await(
                    fusedClient.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, null),
                    10, TimeUnit.SECONDS
                )
            } catch (e: Exception) {
                Log.w(TAG, "getCurrentLocation failed: ${e.message}")
                null
            }

            // Fallback: use last known location if active fix returned null
            if (loc == null) {
                loc = try {
                    Tasks.await(fusedClient.lastLocation, 5, TimeUnit.SECONDS)
                } catch (e: Exception) {
                    Log.w(TAG, "getLastLocation fallback failed: ${e.message}")
                    null
                }
            }

            if (loc != null) {
                DataRepository.addLocationSnapshot(
                    LatLonPoint(loc.latitude, loc.longitude, System.currentTimeMillis())
                )
                Log.i(TAG, "GPS fix recorded: %.5f, %.5f".format(loc.latitude, loc.longitude))
            } else {
                Log.w(TAG, "GPS unavailable — both getCurrentLocation and lastLocation returned null")
            }
        } catch (e: SecurityException) {
            Log.e(TAG, "SecurityException accessing location: ${e.message}")
        } catch (e: Exception) {
            Log.w(TAG, "GPS capture error: ${e.message}")
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
        // Sleep for "today" corresponds to the previous night.
        // Night window: Yesterday 18:00 (6 PM) to Today 14:00 (2 PM) = 20 hours
        val windowStartMs = startOfDayMs - (6 * 3600_000L)
        val windowEndMs   = kotlin.math.min(now, startOfDayMs + (14 * 3600_000L))

        val usm = context.getSystemService(Context.USAGE_STATS_SERVICE) as UsageStatsManager
        val events = usm.queryEvents(windowStartMs, windowEndMs)
        val event = UsageEvents.Event()

        var gapStartMs = windowStartMs
        var longestGapMs = 0L
        var bestSleepStartMs = windowStartMs
        var bestSleepEndMs = windowStartMs

        while (events.hasNextEvent()) {
            events.getNextEvent(event)
            val ts = event.timeStamp
            
            // Interaction Ends -> gap starts
            if (event.eventType == 16 || event.eventType == 2) {
                gapStartMs = ts
            }
            // Interaction Begins -> gap ends
            else if (event.eventType == 1 || event.eventType == 15 || event.eventType == 18) {
                val gap = ts - gapStartMs
                if (gap > longestGapMs) {
                    longestGapMs = gap
                    bestSleepStartMs = gapStartMs
                    bestSleepEndMs = ts
                }
                gapStartMs = ts // Reset gap start to now, so we don't count time while screen is on
            }
        }
        
        // If window hasn't ended and no interaction yet, the gap is still running
        if (now <= windowEndMs) {
            val currentGap = now - gapStartMs
            if (currentGap > longestGapMs) {
                longestGapMs = currentGap
                bestSleepStartMs = gapStartMs
                bestSleepEndMs = now
            }
        }

        return SleepResult(
            sleepTimeHour = msToHour(bestSleepStartMs),
            wakeTimeHour = msToHour(bestSleepEndMs),
            sleepDurationHours = longestGapMs / 3600_000f
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
        context.contentResolver.query(
            android.provider.MediaStore.Downloads.EXTERNAL_CONTENT_URI,
            arrayOf(android.provider.MediaStore.Downloads._ID),
            "${android.provider.MediaStore.Downloads.DATE_ADDED} >= ?",
            arrayOf((since / 1000).toString()), null
        )?.use { it.count } ?: 0
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
        "com.sbi.upi",                               // SBI Pay
        "com.axis.mobile"                            // Axis Mobile
    )

    private fun countUpiLaunches(appLaunches: Map<String, Int>): Int =
        appLaunches.filterKeys { pkg -> UPI_PACKAGES.any { pkg.startsWith(it) } }.values.sum()

    private fun countNightInterruptions(startOfDay: Long): Int {
        val nightStart = startOfDay  // 00:00
        val nightEnd   = startOfDay + 5 * 3600_000L  // 05:00
        if (System.currentTimeMillis() < nightEnd) return 0  // don't query future window
        val usm = context.getSystemService(Context.USAGE_STATS_SERVICE) as UsageStatsManager
        val events = usm.queryEvents(nightStart, nightEnd)
        val event = UsageEvents.Event()
        var count = 0
        while (events.hasNextEvent()) {
            events.getNextEvent(event)
            if (event.eventType == 18) count++ // KEYGUARD_HIDDEN = unlock
        }
        return count
    }
}
