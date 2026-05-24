package com.example.mhealth.logic

import android.app.usage.UsageEvents
import android.app.usage.UsageStatsManager
import android.content.Context
import com.example.mhealth.logic.db.AppSessionEntity
import com.example.mhealth.logic.db.DailyDnaSnapshotEntity
import com.example.mhealth.logic.db.MHealthDatabase
import com.example.mhealth.logic.db.NotificationEventEntity
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import org.json.JSONObject
import org.json.JSONArray

/**
 * Lightweight helper that computes real-time App DNA and Phone DNA metrics
 * for the current day from Room session and notification event data.
 *
 * This is used by the Sensors tab to display today's behavioral fingerprint
 * (NOT baseline — baseline lives in the Monitor tab).
 */
class AppDnaComputer(private val context: Context) {

    private val db = MHealthDatabase.getInstance(context)
    private val dateFmt = SimpleDateFormat("yyyy-MM-dd", Locale.US)

    /**
     * Per-app DNA summary for today (displayed in Sensors expandable list).
     */
    data class TodayAppDna(
        val appPackage: String,
        val appLabel: String,
        val totalScreenTimeMinutes: Long,
        val sessionCount: Int,
        val avgSessionMinutes: Float,
        val minSessionMinutes: Float,
        val maxSessionMinutes: Float,
        val primaryTimeRange: String,        // e.g., "9 AM – 6 PM"
        val selfOpenRatio: Float,            // 0..1 fraction of sessions opened by user (not notification)
        val notificationOpenRatio: Float,    // 0..1 fraction opened via notification tap
        val notificationCount: Int,          // notifications received from this app today
        val notificationTapCount: Int,       // notifications tapped for this app today
        val avgTapLatencyMinutes: Float?,    // average minutes to tap notification (null if no taps)
        val launchCount: Int                 // from PersonalityVector breakdown
    )

    /**
     * Phone-level DNA summary for today (device-wide behavioral fingerprint).
     */
    data class TodayPhoneDna(
        val totalSessions: Int,
        val totalScreenTimeHours: Float,
        val firstPickupHour: Float?,               // hour of first session today (null if no sessions)
        val lastActivityHour: Float?,               // hour of last session close
        val activeWindowHours: Float?,              // lastActivity - firstPickup
        val avgSessionMinutes: Float,
        val microSessionPct: Float,                 // % sessions < 2 min
        val shortSessionPct: Float,                 // % sessions 2–15 min
        val mediumSessionPct: Float,                // % sessions 15–30 min
        val deepSessionPct: Float,                  // % sessions 30–60 min
        val marathonSessionPct: Float,              // % sessions > 60 min
        val selfOpenPct: Float,                     // % sessions triggered by user
        val notificationOpenPct: Float,             // % sessions triggered by notification
        val totalNotifications: Int,                // total notification events today
        val notificationTapRate: Float,             // tapped / total arrivals
        val notificationDismissRate: Float,         // dismissed / total arrivals
        val notificationIgnoreRate: Float,          // ignored / total arrivals
        val uniqueAppsUsed: Int,                    // distinct apps with sessions today
        val topAppPackage: String?,                 // app with most screen time today
        val nightChecks: Int                        // unlocks during sleep window
    )

    /**
     * Compute today's phone-level DNA from sessions + notification events.
     */
    suspend fun computeTodayPhoneDna(): TodayPhoneDna {
        val today = dateFmt.format(Date())
        return computePhoneDnaForDate(today)
    }

    /**
     * Compute phone-level DNA for an arbitrary date (used for midnight snapshot of previous day).
     */
    suspend fun computePhoneDnaForDate(date: String): TodayPhoneDna {
        val sessionsRaw = db.appSessionDao().getByDate(date)
        val sessions = sessionsRaw.distinctBy { "${it.app_package}_${it.open_timestamp}" }

        val notificationsRaw = db.notificationEventDao().getByDate(date)
        val notifications = notificationsRaw.distinctBy { "${it.app_package}_${it.arrival_timestamp}_${it.action}" }

        val totalSessions = sessions.size
        val totalScreenMs = sessions.sumOf { (it.close_timestamp - it.open_timestamp).coerceAtLeast(0) }
        val totalScreenHours = totalScreenMs / 3_600_000f

        val firstPickup = sessions.minByOrNull { it.open_timestamp }
        val lastActivity = sessions.maxByOrNull { it.close_timestamp }

        val firstPickupHour = firstPickup?.let { hourOfDay(it.open_timestamp) }
        val lastActivityHour = lastActivity?.let { hourOfDay(it.close_timestamp) }
        val activeWindowHours = if (firstPickupHour != null && lastActivityHour != null) {
            (lastActivityHour - firstPickupHour).coerceAtLeast(0f)
        } else null

        val sessionDurationsMins = sessions.map {
            ((it.close_timestamp - it.open_timestamp).coerceAtLeast(0)) / 60_000f
        }
        val avgSessionMin = if (sessionDurationsMins.isNotEmpty()) sessionDurationsMins.average().toFloat() else 0f

        val microCount = sessionDurationsMins.count { it < 2f }
        val shortCount = sessionDurationsMins.count { it in 2f..15f }
        val mediumCount = sessionDurationsMins.count { it in 15f..30f }
        val deepCount = sessionDurationsMins.count { it in 30f..60f }
        val marathonCount = sessionDurationsMins.count { it > 60f }
        val total = totalSessions.toFloat().coerceAtLeast(1f)

        val selfCount = sessions.count { it.trigger == "SELF" }
        val notifOpenCount = sessions.count { it.trigger == "NOTIFICATION" }

        val arrivals = notifications.count { it.action == "ARRIVAL" }
        val taps = notifications.count { it.action == "TAP" }
        val dismisses = notifications.count { it.action == "DISMISS" }
        val ignores = notifications.count { it.action == "IGNORE" }

        val totalArrivals = arrivals.coerceAtLeast(taps + dismisses + ignores).toFloat().coerceAtLeast(1f)

        val uniqueApps = sessions.map { it.app_package }.distinct().size

        val appScreenTime = sessions.groupBy { it.app_package }
            .mapValues { (_, s) -> s.sumOf { (it.close_timestamp - it.open_timestamp).coerceAtLeast(0) } }
        val topApp = appScreenTime.maxByOrNull { it.value }?.key

        // Night checks: count unlocks during sleep window
        val nightChecks = countNightChecks(date)

        return TodayPhoneDna(
            totalSessions = totalSessions,
            totalScreenTimeHours = totalScreenHours,
            firstPickupHour = firstPickupHour,
            lastActivityHour = lastActivityHour,
            activeWindowHours = activeWindowHours,
            avgSessionMinutes = avgSessionMin,
            microSessionPct = microCount / total * 100f,
            shortSessionPct = shortCount / total * 100f,
            mediumSessionPct = mediumCount / total * 100f,
            deepSessionPct = deepCount / total * 100f,
            marathonSessionPct = marathonCount / total * 100f,
            selfOpenPct = selfCount / total * 100f,
            notificationOpenPct = notifOpenCount / total * 100f,
            totalNotifications = totalArrivals.toInt(),
            notificationTapRate = (taps / totalArrivals).coerceIn(0f, 1f),
            notificationDismissRate = (dismisses / totalArrivals).coerceIn(0f, 1f),
            notificationIgnoreRate = (ignores / totalArrivals).coerceIn(0f, 1f),
            uniqueAppsUsed = uniqueApps,
            topAppPackage = topApp,
            nightChecks = nightChecks
        )
    }

    /**
     * Compute per-app DNA summaries for today.
     * Returns a list sorted by total screen time (descending).
     */
    suspend fun computeTodayAppDnaList(): List<TodayAppDna> {
        val today = dateFmt.format(Date())
        return computeAppDnaListForDate(today)
    }

    /**
     * Compute per-app DNA summaries for an arbitrary date (used for midnight snapshot).
     */
    suspend fun computeAppDnaListForDate(date: String): List<TodayAppDna> {
        val sessionsRaw = db.appSessionDao().getByDate(date)
        val sessions = sessionsRaw.distinctBy { "${it.app_package}_${it.open_timestamp}" }

        val notificationsRaw = db.notificationEventDao().getByDate(date)
        val notifications = notificationsRaw.distinctBy { "${it.app_package}_${it.arrival_timestamp}_${it.action}" }

        val pm = context.packageManager

        if (sessions.isEmpty()) return emptyList()

        val sessionsByApp = sessions.groupBy { it.app_package }
        val notifsByApp = notifications.groupBy { it.app_package }

        return sessionsByApp.map { (pkg, appSessions) ->
            val appNotifs = notifsByApp[pkg] ?: emptyList()

            val totalScreenMs = appSessions.sumOf { (it.close_timestamp - it.open_timestamp).coerceAtLeast(0) }
            val totalScreenMin = totalScreenMs / 60_000L

            val sessionDurationsMins = appSessions.map {
                ((it.close_timestamp - it.open_timestamp).coerceAtLeast(0)) / 60_000f
            }
            val avgSessionMin = if (sessionDurationsMins.isNotEmpty()) sessionDurationsMins.average().toFloat() else 0f
            val minSessionMin = sessionDurationsMins.minOrNull() ?: 0f
            val maxSessionMin = sessionDurationsMins.maxOrNull() ?: 0f

            // Primary time range: density-based active window
            val primaryRange = computeActiveWindow(appSessions)

            val sessionCount = appSessions.size.toFloat().coerceAtLeast(1f)
            val selfOpen = appSessions.count { it.trigger == "SELF" } / sessionCount
            val notifOpen = appSessions.count { it.trigger == "NOTIFICATION" } / sessionCount

            val arrivals = appNotifs.count { it.action == "ARRIVAL" }
            val taps = appNotifs.count { it.action == "TAP" }
            val tapLatencies = appNotifs.filter { it.action == "TAP" && it.tap_latency_min != null }
                .mapNotNull { it.tap_latency_min }
            val avgTapLatency = if (tapLatencies.isNotEmpty()) tapLatencies.average().toFloat() else null

            val appLabel = try {
                pm.getApplicationLabel(pm.getApplicationInfo(pkg, 0)).toString()
            } catch (_: Exception) {
                pkg.substringAfterLast(".")
            }

            TodayAppDna(
                appPackage = pkg,
                appLabel = appLabel,
                totalScreenTimeMinutes = totalScreenMin,
                sessionCount = appSessions.size,
                avgSessionMinutes = avgSessionMin,
                minSessionMinutes = minSessionMin,
                maxSessionMinutes = maxSessionMin,
                primaryTimeRange = primaryRange,
                selfOpenRatio = selfOpen,
                notificationOpenRatio = notifOpen,
                notificationCount = arrivals.coerceAtLeast(taps),
                notificationTapCount = taps,
                avgTapLatencyMinutes = avgTapLatency,
                launchCount = appSessions.size
            )
        }
            .sortedByDescending { it.totalScreenTimeMinutes }
            .take(15)
    }

    /**
     * Compute and store a daily DNA snapshot for the given date.
     * Called at midnight before purging raw sessions.
     * Returns the snapshot entity (null if no sessions).
     */
    suspend fun computeAndStoreDnaSnapshot(userId: String, date: String): DailyDnaSnapshotEntity? {
        val phoneDna = computePhoneDnaForDate(date)
        val appDnaList = computeAppDnaListForDate(date)

        if (phoneDna.totalSessions == 0) return null

        val appDnaJson = JSONArray().apply {
            appDnaList.forEach { app ->
                put(JSONObject().apply {
                    put("appPackage", app.appPackage)
                    put("appLabel", app.appLabel)
                    put("totalScreenTimeMinutes", app.totalScreenTimeMinutes)
                    put("sessionCount", app.sessionCount)
                    put("avgSessionMinutes", app.avgSessionMinutes)
                    put("selfOpenRatio", app.selfOpenRatio)
                    put("notificationOpenRatio", app.notificationOpenRatio)
                    put("primaryTimeRange", app.primaryTimeRange)
                    put("notificationCount", app.notificationCount)
                    put("notificationTapCount", app.notificationTapCount)
                    put("avgTapLatencyMinutes", app.avgTapLatencyMinutes?.toDouble() ?: org.json.JSONObject.NULL)
                })
            }
        }.toString()

        val entity = DailyDnaSnapshotEntity(
            userId = userId,
            date = date,
            totalSessions = phoneDna.totalSessions,
            totalScreenTimeHours = phoneDna.totalScreenTimeHours,
            firstPickupHour = phoneDna.firstPickupHour,
            lastActivityHour = phoneDna.lastActivityHour,
            activeWindowHours = phoneDna.activeWindowHours,
            avgSessionMinutes = phoneDna.avgSessionMinutes,
            microSessionPct = phoneDna.microSessionPct,
            shortSessionPct = phoneDna.shortSessionPct,
            mediumSessionPct = phoneDna.mediumSessionPct,
            deepSessionPct = phoneDna.deepSessionPct,
            marathonSessionPct = phoneDna.marathonSessionPct,
            selfOpenPct = phoneDna.selfOpenPct,
            notificationOpenPct = phoneDna.notificationOpenPct,
            totalNotifications = phoneDna.totalNotifications,
            notificationTapRate = phoneDna.notificationTapRate,
            notificationDismissRate = phoneDna.notificationDismissRate,
            notificationIgnoreRate = phoneDna.notificationIgnoreRate,
            uniqueAppsUsed = phoneDna.uniqueAppsUsed,
            topAppPackage = phoneDna.topAppPackage,
            nightChecks = phoneDna.nightChecks,
            appDnaJson = appDnaJson,
            createdAt = System.currentTimeMillis()
        )

        db.dailyDnaSnapshotDao().insert(entity)
        return entity
    }

    private fun hourOfDay(epochMs: Long): Float {
        val cal = java.util.Calendar.getInstance().apply { timeInMillis = epochMs }
        return cal.get(java.util.Calendar.HOUR_OF_DAY) + cal.get(java.util.Calendar.MINUTE) / 60f
    }

    /**
     * Count unlocks (KEYGUARD_HIDDEN = event type 18) during the sleep window.
     * Sleep window is derived from the previous day's sleep proxy data:
     *   - Sleep start = yesterday's sleepTimeHour (e.g., 23.5 = 11:30 PM)
     *   - Sleep end = today's wakeTimeHour (e.g., 7.25 = 7:15 AM)
     * If no sleep data available, uses a default 11 PM – 7 AM window.
     */
    private fun countNightChecks(date: String): Int {
        try {
            val sdf = SimpleDateFormat("yyyy-MM-dd", Locale.US)
            val dayStart = sdf.parse(date)?.time ?: return 0
            val dayEnd = dayStart + 86_400_000L

            // Get sleep window from DataRepository (today's sleep data)
            val sleepTimeHour = com.example.mhealth.logic.DataRepository.latestVector.value?.sleepTimeHour
            val wakeTimeHour = com.example.mhealth.logic.DataRepository.latestVector.value?.wakeTimeHour

            // Use defaults if no sleep data yet
            val sleepStart = sleepTimeHour ?: 23f
            val sleepEnd = wakeTimeHour ?: 7f

            // Convert sleep hours to epoch ranges within the day
            val nightStartMs = dayStart + (sleepStart * 3_600_000L).toLong()
            // If sleep time is before midnight, the night window extends into the next day
            val nightEndMs = if (sleepEnd < sleepStart) {
                dayEnd + (sleepEnd * 3_600_000L).toLong()  // wraps past midnight
            } else {
                dayStart + (sleepEnd * 3_600_000L).toLong()
            }

            // Query UsageEvents for the overnight window
            val usm = context.getSystemService(UsageStatsManager::class.java) ?: return 0
            val queryStart = nightStartMs.coerceAtLeast(dayStart)
            val queryEnd = nightEndMs.coerceAtMost(dayEnd + 86_400_000L)  // allow into next day
            val events = usm.queryEvents(queryStart, queryEnd)
            val event = UsageEvents.Event()

            var nightUnlocks = 0
            while (events.hasNextEvent()) {
                events.getNextEvent(event)
                // KEYGUARD_HIDDEN = 18 (screen unlock)
                if (event.eventType == 18) {
                    nightUnlocks++
                }
            }
            return nightUnlocks
        } catch (e: Exception) {
            return 0
        }
    }

    /**
     * Density-based active window: bins sessions into 30-min slots,
     * finds longest contiguous run of occupied slots, expands by 15 min.
     * Returns "9:30 AM – 11:00 AM" style range.
     */
    private fun computeActiveWindow(sessions: List<AppSessionEntity>): String {
        if (sessions.isEmpty()) return "—"
        if (sessions.size == 1) {
            return formatHourPrecise(hourOfDay(sessions[0].open_timestamp))
        }

        val SLOT_DURATION_MIN = 30
        val SLOTS_PER_DAY = 48

        // Bin sessions into 30-min slots
        val occupiedSlots = mutableSetOf<Int>()
        for (s in sessions) {
            val openHourF = hourOfDay(s.open_timestamp)
            val closeHourF = hourOfDay(s.close_timestamp)
            val openSlot = (openHourF * 60f / SLOT_DURATION_MIN).toInt().coerceIn(0, SLOTS_PER_DAY - 1)
            val closeSlot = (closeHourF * 60f / SLOT_DURATION_MIN).toInt().coerceIn(0, SLOTS_PER_DAY - 1)
            for (slot in openSlot..closeSlot) {
                occupiedSlots.add(slot)
            }
        }

        if (occupiedSlots.isEmpty()) return "—"

        // Find longest contiguous run (gap of 1 slot = 30 min allowed)
        val sortedSlots = occupiedSlots.sorted()
        var bestStart = sortedSlots[0]
        var bestEnd = sortedSlots[0]
        var curStart = sortedSlots[0]
        var curEnd = sortedSlots[0]

        for (i in 1 until sortedSlots.size) {
            if (sortedSlots[i] <= curEnd + 1) {
                curEnd = sortedSlots[i]
            } else {
                if (curEnd - curStart > bestEnd - bestStart) {
                    bestStart = curStart; bestEnd = curEnd
                }
                curStart = sortedSlots[i]; curEnd = sortedSlots[i]
            }
        }
        if (curEnd - curStart > bestEnd - bestStart) {
            bestStart = curStart; bestEnd = curEnd
        }

        // Expand by 1 slot (15 min) on each side, clamped
        val windowStartSlot = (bestStart - 1).coerceIn(0, SLOTS_PER_DAY - 1)
        val windowEndSlot = (bestEnd + 1).coerceIn(0, SLOTS_PER_DAY - 1)

        val windowStartHour = windowStartSlot * SLOT_DURATION_MIN / 60f
        val windowEndHour = (windowEndSlot + 1) * SLOT_DURATION_MIN / 60f

        // Cap at 16 hours
        val clampedEnd = if (windowEndHour - windowStartHour > 16f) windowStartHour + 16f else windowEndHour

        return "${formatHourPrecise(windowStartHour)} – ${formatHourPrecise(clampedEnd)}"
    }

    private fun formatHourPrecise(hourFloat: Float): String {
        val totalMinutes = (hourFloat * 60f).toInt().coerceIn(0, 1439)
        val hours = totalMinutes / 60
        val minutes = totalMinutes % 60
        return when {
            hours == 0 -> {
                if (minutes == 0) "12 AM" else "12:%02d AM".format(minutes)
            }
            hours < 12 -> {
                if (minutes == 0) "$hours AM" else "$hours:%02d AM".format(hours, minutes)
            }
            hours == 12 -> {
                if (minutes == 0) "12 PM" else "12:%02d PM".format(minutes)
            }
            else -> {
                val h = hours - 12
                if (minutes == 0) "$h PM" else "$h:%02d PM".format(h, minutes)
            }
        }
    }

    private fun formatHour(hour: Int): String {
        return when {
            hour == 0 -> "12 AM"
            hour < 12 -> "$hour AM"
            hour == 12 -> "12 PM"
            else -> "${hour - 12} PM"
        }
    }
}
