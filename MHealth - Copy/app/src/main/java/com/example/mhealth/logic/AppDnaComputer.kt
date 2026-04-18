package com.example.mhealth.logic

import android.content.Context
import com.example.mhealth.logic.db.AppSessionEntity
import com.example.mhealth.logic.db.MHealthDatabase
import com.example.mhealth.logic.db.NotificationEventEntity
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

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
        val nightChecks: Int                        // phone unlocks during estimated sleep window
    )

    /**
     * Compute today's phone-level DNA from sessions + notification events.
     */
    suspend fun computeTodayPhoneDna(): TodayPhoneDna {
        val today = dateFmt.format(Date())
        // Deduplicate sessions by package and open time to handle old junk data
        val sessionsRaw = db.appSessionDao().getByDate(today)
        val sessions = sessionsRaw.distinctBy { "${it.app_package}_${it.open_timestamp}" }
        
        // Deduplicate notifications by package, arrival time, and action
        val notificationsRaw = db.notificationEventDao().getByDate(today)
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

        // Session duration distribution
        val microCount = sessionDurationsMins.count { it < 2f }
        val shortCount = sessionDurationsMins.count { it in 2f..15f }
        val mediumCount = sessionDurationsMins.count { it in 15f..30f }
        val deepCount = sessionDurationsMins.count { it in 30f..60f }
        val marathonCount = sessionDurationsMins.count { it > 60f }
        val total = totalSessions.toFloat().coerceAtLeast(1f)

        // Trigger breakdown
        val selfCount = sessions.count { it.trigger == "SELF" }
        val notifOpenCount = sessions.count { it.trigger == "NOTIFICATION" }

        // Notification action breakdown
        // Use arrivals as the base for rates
        val arrivals = notifications.count { it.action == "ARRIVAL" }
        val taps = notifications.count { it.action == "TAP" }
        val dismisses = notifications.count { it.action == "DISMISS" }
        val ignores = notifications.count { it.action == "IGNORE" }
        
        // If arrivals is 0 (due to logging lag), but we have taps/dismisses, 
        // fallback to max of (arrivals, taps + dismisses + ignores) for a more realistic rate
        val totalArrivals = arrivals.coerceAtLeast(taps + dismisses + ignores).toFloat().coerceAtLeast(1f)

        // Unique apps
        val uniqueApps = sessions.map { it.app_package }.distinct().size

        // Top app by screen time
        val appScreenTime = sessions.groupBy { it.app_package }
            .mapValues { (_, s) -> s.sumOf { (it.close_timestamp - it.open_timestamp).coerceAtLeast(0) } }
        val topApp = appScreenTime.maxByOrNull { it.value }?.key

        // Night Checks: count sessions during sleep window
        // Use the estimated sleep/wake from the latest vector, or default to 23:00-07:00
        val latestVec = DataRepository.latestVector.value
        val sleepHour = latestVec?.sleepTimeHour ?: 23f
        val wakeHour = latestVec?.wakeTimeHour ?: 7f

        val nightChecks = sessions.count { session ->
            val sessionHour = hourOfDay(session.open_timestamp)
            // Sleep window wraps around midnight (e.g. 23:00 → 07:00)
            if (sleepHour > wakeHour) {
                // e.g. sleep=23, wake=7: night is 23..24 or 0..7
                sessionHour >= sleepHour || sessionHour < wakeHour
            } else {
                // e.g. sleep=1, wake=9: night is 1..9
                sessionHour in sleepHour..wakeHour
            }
        }

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
        val sessionsRaw = db.appSessionDao().getByDate(today)
        val sessions = sessionsRaw.distinctBy { "${it.app_package}_${it.open_timestamp}" }

        val notificationsRaw = db.notificationEventDao().getByDate(today)
        val notifications = notificationsRaw.distinctBy { "${it.app_package}_${it.arrival_timestamp}_${it.action}" }

        val pm = context.packageManager

        if (sessions.isEmpty()) return emptyList()

        // Group sessions by app package
        val sessionsByApp = sessions.groupBy { it.app_package }
        // Group notifications by app package
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

            // Dynamic time range: use IQR (25th–75th percentile) to ignore outlier sessions
            val hours = appSessions.map { hourOfDay(it.open_timestamp).toInt() }.sorted()
            val primaryRange = if (hours.size >= 4) {
                val q1 = hours[hours.size / 4]
                val q3 = hours[3 * hours.size / 4]
                "${formatHour(q1)} – ${formatHour(q3)}"
            } else if (hours.isNotEmpty()) {
                val minH = hours.first()
                val maxH = hours.last()
                "${formatHour(minH)} – ${formatHour(maxH)}"
            } else "—"

            // Trigger ratios
            val sessionCount = appSessions.size.toFloat().coerceAtLeast(1f)
            val selfOpen = appSessions.count { it.trigger == "SELF" } / sessionCount
            val notifOpen = appSessions.count { it.trigger == "NOTIFICATION" } / sessionCount

            // Notification data
            val arrivals = appNotifs.count { it.action == "ARRIVAL" }
            val taps = appNotifs.count { it.action == "TAP" }
            val tapLatencies = appNotifs.filter { it.action == "TAP" && it.tap_latency_min != null }
                .mapNotNull { it.tap_latency_min }
            val avgTapLatency = if (tapLatencies.isNotEmpty()) tapLatencies.average().toFloat() else null

            // App label
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
                notificationCount = arrivals,
                notificationTapCount = taps,
                avgTapLatencyMinutes = avgTapLatency,
                launchCount = appSessions.size
            )
        }
            .sortedByDescending { it.totalScreenTimeMinutes }
            .take(15) // Limit to top 15 apps for UI performance
    }

    private fun hourOfDay(epochMs: Long): Float {
        val cal = java.util.Calendar.getInstance().apply { timeInMillis = epochMs }
        return cal.get(java.util.Calendar.HOUR_OF_DAY) + cal.get(java.util.Calendar.MINUTE) / 60f
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
