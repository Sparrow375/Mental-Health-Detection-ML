package com.example.mhealth.services

import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import android.util.Log
import com.example.mhealth.logic.DataCollector
import com.example.mhealth.logic.DataRepository
import com.example.mhealth.logic.db.MHealthDatabase
import com.example.mhealth.logic.db.NotificationEventEntity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

/**
 * NotificationListenerService for Level 2 Digital DNA.
 *
 * Dual purpose:
 * 1. Provides a valid ComponentName for MediaSessionManager (background audio tracking).
 * 2. Logs notification arrival, dismissal, and tap events to Room for Digital DNA computation.
 *
 * Notification DNA feeds into:
 *   - PhoneDNA: notification_open_rate, dismiss_rate, ignore_rate
 *   - AppDNA: notification_response_latency per app
 *   - L2 Texture: notification_response_latency_shift, notification_to_session_ratio
 *
 * User setup: Settings → Notifications → Notification access → MHealth → Allow.
 */
class MHealthNotificationListenerService : NotificationListenerService() {

    companion object {
        private const val TAG = "MHealth.NLS"
    }

    /**
     * Internal record for tracking notification state.
     * Not persisted — only held in memory for real-time action attribution.
     */
    data class NotificationRecord(
        val key: String,
        val appPackage: String,
        val arrivalTimestampMs: Long,
        val dateStr: String
    )

    /** In-memory map of active notifications: sbn_key → NotificationRecord. */
    private val activeNotifications = mutableMapOf<String, NotificationRecord>()

    /** Track the most recently arrived notification per package for tap attribution. */
    private val lastArrivalPerPackage = mutableMapOf<String, NotificationRecord>()

    private val serviceScope = CoroutineScope(Dispatchers.IO + Job())
    private val dateFmt = SimpleDateFormat("yyyy-MM-dd", Locale.US)

    override fun onListenerConnected() {
        Log.i(TAG, "NotificationListenerService connected — MediaSession access + Notification DNA logging active")
        startIgnoreChecker()
    }

    override fun onListenerDisconnected() {
        Log.i(TAG, "NotificationListenerService disconnected")
        activeNotifications.clear()
        lastArrivalPerPackage.clear()
    }

    /**
     * Called when a new notification is posted.
     * Logs the arrival and tracks it for future action attribution.
     */
    override fun onNotificationPosted(sbn: StatusBarNotification) {
        val pkg = sbn.packageName

        // Skip system/foreground service notifications to avoid noise
        if (pkg == "android" || pkg == "com.android.systemui" || sbn.isOngoing) return

        val now = System.currentTimeMillis()
        val dateStr = dateFmt.format(Date(now))
        val key = sbn.key

        val record = NotificationRecord(
            key = key,
            appPackage = pkg,
            arrivalTimestampMs = now,
            dateStr = dateStr
        )

        activeNotifications[key] = record
        lastArrivalPerPackage[pkg] = record

        // Push to DataRepository so DataCollector.logSessionsFromEvents can use it for trigger detection
        DataRepository.setRecentNotificationTime(pkg, now)

        // Persist ARRIVAL event to Room for notification DNA
        val arrivalEntity = NotificationEventEntity(
            event_id = UUID.randomUUID().toString(),
            app_package = pkg,
            arrival_timestamp = now,
            action = "ARRIVAL",
            tap_latency_min = null,
            date = dateStr
        )
        persistEvent(arrivalEntity)

        Log.d(TAG, "Notification arrived: pkg=$pkg, key=$key")
    }

    /**
     * Called when a notification is removed (dismissed by user or auto-removed).
     * If the user actively dismissed it, we log DISMISS.
     * We cannot directly detect taps from NotificationListenerService — taps are inferred
     * when the user opens the corresponding app shortly after notification arrival.
     * That inference happens in DataCollector during app session tracking.
     */
    override fun onNotificationRemoved(sbn: StatusBarNotification) {
        val key = sbn.key
        val record = activeNotifications.remove(key)

        if (record != null) {
            // Log as DISMISS — if it was a tap, the session event logger will handle that
            // via the trigger=NOTIFICATION mechanism in DataCollector.
            // We still log the dismiss for the ignore/dismiss rate computation.
            val now = System.currentTimeMillis()
            val eventId = UUID.randomUUID().toString()

            val entity = NotificationEventEntity(
                event_id = eventId,
                app_package = record.appPackage,
                arrival_timestamp = record.arrivalTimestampMs,
                action = "DISMISS",
                tap_latency_min = null,
                date = record.dateStr
            )

            persistEvent(entity)
            Log.d(TAG, "Notification dismissed: pkg=${record.appPackage}")

            // Clean up last arrival tracking if this was the most recent
            if (lastArrivalPerPackage[record.appPackage]?.key == key) {
                lastArrivalPerPackage.remove(record.appPackage)
            }
        }
    }

    /**
     * Periodic coroutine that checks for IGNORED notifications (active 4+ hours).
     * Replaces the non-existent onActiveNotificationsChanged API.
     * Runs every 30 minutes while the listener is connected.
     */
    private fun startIgnoreChecker() {
        serviceScope.launch {
            val fourHoursMs = 4 * 60 * 60 * 1000L
            val checkIntervalMs = 30 * 60 * 1000L
            while (isActive) {
                delay(checkIntervalMs)
                val now = System.currentTimeMillis()
                val currentKeys = try {
                    getActiveNotifications()?.map { it.key }?.toSet() ?: emptySet()
                } catch (e: Exception) {
                    emptySet()
                }

                val toRemove = mutableListOf<String>()
                for ((key, record) in activeNotifications) {
                    if (key !in currentKeys) {
                        toRemove.add(key)
                        continue
                    }
                    val age = now - record.arrivalTimestampMs
                    if (age >= fourHoursMs) {
                        val entity = NotificationEventEntity(
                            event_id = UUID.randomUUID().toString(),
                            app_package = record.appPackage,
                            arrival_timestamp = record.arrivalTimestampMs,
                            action = "IGNORE",
                            tap_latency_min = null,
                            date = record.dateStr
                        )
                        persistEvent(entity)
                        Log.d(TAG, "Notification ignored (4h+): pkg=${record.appPackage}")
                        toRemove.add(key)
                        if (lastArrivalPerPackage[record.appPackage]?.key == key) {
                            lastArrivalPerPackage.remove(record.appPackage)
                        }
                    }
                }
                toRemove.forEach { activeNotifications.remove(it) }
            }
        }
    }

    /**
     * Public helper: log a TAP event when the DataCollector detects that an app was opened
     * shortly after a notification arrived from the same package.
     * Called from DataCollector's session tracking logic.
     */
    fun logNotificationTap(appPackage: String) {
        val record = lastArrivalPerPackage.remove(appPackage)
        val now = System.currentTimeMillis()

        if (record != null) {
            val latencyMs = now - record.arrivalTimestampMs
            val latencyMin = latencyMs / 60_000f

            // Remove from active notifications to prevent DISMISS logging
            activeNotifications.remove(record.key)

            val eventId = UUID.randomUUID().toString()
            val entity = NotificationEventEntity(
                event_id = eventId,
                app_package = appPackage,
                arrival_timestamp = record.arrivalTimestampMs,
                action = "TAP",
                tap_latency_min = latencyMin,
                date = record.dateStr
            )

            persistEvent(entity)
            Log.d(TAG, "Notification tapped: pkg=$appPackage, latency=${"%.1f".format(latencyMin)}min")
        } else {
            // No tracked arrival for this package — still log a TAP without latency
            val dateStr = dateFmt.format(Date(now))
            val eventId = UUID.randomUUID().toString()
            val entity = NotificationEventEntity(
                event_id = eventId,
                app_package = appPackage,
                arrival_timestamp = now,
                action = "TAP",
                tap_latency_min = null,
                date = dateStr
            )

            persistEvent(entity)
            Log.d(TAG, "Notification tapped (no arrival tracked): pkg=$appPackage")
        }
    }

    /**
     * Get the most recent arrival time for a package (used by DataCollector for trigger detection).
     */
    fun getLastArrivalTime(appPackage: String): Long? {
        return lastArrivalPerPackage[appPackage]?.arrivalTimestampMs
    }

    private fun persistEvent(entity: NotificationEventEntity) {
        serviceScope.launch {
            try {
                val db = MHealthDatabase.getInstance(this@MHealthNotificationListenerService)
                db.notificationEventDao().insert(entity)
                // Diagnostic: log today's event count for verification
                val todayCount = db.notificationEventDao().getByDate(entity.date).size
                Log.d(TAG, "Persisted ${entity.action} event for ${entity.app_package} | Today's total: $todayCount events")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to persist notification event: ${entity.action} for ${entity.app_package}", e)
            }
        }
    }
}