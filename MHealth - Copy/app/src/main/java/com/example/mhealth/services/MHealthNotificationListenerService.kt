package com.example.mhealth.services

import android.service.notification.NotificationListenerService
import android.util.Log

/**
 * Minimal NotificationListenerService stub.
 *
 * We do NOT process notifications here (MonitoringService already counts them
 * via UsageEvents type-12). The sole purpose of this class is to give
 * MediaSessionManager a valid ComponentName so that:
 *
 *   msm.getActiveSessions(component)
 *   msm.addOnActiveSessionsChangedListener(listener, component)
 *
 * …are granted without requiring the privileged MEDIA_CONTENT_CONTROL
 * permission.  Android gives any app with a registered (and user-enabled)
 * NotificationListenerService the right to observe active media sessions.
 *
 * User setup: Settings → Notifications → Notification access → Cove → Allow.
 * MonitoringService already logs a warning when this hasn't been done.
 */
class MHealthNotificationListenerService : NotificationListenerService() {

    override fun onListenerConnected() {
        Log.i("MHealth.NLS", "NotificationListenerService connected — MediaSession access granted")
    }

    override fun onListenerDisconnected() {
        Log.i("MHealth.NLS", "NotificationListenerService disconnected")
    }
}
