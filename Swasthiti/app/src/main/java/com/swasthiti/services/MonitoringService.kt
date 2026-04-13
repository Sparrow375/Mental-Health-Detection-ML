package com.swasthiti.services

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.ServiceInfo
import android.app.AlarmManager
import android.media.session.PlaybackState
import android.media.session.MediaController
import android.media.session.MediaSessionManager
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import com.swasthiti.MainActivity
import com.swasthiti.logic.AnomalyDetector
import com.swasthiti.logic.DataCollector
import com.swasthiti.logic.DataRepository
import com.swasthiti.logic.GpsStateManager
import com.swasthiti.logic.JsonConverter
import com.swasthiti.logic.db.AnalysisResultEntity
import com.swasthiti.logic.db.BaselineEntity
import com.swasthiti.logic.db.SwasthitiDatabase
import com.swasthiti.logic.db.UserProfileEntity
import com.swasthiti.models.DailyReport
import com.swasthiti.models.PersonalityVector
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.firestore.FirebaseFirestore
import kotlinx.coroutines.tasks.await
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.runBlocking
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

class MonitoringService : Service() {

    companion object {
        private const val TAG = "Swasthiti.Service"
    }

    private lateinit var dataCollector: DataCollector
    private lateinit var gpsStateManager: GpsStateManager
    private var detector: AnomalyDetector? = null

    private val serviceScope = CoroutineScope(Dispatchers.IO + Job())
    private var trackingJob: Job? = null
    private val dateFmt = SimpleDateFormat("yyyy-MM-dd", Locale.US)

    private val collectedDailyVectors = mutableListOf<PersonalityVector>()
    private var collectionTickCount = 0
    private var nightlyWorkerScheduled = false

    private var chargingStartMs: Long = -1L
    private var musicStartMs: Long = -1L
    private var activeMusicPackage: String? = null
    private var preRestartAudioMs: Long = 0L  // Tracks accumulated audio time before service restart
    private var alarmManager: AlarmManager? = null

    // CompletableDeferred ensures runTick() waits for Room rehydration without blocking the main thread.
    private val isRestored = CompletableDeferred<Unit>()

    // ── Screen and Interaction Receiver ──────────────────────────────────────
    // Triggers runTick() on user events so charts stay fresh without polling.
    private val interactiveReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.action) {
                Intent.ACTION_SCREEN_ON, Intent.ACTION_USER_PRESENT -> {
                    // Update UI snapshot when the user is actually looking
                    serviceScope.launch { runTick() }
                }
                Intent.ACTION_SCREEN_OFF -> {
                    // Save state when user puts phone away
                    serviceScope.launch { runTick() }
                }
            }
        }
    }

    // ── Precise charging tracker ──
    private val powerReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.action) {
                Intent.ACTION_POWER_CONNECTED -> {
                    chargingStartMs = System.currentTimeMillis()
                    Log.i("Swasthiti.Service", "Charger connected at $chargingStartMs")
                }
                Intent.ACTION_POWER_DISCONNECTED -> {
                    if (chargingStartMs > 0L) {
                        val sessionMs = System.currentTimeMillis() - chargingStartMs
                        val sessionHrs = sessionMs / 3_600_000f
                        DataRepository.addChargeTime(sessionHrs)
                        Log.i("Swasthiti.Service", "Charger disconnected — added %.2fh (total %.2fh)"
                            .format(sessionHrs, DataRepository.accumulatedChargeHours.value))
                        chargingStartMs = -1L
                    }
                }
            }
        }
    }

    // ── DND (Interruption Filter) Receiver ───────────────────────────────────
    private val dndReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action == NotificationManager.ACTION_INTERRUPTION_FILTER_CHANGED) {
                val notifManager = context.getSystemService(NotificationManager::class.java)
                val filter = notifManager.currentInterruptionFilter
                val now = System.currentTimeMillis()
                
                // If DND is ON (PRIORITY, ALARMS, or NONE)
                if (filter == NotificationManager.INTERRUPTION_FILTER_PRIORITY || 
                    filter == NotificationManager.INTERRUPTION_FILTER_ALARMS || 
                    filter == NotificationManager.INTERRUPTION_FILTER_NONE) {
                    DataRepository.setDndOnTimestamp(now)
                    Log.i("Swasthiti.Service", "DND turned ON at $now")
                } 
                // If DND is OFF (ALL)
                else if (filter == NotificationManager.INTERRUPTION_FILTER_ALL) {
                    DataRepository.setDndOffTimestamp(now)
                    Log.i("Swasthiti.Service", "DND turned OFF at $now")
                }
            }
        }
    }

    // ── Music/Audio Event Listener ───────────────────────────────────────────
    private val sessionListener = MediaSessionManager.OnActiveSessionsChangedListener { controllers ->
        updateMusicSessionState(controllers)
    }

    private fun updateMusicSessionState(controllers: List<MediaController>?) {
        val activeMusicController = controllers?.firstOrNull { controller ->
            val pkg = controller.packageName
            val isPlaying = controller.playbackState?.state == PlaybackState.STATE_PLAYING
            isPlaying && dataCollector.isMusicApp(pkg)
        }
        val isMusicPlaying = activeMusicController != null
        val pkg = activeMusicController?.packageName

        val now = System.currentTimeMillis()
        if (isMusicPlaying && musicStartMs == -1L) {
            // New session starting - reset pre-restart accumulator
            preRestartAudioMs = 0L
            musicStartMs = now
            activeMusicPackage = pkg
            Log.i("Swasthiti.Service", "Music session detected: $activeMusicPackage — tracking started")
        } else if (!isMusicPlaying && musicStartMs > 0L) {
            // Session ending - add both the current session time AND any pre-restart accumulated time
            val sessionMs = (now - musicStartMs) + preRestartAudioMs
            DataRepository.addBgAudioTime(activeMusicPackage, sessionMs)
            Log.i("Swasthiti.Service", "Music session ended: $activeMusicPackage — added ${sessionMs}ms (session: ${now - musicStartMs}ms, pre-restart: ${preRestartAudioMs}ms)")
            musicStartMs = -1L
            activeMusicPackage = null
            preRestartAudioMs = 0L
        }
    }

    override fun onCreate() {
        super.onCreate()
        // FIX 1: START FOREGROUND IMMEDIATELY. Android 12+ requires startForeground
        // within ~5 seconds or the app crashes. Moving this to the top.
        startForegroundNotification()

        dataCollector = DataCollector(this)
        gpsStateManager = GpsStateManager(this)

        // FIX 2: ASYNC REHYDRATION. Moving restoreStateFromRoomSuspend() out of runBlocking.
        // runBlocking blocks the main thread, which can cause ANRs or OS-level kills during startup.
        serviceScope.launch {
            try {
                restoreStateFromRoomSuspend()
            } finally {
                isRestored.complete(Unit) // Unlock runTick() even if restoration fails
            }
        }

        // FIX 3: Prime the UI.
        serviceScope.launch { runTick() }

        scheduleMonitoring()

        // Wire Room-backed StateFlows so AnalysisScreen/InsightsScreen update reactively
        val userId = DataRepository.userProfile.value?.email ?: "default_user"
        DataRepository.initWithDb(applicationContext, userId)

        // ADAPTIVE GPS: Observe GPS state changes and update repository for UI
        serviceScope.launch {
            gpsStateManager.currentState.collect { state ->
                DataRepository.updateGpsState(state.displayName)
                Log.i("Swasthiti.Service", "GPS State changed: ${state.displayName} (interval: ${state.intervalMs / 60_000}min)")
            }
        }

        // FIX 7: Register receivers for event-driven monitoring
        registerReceiver(powerReceiver, IntentFilter().apply {
            addAction(Intent.ACTION_POWER_CONNECTED)
            addAction(Intent.ACTION_POWER_DISCONNECTED)
        })

        registerReceiver(interactiveReceiver, IntentFilter().apply {
            addAction(Intent.ACTION_SCREEN_ON)
            addAction(Intent.ACTION_SCREEN_OFF)
            addAction(Intent.ACTION_USER_PRESENT)
        })

        registerReceiver(dndReceiver, IntentFilter(NotificationManager.ACTION_INTERRUPTION_FILTER_CHANGED))

        // FIX 8: Register for MediaSession changes (requires notification access)
        // We pass a ComponentName pointing to our declared NotificationListenerService.
        // Without this, Android throws SecurityException on any getActiveSessions() call
        // from a 3rd-party app that lacks MEDIA_CONTENT_CONTROL (a privileged permission).
        try {
            val nlsComponent = ComponentName(this, SwasthitiNotificationListenerService::class.java)
            val msm = getSystemService(MediaSessionManager::class.java)
            msm?.addOnActiveSessionsChangedListener(sessionListener, nlsComponent)
            // Initial check for already-playing music when service starts
            updateMusicSessionState(msm?.getActiveSessions(nlsComponent))
        } catch (e: SecurityException) {
            Log.w("Swasthiti.Service", "MediaSession listener failed (Notification Access not granted by user) — background audio tracking disabled until access is granted")
        }

        // FIX 9: Setup Midnight Alarm for exact day transition
        setupMidnightAlarm()

        // If the phone is ALREADY charging/playing when the service starts, anchor state now
        val batteryOnStart = dataCollector.getBatteryInfo()
        if (batteryOnStart.isCharging) {
            chargingStartMs = System.currentTimeMillis()
            Log.i("Swasthiti.Service", "Service started while already charging — anchored at $chargingStartMs")
        }
        // Music state is already anchored via updateMusicSessionState call above

        // ── Reactive slider: two-way status sync on every slider move ─────────────
        serviceScope.launch {
            DataRepository.baselineDaysRequired.collectLatest { newTarget ->
                val collected = collectedDailyVectors.size
                when {
                    collected >= newTarget && DataRepository.isBuildingBaseline.value -> {
                        Log.d("Swasthiti.Service", "Slider -> $newTarget (have $collected) — finalizing baseline now")
                        checkAndFinalizeBaseline(newTarget)
                    }
                    collected < newTarget && !DataRepository.isBuildingBaseline.value -> {
                        Log.d("Swasthiti.Service", "Slider -> $newTarget (have $collected) — reverting to Building Baseline")
                        nightlyWorkerScheduled = false
                        DataRepository.setIsBuildingBaseline(true)
                    }
                }
            }
        }

        // Start passive continuous location tracking with adaptive intervals
        dataCollector.startContinuousLocationTracking()
    }

    // FIX 5: Suspend version of restore — called from runBlocking in onCreate
    // so we BLOCK the main service thread until Room data is fully loaded.
    private suspend fun restoreStateFromRoomSuspend() {
        val userId = DataRepository.userProfile.value?.email ?: "default_user"
        try {
            val db = SwasthitiDatabase.getInstance(this@MonitoringService)
            val profile = db.userProfileDao().get(userId)

            if (profile?.baselineReady == true) {
                val baselineEntities = db.baselineDao().getBaseline(userId)
                if (baselineEntities.isNotEmpty()) {
                    val baselineFields = baselineEntities.associate { it.featureName to it.baselineValue }
                    val variances = baselineEntities.associate { it.featureName to it.stdDeviation }
                    val baseline = PersonalityVector(
                        screenTimeHours = baselineFields["screenTimeHours"] ?: 0f,
                        unlockCount = baselineFields["unlockCount"] ?: 0f,
                        appLaunchCount = baselineFields["appLaunchCount"] ?: 0f,
                        notificationsToday = baselineFields["notificationsToday"] ?: 0f,
                        socialAppRatio = baselineFields["socialAppRatio"] ?: 0f,
                        callsPerDay = baselineFields["callsPerDay"] ?: 0f,
                        callDurationMinutes = baselineFields["callDurationMinutes"] ?: 0f,
                        uniqueContacts = baselineFields["uniqueContacts"] ?: 0f,
                        conversationFrequency = baselineFields["conversationFrequency"] ?: 0f,
                        dailyDisplacementKm = baselineFields["dailyDisplacementKm"] ?: 0f,
                        locationEntropy = baselineFields["locationEntropy"] ?: 0f,
                        homeTimeRatio = baselineFields["homeTimeRatio"] ?: 0f,
                        placesVisited = baselineFields["placesVisited"] ?: 0f,
                        wakeTimeHour = baselineFields["wakeTimeHour"] ?: 0f,
                        sleepTimeHour = baselineFields["sleepTimeHour"] ?: 0f,
                        sleepDurationHours = baselineFields["sleepDurationHours"] ?: 0f,
                        darkDurationHours = baselineFields["darkDurationHours"] ?: 0f,
                        chargeDurationHours = baselineFields["chargeDurationHours"] ?: 0f,
                        memoryUsagePercent = baselineFields["memoryUsagePercent"] ?: 0f,
                        networkWifiMB = baselineFields["networkWifiMB"] ?: 0f,
                        networkMobileMB = baselineFields["networkMobileMB"] ?: 0f,
                        downloadsToday = baselineFields["downloadsToday"] ?: 0f,
                        storageUsedGB = baselineFields["storageUsedGB"] ?: 0f,
                        appUninstallsToday = baselineFields["appUninstallsToday"] ?: 0f,
                        upiTransactionsToday = baselineFields["upiTransactionsToday"] ?: 0f,
                        totalAppsCount = baselineFields["totalAppsCount"] ?: 0f,
                        backgroundAudioHours = baselineFields["backgroundAudioHours"] ?: 0f,
                        mediaCountToday = baselineFields["mediaCountToday"] ?: 0f,
                        appInstallsToday = baselineFields["appInstallsToday"] ?: 0f,
                        dailySteps = baselineFields["dailySteps"] ?: 0f,
                        calendarEventsToday = baselineFields["calendarEventsToday"] ?: 0f,
                        variances = variances.toMutableMap()
                    )
                    DataRepository.setBaseline(baseline)

                    // FIX: Load historical anomaly scores from Room for pattern detection
                    val pastAnalysisResults = db.analysisResultDao().getLatestN(userId, 14)
                        .reversed()  // oldest first
                        .map { it.anomalyScore }

                    detector = AnomalyDetector(baseline, pastAnalysisResults)
                    Log.i(TAG, "AnomalyDetector initialized with ${pastAnalysisResults.size} historical scores")
                    
                    // FIX: Ensure workers are scheduled since baseline is already ready.
                    // Previously they were only scheduled at the exact moment of finalization.
                    scheduleNightlyWorker()
                }
            }

            // Always load recent history for the Recent Trends UI, whether building or actively monitoring
            val pastFeatures = db.dailyFeaturesDao().getLatestN(userId, 60).reversed()
            val pastVectors = pastFeatures.map { JsonConverter.toPersonalityVector(it) }
            collectedDailyVectors.clear()
            collectedDailyVectors.addAll(pastVectors)
            DataRepository.updateBaselineProgress(collectedDailyVectors.size)
            DataRepository.updateCollectedBaselineVectors(collectedDailyVectors)

            // Immediate check for baseline readiness on startup
            if (DataRepository.isBuildingBaseline.value) {
                val target = DataRepository.baselineDaysRequired.value
                checkAndFinalizeBaseline(target)
            }

            // FIX: Re-anchor audio session if music was already playing when the service was
            // killed and restarted. Without this, the in-progress session is silently dropped
            // because musicStartMs resets to -1L on every service start.
            // We capture the already-accumulated time from SharedPreferences so it's not lost,
            // then add the full session duration (pre-restart + post-restart) when music stops.
            val resumingPkg = isMusicAppActiveViaMediaSession()
            if (resumingPkg != null && musicStartMs == -1L) {
                // Capture already-accumulated time before resetting daily counter
                preRestartAudioMs = DataRepository.accumulatedBgAudioMs.value
                musicStartMs = System.currentTimeMillis()
                activeMusicPackage = resumingPkg
                Log.i("Swasthiti.Service", "Audio session re-anchored after service restart: $resumingPkg (pre-accumulated: ${preRestartAudioMs}ms)")
            }

            // Sync any unsynced data from previous sessions on startup
            syncUnstagedDailyFeaturesToFirebase()

            // FIX 1: Prime lastProcessedDay on first run so midnight day-transitions fire correctly.
            // If the service is killed before the first midnight cycle completes, lastProcessedDay
            // stays -1 and the guard (savedDay != -1) permanently skips all future transitions.
            if (DataRepository.lastProcessedDay.value == -1) {
                val todayDoy = Calendar.getInstance().get(Calendar.DAY_OF_YEAR)
                DataRepository.setLastProcessedDay(todayDoy)
                Log.i("Swasthiti.Service", "Primed lastProcessedDay=$todayDoy on first service start")
            }

            // FIX 4: Recover missed yesterday snapshot if the service was killed before the
            // midnight transition had a chance to fire (e.g., Android Doze / battery optimiser).
            recoverMissedDayIfNeeded(userId, db)

        } catch (e: Exception) {
            Log.e("Swasthiti.Service", "Error restoring state from Room", e)
        }
    }

    // Legacy async wrapper kept so the reset flow (which launches a coroutine) still works
    private fun restoreStateFromRoom() {
        serviceScope.launch { restoreStateFromRoomSuspend() }
    }

    private fun startForegroundNotification() {
        val channelId = "Swasthiti_monitoring"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                channelId, "Swasthiti Monitoring", NotificationManager.IMPORTANCE_LOW
            ).apply { description = "Passive mental health pattern monitoring" }
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
        val notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("Swasthiti Active")
            .setContentText("Passively monitoring device patterns")
            .setSmallIcon(android.R.drawable.ic_menu_info_details)
            .setContentIntent(
                PendingIntent.getActivity(
                    this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE
                )
            )
            .setOngoing(true)
            .build()
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(1, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION or ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        } else {
            startForeground(1, notification)
        }
    }

    private fun scheduleMonitoring() {
        // Observers stay active for configuration changes
        serviceScope.launch {
            DataRepository.monitoringIntervalMinutes.collectLatest { intervalMin ->
                Log.d("Swasthiti.Service", "Monitoring loop re-enabled. Tick every $intervalMin min.")
                while (isActive) {
                    runTick()
                    delay(intervalMin * 60 * 1000L)
                }
            }
        }

        // ── Baseline days observer: build baseline immediately if requirement is met ──
        serviceScope.launch {
            DataRepository.baselineDaysRequired.collect { target ->
                checkAndFinalizeBaseline(target)
            }
        }
        
        // dev force new-day trigger listener
        serviceScope.launch {
            DataRepository.forceNewDayTrigger.collect { triggers ->
                if (triggers > 0) {
                    val today = Calendar.getInstance().get(Calendar.DAY_OF_YEAR)
                    DataRepository.setLastProcessedDay(if (today <= 1) 365 else today - 1)
                    runTick(isSimulated = true)
                }
            }
        }

        // dev force reset trigger listener
        serviceScope.launch {
            DataRepository.resetTrigger.collect { triggers ->
                if (triggers > 0) {
                    val userId = DataRepository.userProfile.value?.email ?: "default_user"
                    val db = SwasthitiDatabase.getInstance(this@MonitoringService)
                    try {
                        db.dailyFeaturesDao().clearSimulated(userId)
                        
                        val targetBaselineDays = DataRepository.baselineDaysRequired.value
                        val remainingFeatures = db.dailyFeaturesDao().getLatestN(userId, 60)
                        if (remainingFeatures.size < targetBaselineDays) {
                            db.baselineDao().clearBaseline(userId)
                            db.analysisResultDao().clearAll(userId)
                            val profile = db.userProfileDao().get(userId)
                            if (profile != null) {
                                db.userProfileDao().upsert(profile.copy(baselineReady = false))
                            }
                            DataRepository.setIsBuildingBaseline(true)
                            detector = null
                        }
                        
                        restoreStateFromRoom()

                        // IMPORTANT: Refresh the "Live" UI snapshots immediately so the user 
                        // sees their current day's progress (distance, etc.) right after reset.
                        runTick()
                    } catch (e: Exception) {
                        Log.e("Swasthiti.Service", "Error during master reset", e)
                    }
                }
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        // Flush any remaining charge session on service stop
        if (chargingStartMs > 0L) {
            val sessionMs  = System.currentTimeMillis() - chargingStartMs
            val sessionHrs = sessionMs / 3_600_000f
            DataRepository.addChargeTime(sessionHrs)
            chargingStartMs = -1L
        }
        // Flush any remaining music session
        if (musicStartMs > 0L) {
            DataRepository.addBgAudioTime(activeMusicPackage, System.currentTimeMillis() - musicStartMs)
            musicStartMs = -1L
        }
        try { unregisterReceiver(powerReceiver) } catch (_: Exception) {}
        try { unregisterReceiver(interactiveReceiver) } catch (_: Exception) {}
        try { unregisterReceiver(dndReceiver) } catch (_: Exception) {}
        try {
            val msm = getSystemService(MediaSessionManager::class.java)
            msm?.removeOnActiveSessionsChangedListener(sessionListener)
        } catch (_: Exception) {}
        
        serviceScope.cancel()
        dataCollector.stopContinuousLocationTracking()
    }

    private suspend fun runTick(isSimulated: Boolean = false) {
        // Wait for Room database rehydration to finish so we don't calculate on empty state
        isRestored.await()
        
        try {
            collectionTickCount++

            // BUG FIX: Proactively capture a GPS fix every tick so distance is
            // never 0.0 just because the passive 50m-displacement listener didn't fire.
            // This runs async (non-blocking) and writes into DataRepository before the
            // snapshot is collected below on the next tick.
            if (!isSimulated) {
                dataCollector.captureProactiveLocationSnapshot()
            }

            // 1) Collect a "Live" snapshot (Midnight to Now) for UI updates
            val locationSnaps = DataRepository.locationSnapshots.value
            val liveSnapshot = dataCollector.collectSnapshot(locationSnaps)

            // Always update live data for home screen
            DataRepository.updateLatestVector(liveSnapshot)
            DataRepository.addHourlySnapshot(liveSnapshot)

            // 2) Battery tracking is now handled by the exact-timestamp BroadcastReceiver
            //    (powerReceiver). No per-tick polling here — that caused 15-min rounding errors.

            // Level 2 Digital DNA: Log session events from UsageEvents for behavioral DNA
            dataCollector.logSessionsFromEvents(dataCollector.getStartOfDayMs(), System.currentTimeMillis())

            // 3) Background audio: Event-driven via sessionListener (no tick polling needed)

            // 4) Provisional Analysis (Live Score)
            if (!DataRepository.isBuildingBaseline.value && detector != null) {
                val provisionalReport = detector?.analyze(liveSnapshot, DataRepository.analysisHistory.value.size + 1, isProvisional = true)
                provisionalReport?.let {
                    // Update the Repository so the UI can show the live score
                    DataRepository.updateProvisionalAnalysis(it)
                }
            }

            val today = Calendar.getInstance().get(Calendar.DAY_OF_YEAR)
            val savedDay = DataRepository.lastProcessedDay.value

            // 3) Day Transition Logic: Capture the FULL profile of the day that just ended
            if (today != savedDay && savedDay != -1) {
                // Determine the 24-hour range for the day that ended (Midnight to Midnight)
                val startOfToday = dataCollector.getStartOfDayMs()
                val yesterdayStart = startOfToday - 24 * 3600_000L
                val yesterdayEnd = startOfToday - 1L

                Log.i("Swasthiti.Service", "Day Transition Detected: Capturing full profile for Day $savedDay [Range: $yesterdayStart to $yesterdayEnd]")
                
                // Collect a 100% accurate snapshot for the entire prior day
                val fullDaySnapshot = dataCollector.collectSnapshot(locationSnaps, yesterdayStart, yesterdayEnd)

                // Record this full day in history/baseline
                if (DataRepository.isBuildingBaseline.value) {
                    handleBaselineBuilding(fullDaySnapshot, today, savedDay, isSimulated)
                } else {
                    handleAnomalyDetection(fullDaySnapshot, today, savedDay, isSimulated)
                }

                // Reset daily accumulators for the new day
                DataRepository.resetDailyState()
                gpsStateManager.reset()  // Reset GPS state machine to STATIONARY
                DataRepository.setLastProcessedDay(today)
            } else {
                // Regular tick within the same day: just check if baseline was completed by manual settings change
                if (DataRepository.isBuildingBaseline.value) {
                    val target = DataRepository.baselineDaysRequired.value
                    checkAndFinalizeBaseline(target)
                }
            }
        } catch (e: Exception) {
            Log.e("Swasthiti.Service", "Error in runTick", e)
        }
    }

    private fun handleBaselineBuilding(snapshot: PersonalityVector, today: Int, savedDay: Int, isSimulated: Boolean) {
        val targetBaselineDays = DataRepository.baselineDaysRequired.value
        if (today != savedDay && savedDay != -1) {
            // Only add the vector if we still need more days for the baseline
            if (collectedDailyVectors.size < targetBaselineDays) {
                collectedDailyVectors.add(snapshot)
                DataRepository.updateBaselineProgress(collectedDailyVectors.size)
                DataRepository.updateCollectedBaselineVectors(collectedDailyVectors)

                // Persist end-of-day snapshot to Room
                persistDailySnapshot(snapshot, savedDay, isSimulated)
            }
            // FIX 2: Always attempt finalization after a day transition — not just when the last
            // vector was added. This handles the case where the user lowered baselineDaysRequired
            // before the previous midnight cycle ran (size was already >= target at transition).
            checkAndFinalizeBaseline(targetBaselineDays)
        } else {
            // Same-day tick: check if baseline is now ready (e.g. user lowered the setting mid-day)
            checkAndFinalizeBaseline(targetBaselineDays)
        }
    }

    /**
     * Finalizes the baseline building phase if requirements are met.
     * Sets P₀ in DataRepository and Room, and flips the isBuilding flag.
     */
    private fun checkAndFinalizeBaseline(targetBaselineDays: Int) {
        if (DataRepository.isBuildingBaseline.value && collectedDailyVectors.size >= targetBaselineDays && targetBaselineDays > 0) {
            val baseline = buildBaseline(collectedDailyVectors)

            // FIX 1: Do NOT flip isBuildingBaseline or set detector here.
            // If we do it before Room is written and the coroutine fails silently,
            // isBuildingBaseline stays false, this block never re-enters, and the
            // baseline is permanently lost in memory only.
            // Instead, set the flag first to prevent concurrent launch, then
            // flip state only AFTER Room write succeeds.
            if (!nightlyWorkerScheduled) {
                nightlyWorkerScheduled = true // guard against double-launch
                serviceScope.launch {
                    try {
                        persistBaselineToRoom(baseline, targetBaselineDays)
                        // Only flip the live state AFTER Room is confirmed written
                        DataRepository.setBaseline(baseline)
                        detector = AnomalyDetector(baseline)
                        scheduleNightlyWorker()
                        Log.i("Swasthiti.Service", "Baseline established and persisted to Room (${targetBaselineDays}d)")
                    } catch (e: Exception) {
                        // Allow retry on the next tick by resetting the guard
                        nightlyWorkerScheduled = false
                        Log.e("Swasthiti.Service", "Failed to persist baseline to Room — will retry on next tick: ${e.message}", e)
                    }
                }
            }
        }
    }

    private fun handleAnomalyDetection(snapshot: PersonalityVector, today: Int, savedDay: Int, isSimulated: Boolean) {
        val report = detector?.analyze(snapshot, DataRepository.reports.value.size + 1)
        report?.let {
            if (today != savedDay && savedDay != -1) {
                // Persist end-of-monitoring-day snapshot to Room
                persistDailySnapshot(snapshot, savedDay, isSimulated)

                // FIX: Persist anomaly score to AnalysisResultEntity (not just in-memory)
                persistAnomalyResultToRoom(report, savedDay, isSimulated)

                DataRepository.addReport(it)
                if (it.alertLevel == "orange" || it.alertLevel == "red") {
                    sendAlertNotification(it.alertLevel, it.notes)
                }
            }
        }
    }

    // ── Room persistence helpers ──────────────────────────────────────────────

    private fun persistDailySnapshot(snapshot: PersonalityVector, dayOfYear: Int, isSimulated: Boolean) {
        val userId = DataRepository.userProfile.value?.email ?: "default_user"

        // FIX 3: Safe year-boundary-proof date computation.
        // set(DAY_OF_YEAR, x) is wrong on Jan 1 when dayOfYear refers to the previous Dec 31.
        // Instead, compute how many days ago dayOfYear was and subtract that many days from today.
        val nowCal   = Calendar.getInstance()
        val todayDoy = nowCal.get(Calendar.DAY_OF_YEAR)
        val daysAgo  = if (todayDoy >= dayOfYear) {
            todayDoy - dayOfYear
        } else {
            // dayOfYear is from the previous year (e.g., recording wraps Dec->Jan)
            val prevYearCal      = Calendar.getInstance().apply { set(Calendar.YEAR, nowCal.get(Calendar.YEAR) - 1) }
            val daysInPrevYear   = prevYearCal.getActualMaximum(Calendar.DAY_OF_YEAR)
            daysInPrevYear - dayOfYear + todayDoy
        }
        val cal     = Calendar.getInstance().apply { add(Calendar.DAY_OF_YEAR, -daysAgo.coerceAtLeast(0)) }
        val dateStr = dateFmt.format(cal.time)

        val entity = JsonConverter.fromPersonalityVector(userId, dateStr, snapshot, isSimulated)
        serviceScope.launch {
            try {
                SwasthitiDatabase.getInstance(this@MonitoringService)
                    .dailyFeaturesDao().insert(entity)

                // Automatically push un-synced data to Firebase database
                syncUnstagedDailyFeaturesToFirebase()
            } catch (e: Exception) {
                Log.e("Swasthiti.Service", "Error persisting daily snapshot", e)
            }
        }
    }

    /**
     * FIX: Persist anomaly detection results to Room immediately when detected.
     * Previously, only NightlyAnalysisWorker wrote to analysis_results table,
     * so days analyzed by MonitoringService had no anomaly scores in the export.
     */
    private fun persistAnomalyResultToRoom(report: DailyReport, dayOfYear: Int, isSimulated: Boolean) {
        val userId = DataRepository.userProfile.value?.email ?: "default_user"

        // Compute the date string for the day that just ended
        val nowCal   = Calendar.getInstance()
        val todayDoy = nowCal.get(Calendar.DAY_OF_YEAR)
        val daysAgo  = if (todayDoy >= dayOfYear) {
            todayDoy - dayOfYear
        } else {
            val prevYearCal = Calendar.getInstance().apply { set(Calendar.YEAR, nowCal.get(Calendar.YEAR) - 1) }
            val daysInPrevYear = prevYearCal.getActualMaximum(Calendar.DAY_OF_YEAR)
            daysInPrevYear - dayOfYear + todayDoy
        }
        val cal = Calendar.getInstance().apply { add(Calendar.DAY_OF_YEAR, -daysAgo.coerceAtLeast(0)) }
        val dateStr = dateFmt.format(cal.time)

        serviceScope.launch {
            try {
                val db = SwasthitiDatabase.getInstance(this@MonitoringService)

                // Check if result already exists for this date (avoid duplicates)
                val existing = db.analysisResultDao().getByDate(userId, dateStr)
                if (existing != null) {
                    Log.w(TAG, "Anomaly result already exists for $dateStr, skipping duplicate")
                    return@launch
                }

                val resultEntity = AnalysisResultEntity(
                    userId = userId,
                    date = dateStr,
                    anomalyDetected = report.alertLevel != "green",
                    anomalyMessage = report.notes,
                    anomalyScore = report.anomalyScore,
                    sustainedDays = report.sustainedDeviationDays,
                    alertLevel = report.alertLevel,
                    prototypeMatch = report.patternType,
                    matchMessage = report.flaggedFeatures.joinToString(", "),
                    prototypeConfidence = report.evidenceAccumulated,
                    gateResults = "{}"
                )
                db.analysisResultDao().insert(resultEntity)
                Log.i(TAG, "Anomaly result persisted for $dateStr: score=${report.anomalyScore}, level=${report.alertLevel}")
            } catch (e: Exception) {
                Log.e(TAG, "Error persisting anomaly result to Room", e)
            }
        }
    }

    private suspend fun persistBaselineToRoom(
        baseline: PersonalityVector,
        baselineDays: Int
    ) {
        val userId = DataRepository.userProfile.value?.email ?: "default_user"
        val db = SwasthitiDatabase.getInstance(this@MonitoringService)
        val today = dateFmt.format(Date())

        val entities = baseline.toMap().map { (feature, mean) ->
            BaselineEntity(
                userId         = userId,
                featureName    = feature,
                baselineValue  = mean,
                stdDeviation   = baseline.variances[feature] ?: 1f,
                baselineStart  = today,
                baselineEnd    = today
            )
        }
        db.baselineDao().insertAll(entities)
        db.userProfileDao().upsert(
            UserProfileEntity(
                userId        = userId,
                baselineReady = true,
                baselineDays  = baselineDays,
                currentStatus = "Monitoring"
            )
        )

        // Upload firmly established baseline to Cloud Backup
        val uid = FirebaseAuth.getInstance().currentUser?.uid
        if (uid != null) {
            try {
                val firestore = FirebaseFirestore.getInstance()
                // Use set(merge=true) instead of update() so this works even when the
                // user document doesn't exist yet in Firestore (update() throws if missing).
                firestore.collection("users").document(uid)
                    .set(mapOf("baseline_ready" to true), com.google.firebase.firestore.SetOptions.merge()).await()
                
                val baselineRef = firestore.collection("users").document(uid).collection("baseline")
                baseline.toMap().forEach { (feature, mean) ->
                    val std = baseline.variances[feature] ?: 1f
                    val data = hashMapOf(
                        "featureName" to feature,
                        "baselineValue" to mean,
                        "stdDeviation" to std,
                        "baselineStart" to today,
                        "baselineEnd" to today
                    )
                    baselineRef.document(feature).set(data).await()
                }
            } catch (e: Exception) {
                Log.e("Swasthiti.Service", "Error syncing baseline to Firebase", e)
            }
        }
    }

    private fun scheduleNightlyWorker() {
        val userId = DataRepository.userProfile.value?.email ?: "default_user"
        NightlyAnalysisWorker.schedule(this@MonitoringService, userId)
        // Also ensure the 4-hour periodic cloud sync is active
        CloudSyncWorker.schedulePeriodic(this@MonitoringService)
    }

    /**
     * Normalizes a clock hour to a "Noon-Offset" scale to eliminate the midnight cliff.
     * Standard:     Midnight=0,  6PM=18,  11PM=23
     * Noon-Offset:  Noon=0,      6PM=6,   Midnight=12,  6AM=18,  11:59AM≈24
     *
     * This keeps the entire 6PM→12PM sleep window on one continuous linear scale,
     * making means and standard deviations mathematically correct.
     * The raw value stored in the DB / shown in the UI is never changed.
     */
    private fun normalizeTimeToNoon(rawHour: Float): Float = (rawHour - 12f + 24f) % 24f

    private fun buildBaseline(vectors: List<PersonalityVector>): PersonalityVector {
        if (vectors.isEmpty()) return PersonalityVector()
        val features = vectors.first().toMap().keys
        val averages = mutableMapOf<String, Float>()
        val variances = mutableMapOf<String, Float>()

        val n = vectors.size
        // 10% Trimmed Mean: Removes extreme 10% high & 10% low outliers from the calibration period
        val trimCount = (n * 0.10).toInt().coerceAtLeast(0)

        // Features whose raw 24h value crosses midnight — apply Noon-Offset before any math.
        val circularTimeFeatures = setOf("sleepTimeHour", "wakeTimeHour")

        features.forEach { feature ->
            val vals = vectors.map {
                val raw = it.toMap()[feature] ?: 0f
                if (feature in circularTimeFeatures) normalizeTimeToNoon(raw) else raw
            }

            // Trim outliers for robust mean calculation
            val sortedVals = vals.sorted()
            val trimmedVals = if (n > 4 && trimCount > 0) {
                sortedVals.subList(trimCount, n - trimCount)
            } else {
                sortedVals
            }

            val robustAvg = trimmedVals.average().toFloat()

            // IMPORTANT: variance is computed in Noon-Offset space (correct scale).
            // The MEAN is stored back in raw 0-24h format so the downstream AnomalyDetector
            // can normalize both sides (current + baseline) from raw — preventing double-normalization.
            averages[feature] = if (feature in circularTimeFeatures) {
                (robustAvg + 12f) % 24f   // de-normalize noon-offset → raw
            } else {
                robustAvg
            }

            // Variance is computed against the noon-offset average using the full valid dataset
            variances[feature] = calculateSD(vals, robustAvg)
        }

        return PersonalityVector(
            screenTimeHours = averages["screenTimeHours"] ?: 0f,
            unlockCount = averages["unlockCount"] ?: 0f,
            appLaunchCount = averages["appLaunchCount"] ?: 0f,
            notificationsToday = averages["notificationsToday"] ?: 0f,
            socialAppRatio = averages["socialAppRatio"] ?: 0f,
            callsPerDay = averages["callsPerDay"] ?: 0f,
            callDurationMinutes = averages["callDurationMinutes"] ?: 0f,
            uniqueContacts = averages["uniqueContacts"] ?: 0f,
            dailyDisplacementKm = averages["dailyDisplacementKm"] ?: 0f,
            locationEntropy = averages["locationEntropy"] ?: 0f,
            homeTimeRatio = averages["homeTimeRatio"] ?: 0f,
            placesVisited = averages["placesVisited"] ?: 0f,
            wakeTimeHour = averages["wakeTimeHour"] ?: 0f,
            sleepTimeHour = averages["sleepTimeHour"] ?: 0f,
            sleepDurationHours = averages["sleepDurationHours"] ?: 0f,
            darkDurationHours = averages["darkDurationHours"] ?: 0f,
            chargeDurationHours = averages["chargeDurationHours"] ?: 0f,
            conversationFrequency = averages["conversationFrequency"] ?: 0f,
            memoryUsagePercent = averages["memoryUsagePercent"] ?: 0f,
            networkWifiMB = averages["networkWifiMB"] ?: 0f,
            networkMobileMB = averages["networkMobileMB"] ?: 0f,
            downloadsToday = averages["downloadsToday"] ?: 0f,
            storageUsedGB = averages["storageUsedGB"] ?: 0f,
            appUninstallsToday = averages["appUninstallsToday"] ?: 0f,
            upiTransactionsToday = averages["upiTransactionsToday"] ?: 0f,
            totalAppsCount = averages["totalAppsCount"] ?: 0f,
            backgroundAudioHours = averages["backgroundAudioHours"] ?: 0f,
            mediaCountToday = averages["mediaCountToday"] ?: 0f,
            appInstallsToday = averages["appInstallsToday"] ?: 0f,
            dailySteps = averages["dailySteps"] ?: 0f,
            calendarEventsToday = averages["calendarEventsToday"] ?: 0f,
            variances = variances
        )
    }

    /** Returns standard deviation (used as variance bound in AnomalyDetector) */
    private fun calculateSD(values: List<Float>, mean: Float): Float {
        if (values.size < 2) return 1.0f
        val sd = kotlin.math.sqrt(values.map { (it - mean) * (it - mean) }.average()).toFloat()
        return if (sd < 0.01f) 0.01f else sd
    }

    private fun sendAlertNotification(level: String, notes: String) {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.notify(
            2, NotificationCompat.Builder(this, "Swasthiti_monitoring")
                .setContentTitle("Pattern Change: ${level.uppercase()}")
                .setContentText(notes)
                .setSmallIcon(android.R.drawable.ic_dialog_alert)
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setAutoCancel(true)
                .build()
        )
    }

    private suspend fun syncUnstagedDailyFeaturesToFirebase() {
        val uid = FirebaseAuth.getInstance().currentUser?.uid ?: return
        val db = SwasthitiDatabase.getInstance(this@MonitoringService)
        val userId = DataRepository.userProfile.value?.email ?: return
        
        try {
            val unsynced = db.dailyFeaturesDao().getUnsynced(userId)
            if (unsynced.isEmpty()) return
            
            val firestore = FirebaseFirestore.getInstance()
            val collectionRef = firestore.collection("users").document(uid).collection("daily_features")
            
            for (entity in unsynced) {
                // Ensure simulated testing data does not contaminate Firebase
                if (entity.isSimulated) {
                    db.dailyFeaturesDao().markSynced(entity.id)
                    continue
                }
                
                val data = hashMapOf(
                    "date" to entity.date,
                    "screenTimeHours" to entity.screenTimeHours,
                    "unlockCount" to entity.unlockCount,
                    "appLaunchCount" to entity.appLaunchCount,
                    "notificationsToday" to entity.notificationsToday,
                    "socialAppRatio" to entity.socialAppRatio,
                    "callsPerDay" to entity.callsPerDay,
                    "callDurationMinutes" to entity.callDurationMinutes,
                    "uniqueContacts" to entity.uniqueContacts,
                    "conversationFrequency" to entity.conversationFrequency,
                    "dailyDisplacementKm" to entity.dailyDisplacementKm,
                    "locationEntropy" to entity.locationEntropy,
                    "homeTimeRatio" to entity.homeTimeRatio,
                    "placesVisited" to entity.placesVisited,
                    "wakeTimeHour" to entity.wakeTimeHour,
                    "sleepTimeHour" to entity.sleepTimeHour,
                    "sleepDurationHours" to entity.sleepDurationHours,
                    "darkDurationHours" to entity.darkDurationHours,
                    "chargeDurationHours" to entity.chargeDurationHours,
                    "memoryUsagePercent" to entity.memoryUsagePercent,
                    "networkWifiMB" to entity.networkWifiMB,
                    "networkMobileMB" to entity.networkMobileMB,
                    "downloadsToday" to entity.downloadsToday,
                    "storageUsedGB" to entity.storageUsedGB,
                    "appUninstallsToday" to entity.appUninstallsToday,
                    "upiTransactionsToday" to entity.upiTransactionsToday,
                    "totalAppsCount" to entity.totalAppsCount,
                    "backgroundAudioHours" to entity.backgroundAudioHours,
                    "mediaCountToday" to entity.mediaCountToday,
                    "appInstallsToday" to entity.appInstallsToday,
                    "calendarEventsToday" to entity.calendarEventsToday,
                    "dailySteps" to entity.dailySteps,
                    "appBreakdownJson" to entity.appBreakdownJson,
                    "notificationBreakdownJson" to entity.notificationBreakdownJson,
                    "appLaunchesBreakdownJson" to entity.appLaunchesBreakdownJson,
                    "bgAudioBreakdownJson" to entity.bgAudioBreakdownJson
                )
                
                collectionRef.document(entity.date).set(data).await()
                db.dailyFeaturesDao().markSynced(entity.id)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    // ── Missed-day recovery ───────────────────────────────────────────────────

    /**
     * FIX 4: If the MonitoringService was killed by Android (Doze mode, battery optimiser,
     * or a reboot) before the midnight day-transition fired, yesterday's data will be missing
     * from Room. This function detects that gap and re-collects it from UsageEvents.
     *
     * UsageStatsManager retains a 14-day rolling window, so recovery is possible for up to
     * 14 days after the missed night.
     */
    private suspend fun recoverMissedDayIfNeeded(userId: String, db: SwasthitiDatabase) {
        val lastDay = DataRepository.lastProcessedDay.value
        if (lastDay == -1) return // Fresh install — no prior day to recover.

        val yesterdayCal = Calendar.getInstance().apply { add(Calendar.DAY_OF_YEAR, -1) }
        val yesterdayStr = dateFmt.format(yesterdayCal.time)
        val savedYesterday = db.dailyFeaturesDao().getByDate(userId, yesterdayStr)

        if (savedYesterday == null) {
            Log.w("Swasthiti.Service", "Missed-day recovery: no snapshot found for $yesterdayStr — recovering from UsageEvents")
            try {
                val startOfToday   = dataCollector.getStartOfDayMs()
                val yesterdayStart = startOfToday - 24 * 3600_000L
                val yesterdayEnd   = startOfToday - 1L
                val locationSnaps  = DataRepository.locationSnapshots.value

                // Re-collect the full prior day from system APIs — still available in the 14-day window
                val missedSnapshot = dataCollector.collectSnapshot(locationSnaps, yesterdayStart, yesterdayEnd)
                val entity = JsonConverter.fromPersonalityVector(userId, yesterdayStr, missedSnapshot, false)
                db.dailyFeaturesDao().insert(entity)
                Log.i("Swasthiti.Service", "Missed-day recovery: saved snapshot for $yesterdayStr — syncing to Firebase")

                // Immediately push the recovered data to Firestore
                syncUnstagedDailyFeaturesToFirebase()
            } catch (e: Exception) {
                Log.e("Swasthiti.Service", "Missed-day recovery failed for $yesterdayStr: ${e.message}", e)
            }
        }
    }

    /**
     * FIX 8: Music-only audio detection via MediaSessionManager.
     *
     * Why MediaSession instead of AudioManager.isMusicActive():
     *   • isMusicActive() returns true for ANY audio (YouTube, game sounds, ads).
     *   • MediaSessionManager gives us the PACKAGE NAME of the app controlling playback.
     *   • We validate that package against DataCollector.isMusicApp() (3-layer check:
     *     exact package list, keyword scan, OS CATEGORY_AUDIO flag).
     *
     * This ensures only Spotify, Gaana, OuerTune, etc. add to backgroundAudioHours.
     */
    private fun isMusicAppActiveViaMediaSession(): String? {
        return try {
            val msm = getSystemService(MediaSessionManager::class.java) ?: return null
            val nlsComponent = ComponentName(this, SwasthitiNotificationListenerService::class.java)
            val sessions = msm.getActiveSessions(nlsComponent)
            val controller = sessions.firstOrNull { controller ->
                val pkg = controller.packageName
                val state = controller.playbackState
                val isPlaying = state?.state == android.media.session.PlaybackState.STATE_PLAYING
                isPlaying && dataCollector.isMusicApp(pkg)
            }
            controller?.packageName
        } catch (e: SecurityException) {
            // Notification access not granted — AudioManager fallback (no package name)
            val audioManager = getSystemService(android.media.AudioManager::class.java)
            if (audioManager?.isMusicActive == true) "unknown_music_app" else null
        } catch (e: Exception) {
            Log.w("Swasthiti.Service", "isMusicAppActiveViaMediaSession error: ${e.message}")
            null
        }
    }

    private fun setupMidnightAlarm() {
        alarmManager = getSystemService(AlarmManager::class.java)
        val midnightIntent = Intent(this, MidnightReceiver::class.java)
        val pendingIntent = PendingIntent.getBroadcast(this, 1001, midnightIntent, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)

        val cal = Calendar.getInstance().apply {
            add(Calendar.DAY_OF_YEAR, 1)
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 1)
            set(Calendar.MILLISECOND, 0)
        }

        alarmManager?.let { am ->
            try {
                val canSchedule = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    am.canScheduleExactAlarms()
                } else true

                if (canSchedule) {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                        am.setExactAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, cal.timeInMillis, pendingIntent)
                    } else {
                        am.setExact(AlarmManager.RTC_WAKEUP, cal.timeInMillis, pendingIntent)
                    }
                } else {
                    // Fallback to inexact alarm if permission denied
                    am.set(AlarmManager.RTC_WAKEUP, cal.timeInMillis, pendingIntent)
                    Log.w("Swasthiti.Service", "Exact Alarm permission missing — falling back to standard alarm")
                }
            } catch (e: Exception) {
                Log.e("Swasthiti.Service", "Failed to set midnight alarm: ${e.message}")
                am.set(AlarmManager.RTC_WAKEUP, cal.timeInMillis, pendingIntent)
            }
        }
        Log.i("Swasthiti.Service", "Midnight transition alarm set for ${cal.time}")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == "ACTION_MIDNIGHT_TRANSITION") {
            // Force a tick to detect the day change logic
            serviceScope.launch { runTick() }
            // Reschedule alarm for the next night
            setupMidnightAlarm()
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null
}

/** Standalone receiver to handle the exact midnight alarm */
class MidnightReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        Log.i("Swasthiti.Service", "Midnight Alarm Fired! Triggering day transition logic.")
        val serviceIntent = Intent(context, MonitoringService::class.java).apply {
            action = "ACTION_MIDNIGHT_TRANSITION"
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            context.startForegroundService(serviceIntent)
        } else {
            context.startService(serviceIntent)
        }
    }
}

