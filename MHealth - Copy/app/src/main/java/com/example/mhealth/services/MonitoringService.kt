package com.example.mhealth.services

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
import android.content.SharedPreferences
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.example.mhealth.MainActivity
import com.example.mhealth.R
import com.example.mhealth.logic.AnomalyDetector
import com.example.mhealth.logic.PythonEngine
import com.example.mhealth.logic.DataCollector
import com.example.mhealth.logic.DataRepository
import com.example.mhealth.logic.GpsStateManager
import com.example.mhealth.logic.JsonConverter
import com.example.mhealth.logic.db.AnalysisResultEntity
import com.example.mhealth.logic.db.BaselineEntity
import com.example.mhealth.logic.db.MHealthDatabase
import com.example.mhealth.logic.db.UserProfileEntity
import com.example.mhealth.logic.db.ObservationEntity
import com.example.mhealth.models.DailyReport
import com.example.mhealth.models.PersonalityVector
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.debounce
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

class MonitoringService : Service() {

    companion object {
        private const val TAG = "MHealth.Service"
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
    private var lastMusicPollMs: Long = 0L    // Last tick where music was confirmed playing
    private var alarmManager: AlarmManager? = null

    // CompletableDeferred ensures runTick() waits for Room rehydration without blocking the main thread.
    private val isRestored = CompletableDeferred<Unit>()
    private val tickMutex = Mutex()
    private var lastTickMs = 0L

    // ── Screen and Interaction Receiver ──────────────────────────────────────
    // Triggers runTick() on user events so charts stay fresh without polling.
    private val interactiveReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.action) {
                Intent.ACTION_SCREEN_ON, Intent.ACTION_USER_PRESENT -> {
                    // Update UI snapshot when the user is actually looking
                    serviceScope.launch { runTick(isEventTriggered = true) }
                }
                Intent.ACTION_SCREEN_OFF -> {
                    // Save state when user puts phone away
                    serviceScope.launch { runTick(isEventTriggered = true) }
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
                    Log.i("MHealth.Service", "Charger connected at $chargingStartMs")
                    DataRepository.recordChargingStart(chargingStartMs)
                }
                Intent.ACTION_POWER_DISCONNECTED -> {
                    if (chargingStartMs > 0L) {
                        val sessionMs = System.currentTimeMillis() - chargingStartMs
                        val sessionHrs = sessionMs / 3_600_000f
                        DataRepository.addChargeTime(sessionHrs)
                        Log.i("MHealth.Service", "Charger disconnected — added %.2fh (total %.2fh)"
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
                    Log.i("MHealth.Service", "DND turned ON at $now")
                } 
                // If DND is OFF (ALL)
                else if (filter == NotificationManager.INTERRUPTION_FILTER_ALL) {
                    DataRepository.setDndOffTimestamp(now)
                    Log.i("MHealth.Service", "DND turned OFF at $now")
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
            Log.i("MHealth.Service", "Music session detected: $activeMusicPackage — tracking started")
        } else if (!isMusicPlaying && musicStartMs > 0L) {
            // Session ending - add both the current session time AND any pre-restart accumulated time
            val sessionMs = (now - musicStartMs) + preRestartAudioMs
            DataRepository.addBgAudioTime(activeMusicPackage, sessionMs)
            Log.i("MHealth.Service", "Music session ended: $activeMusicPackage — added ${sessionMs}ms (session: ${now - musicStartMs}ms, pre-restart: ${preRestartAudioMs}ms)")
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

        try { // Safety wrapper — prevent secondary crashes from killing the service
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
                Log.i("MHealth.Service", "GPS State changed: ${state.displayName} (interval: ${state.intervalMs / 60_000}min)")
            }
        }

        // FIX 7: Register receivers for event-driven monitoring
        ContextCompat.registerReceiver(this, powerReceiver, IntentFilter().apply {
            addAction(Intent.ACTION_POWER_CONNECTED)
            addAction(Intent.ACTION_POWER_DISCONNECTED)
        }, ContextCompat.RECEIVER_NOT_EXPORTED)

        ContextCompat.registerReceiver(this, interactiveReceiver, IntentFilter().apply {
            addAction(Intent.ACTION_SCREEN_ON)
            addAction(Intent.ACTION_SCREEN_OFF)
            addAction(Intent.ACTION_USER_PRESENT)
        }, ContextCompat.RECEIVER_NOT_EXPORTED)

        ContextCompat.registerReceiver(this, dndReceiver,
            IntentFilter(NotificationManager.ACTION_INTERRUPTION_FILTER_CHANGED),
            ContextCompat.RECEIVER_NOT_EXPORTED)

        // FIX 8: Register for MediaSession changes (requires notification access)
        // We pass a ComponentName pointing to our declared NotificationListenerService.
        // Without this, Android throws SecurityException on any getActiveSessions() call
        // from a 3rd-party app that lacks MEDIA_CONTENT_CONTROL (a privileged permission).
        try {
            val nlsComponent = ComponentName(this, MHealthNotificationListenerService::class.java)
            val msm = getSystemService(MediaSessionManager::class.java)
            msm?.addOnActiveSessionsChangedListener(sessionListener, nlsComponent)
            // Initial check for already-playing music when service starts
            updateMusicSessionState(msm?.getActiveSessions(nlsComponent))
        } catch (e: SecurityException) {
            Log.w("MHealth.Service", "MediaSession listener failed (Notification Access not granted by user) — background audio tracking disabled until access is granted")
        }

        // FIX 9: Setup Midnight Alarm for exact day transition
        setupMidnightAlarm()

        // If the phone is ALREADY charging/playing when the service starts, anchor state now
        val batteryOnStart = dataCollector.getBatteryInfo()
        if (batteryOnStart.isCharging) {
            chargingStartMs = System.currentTimeMillis()
            Log.i("MHealth.Service", "Service started while already charging — anchored at $chargingStartMs")
        }
        // Music state is already anchored via updateMusicSessionState call above

        // ── Reactive slider: two-way status sync on every slider move ─────────────
        // ── Reactive slider: status sync is handled by the scheduleMonitoring observer ─────────────

        // Start passive continuous location tracking with adaptive intervals
        dataCollector.startContinuousLocationTracking()
        } catch (e: Exception) {
            Log.e(TAG, "Critical error during MonitoringService initialization", e)
        }
    }

    // FIX 5: Suspend version of restore — called from runBlocking in onCreate
    // so we BLOCK the main service thread until Room data is fully loaded.
    private suspend fun restoreStateFromRoomSuspend() {
        val userId = DataRepository.userProfile.value?.email ?: "default_user"
        try {
            val db = MHealthDatabase.getInstance(this@MonitoringService)
            val profile = db.userProfileDao().getProfile(userId)

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
                        wakeTimeHour = baselineFields["wakeTimeHour"] ?: 0f,
                        sleepTimeHour = baselineFields["sleepTimeHour"] ?: 0f,
                        sleepDurationHours = baselineFields["sleepDurationHours"] ?: 0f,
                        dailyStepCount = baselineFields["dailyStepCount"] ?: 0f,
                        activeMinutes = baselineFields["activeMinutes"] ?: 0f,
                        keystrokeSpeed = baselineFields["keystrokeSpeed"] ?: 0f,
                        backspaceRatio = baselineFields["backspaceRatio"] ?: 0f,
                        scrollVelocity = baselineFields["scrollVelocity"] ?: 0f,
                        daylightExposureMinutes = baselineFields["daylightExposureMinutes"] ?: 0f,
                        chargeRegularity = baselineFields["chargeRegularity"] ?: 0f,
                        chargeDurationHours = baselineFields["chargeDurationHours"] ?: 0f,
                        upiTransactionsToday = baselineFields["upiTransactionsToday"] ?: 0f,
                        appUninstallsToday = baselineFields["appUninstallsToday"] ?: 0f,
                        appInstallsToday = baselineFields["appInstallsToday"] ?: 0f,
                        calendarEventsToday = baselineFields["calendarEventsToday"] ?: 0f,
                        mediaCountToday = baselineFields["mediaCountToday"] ?: 0f,
                        downloadsToday = baselineFields["downloadsToday"] ?: 0f,
                        musicTimeMinutes = baselineFields["musicTimeMinutes"] ?: 0f,
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
            // Progress = actual distinct days of data (today's live row is already in Room).
            // No +1 needed — today is already counted in the Room query.
            val currentProg = collectedDailyVectors.size.coerceAtLeast(1)
            DataRepository.updateBaselineProgress(currentProg)
            DataRepository.updateDnaBaselineProgress(currentProg)
            DataRepository.updateCollectedBaselineVectors(collectedDailyVectors)

            // Immediate check for baseline readiness on startup
            if (DataRepository.isBuildingBaseline.value) {
                val liveSnapshot = dataCollector.collectSnapshot(DataRepository.locationSnapshots.value)
                checkAndFinalizeBaseline(liveSnapshot)
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
                Log.i("MHealth.Service", "Audio session re-anchored after service restart: $resumingPkg (pre-accumulated: ${preRestartAudioMs}ms)")
            }

            // Sync any unsynced data from previous sessions on startup
            syncUnstagedDailyFeaturesToFirebase()

            // FIX 1: Prime lastProcessedDay on first run so midnight day-transitions fire correctly.
            // If the service is killed before the first midnight cycle completes, lastProcessedDay
            // stays -1 and the guard (savedDay != -1) permanently skips all future transitions.
            var isFirstServiceStart = false
            if (DataRepository.lastProcessedDay.value == -1) {
                val todayDoy = Calendar.getInstance().get(Calendar.DAY_OF_YEAR)
                DataRepository.setLastProcessedDay(todayDoy)
                Log.i("MHealth.Service", "Primed lastProcessedDay=$todayDoy on first service start")
                isFirstServiceStart = true
            }

            // FIX 4: Recover missed yesterday snapshot if the service was killed before the
            // midnight transition had a chance to fire (e.g., Android Doze / battery optimiser).
            if (!isFirstServiceStart) {
                recoverMissedDayIfNeeded(userId, db)
            }

        } catch (e: Exception) {
            Log.e("MHealth.Service", "Error restoring state from Room", e)
        }
    }

    // Legacy async wrapper kept so the reset flow (which launches a coroutine) still works
    private fun restoreStateFromRoom() {
        serviceScope.launch { restoreStateFromRoomSuspend() }
    }

    private fun startForegroundNotification() {
        val channelId = "mhealth_monitoring"
        val appName = getString(R.string.app_name)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                channelId, "$appName Monitoring", NotificationManager.IMPORTANCE_LOW
            ).apply { description = "Passive mental health pattern monitoring" }
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
        val notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("$appName Active")
            .setContentText("Passively monitoring device patterns")
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentIntent(
                PendingIntent.getActivity(
                    this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE
                )
            )
            .setOngoing(true)
            .build()
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(1, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION)
        } else {
            startForeground(1, notification)
        }
    }

    private fun scheduleMonitoring() {
        // Observers stay active for configuration changes
        serviceScope.launch {
            DataRepository.monitoringIntervalMinutes.collectLatest { intervalMin ->
                Log.d("MHealth.Service", "Monitoring loop re-enabled. Tick every $intervalMin min.")
                while (isActive) {
                    runTick()
                    delay(intervalMin * 60 * 1000L)
                }
            }
        }

        // ── Baseline: auto-finalize is handled within runTick and handleBaselineBuilding ──
        // No slider observer needed — baseline builds automatically from Day 1.

        // ── DNA: auto-build with no threshold gating ──────────────────────────
        // DNA readiness is now set automatically when the nightly worker
        // produces a valid profile. No slider observer needed.

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

        // Manual DNA Finalize trigger
        serviceScope.launch {
            DataRepository.dnaFinalizeTrigger.collect { triggers ->
                if (triggers > 0) {
                    val userId = DataRepository.userProfile.value?.email ?: "default_user"
                    val now = Date()
                    val todayDoy = Calendar.getInstance().get(Calendar.DAY_OF_YEAR)
                    val todayStr = dateFmt.format(now)

                    Log.i("MHealth.Service", "Manual DNA Finalize Clicked. triggers=$triggers, userId=$userId")
                    Log.i("MHealth.Service", "Force-saving snapshot for Day $todayDoy ($todayStr) to ensure worker has data.")

                    // 1. Collect current data and save as a formal row for TODAY
                    val currentSnapshot = dataCollector.collectSnapshot(DataRepository.locationSnapshots.value)
                    persistDailySnapshot(currentSnapshot, todayDoy, isSimulated = true)

                    // 2. Compute and store DNA snapshot for today (populates daily_dna_snapshot table)
                    try {
                        val dnaComputer = com.example.mhealth.logic.AppDnaComputer(this@MonitoringService)
                        val snapshot = dnaComputer.computeAndStoreDnaSnapshot(userId, todayStr)
                        if (snapshot != null) {
                            Log.i("MHealth.Service", "DNA snapshot stored for manual finalize: ${snapshot.totalSessions} sessions, ${snapshot.totalScreenTimeHours}h")
                            val db = MHealthDatabase.getInstance(this@MonitoringService)
                            val dnaDays = db.dailyDnaSnapshotDao().countDistinctDays(userId)
                            DataRepository.updateDnaBaselineProgress(dnaDays)
                        } else {
                            Log.i("MHealth.Service", "No sessions for today — DNA snapshot skipped for manual finalize")
                        }
                    } catch (e: Exception) {
                        Log.e("MHealth.Service", "Failed to compute DNA snapshot for manual finalize: ${e.message}", e)
                    }

                    Log.i("MHealth.Service", "Spawning NightlyAnalysisWorker (force) for $todayStr...")

                    // 3. Trigger the nightly worker with force_run
                    DataRepository.setDnaAnalysing(true)
                    NightlyAnalysisWorker.runNow(this@MonitoringService, userId, todayStr, forceRun = true)
                }
            }
        }

        // Manual Cluster Reset trigger — re-discovers archetypes from scratch
        serviceScope.launch {
            DataRepository.clusterResetTrigger.collect { triggers ->
                if (triggers > 0) {
                    val userId = DataRepository.userProfile.value?.email ?: "default_user"
                    val now = Date()
                    val todayStr = dateFmt.format(now)

                    Log.i("MHealth.Service", "Cluster Reset triggered — will rebuild clusters from scratch for $todayStr")

                    // 1. Collect current data and save as a formal row for TODAY
                    val currentSnapshot = dataCollector.collectSnapshot(DataRepository.locationSnapshots.value)
                    persistDailySnapshot(currentSnapshot, Calendar.getInstance().get(Calendar.DAY_OF_YEAR), isSimulated = true)

                    // 2. Compute and store DNA snapshot for today
                    try {
                        val dnaComputer = com.example.mhealth.logic.AppDnaComputer(this@MonitoringService)
                        dnaComputer.computeAndStoreDnaSnapshot(userId, todayStr)
                    } catch (e: Exception) {
                        Log.e("MHealth.Service", "DNA snapshot for cluster reset failed: ${e.message}", e)
                    }

                    // 3. Trigger nightly worker with forceResetClusters = true
                    DataRepository.setDnaAnalysing(true)
                    NightlyAnalysisWorker.runNow(
                        this@MonitoringService, userId, todayStr,
                        forceRun = true, forceResetClusters = true
                    )
                }
            }
        }

        // dev force reset trigger listener
        serviceScope.launch {
            DataRepository.resetTrigger.collect { triggers ->
                if (triggers > 0) {
                    val userId = DataRepository.userProfile.value?.email ?: "default_user"
                    val db = MHealthDatabase.getInstance(this@MonitoringService)
                    try {
                        Log.i("MHealth.Service", "Reset triggered: rebuilding baseline from all real days")

                        // 1. Clear ONLY simulated features (non-destructive)
                        db.dailyFeaturesDao().clearSimulated(userId)
                        // 2. Clear old state tables permanently to avoid merging conflicts
                        db.baselineDao().clearBaseline(userId)
                        db.userProfileDao().upsert(
                            UserProfileEntity(
                                userId = userId,
                                baselineReady = false,
                                baselineDays = 1,
                                currentStatus = "Learning Baseline"
                            )
                        )
                        DataRepository.clearReports()
                        DataRepository.updateProvisionalAnalysis(null)  // Clear stale provisional

                        Log.i("MHealth.Service", "Reset triggered: Simulated data removed, old baseline wiped.")

                        // 3. Reload real features into memory to update progress
                        val allRealFeatures = db.dailyFeaturesDao().getAllFeatures(userId)
                            .sortedBy { it.date }
                        
                        collectedDailyVectors.clear()
                        collectedDailyVectors.addAll(allRealFeatures.map { JsonConverter.toPersonalityVector(it) })
                        
                        // Update UI progress: actual saved days (no +1)
                        val currentProg = collectedDailyVectors.size.coerceAtLeast(1)
                        DataRepository.updateBaselineProgress(currentProg)
                        DataRepository.updateDnaBaselineProgress(currentProg)
                        DataRepository.updateCollectedBaselineVectors(collectedDailyVectors)

                        // 5. Auto-build baseline from ALL available real data
                        if (collectedDailyVectors.isNotEmpty()) {
                            DataRepository.setIsBuildingBaseline(true)
                            
                            // Rebuild Mathematical Baseline using ALL vectors
                            val baseline = buildBaseline(collectedDailyVectors)
                            val totalDays = collectedDailyVectors.size
                            persistBaselineToRoom(baseline, totalDays)
                            DataRepository.setBaseline(baseline)
                            detector = AnomalyDetector(baseline)

                            scheduleNightlyWorker()
                            Log.i("MHealth.Service", "Reset built new baseline from ${totalDays} real days.")
                        } else {
                            // If we don't have any days, remain in Learning Mode
                            DataRepository.setIsBuildingBaseline(true)
                            DataRepository.clearBaseline()
                            detector = null
                            Log.i("MHealth.Service", "No data after wipe. Waiting for real telemetry.")
                        }

                        // 6. Refresh the "Live" UI snapshots immediately 
                        runTick()
                    } catch (e: Exception) {
                        Log.e("MHealth.Service", "Error during master reset", e)
                    }
                }
            }
        }

        // ── Hard Reset: Complete nuclear wipe of ALL data ────────────────────
        // Triggered from the "Clear All Data" button in Settings.
        // Automatically called AFTER exportDataAsJson() saves a JSON backup.
        serviceScope.launch {
            DataRepository.hardResetTrigger.collect { triggers ->
                if (triggers > 0) {
                    val userId = DataRepository.userProfile.value?.email ?: "default_user"
                    val db = MHealthDatabase.getInstance(this@MonitoringService)
                    try {
                        Log.w(TAG, "⚠️ HARD RESET triggered for $userId — wiping all data tables")

                        // 1. Wipe ALL L1 daily feature vectors
                        val l1Count = db.dailyFeaturesDao().clearAll(userId)
                        Log.i(TAG, "  Cleared $l1Count L1 daily feature rows")

                        // 2. Wipe ALL L2 raw session events
                        val sessCount = db.appSessionDao().clearAll()
                        Log.i(TAG, "  Cleared $sessCount app session rows")

                        // 3. Wipe ALL L2 notification events
                        val notifCount = db.notificationEventDao().clearAll()
                        Log.i(TAG, "  Cleared $notifCount notification event rows")

                        // 4. Wipe ALL L2 computed DNA snapshots
                        val snapCount = db.dailyDnaSnapshotDao().clearAll(userId)
                        Log.i(TAG, "  Cleared $snapCount DNA snapshot rows")

                        // 5. Wipe the L2 Person DNA baseline clusters
                        db.personDnaDao().deleteByUserId(userId)
                        Log.i(TAG, "  Cleared Person DNA baseline")

                        // 6. Wipe the L1 mathematical baseline (mean/std per feature)
                        db.baselineDao().clearBaseline(userId)
                        Log.i(TAG, "  Cleared L1 baseline")

                        // 7. Wipe all anomaly analysis results / history
                        db.analysisResultDao().clearAll(userId)
                        Log.i(TAG, "  Cleared analysis result history")

                        // 8. Reset UserProfile to fresh onboarding defaults
                        db.userProfileDao().upsert(
                            UserProfileEntity(
                                userId = userId,
                                baselineReady = false,
                                dnaReady = false,
                                baselineDays = 1,
                                currentStatus = "Learning Baseline"
                            )
                        )

                        // 9. Clear all in-memory accumulators and local caches
                        collectedDailyVectors.clear()
                        detector = null
                        nightlyWorkerScheduled = false

                        // 10. Reset all DataRepository UI state flows to Day-1 defaults
                        DataRepository.clearAllState()

                        Log.i(TAG, "✅ Hard Reset complete — app is back to Day 1 baseline collection")

                        // 11. Kick off a fresh telemetry tick so sensors tab shows live data
                        runTick()

                    } catch (e: Exception) {
                        Log.e(TAG, "Error during hard reset", e)
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

    private suspend fun runTick(isSimulated: Boolean = false, isEventTriggered: Boolean = false) = tickMutex.withLock {
        // Wait for Room database rehydration to finish so we don't calculate on empty state
        isRestored.await()
        
        val now = System.currentTimeMillis()
        if (isEventTriggered && (now - lastTickMs < 30_000L)) {
            // Throttle rapid-fire events (e.g. rapid screen on/off)
            return@withLock
        }
        lastTickMs = now
        
        try {
            collectionTickCount++

            // Check daily and monthly check-in notification triggers
            checkAndSendCheckinNotifications()

            // BUG FIX: Proactively capture a GPS fix every tick so distance is
            // never 0.0 just because the passive 50m-displacement listener didn't fire.
            if (!isSimulated) {
                dataCollector.captureProactiveLocationSnapshot()
            }

            // 1) Collect a "Live" snapshot (Midnight to Now) for UI updates
            val locationSnaps = DataRepository.locationSnapshots.value
            val liveSnapshot = dataCollector.collectSnapshot(locationSnaps)

            // Always update live data for home screen
            DataRepository.updateLatestVector(liveSnapshot)
            DataRepository.addHourlySnapshot(liveSnapshot)

            // Level 2 Digital DNA: Log session events from UsageEvents for behavioral DNA
            dataCollector.logSessionsFromEvents(dataCollector.getStartOfDayMs(), System.currentTimeMillis())

            // ── Periodic music polling — credits time incrementally each tick ──
            // Fixes: MediaSession listener unreliable, background audio always 0.
            // This polls isMusicAppActiveViaMediaSession() every tick and credits
            // elapsed time if music is playing, capped at 30 min per tick.
            if (!isSimulated) {
                val currentlyPlayingPkg = isMusicAppActiveViaMediaSession()
                val pollNow = System.currentTimeMillis()
                if (currentlyPlayingPkg != null) {
                    if (musicStartMs == -1L) {
                        // New session detected via poll (listener may have missed it)
                        preRestartAudioMs = 0L
                        musicStartMs = pollNow
                        activeMusicPackage = currentlyPlayingPkg
                    }
                    // Credit elapsed time since last successful poll
                    if (lastMusicPollMs > 0L) {
                        val elapsed = pollNow - lastMusicPollMs
                        if (elapsed in 1..30 * 60_000L) {
                            DataRepository.addBgAudioTime(currentlyPlayingPkg, elapsed)
                        }
                    }
                    lastMusicPollMs = pollNow
                } else {
                    if (musicStartMs > 0L) {
                        musicStartMs = -1L
                        activeMusicPackage = null
                        preRestartAudioMs = 0L
                    }
                    lastMusicPollMs = 0L
                }
            }

            try {
                val userId = DataRepository.userProfile.value?.email ?: "default_user"
                val todayStr = dateFmt.format(Date())
                val entity = JsonConverter.fromPersonalityVector(userId, todayStr, liveSnapshot, isSimulated = false)
                val db = MHealthDatabase.getInstance(this@MonitoringService)
                db.dailyFeaturesDao().insert(entity)

                // Sync collectedDailyVectors in memory with the DB rows to ensure today's live snapshot
                // is immediately visible to baseline builders and UI progress indicators.
                val pastFeatures = db.dailyFeaturesDao().getLatestN(userId, 60).reversed()
                val pastVectors = pastFeatures.map { JsonConverter.toPersonalityVector(it) }
                collectedDailyVectors.clear()
                collectedDailyVectors.addAll(pastVectors)
                
                DataRepository.updateCollectedBaselineVectors(collectedDailyVectors)
                
                val weeklyVectors = pastVectors.takeLast(7)
                DataRepository.updateWeeklyFeatureHistory(weeklyVectors)
            } catch (e: Exception) {
                Log.d("MHealth.Service", "Live daily features save/reload failed: ${e.message}")
            }

            // 2) Provisional Analysis (Authoritative Python Live Score)
            if (!DataRepository.isBuildingBaseline.value) {
                runProvisionalAnalysisAsync(liveSnapshot)
            }

            val today = Calendar.getInstance().get(Calendar.DAY_OF_YEAR)
            // Re-fetch savedDay inside the lock to prevent double transitions
            val savedDay = DataRepository.lastProcessedDay.value

            // 3) Day Transition Logic: Capture the FULL profile of the day that just ended
            if (today != savedDay && savedDay != -1) {
                // Determine the 24-hour range for the day that ended (Midnight to Midnight)
                val startOfToday = dataCollector.getStartOfDayMs()
                val yesterdayStart = startOfToday - 24 * 3600_000L
                val yesterdayEnd = startOfToday - 1L

                // Compute the date string for the day that just ended
                val yesterdayCal = Calendar.getInstance().apply { add(Calendar.DAY_OF_YEAR, -1) }
                val yesterdayStr = dateFmt.format(yesterdayCal.time)

                Log.i("MHealth.Service", "Day Transition Detected: Capturing full profile for Day $savedDay ($yesterdayStr) [Range: $yesterdayStart to $yesterdayEnd]")

                // Collect a 100% accurate snapshot for the entire prior day
                val fullDaySnapshot = dataCollector.collectSnapshot(locationSnaps, yesterdayStart, yesterdayEnd)

                // ── DNA: Compute daily DNA snapshot from raw sessions BEFORE purge ──
                try {
                    val userId = DataRepository.userProfile.value?.email ?: "default_user"
                    val dnaComputer = com.example.mhealth.logic.AppDnaComputer(this@MonitoringService)
                    val snapshot = dnaComputer.computeAndStoreDnaSnapshot(userId, yesterdayStr)
                    if (snapshot != null) {
                        Log.i("MHealth.Service", "DNA snapshot stored for $yesterdayStr: ${snapshot.totalSessions} sessions, ${snapshot.totalScreenTimeHours}h screen")
                        // Update DNA progress from the daily_dna_snapshot table
                        val db = MHealthDatabase.getInstance(this@MonitoringService)
                        val dnaDays = db.dailyDnaSnapshotDao().countDistinctDays(userId)
                        DataRepository.updateDnaBaselineProgress(dnaDays)
                    } else {
                        Log.i("MHealth.Service", "No sessions for $yesterdayStr — DNA snapshot skipped")
                    }
                } catch (e: Exception) {
                    Log.e("MHealth.Service", "Failed to compute DNA snapshot for $yesterdayStr: ${e.message}", e)
                }

                // ── DNA: Keep a rolling 60-day window of raw sessions and notifications; purge older than 60 days ──
                try {
                    val db = MHealthDatabase.getInstance(this@MonitoringService)
                    val sixtyDaysAgoMs = System.currentTimeMillis() - 60L * 24L * 3600_000L
                    val sessionsDeleted = db.appSessionDao().deleteOlderThan(sixtyDaysAgoMs)
                    val notifsDeleted = db.notificationEventDao().deleteOlderThan(sixtyDaysAgoMs)
                    Log.i("MHealth.Service", "Purged $sessionsDeleted old sessions and $notifsDeleted old notification events older than 60 days")
                } catch (e: Exception) {
                    Log.e("MHealth.Service", "Failed to purge old DNA raw data: ${e.message}", e)
                }

                // Record this full day in history/baseline
                if (DataRepository.isBuildingBaseline.value) {
                    handleBaselineBuilding(fullDaySnapshot, today, savedDay, isSimulated)
                } else {
                    handleAnomalyDetection(fullDaySnapshot, today, savedDay, isSimulated)
                }

                // Evaluate Passive Habits
                evaluateHabitsForDay(fullDaySnapshot, yesterdayStart, yesterdayEnd, yesterdayStr)

                // Reset daily accumulators for the new day
                DataRepository.resetDailyState()
                gpsStateManager.reset()  // Reset GPS state machine to STATIONARY
                dataCollector.resetDisplacementGuard()  // Reset monotonic displacement for new day
                com.example.mhealth.services.MHealthAccessibilityService.resetDailyMetrics(this@MonitoringService)
                DataRepository.setLastProcessedDay(today)
                Log.i("MHealth.Service", "Day transition logic for Day $savedDay complete.")
            } else {
                // Regular tick within the same day
                // Progress = actual collected vectors (today is already in Room)
                val currentProg = collectedDailyVectors.size.coerceAtLeast(1)
                DataRepository.updateBaselineProgress(currentProg)
                DataRepository.updateDnaBaselineProgress(currentProg)

                if (DataRepository.isBuildingBaseline.value) {
                    checkAndFinalizeBaseline(liveSnapshot)
                }
            }
        } catch (e: Exception) {
            Log.e("MHealth.Service", "Error in runTick: ${e.message}", e)
        }
    }

    private fun evaluateHabitsForDay(snapshot: PersonalityVector, yesterdayStartMs: Long, yesterdayEndMs: Long, dateStr: String) {
        val prefs = getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE)
        val editor = prefs.edit()
        
        Log.i("MHealth.Service", "Evaluating habits for $dateStr...")

        // 1. Digital Sunset
        if (prefs.getBoolean("habit_digital_sunset_enabled", false)) {
            val targetMin = prefs.getInt("habit_digital_sunset_target", 30)
            val screenMinAfter9 = dataCollector.getScreenTimeAfter9PM(yesterdayStartMs, yesterdayEndMs)
            val success = screenMinAfter9 <= targetMin
            val streak = prefs.getInt("habit_digital_sunset_streak", 0)
            val newStreak = if (success) streak + 1 else 0
            editor.putInt("habit_digital_sunset_streak", newStreak)
            editor.putBoolean("habit_digital_sunset_status_last", success)
            Log.i("MHealth.Service", "  Digital Sunset: $screenMinAfter9 mins (target: $targetMin mins) -> success=$success, streak=$newStreak")
        } else {
            editor.putInt("habit_digital_sunset_streak", 0)
        }

        // 2. Circadian Anchor
        if (prefs.getBoolean("habit_circadian_anchor_enabled", false)) {
            val targetHour = prefs.getFloat("habit_circadian_anchor_target", 23.0f) // default 11 PM
            val bedtime = snapshot.sleepTimeHour
            val success = if (bedtime >= 18.0f) {
                bedtime <= targetHour || targetHour < 12.0f
            } else if (bedtime >= 0.0f) {
                targetHour < 12.0f && bedtime <= targetHour
            } else {
                false
            }
            val streak = prefs.getInt("habit_circadian_anchor_streak", 0)
            val newStreak = if (success) streak + 1 else 0
            editor.putInt("habit_circadian_anchor_streak", newStreak)
            editor.putBoolean("habit_circadian_anchor_status_last", success)
            Log.i("MHealth.Service", "  Circadian Anchor: bedtime $bedtime (target: $targetHour) -> success=$success, streak=$newStreak")
        } else {
            editor.putInt("habit_circadian_anchor_streak", 0)
        }

        // 3. Movement Boost
        if (prefs.getBoolean("habit_movement_boost_enabled", false)) {
            val targetSteps = prefs.getInt("habit_movement_boost_target", 6000)
            val steps = snapshot.dailyStepCount.toInt()
            val success = steps >= targetSteps
            val streak = prefs.getInt("habit_movement_boost_streak", 0)
            val newStreak = if (success) streak + 1 else 0
            editor.putInt("habit_movement_boost_streak", newStreak)
            editor.putBoolean("habit_movement_boost_status_last", success)
            Log.i("MHealth.Service", "  Movement Boost: $steps steps (target: $targetSteps) -> success=$success, streak=$newStreak")
        } else {
            editor.putInt("habit_movement_boost_streak", 0)
        }

        // 4. Focus Mode
        if (prefs.getBoolean("habit_focus_mode_enabled", false)) {
            val targetRatio = prefs.getFloat("habit_focus_mode_target", 0.20f)
            val socialRatio = snapshot.socialAppRatio
            val success = socialRatio <= targetRatio
            val streak = prefs.getInt("habit_focus_mode_streak", 0)
            val newStreak = if (success) streak + 1 else 0
            editor.putInt("habit_focus_mode_streak", newStreak)
            editor.putBoolean("habit_focus_mode_status_last", success)
            Log.i("MHealth.Service", "  Focus Mode: social ratio $socialRatio (target: $targetRatio) -> success=$success, streak=$newStreak")
        } else {
            editor.putInt("habit_focus_mode_streak", 0)
        }

        editor.putString("habit_last_checked_date", dateStr)
        editor.apply()
    }

    private suspend fun handleBaselineBuilding(snapshot: PersonalityVector, today: Int, savedDay: Int, isSimulated: Boolean) {
        if (today != savedDay && savedDay != -1) {
            // Always add the completed day's vector to history
            collectedDailyVectors.add(snapshot)
            val prog = collectedDailyVectors.size
            DataRepository.updateBaselineProgress(prog)
            DataRepository.updateDnaBaselineProgress(prog)
            DataRepository.updateCollectedBaselineVectors(collectedDailyVectors)

            // Persist end-of-day snapshot to Room
            persistDailySnapshot(snapshot, savedDay, isSimulated)

            // Auto-finalize: no minimum day requirement
            checkAndFinalizeBaseline(snapshot)
        } else {
            // Same-day tick: check if baseline can be built
            checkAndFinalizeBaseline(snapshot)
        }
    }

    private var isPersistingBaseline = false

    /**
     * Auto-finalize baseline whenever we have >= 1 collected day vector.
     * Uses ALL collected vectors (no slider-gated take(N)).
     * The Python Bayesian warm-start system handles progressive refinement.
     */
    private fun checkAndFinalizeBaseline(liveSnapshot: PersonalityVector) {
        if (DataRepository.isBuildingBaseline.value && collectedDailyVectors.size >= 7) {
            if (isPersistingBaseline) return // Prevent duplicate Coroutine launches
            isPersistingBaseline = true

            // Use ALL collected vectors — no artificial cap
            val baseline = buildBaseline(collectedDailyVectors)
            val totalDays = collectedDailyVectors.size

            serviceScope.launch {
                try {
                    persistBaselineToRoom(baseline, totalDays)
                    // Only flip the live state AFTER Room is confirmed written
                    DataRepository.setBaseline(baseline)
                    detector = AnomalyDetector(baseline)
                    scheduleNightlyWorker()
                    Log.i("MHealth.Service", "Baseline auto-established and persisted to Room (${totalDays}d)")
                    
                    // Immediately trigger provisional analysis so UI gauge activates in real-time
                    runProvisionalAnalysisAsync(liveSnapshot)
                } catch (e: Exception) {
                    Log.e("MHealth.Service", "Failed to persist baseline to Room — will retry on next tick: ${e.message}", e)
                } finally {
                    isPersistingBaseline = false
                }
            }
        }
    }

    private fun runProvisionalAnalysisAsync(liveSnapshot: PersonalityVector) {
        val userId = DataRepository.userProfile.value?.email ?: "default_user"
        serviceScope.launch(Dispatchers.Default) {
            try {
                val db = MHealthDatabase.getInstance(this@MonitoringService)
                
                // 1. Load baseline entities
                val baselineEntities = db.baselineDao().getBaseline(userId)
                if (baselineEntities.isEmpty()) return@launch
                
                // 2. Load daily features history (excluding today)
                val todayStr = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date())
                val history = db.dailyFeaturesDao().getAllFeatures(userId)
                    .filter { it.date != todayStr }
                    .sortedBy { it.date }
                
                // 3. Load historical anomaly scores and L2 modifiers
                val historicalResults = db.analysisResultDao().getLatestN(userId, 14).reversed()
                val historicalScores = historicalResults.map { it.effectiveScore }
                val historicalL2Modifiers = historicalResults.map { it.l2Modifier }
                
                // 4. Calculate day number
                val priorAnalysisCount = db.analysisResultDao().count(userId)
                val dayNumber = priorAnalysisCount + 1
                
                // 5. Construct JSON input via JsonConverter
                val todayFeatures = JsonConverter.fromPersonalityVector(userId, todayStr, liveSnapshot, isSimulated = false)
                val inputJsonStr = JsonConverter.toEngineJson(this@MonitoringService, todayFeatures, baselineEntities, history)
                
                // 6. Build meta JSON
                val root = org.json.JSONObject(inputJsonStr)
                root.put("day_number", dayNumber)
                val profile = db.userProfileDao().getProfile(userId)
                root.put("baseline_contaminated", profile?.baselineContaminated ?: false)
                root.put("is_provisional", true)
                root.put("user_id", userId)
                root.put("target_date", todayStr)
                
                val latestResult = db.analysisResultDao().getLatest(userId)
                val gateState = try {
                    org.json.JSONObject(latestResult?.gateResults ?: "{}")
                } catch (e: Exception) {
                    org.json.JSONObject()
                }
                root.put("gate_state", gateState)
                
                if (historicalScores.isNotEmpty()) {
                    val scoresArray = org.json.JSONArray()
                    historicalScores.forEach { scoresArray.put(it.toDouble()) }
                    root.put("historical_anomaly_scores", scoresArray)
                }
                
                if (historicalL2Modifiers.isNotEmpty()) {
                    val l2ModifiersArray = org.json.JSONArray()
                    historicalL2Modifiers.forEach { l2ModifiersArray.put(it.toDouble()) }
                    root.put("historical_l2_modifiers", l2ModifiersArray)
                }

                if (historicalResults.isNotEmpty()) {
                    val feedbacksArray = org.json.JSONArray()
                    historicalResults.forEach { res ->
                        val feedbackObj = org.json.JSONObject().apply {
                            put("date", res.date)
                            put("state", res.userFeedbackState)
                            put("category", res.userFeedbackCategory)
                            put("notes", res.userFeedbackNotes)
                            
                            val flaggedJson = try {
                                org.json.JSONArray(res.flaggedFeatures)
                            } catch (e: Exception) {
                                org.json.JSONArray()
                            }
                            put("flagged_features", flaggedJson)
                        }
                        feedbacksArray.put(feedbackObj)
                    }
                    root.put("user_feedbacks", feedbacksArray)
                }
                
                // 7. Inject sessions
                val cal = Calendar.getInstance().apply { add(Calendar.DAY_OF_YEAR, -60) }
                val dateFormat = SimpleDateFormat("yyyy-MM-dd", Locale.US)
                val startDate = dateFormat.format(cal.time)
                
                val allSessions = db.appSessionDao().getByDateRange(startDate, todayStr)
                root.put("sessions", org.json.JSONArray(JsonConverter.sessionsToJson(allSessions)))
                
                val todaySessions = db.appSessionDao().getByDate(todayStr)
                root.put("sessions_today", org.json.JSONArray(JsonConverter.sessionsToJson(todaySessions)))
                
                val existingDna = db.personDnaDao().getByUserId(userId)
                if (existingDna != null) {
                    try {
                        root.put("existing_profile", org.json.JSONObject(existingDna.dna_json))
                    } catch (_: Exception) {}
                }
                
                // 8. Call Python Engine in real-time
                val result = PythonEngine.runAnalysis(root.toString())
                if (result.engineStatus == "ok") {
                    val provisionalEntity = AnalysisResultEntity(
                        userId = userId,
                        date = todayStr,
                        anomalyDetected = result.anomalyDetected,
                        anomalyMessage = result.anomalyMessage,
                        anomalyScore = result.anomalyScore,
                        sustainedDays = result.sustainedDays,
                        alertLevel = result.alertLevel,
                        prototypeMatch = result.prototypeMatch,
                        matchMessage = result.matchMessage,
                        prototypeConfidence = result.prototypeConfidence,
                        gateResults = result.gateResultsJson,
                        l2Modifier = result.l2Modifier,
                        coherence = result.coherence,
                        rhythmDissolution = result.rhythmDissolution,
                        sessionIncoherence = result.sessionIncoherence,
                        effectiveScore = result.anomalyScore * result.l2Modifier,
                        evidenceAccumulated = result.evidence,
                        patternType = result.patternType,
                        flaggedFeatures = org.json.JSONArray(result.flaggedFeatures).toString()
                    )
                    
                    // Push live results to DataRepository
                    DataRepository.updateProvisionalAnalysis(provisionalEntity)

                    // Store provisional ObservationEntity for Home Hero Card real-time update
                    val flaggedList = result.flaggedFeatures
                    val category = when {
                        flaggedList.any { it.contains("sleep", ignoreCase = true) || it.contains("wake", ignoreCase = true) } -> "Sleep"
                        flaggedList.any { it.contains("step", ignoreCase = true) || it.contains("active", ignoreCase = true) } -> "Activity"
                        flaggedList.any { it.contains("screen", ignoreCase = true) || it.contains("app", ignoreCase = true) || it.contains("unlock", ignoreCase = true) } -> "Digital"
                        flaggedList.any { it.contains("displacement", ignoreCase = true) || it.contains("location", ignoreCase = true) || it.contains("home", ignoreCase = true) } -> "Mobility"
                        else -> "General"
                    }
                    val title = when (category) {
                        "Sleep" -> "Sleep & Bedtime Routine"
                        "Activity" -> "Daily Physical Activity"
                        "Digital" -> "Screen & App Habits"
                        "Mobility" -> "Movement & Mobility"
                        else -> "Daily Rhythm Summary"
                    }

                    val observationEntity = ObservationEntity(
                        userId = userId,
                        date = todayStr,
                        category = category,
                        title = title,
                        narrative = result.observationStory,
                        feedbackState = "unresolved",
                        feedbackCategory = "",
                        feedbackNotes = "",
                        baselineConfidence = result.baselineConfidence,
                        isQuietDay = result.alertLevel == "green",
                        flaggedFeatures = org.json.JSONArray(result.flaggedFeatures).toString()
                    )
                    val existingObs = db.observationDao().getByDate(userId, todayStr)
                    if (existingObs != null) {
                        db.observationDao().update(observationEntity.copy(id = existingObs.id))
                    } else {
                        db.observationDao().insert(observationEntity)
                    }
                    
                    // Construct live baseline PersonalityVector using bayesianMeans/bayesianStds
                    if (result.bayesianMeans.isNotEmpty() && result.bayesianStds.isNotEmpty()) {
                        val provisionalBaselineVector = PersonalityVector(
                            screenTimeHours = result.bayesianMeans["screenTimeHours"] ?: 0f,
                            unlockCount = result.bayesianMeans["unlockCount"] ?: 0f,
                            appLaunchCount = result.bayesianMeans["appLaunchCount"] ?: 0f,
                            notificationsToday = result.bayesianMeans["notificationsToday"] ?: 0f,
                            socialAppRatio = result.bayesianMeans["socialAppRatio"] ?: 0f,
                            callsPerDay = result.bayesianMeans["callsPerDay"] ?: 0f,
                            callDurationMinutes = result.bayesianMeans["callDurationMinutes"] ?: 0f,
                            uniqueContacts = result.bayesianMeans["uniqueContacts"] ?: 0f,
                            conversationFrequency = result.bayesianMeans["conversationFrequency"] ?: 0f,
                            dailyDisplacementKm = result.bayesianMeans["dailyDisplacementKm"] ?: 0f,
                            locationEntropy = result.bayesianMeans["locationEntropy"] ?: 0f,
                            homeTimeRatio = result.bayesianMeans["homeTimeRatio"] ?: 0f,
                            wakeTimeHour = result.bayesianMeans["wakeTimeHour"] ?: 0f,
                            sleepTimeHour = result.bayesianMeans["sleepTimeHour"] ?: 0f,
                            sleepDurationHours = result.bayesianMeans["sleepDurationHours"] ?: 0f,
                            dailyStepCount = result.bayesianMeans["dailyStepCount"] ?: 0f,
                            activeMinutes = result.bayesianMeans["activeMinutes"] ?: 0f,
                            keystrokeSpeed = result.bayesianMeans["keystrokeSpeed"] ?: 0f,
                            backspaceRatio = result.bayesianMeans["backspaceRatio"] ?: 0f,
                            scrollVelocity = result.bayesianMeans["scrollVelocity"] ?: 0f,
                            daylightExposureMinutes = result.bayesianMeans["daylightExposureMinutes"] ?: 0f,
                            chargeRegularity = result.bayesianMeans["chargeRegularity"] ?: 0f,
                            chargeDurationHours = result.bayesianMeans["chargeDurationHours"] ?: 0f,
                            upiTransactionsToday = result.bayesianMeans["upiTransactionsToday"] ?: 0f,
                            appUninstallsToday = result.bayesianMeans["appUninstallsToday"] ?: 0f,
                            appInstallsToday = result.bayesianMeans["appInstallsToday"] ?: 0f,
                            calendarEventsToday = result.bayesianMeans["calendarEventsToday"] ?: 0f,
                            mediaCountToday = result.bayesianMeans["mediaCountToday"] ?: 0f,
                            downloadsToday = result.bayesianMeans["downloadsToday"] ?: 0f,
                            musicTimeMinutes = result.bayesianMeans["musicTimeMinutes"] ?: 0f,
                            variances = result.bayesianStds
                        )
                        DataRepository.updateProvisionalBaseline(provisionalBaselineVector)
                    }
                }
            } catch (e: Exception) {
                Log.e("MHealth.Service", "Provisional real-time analysis failed: ${e.message}", e)
            }
        }
    }

    private suspend fun handleAnomalyDetection(snapshot: PersonalityVector, today: Int, savedDay: Int, isSimulated: Boolean) {
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

    private suspend fun persistDailySnapshot(snapshot: PersonalityVector, dayOfYear: Int, isSimulated: Boolean) {
        val userId = DataRepository.userProfile.value?.email ?: "default_user"

        // FIX 3: Safe year-boundary-proof date computation.
        val nowCal   = Calendar.getInstance()
        val todayDoy = nowCal.get(Calendar.DAY_OF_YEAR)
        val daysAgo  = if (todayDoy >= dayOfYear) {
            todayDoy - dayOfYear
        } else {
            val prevYearCal      = Calendar.getInstance().apply { set(Calendar.YEAR, nowCal.get(Calendar.YEAR) - 1) }
            val daysInPrevYear   = prevYearCal.getActualMaximum(Calendar.DAY_OF_YEAR)
            daysInPrevYear - dayOfYear + todayDoy
        }
        val cal     = Calendar.getInstance().apply { add(Calendar.DAY_OF_YEAR, -daysAgo.coerceAtLeast(0)) }
        val dateStr = dateFmt.format(cal.time)

        val entity = JsonConverter.fromPersonalityVector(userId, dateStr, snapshot, isSimulated)
        try {
            MHealthDatabase.getInstance(this@MonitoringService)
                .dailyFeaturesDao().insert(entity)

            // Automatically push un-synced data to Firebase database
            syncUnstagedDailyFeaturesToFirebase()
        } catch (e: Exception) {
            Log.e("MHealth.Service", "Error persisting daily snapshot", e)
        }
    }

    /**
     * FIX: Persist anomaly detection results to Room immediately when detected.
     * Previously, only NightlyAnalysisWorker wrote to analysis_results table,
     * so days analyzed by MonitoringService had no anomaly scores in the export.
     */
    private suspend fun persistAnomalyResultToRoom(report: DailyReport, dayOfYear: Int, isSimulated: Boolean) {
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
        persistAnomalyResultToRoom(report, dateStr, isSimulated)
    }

    private suspend fun persistAnomalyResultToRoom(report: DailyReport, dateStr: String, isSimulated: Boolean) {
        val userId = DataRepository.userProfile.value?.email ?: "default_user"

        try {
            val db = MHealthDatabase.getInstance(this@MonitoringService)

                // Check if result already exists for this date (avoid duplicates)
                val existing = db.analysisResultDao().getByDate(userId, dateStr)
                if (existing != null) {
                    Log.w(TAG, "Anomaly result already exists for $dateStr, skipping duplicate")
                    return
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
                    prototypeConfidence = (report.evidenceAccumulated / 10f).coerceIn(0f, 1f),
                    gateResults = "{}"
                )
                db.analysisResultDao().insert(resultEntity)
            Log.i(TAG, "Anomaly result persisted for $dateStr: score=${report.anomalyScore}, level=${report.alertLevel}")
        } catch (e: Exception) {
            Log.e(TAG, "Error persisting anomaly result to Room", e)
        }
    }

    private suspend fun persistBaselineToRoom(
        baseline: PersonalityVector,
        baselineDays: Int
    ) {
        val userId = DataRepository.userProfile.value?.email ?: "default_user"
        val db = MHealthDatabase.getInstance(this@MonitoringService)
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

        // Upload firmly established baseline to Cloud Backup (no-op in release)
        FirebaseSyncHelper.syncBaseline(this@MonitoringService, baseline, today)
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

        val circularTimeFeatures = setOf("sleepTimeHour", "wakeTimeHour")

        features.forEach { feature ->
            val vals = vectors.map { it.toMap()[feature] ?: 0f }

            if (feature in circularTimeFeatures) {
                // Circular vector average (summing sines and cosines to avoid boundary wraps)
                var sinSum = 0.0
                var cosSum = 0.0
                vals.forEach { v ->
                    val radians = v * (2.0 * Math.PI / 24.0)
                    sinSum += Math.sin(radians)
                    cosSum += Math.cos(radians)
                }
                val avgAngle = Math.atan2(sinSum, cosSum)
                val circularAvg = ((avgAngle * (24.0 / (2.0 * Math.PI))) + 24.0) % 24.0
                averages[feature] = circularAvg.toFloat()

                // Calculate standard deviation using circular differences
                val diffsSq = vals.map { v ->
                    val diff = v - circularAvg
                    val diffCirc = ((diff + 12.0) % 24.0) - 12.0
                    diffCirc * diffCirc
                }
                val varianceVal = diffsSq.average().toFloat()
                val sd = kotlin.math.sqrt(varianceVal)
                variances[feature] = if (sd < 0.01f) 0.01f else sd
            } else {
                // Trim outliers for robust mean calculation
                val sortedVals = vals.sorted()
                val trimmedVals = if (n > 4 && trimCount > 0) {
                    sortedVals.subList(trimCount, n - trimCount)
                } else {
                    sortedVals
                }
                val robustAvg = trimmedVals.average().toFloat()
                averages[feature] = robustAvg
                variances[feature] = calculateSD(vals, robustAvg)
            }
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
            conversationFrequency = averages["conversationFrequency"] ?: 0f,
            dailyDisplacementKm = averages["dailyDisplacementKm"] ?: 0f,
            locationEntropy = averages["locationEntropy"] ?: 0f,
            homeTimeRatio = averages["homeTimeRatio"] ?: 0f,
            wakeTimeHour = averages["wakeTimeHour"] ?: 0f,
            sleepTimeHour = averages["sleepTimeHour"] ?: 0f,
            sleepDurationHours = averages["sleepDurationHours"] ?: 0f,
            dailyStepCount = averages["dailyStepCount"] ?: 0f,
            activeMinutes = averages["activeMinutes"] ?: 0f,
            keystrokeSpeed = averages["keystrokeSpeed"] ?: 0f,
            backspaceRatio = averages["backspaceRatio"] ?: 0f,
            scrollVelocity = averages["scrollVelocity"] ?: 0f,
            daylightExposureMinutes = averages["daylightExposureMinutes"] ?: 0f,
            chargeRegularity = averages["chargeRegularity"] ?: 0f,
            chargeDurationHours = averages["chargeDurationHours"] ?: 0f,
            upiTransactionsToday = averages["upiTransactionsToday"] ?: 0f,
            appUninstallsToday = averages["appUninstallsToday"] ?: 0f,
            appInstallsToday = averages["appInstallsToday"] ?: 0f,
            calendarEventsToday = averages["calendarEventsToday"] ?: 0f,
            mediaCountToday = averages["mediaCountToday"] ?: 0f,
            downloadsToday = averages["downloadsToday"] ?: 0f,
            musicTimeMinutes = averages["musicTimeMinutes"] ?: 0f,
            variances = variances
        )
    }

    /** Returns standard deviation (used as variance bound in AnomalyDetector) */
    private fun calculateSD(values: List<Float>, mean: Float): Float {
        if (values.size < 2) return 1.0f
        val sd = kotlin.math.sqrt(values.map { (it - mean) * (it - mean) }.average()).toFloat()
        return if (sd < 0.01f) 0.01f else sd
    }

    private suspend fun syncUnstagedDailyFeaturesToFirebase() {
        val userId = DataRepository.userProfile.value?.email ?: return
        FirebaseSyncHelper.syncUnstagedDailyFeatures(this@MonitoringService, userId)
    }

    private fun sendAlertNotification(level: String, notes: String) {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.notify(
            2, NotificationCompat.Builder(this, "mhealth_monitoring")
                .setContentTitle("Pattern Change: ${level.uppercase()}")
                .setContentText(notes)
                .setSmallIcon(R.mipmap.ic_launcher)
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setAutoCancel(true)
                .build()
        )
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
    private suspend fun recoverMissedDayIfNeeded(userId: String, db: MHealthDatabase) {
        val lastDay = DataRepository.lastProcessedDay.value
        if (lastDay == -1) return // Fresh install — no prior day to recover.

        // Guard: Don't fabricate data for a fresh/reset account with no prior data.
        // Without this, a hard-reset user gets yesterday's UsageStats data
        // injected as if it belonged to the new account.
        val existingDayCount = db.dailyFeaturesDao().countDistinctDays(userId)
        if (existingDayCount == 0) {
            Log.i("MHealth.Service", "Missed-day recovery skipped: fresh account with no prior data")
            return
        }

        val yesterdayCal = Calendar.getInstance().apply { add(Calendar.DAY_OF_YEAR, -1) }
        val yesterdayStr = dateFmt.format(yesterdayCal.time)
        val savedYesterday = db.dailyFeaturesDao().getByDate(userId, yesterdayStr)

        if (savedYesterday == null) {
            Log.w("MHealth.Service", "Missed-day recovery: no snapshot found for $yesterdayStr — recovering from UsageEvents")
            try {
                val startOfToday   = dataCollector.getStartOfDayMs()
                val yesterdayStart = startOfToday - 24 * 3600_000L
                val yesterdayEnd   = startOfToday - 1L
                val locationSnaps  = DataRepository.locationSnapshots.value

                // Re-collect the full prior day from system APIs — still available in the 14-day window
                val missedSnapshot = dataCollector.collectSnapshot(locationSnaps, yesterdayStart, yesterdayEnd)
                val entity = JsonConverter.fromPersonalityVector(userId, yesterdayStr, missedSnapshot, false)
                db.dailyFeaturesDao().insert(entity)
                Log.i("MHealth.Service", "Missed-day recovery: saved snapshot for $yesterdayStr — syncing to Firebase")

                // Immediately push the recovered data to Firestore
                syncUnstagedDailyFeaturesToFirebase()
            } catch (e: Exception) {
                Log.e("MHealth.Service", "Missed-day recovery failed for $yesterdayStr: ${e.message}", e)
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
     * This ensures only Spotify, Gaana, OuerTune, etc. add to musicTimeMinutes.
     */
    private fun isMusicAppActiveViaMediaSession(): String? {
        return try {
            val msm = getSystemService(MediaSessionManager::class.java) ?: return null
            val nlsComponent = ComponentName(this, MHealthNotificationListenerService::class.java)
            val sessions = msm.getActiveSessions(nlsComponent)
            val controller = sessions.firstOrNull { controller ->
                val pkg = controller.packageName
                val state = controller.playbackState
                val isPlaying = state?.state == android.media.session.PlaybackState.STATE_PLAYING
                isPlaying && dataCollector.isMusicApp(pkg)
            }
            controller?.packageName
        } catch (e: SecurityException) {
            // Notification access not granted — cannot identify package, so don't count.
            // isMusicActive() fallback REMOVED: it returns true for ANY audio (Instagram
            // reels, YouTube videos, game sounds, ads) which inflates musicTimeMinutes.
            Log.w("MHealth.Service", "MediaSession access denied (no NotificationListener) — music tracking disabled until granted")
            null
        } catch (e: Exception) {
            Log.w("MHealth.Service", "isMusicAppActiveViaMediaSession error: ${e.message}")
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
                    Log.w("MHealth.Service", "Exact Alarm permission missing — falling back to standard alarm")
                }
            } catch (e: Exception) {
                Log.e("MHealth.Service", "Failed to set midnight alarm: ${e.message}")
                am.set(AlarmManager.RTC_WAKEUP, cal.timeInMillis, pendingIntent)
            }
        }
        Log.i("MHealth.Service", "Midnight transition alarm set for ${cal.time}")
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

    private fun checkAndSendCheckinNotifications() {
        try {
            val prefs = getSharedPreferences("mhealth_data_store", Context.MODE_PRIVATE)

            val calendar = Calendar.getInstance()
            val hour = calendar.get(Calendar.HOUR_OF_DAY)
            val todayStr = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(calendar.time)

            // 1. Daily Check-in Notification (Send after 7 PM / 19:00 if not done and not sent today)
            val dailyRemindersEnabled = prefs.getBoolean("daily_reminders_enabled", true)
            if (dailyRemindersEnabled && hour >= 19) {
                val lastCheckinDate = prefs.getString("daily_checkin_date_last", "") ?: ""
                val lastSentDate = prefs.getString("daily_checkin_notification_sent_date", "") ?: ""
                if (lastCheckinDate != todayStr && lastSentDate != todayStr) {
                    sendDailyCheckinNotification()
                    prefs.edit().putString("daily_checkin_notification_sent_date", todayStr).apply()
                }
            }

            // 2. Monthly Check-in Notification (Send after 12 PM / 12:00 if due, not done today, and not sent for this cycle)
            val monthlyRemindersEnabled = prefs.getBoolean("monthly_reminders_enabled", true)
            if (monthlyRemindersEnabled) {
                val lastMonthlyCheckinDate = prefs.getString("monthly_checkin_last_date", "") ?: ""
                val lastMonthlySentDate = prefs.getString("monthly_checkin_notification_sent_date", "") ?: ""
                
                val isMonthlyDue = getMonthlyCooldownDays(prefs) <= 0
                val hasSentForThisCycle = if (lastMonthlySentDate.isNotEmpty() && lastMonthlyCheckinDate.isNotEmpty()) {
                    val sdf = SimpleDateFormat("yyyy-MM-dd", Locale.US)
                    try {
                        val sentTime = sdf.parse(lastMonthlySentDate)?.time ?: 0L
                        val checkinTime = sdf.parse(lastMonthlyCheckinDate)?.time ?: 0L
                        sentTime > checkinTime
                    } catch (e: Exception) {
                        false
                    }
                } else lastMonthlySentDate == todayStr

                if (isMonthlyDue && lastMonthlyCheckinDate != todayStr && !hasSentForThisCycle) {
                    if (hour >= 12) {
                        sendMonthlyCheckinNotification()
                        prefs.edit().putString("monthly_checkin_notification_sent_date", todayStr).apply()
                    }
                }
            }

            // 3. Weekly Summary Notification (Send on Sunday after 6 PM / 18:00 if not sent this week)
            val weeklySummaryEnabled = prefs.getBoolean("weekly_summary_notifications_enabled", true)
            if (weeklySummaryEnabled) {
                val dayOfWeek = calendar.get(Calendar.DAY_OF_WEEK)
                if (dayOfWeek == Calendar.SUNDAY && hour >= 18) {
                    val lastWeeklySentDate = prefs.getString("weekly_summary_notification_sent_date", "") ?: ""
                    if (lastWeeklySentDate != todayStr) {
                        sendWeeklySummaryNotification()
                        prefs.edit().putString("weekly_summary_notification_sent_date", todayStr).apply()
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("MHealth.Service", "Error checking checkin notifications: ${e.message}", e)
        }
    }

    private fun sendWeeklySummaryNotification() {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            putExtra("navigate_to", "insights")
        }
        val pendingIntent = PendingIntent.getActivity(
            this, 103, intent, PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        
        val notification = NotificationCompat.Builder(this, "mhealth_monitoring")
            .setContentTitle("Your Weekly Wellness Insights")
            .setContentText("Your weekly qualitative rhythm summary is ready. Tap to view.")
            .setSmallIcon(R.mipmap.ic_launcher)
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setAutoCancel(true)
            .setContentIntent(pendingIntent)
            .build()
            
        nm.notify(5, notification)
    }

    private fun getMonthlyCooldownDays(prefs: SharedPreferences): Int {
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

    private fun sendDailyCheckinNotification() {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        }
        val pendingIntent = PendingIntent.getActivity(
            this, 101, intent, PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        
        val notification = NotificationCompat.Builder(this, "mhealth_monitoring")
            .setContentTitle("Time for your Daily Check-in")
            .setContentText("Take a moment to reflect on your day and log your wellness.")
            .setSmallIcon(R.mipmap.ic_launcher)
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setAutoCancel(true)
            .setContentIntent(pendingIntent)
            .build()
            
        nm.notify(3, notification)
    }

    private fun sendMonthlyCheckinNotification() {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        }
        val pendingIntent = PendingIntent.getActivity(
            this, 102, intent, PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        
        val notification = NotificationCompat.Builder(this, "mhealth_monitoring")
            .setContentTitle("Monthly Wellness Check-in")
            .setContentText("It's time for your monthly well-being assessment.")
            .setSmallIcon(R.mipmap.ic_launcher)
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setAutoCancel(true)
            .setContentIntent(pendingIntent)
            .build()
            
        nm.notify(4, notification)
    }
}

/** Standalone receiver to handle the exact midnight alarm */
class MidnightReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        Log.i("MHealth.Service", "Midnight Alarm Fired! Triggering day transition logic.")
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
