package com.example.mhealth.services

import android.content.Context
import android.util.Log
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.example.mhealth.logic.JsonConverter
import com.example.mhealth.logic.PythonEngine
import com.example.mhealth.logic.DataRepository
import com.example.mhealth.logic.db.AnalysisResultEntity
import com.example.mhealth.logic.db.MHealthDatabase
import com.example.mhealth.logic.db.PersonDnaEntity
import com.example.mhealth.models.DailyReport
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale
import java.util.concurrent.TimeUnit

/**
 * NightlyAnalysisWorker — WorkManager worker that runs once per day.
 *
 * Pipeline:
 *   1. Query today's DailyFeaturesEntity from Room
 *   2. Check baseline readiness (configurable threshold)
 *   3. Build JSON input via JsonConverter
 *   4. Call PythonEngine.runAnalysis()
 *   5. Store AnalysisResultEntity in Room
 *   6. Trigger immediate CloudSyncWorker sync to Firestore
 *   7. Update DataRepository live state (for UI)
 *
 * Input data keys:
 *   KEY_USER_ID — the user ID string (email)
 */
class NightlyAnalysisWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    companion object {
        const val TAG          = "MHealth.NightlyWorker"
        const val KEY_USER_ID  = "user_id"
        const val KEY_TARGET_DATE = "target_date"
        const val KEY_FORCE_RUN = "force_run"
        const val WORK_NAME    = "nightly_analysis"


        private val DATE_FMT   = SimpleDateFormat("yyyy-MM-dd", Locale.US)

        /**
         * Schedule (or replace) the nightly analysis to run ~midnight,
         * constrained to battery-not-low.
         */
        fun schedule(context: Context, userId: String) {
            val data = workDataOf(KEY_USER_ID to userId)

            val constraints = Constraints.Builder()
                .setRequiresBatteryNotLow(true)
                .build()

            val request = PeriodicWorkRequestBuilder<NightlyAnalysisWorker>(
                1, TimeUnit.DAYS
            )
                .setInitialDelay(delayUntilMidnight(), TimeUnit.MILLISECONDS)
                .setConstraints(constraints)
                .setInputData(data)
                .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30, TimeUnit.MINUTES)
                .build()

            WorkManager.getInstance(context)
                .enqueueUniquePeriodicWork(
                    WORK_NAME,
                    ExistingPeriodicWorkPolicy.UPDATE,
                    request
                )
            Log.i(TAG, "Nightly analysis scheduled for user=$userId")
        }

        fun runNow(context: Context, userId: String, targetDate: String? = null, forceRun: Boolean = false) {
            val data = workDataOf(
                KEY_USER_ID to userId,
                KEY_TARGET_DATE to targetDate,
                KEY_FORCE_RUN to forceRun
            )
            val request = OneTimeWorkRequestBuilder<NightlyAnalysisWorker>()
                .setInputData(data)
                .build()
            WorkManager.getInstance(context).enqueue(request)
            Log.i(TAG, "Manual analysis triggered for user=$userId date=$targetDate")
        }


        private fun delayUntilMidnight(): Long {
            val now      = Calendar.getInstance()
            val midnight = Calendar.getInstance().apply {
                set(Calendar.HOUR_OF_DAY, 0)
                set(Calendar.MINUTE, 5)   // 00:05 to let the phone settle
                set(Calendar.SECOND, 0)
                set(Calendar.MILLISECOND, 0)
                if (before(now)) add(Calendar.DAY_OF_MONTH, 1)
            }
            return midnight.timeInMillis - now.timeInMillis
        }
    }

    private val db = MHealthDatabase.getInstance(applicationContext)

    override suspend fun doWork(): Result {
        val userId = inputData.getString(KEY_USER_ID) ?: run {
            Log.e(TAG, "No userId in worker input data")
            return Result.failure()
        }

        // Date selection: manual targetDate or default to "yesterday" (for nightly runs)
        val manualDate = inputData.getString(KEY_TARGET_DATE)
        val targetDate = manualDate ?: run {
            val yesterdayCal = Calendar.getInstance().apply { add(Calendar.DAY_OF_YEAR, -1) }
            DATE_FMT.format(yesterdayCal.time)
        }
        
        Log.i(TAG, "Running nightly analysis for user=$userId date=$targetDate")


        return try {
            // ── 1. Load today's features ───────────────────────────────────────
            val todayFeatures = db.dailyFeaturesDao().getByDate(userId, targetDate)
            if (todayFeatures == null) {
                Log.w(TAG, "No feature data for $targetDate — skipping analysis")
                return Result.success()
            }

            // ── 2. Check baseline readiness ────────────────────────────────────
            val profileEntity = db.userProfileDao().getProfile(userId)
            val l1BaselineReady = profileEntity?.baselineReady ?: false
            val baselineEntities = db.baselineDao().getBaseline(userId)

            // DNA (L2) readiness uses daily_dna_snapshot count, NOT daily_features count
            val dnaDaysCollected = db.dailyDnaSnapshotDao().countDistinctDays(userId)
            val dnaThreshold = DataRepository.dnaBaselineDaysRequired.value
            val dnaReady = dnaDaysCollected >= dnaThreshold

            Log.i(TAG, "DNA Progress Check: dnaDays=$dnaDaysCollected, threshold=$dnaThreshold, dnaReady=$dnaReady, l1Ready=$l1BaselineReady")

            val forceRun = inputData.getBoolean(KEY_FORCE_RUN, false)

            if (!dnaReady && !l1BaselineReady && !forceRun) {
                Log.w(TAG, "Neither baseline nor DNA ready yet (dnaDays=$dnaDaysCollected/$dnaThreshold) — aborting analysis")
                return Result.success()
            }

            if (forceRun) {
                Log.i(TAG, "Force run requested — proceeding with analysis regardless of readiness")
            }

            // ── 3. Fetch history (last 14 days) ────────────────────────────────
            val history = db.dailyFeaturesDao().getLatestN(userId, limit = 15)
                .filter { it.date != targetDate }   // exclude the analysis day from history
                .sortedBy { it.date }           // oldest first

            // ── 3b. Fetch historical anomaly scores (for pattern detection) ─────
            val historicalScores = db.analysisResultDao().getLatestN(userId, 14)
                .reversed()  // oldest first
                .map { it.anomalyScore }

            // ── 4. Build JSON input ────────────────────────────────────────────
            val dayNumber = dnaDaysCollected + 1
            val inputJson = JsonConverter.toEngineJson(todayFeatures, baselineEntities, history)

            // Inject day_number, gate_state, historical anomaly scores, and session data
            val jsonWithMeta = injectMetadata(
                inputJson = inputJson,
                dayNumber = dayNumber,
                contaminated = profileEntity?.baselineContaminated ?: false,
                gateResultsJson = db.analysisResultDao().getLatest(userId)?.gateResults ?: "{}",
                historicalScores = historicalScores
            )

            // ── 4b. Inject session data for Phone DNA and App DNA computation ──
            val jsonWithSessions = try {
                val obj = org.json.JSONObject(jsonWithMeta)
                // Get sessions for the baseline period (last 30 days)
                val dateFormat = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.US)
                val cal = java.util.Calendar.getInstance()
                cal.add(java.util.Calendar.DAY_OF_YEAR, -30)
                val startDate = dateFormat.format(cal.time)
                val cal2 = java.util.Calendar.getInstance()
                val endDate = dateFormat.format(cal2.time)
                val allSessions = db.appSessionDao().getByDateRange(startDate, endDate)
                obj.put("sessions", org.json.JSONArray(JsonConverter.sessionsToJson(allSessions)))
                Log.d(TAG, "Injected ${allSessions.size} sessions for DNA computation")

                // Get today's sessions specifically
                val todaySessions = db.appSessionDao().getByDate(targetDate)
                obj.put("sessions_today", org.json.JSONArray(JsonConverter.sessionsToJson(todaySessions)))
                obj.toString()
            } catch (e: Exception) {
                Log.w(TAG, "Failed to inject sessions: ${e.message}")
                jsonWithMeta
            }

            // ── 5. Call Python engine ──────────────────────────────────────────
            val engineResult = PythonEngine.runAnalysis(jsonWithSessions)
            Log.i(TAG, "Engine result: status=${engineResult.engineStatus} " +
                    "alert=${engineResult.alertLevel} match=${engineResult.prototypeMatch}")

            if (engineResult.engineStatus == "error") {
                Log.e(TAG, "Engine error: ${engineResult.errorMessage}")
                return Result.retry()
            }

            // ── 6. Store result in Room ────────────────────────────────────────
            val resultEntity = AnalysisResultEntity(
                userId              = userId,
                date                = targetDate,
                anomalyDetected     = engineResult.anomalyDetected,
                anomalyMessage      = engineResult.anomalyMessage,
                anomalyScore        = engineResult.anomalyScore,
                sustainedDays       = engineResult.sustainedDays,
                alertLevel          = engineResult.alertLevel,
                prototypeMatch      = engineResult.prototypeMatch,
                matchMessage        = engineResult.matchMessage,
                prototypeConfidence = (engineResult.prototypeConfidence / 10f).coerceIn(0f, 1f),
                gateResults         = engineResult.gateResultsJson,
                l2Modifier          = engineResult.l2Modifier,
                coherence           = engineResult.coherence,
                rhythmDissolution   = engineResult.rhythmDissolution,
                sessionIncoherence  = engineResult.sessionIncoherence,
                effectiveScore      = engineResult.anomalyScore * engineResult.l2Modifier,
                evidenceAccumulated = engineResult.evidence,
                patternType         = engineResult.patternType,
                flaggedFeatures     = org.json.JSONArray(engineResult.flaggedFeatures).toString()
            )
            // Dedup: if result already exists for this date, update instead of inserting duplicate
            val existingResult = db.analysisResultDao().getByDate(userId, targetDate)
            if (existingResult != null) {
                db.analysisResultDao().update(resultEntity.copy(id = existingResult.id))
                Log.i(TAG, "Analysis result updated for $targetDate | anomaly_score: ${engineResult.anomalyScore} | alert: ${engineResult.alertLevel}")
            } else {
                db.analysisResultDao().insert(resultEntity)
                Log.i(TAG, "Analysis result saved to Room for $targetDate | anomaly_score: ${engineResult.anomalyScore} | alert: ${engineResult.alertLevel}")
            }

            // Don't wait for the scheduled CloudSyncWorker - sync immediately after saving
            try {
                val syncWork = OneTimeWorkRequestBuilder<CloudSyncWorker>()
                    .setConstraints(
                        Constraints.Builder()
                            .setRequiredNetworkType(NetworkType.CONNECTED)
                            .build()
                    )
                    .build()
                WorkManager.getInstance(applicationContext).enqueue(syncWork)
                Log.d(TAG, "Immediate sync triggered for analysis result")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to enqueue immediate sync: ${e.message}", e)
                // Don't fail the worker - CloudSyncWorker will run on its schedule
            }

            // ── 7. Update live DataRepository state (feeds existing UI) ─────────
            val dailyReport = DailyReport(
                dayNumber             = dayNumber,
                date                  = Date(),
                anomalyScore          = engineResult.anomalyScore,
                alertLevel            = engineResult.alertLevel,
                flaggedFeatures       = engineResult.flaggedFeatures,
                patternType           = engineResult.patternType,
                sustainedDeviationDays = engineResult.sustainedDays,
                evidenceAccumulated   = engineResult.evidence,
                topDeviations         = emptyMap(),   // populated by Python; use notes for display
                notes                 = buildNotes(engineResult)
            )
            DataRepository.addReport(dailyReport)

            // ── 7b. Persist System 1 Profile (DNA Baseline, Clusters, Texture) ──
            val profileJson = engineResult.profileJson
            Log.i(TAG, "Engine returned Profile JSON | Length: ${profileJson.length}")
            if (profileJson.length > 2) {
                Log.d(TAG, "Profile JSON Snippet: ${profileJson.take(200)}...")
            }

            // Save whenever the profile is a real JSON object (not "{}" or empty/null)
            val hasProfile = profileJson.length > 2 &&
                profileJson.trimStart().startsWith("{") &&
                profileJson != "{}"
            
            if (hasProfile) {
                try {
                    val existingDna = db.personDnaDao().getByUserId(userId)
                    val now = System.currentTimeMillis()
                    if (existingDna != null) {
                        db.personDnaDao().updateDna(userId, engineResult.profileJson, now)
                    } else {
                        db.personDnaDao().insert(
                            PersonDnaEntity(
                                person_id = userId,
                                dna_json = engineResult.profileJson,
                                created_at = now,
                                last_updated = now
                            )
                        )
                    }
                    DataRepository.updateS1Profile(engineResult.profileJson)
                    Log.i(TAG, "System 1 profile persisted to person_dna for $userId")

                    // Mark DNA as ready in the database if it isn't already
                    val profile = db.userProfileDao().getProfile(userId)
                    if (profile != null && !profile.dnaReady) {
                        db.userProfileDao().updateDnaReady(userId, true)
                        DataRepository.setIsDnaBaselineReady(true)
                        Log.i(TAG, "DNA Baseline marked as READY for $userId")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to persist profile: ${e.message}", e)
                }
            } else {
                Log.w(TAG, "Profile JSON was empty or null — not saving to person_dna. profileJson='${engineResult.profileJson}'")
            }

            // ── 8. Trigger immediate sync to Firestore ────────────────────────
            // Don't wait for the scheduled CloudSyncWorker - sync immediately after analysis
            try {
                val constraints = Constraints.Builder()
                    .setRequiredNetworkType(NetworkType.CONNECTED)
                    .build()
                val syncWork = OneTimeWorkRequestBuilder<CloudSyncWorker>()
                    .setConstraints(constraints)
                    .build()
                WorkManager.getInstance(applicationContext).enqueue(syncWork)
                Log.d(TAG, "Immediate sync triggered after analysis and DNA persistence")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to enqueue immediate sync: ${e.message}", e)
            }

            Log.i(TAG, "Nightly analysis complete for $targetDate")
            Result.success()

        } catch (e: Exception) {
            Log.e(TAG, "Worker failed: ${e.message}", e)
            Result.retry()
        } finally {
            DataRepository.setDnaAnalysing(false)
        }
    }

    // ─── Helpers ──────────────────────────────────────────────────────────────

    /** Injects day_number, gate_state, baseline_contaminated, and historical anomaly scores into JSON. */
    private fun injectMetadata(
        inputJson: String,
        dayNumber: Int,
        contaminated: Boolean,
        gateResultsJson: String,
        historicalScores: List<Float> = emptyList()
    ): String {
        return try {
            val obj = org.json.JSONObject(inputJson)
            obj.put("day_number", dayNumber)
            obj.put("baseline_contaminated", contaminated)
            // Parse gate_state from last result
            val gateState = try {
                org.json.JSONObject(gateResultsJson)
            } catch (e: Exception) {
                org.json.JSONObject()
            }
            obj.put("gate_state", gateState)

            // Add historical anomaly scores for pattern detection
            if (historicalScores.isNotEmpty()) {
                val scoresArray = org.json.JSONArray()
                historicalScores.forEach { scoresArray.put(it.toDouble()) }
                obj.put("historical_anomaly_scores", scoresArray)
            }

            obj.toString()
        } catch (e: Exception) {
            inputJson   // fallback: pass as-is
        }
    }

    private fun buildNotes(r: PythonEngine.AnalysisResult): String {
        val parts = mutableListOf<String>()
        if (r.anomalyDetected) parts.add("Anomaly: ${r.alertLevel.uppercase()}")
        if (r.prototypeMatch != "Normal" && r.prototypeMatch != "Situational") {
            parts.add("Pattern: ${r.prototypeMatch} (${r.confidenceLabel})")
        }
        if (r.engineStatus == "error") parts.add("⚠ Engine error: ${r.errorMessage}")
        return if (parts.isEmpty()) r.anomalyMessage else parts.joinToString(" | ")
    }
}
