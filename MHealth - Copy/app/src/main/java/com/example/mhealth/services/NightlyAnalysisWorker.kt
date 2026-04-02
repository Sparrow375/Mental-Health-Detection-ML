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
 *   2. Check baseline readiness (28 days threshold)
 *   3. Build JSON input via JsonConverter
 *   4. Call PythonEngine.runAnalysis()
 *   5. Store AnalysisResultEntity in Room
 *   6. Update DataRepository live state (for UI)
 *   7. Enqueue CloudSyncWorker stub (Phase 6)
 *
 * Input data keys:
 *   KEY_USER_ID — the user ID string
 */
class NightlyAnalysisWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    companion object {
        const val TAG          = "MHealth.NightlyWorker"
        const val KEY_USER_ID  = "user_id"
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
                    ExistingPeriodicWorkPolicy.KEEP,
                    request
                )
            Log.i(TAG, "Nightly analysis scheduled for user=$userId")
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

        val today = DATE_FMT.format(Date())
        Log.i(TAG, "Running nightly analysis for user=$userId date=$today")

        return try {
            // ── 1. Load today's features ───────────────────────────────────────
            val todayFeatures = db.dailyFeaturesDao().getByDate(userId, today)
            if (todayFeatures == null) {
                Log.w(TAG, "No feature data for today ($today) — skipping analysis")
                return Result.success()
            }

            // ── 2. Check baseline readiness ────────────────────────────────────
            val profileEntity = db.userProfileDao().get(userId)
            val baselineReady = profileEntity?.baselineReady ?: false
            val baselineEntities = db.baselineDao().getBaseline(userId)

            if (!baselineReady || baselineEntities.isEmpty()) {
                Log.i(TAG, "Baseline not ready yet — storing data only")
                return Result.success()
            }

            // ── 3. Fetch history (last 14 days) ────────────────────────────────
            val history = db.dailyFeaturesDao().getLatestN(userId, limit = 15)
                .filter { it.date != today }   // exclude today from history
                .sortedBy { it.date }           // oldest first

            // ── 4. Build JSON input ────────────────────────────────────────────
            val dayNumber = DataRepository.reports.value.size + 1
            val inputJson = JsonConverter.toEngineJson(todayFeatures, baselineEntities, history)

            // Inject day_number and gate_state into the JSON before passing to engine
            val jsonWithMeta = injectMetadata(
                inputJson = inputJson,
                dayNumber = dayNumber,
                contaminated = profileEntity?.baselineContaminated ?: false,
                gateResultsJson = db.analysisResultDao().getLatest(userId)?.gateResults ?: "{}"
            )

            // ── 5. Call Python engine ──────────────────────────────────────────
            val engineResult = PythonEngine.runAnalysis(jsonWithMeta)
            Log.i(TAG, "Engine result: status=${engineResult.engineStatus} " +
                    "alert=${engineResult.alertLevel} match=${engineResult.prototypeMatch}")

            if (engineResult.engineStatus == "error") {
                Log.e(TAG, "Engine error: ${engineResult.errorMessage}")
                return Result.retry()
            }

            // ── 6. Store result in Room ────────────────────────────────────────
            val resultEntity = AnalysisResultEntity(
                userId              = userId,
                date                = today,
                anomalyDetected     = engineResult.anomalyDetected,
                anomalyMessage      = engineResult.anomalyMessage,
                anomalyScore        = engineResult.anomalyScore,
                sustainedDays       = engineResult.sustainedDays,
                alertLevel          = engineResult.alertLevel,
                prototypeMatch      = engineResult.prototypeMatch,
                matchMessage        = engineResult.matchMessage,
                prototypeConfidence = engineResult.prototypeConfidence,
                gateResults         = engineResult.gateResultsJson
            )
            db.analysisResultDao().insert(resultEntity)

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

            // ── 8. Enqueue CloudSyncWorker (Phase 6) ──────────────────────
            val syncWork = OneTimeWorkRequestBuilder<CloudSyncWorker>()
                .setConstraints(
                    Constraints.Builder()
                        .setRequiredNetworkType(NetworkType.CONNECTED)
                        .build()
                )
                .build()
            WorkManager.getInstance(applicationContext).enqueue(syncWork)
            Log.d(TAG, "Enqueued CloudSyncWorker for data upload")

            Log.i(TAG, "Nightly analysis complete for $today")
            Result.success()

        } catch (e: Exception) {
            Log.e(TAG, "Nightly worker failed: ${e.message}", e)
            Result.retry()
        }
    }

    // ─── Helpers ──────────────────────────────────────────────────────────────

    /** Injects day_number, gate_state, and baseline_contaminated into existing JSON string. */
    private fun injectMetadata(
        inputJson: String,
        dayNumber: Int,
        contaminated: Boolean,
        gateResultsJson: String
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
