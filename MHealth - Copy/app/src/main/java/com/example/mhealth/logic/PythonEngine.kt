package com.example.mhealth.logic

import android.util.Log
import com.chaquo.python.Python
import org.json.JSONObject

/**
 * PythonEngine — Kotlin bridge to the Python analysis engine via Chaquopy.
 *
 * Calls engine.run_analysis(inputJsonString) and parses the structured JSON result
 * into a strongly-typed [AnalysisResult] data class.
 *
 * Must only be called after Python.start() has been invoked (done automatically
 * by Chaquopy once declared in build.gradle — no manual init needed).
 */
object PythonEngine {

    private const val TAG = "MHealth.PythonEngine"

    /** Structured result returned to Kotlin callers. */
    data class AnalysisResult(
        // System 1
        val anomalyDetected: Boolean       = false,
        val anomalyScore: Float            = 0f,
        val alertLevel: String             = "green",   // "green"|"yellow"|"orange"|"red"
        val sustainedDays: Int             = 0,
        val evidence: Float                = 0f,
        val flaggedFeatures: List<String>  = emptyList(),
        val patternType: String            = "stable",
        val anomalyMessage: String         = "",

        // System 2
        val prototypeMatch: String         = "Normal",
        val prototypeConfidence: Float     = 0f,
        val confidenceLabel: String        = "HIGH",
        val matchMessage: String           = "",

        // Gate state (serialised back to JSON for persistence)
        val gateResultsJson: String        = "{}",

        // Meta
        val engineStatus: String           = "ok",
        val errorMessage: String           = ""
    )

    /**
     * Run the full System 1 + System 2 analysis.
     *
     * @param inputJson JSON string conforming to engine.py's input schema
     * @return          Parsed [AnalysisResult]. Never throws — errors are captured in result.
     */
    fun runAnalysis(inputJson: String): AnalysisResult {
        return try {
            val py     = Python.getInstance()
            val engine = py.getModule("engine")
            val rawJson = engine.callAttr("run_analysis", inputJson).toString()
            Log.d(TAG, "Engine raw result: $rawJson")
            parseResult(rawJson)
        } catch (e: Exception) {
            Log.e(TAG, "Python engine call failed: ${e.message}", e)
            AnalysisResult(
                engineStatus  = "error",
                errorMessage  = e.message ?: "Unknown Python error"
            )
        }
    }

    // ─── JSON → AnalysisResult ─────────────────────────────────────────────────

    private fun parseResult(rawJson: String): AnalysisResult {
        return try {
            val root      = JSONObject(rawJson)
            val status    = root.optString("status", "ok")

            if (status == "error") {
                return AnalysisResult(
                    engineStatus = "error",
                    errorMessage = root.optString("error_message", "Unknown error")
                )
            }

            val anomaly   = root.optJSONObject("anomaly")  ?: JSONObject()
            val prototype = root.optJSONObject("prototype") ?: JSONObject()
            val gate      = root.optJSONObject("gate")      ?: JSONObject()

            // Flagged features array → List<String>
            val flaggedArr = anomaly.optJSONArray("flagged_features")
            val flagged = buildList {
                flaggedArr?.let { arr ->
                    for (i in 0 until arr.length()) add(arr.getString(i))
                }
            }

            AnalysisResult(
                anomalyDetected     = anomaly.optBoolean("detected",      false),
                anomalyScore        = anomaly.optDouble("anomaly_score",   0.0).toFloat(),
                alertLevel          = anomaly.optString("alert_level",     "green"),
                sustainedDays       = anomaly.optInt("sustained_days",     0),
                evidence            = anomaly.optDouble("evidence",        0.0).toFloat(),
                flaggedFeatures     = flagged,
                patternType         = anomaly.optString("pattern_type",   "stable"),
                anomalyMessage      = anomaly.optString("message",         ""),
                prototypeMatch      = prototype.optString("match",         "Normal"),
                prototypeConfidence = prototype.optDouble("confidence",    0.0).toFloat(),
                confidenceLabel     = prototype.optString("confidence_label","HIGH"),
                matchMessage        = prototype.optString("message",       ""),
                gateResultsJson     = gate.toString(),
                engineStatus        = status,
                errorMessage        = root.optString("error_message", "")
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse Python result: ${e.message}")
            AnalysisResult(engineStatus = "error", errorMessage = "Parse error: ${e.message}")
        }
    }
}
