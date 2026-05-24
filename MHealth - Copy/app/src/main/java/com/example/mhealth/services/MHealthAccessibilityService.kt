package com.example.mhealth.services

import android.accessibilityservice.AccessibilityService
import android.content.Context
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo

class MHealthAccessibilityService : AccessibilityService() {

    companion object {
        private const val PREFS_NAME = "accessibility_stats"
        private const val KEY_CHARS_TYPED = "chars_typed"
        private const val KEY_BACKSPACES = "backspaces"
        private const val KEY_TYPING_DURATION_MS = "typing_duration_ms"
        private const val KEY_SCROLL_DISTANCE_PX = "scroll_distance_px"
        private const val KEY_SCROLL_DURATION_MS = "scroll_duration_ms"

        private var lastTypeTimeMs: Long = 0L

        fun getDailyMetrics(context: Context): Triple<Float, Float, Float> {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            val chars = prefs.getInt(KEY_CHARS_TYPED, 0)
            val backspaces = prefs.getInt(KEY_BACKSPACES, 0)
            val typingTimeMs = prefs.getLong(KEY_TYPING_DURATION_MS, 0L)
            val scrollDist = prefs.getFloat(KEY_SCROLL_DISTANCE_PX, 0f)
            val scrollTimeMs = prefs.getLong(KEY_SCROLL_DURATION_MS, 0L)

            // Calculate Keystroke Speed (char/sec)
            val keystrokeSpeed = if (typingTimeMs > 0) {
                (chars.toFloat() / (typingTimeMs / 1000f)).coerceIn(0f, 20f)
            } else 0f

            // Calculate Backspace Ratio
            val totalKeystrokes = chars + backspaces
            val backspaceRatio = if (totalKeystrokes > 0) {
                (backspaces.toFloat() / totalKeystrokes).coerceIn(0f, 1f)
            } else 0f

            // Calculate Scroll Velocity (pixels/sec)
            val scrollVelocity = if (scrollTimeMs > 0) {
                (scrollDist / (scrollTimeMs / 1000f)).coerceIn(0f, 5000f)
            } else 0f

            return Triple(keystrokeSpeed, backspaceRatio, scrollVelocity)
        }

        fun resetDailyMetrics(context: Context) {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            prefs.edit().clear().apply()
        }
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        if (event == null) return

        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

        when (event.eventType) {
            AccessibilityEvent.TYPE_VIEW_TEXT_CHANGED -> {
                // Keystroke dynamics
                // Security guardrail: skip passwords and sensitive inputs
                if (event.isPassword) return
                val sourceNode = event.source
                if (sourceNode != null) {
                    val inputType = sourceNode.inputType
                    // PASSWORD inputs or similar sensitive fields
                    if (inputType and 0x80 != 0 || inputType and 0x90 != 0) {
                        return
                    }
                }

                val beforeText = event.beforeText ?: ""
                val text = event.text?.firstOrNull() ?: ""
                val textLen = text.length
                val beforeLen = beforeText.length

                val now = System.currentTimeMillis()
                val isCharAdded = textLen > beforeLen
                val isCharDeleted = beforeLen > textLen

                val editor = prefs.edit()
                if (isCharAdded) {
                    val addedCount = textLen - beforeLen
                    val totalChars = prefs.getInt(KEY_CHARS_TYPED, 0) + addedCount
                    editor.putInt(KEY_CHARS_TYPED, totalChars)

                    if (lastTypeTimeMs > 0) {
                        val diff = now - lastTypeTimeMs
                        if (diff in 50L..2500L) { // reasonable gap between typings
                            val totalTime = prefs.getLong(KEY_TYPING_DURATION_MS, 0L) + diff
                            editor.putLong(KEY_TYPING_DURATION_MS, totalTime)
                        }
                    }
                    lastTypeTimeMs = now
                } else if (isCharDeleted) {
                    val deletedCount = beforeLen - textLen
                    val totalBackspaces = prefs.getInt(KEY_BACKSPACES, 0) + deletedCount
                    editor.putInt(KEY_BACKSPACES, totalBackspaces)
                    lastTypeTimeMs = now
                }
                editor.apply()
            }
            AccessibilityEvent.TYPE_VIEW_SCROLLED -> {
                // Scroll velocity dynamics
                val dy = kotlin.math.abs(event.toIndex - event.fromIndex).toFloat() * 50f // proxy distance in px
                val editor = prefs.edit()
                val currentDist = prefs.getFloat(KEY_SCROLL_DISTANCE_PX, 0f) + dy
                editor.putFloat(KEY_SCROLL_DISTANCE_PX, currentDist)

                // Scroll time proxy: each scroll event takes roughly 200ms
                val currentTimeMs = prefs.getLong(KEY_SCROLL_DURATION_MS, 0L) + 200L
                editor.putLong(KEY_SCROLL_DURATION_MS, currentTimeMs)
                editor.apply()
            }
        }
    }

    override fun onInterrupt() {}
}
