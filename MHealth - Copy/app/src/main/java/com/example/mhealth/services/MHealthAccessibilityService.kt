package com.example.mhealth.services

import android.accessibilityservice.AccessibilityService
import android.content.Context
import android.text.InputType
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
            lastTypeTimeMs = 0L
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

                // FIX: Properly check password inputType using Android's InputType constants.
                // Previous bitmask logic was broken due to operator precedence — it evaluated
                // `inputType and (0x80 != 0)` → `inputType and true` → always non-zero,
                // causing EVERY text field to be treated as a password and rejected.
                var sourceNode: AccessibilityNodeInfo? = null
                try {
                    sourceNode = event.source
                    if (sourceNode != null) {
                        val inputTypeMask = sourceNode.inputType
                        val variation = inputTypeMask and InputType.TYPE_MASK_VARIATION
                        // Skip password, visible-password, and web-password variations
                        if (variation == InputType.TYPE_TEXT_VARIATION_PASSWORD ||
                            variation == InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD ||
                            variation == InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD ||
                            variation == InputType.TYPE_NUMBER_VARIATION_PASSWORD) {
                            return
                        }
                    }
                } finally {
                    // FIX: Always recycle the AccessibilityNodeInfo to prevent resource leaks.
                    // Leaked nodes cause the OS to stop delivering events after ~500 leaks.
                    sourceNode?.recycle()
                }

                val added = event.addedCount
                val removed = event.removedCount
                val now = System.currentTimeMillis()

                val editor = prefs.edit()
                if (added > 0) {
                    val totalChars = prefs.getInt(KEY_CHARS_TYPED, 0) + added
                    editor.putInt(KEY_CHARS_TYPED, totalChars)

                    if (lastTypeTimeMs > 0) {
                        val diff = now - lastTypeTimeMs
                        if (diff in 50L..2500L) { // reasonable gap between typings
                            val totalTime = prefs.getLong(KEY_TYPING_DURATION_MS, 0L) + diff
                            editor.putLong(KEY_TYPING_DURATION_MS, totalTime)
                        }
                    }
                    lastTypeTimeMs = now
                }
                if (removed > 0) {
                    val totalBackspaces = prefs.getInt(KEY_BACKSPACES, 0) + removed
                    editor.putInt(KEY_BACKSPACES, totalBackspaces)

                    // FIX: Also accumulate typing duration for backspace events.
                    // Previously, rapid type-delete cycles lost timing data because
                    // only character additions accumulated duration.
                    if (lastTypeTimeMs > 0) {
                        val diff = now - lastTypeTimeMs
                        if (diff in 50L..2500L) {
                            val totalTime = prefs.getLong(KEY_TYPING_DURATION_MS, 0L) + diff
                            editor.putLong(KEY_TYPING_DURATION_MS, totalTime)
                        }
                    }
                    lastTypeTimeMs = now
                }
                editor.apply()
            }
            AccessibilityEvent.TYPE_VIEW_SCROLLED -> {
                // Scroll velocity dynamics
                var dy = kotlin.math.abs(event.toIndex - event.fromIndex).toFloat() * 50f // proxy distance in px
                if (dy == 0f) {
                    dy = 300f // robust fallback for minor scrolls or index-less containers (e.g. WebViews, single-item scrolls)
                }
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
