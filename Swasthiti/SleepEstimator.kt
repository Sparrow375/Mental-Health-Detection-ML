package com.swasthiti.logic

import java.util.concurrent.TimeUnit

/**
 * Represents a single screen event from Android's UsageStatsManager.
 * @param timestamp The time the event occurred (in milliseconds)
 * @param isInteractive True if the screen turned ON / device was unlocked. False if the screen turned OFF.
 */
data class ScreenEvent(
    val timestamp: Long,
    val isInteractive: Boolean
)

/**
 * Represents a calculated sleep session.
 * @param startTime Millisecond timestamp of sleep start
 * @param endTime Millisecond timestamp of sleep end
 * @param durationMs Total duration in milliseconds
 */
data class SleepSession(
    val startTime: Long,
    val endTime: Long,
    val durationMs: Long
) {
    val durationHours: Double
        get() = durationMs / (1000.0 * 60 * 60)
}

class SleepEstimator {
    companion object {
        /**
         * Estimates sleep duration given a block of screen events (usually a 24-hour block like 6 PM to 6 PM).
         * 
         * @param events List of screen interactive events.
         * @param minorInterruptionLimitMs The maximum duration (in ms) of phone usage that will NOT break a sleep block. Default is 2 minutes.
         * @return The longest merged SleepSession, or null if no sleep blocks were found.
         */
        fun estimateSleep(
            events: List<ScreenEvent>,
            minorInterruptionLimitMs: Long = TimeUnit.MINUTES.toMillis(2)
        ): SleepSession? {
            if (events.isEmpty()) return null
            
            // 1. Sort events chronologically
            val sortedEvents = events.sortedBy { it.timestamp }
            
            // Data structure to hold inactivity gaps
            data class Gap(val start: Long, val end: Long) {
                val duration: Long get() = end - start
            }
            
            val gaps = mutableListOf<Gap>()
            var lastInactivityStart: Long? = null
            
            // 2. Identify all inactivity gaps
            for (event in sortedEvents) {
                if (!event.isInteractive) {
                    // Screen turned OFF. Start of a gap (if one hasn't already started)
                    if (lastInactivityStart == null) {
                        lastInactivityStart = event.timestamp
                    }
                } else {
                    // Screen turned ON / Unlocked. End of the current gap
                    if (lastInactivityStart != null) {
                        gaps.add(Gap(start = lastInactivityStart, end = event.timestamp))
                        lastInactivityStart = null
                    }
                }
            }
            
            // If the user goes to sleep and hasn't unlocked their phone yet at the time the query runs,
            // we close the gap at the last known timestamp in our event window.
            val lastEvent = sortedEvents.last()
            if (lastInactivityStart != null && lastEvent.timestamp > lastInactivityStart) {
                gaps.add(Gap(start = lastInactivityStart, end = lastEvent.timestamp))
            }
            
            if (gaps.isEmpty()) return null
            
            // 3. Merge gaps separated by short interruptions (Phone usage < 2 minutes)
            val mergedGaps = mutableListOf<Gap>()
            var currentMergedBlock = gaps[0]
            
            for (i in 1 until gaps.size) {
                val nextGap = gaps[i]
                val interactiveDurationBetweenGaps = nextGap.start - currentMergedBlock.end
                
                if (interactiveDurationBetweenGaps < minorInterruptionLimitMs) {
                    // Merge! The interactive period was very short (e.g. checking time).
                    // The new block stretches from the start of the first gap to the end of the second gap.
                    currentMergedBlock = Gap(start = currentMergedBlock.start, end = nextGap.end)
                } else {
                    // Do not merge. The interactive period was long (person fully woke up).
                    // Save the completed block and start tracking a new one.
                    mergedGaps.add(currentMergedBlock)
                    currentMergedBlock = nextGap
                }
            }
            // Add the final remaining block
            mergedGaps.add(currentMergedBlock)
            
            // 4. Find the max continuous/merged segment, which represents the primary night sleep
            val longestSleepBlock = mergedGaps.maxByOrNull { it.duration }
            
            return longestSleepBlock?.let {
                SleepSession(
                    startTime = it.start,
                    endTime = it.end,
                    durationMs = it.duration
                )
            }
        }
    }
}

