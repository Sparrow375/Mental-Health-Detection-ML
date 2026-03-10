package com.example.mhealth.logic

import android.app.usage.UsageEvents
import android.app.usage.UsageStatsManager
import android.content.Context
import android.provider.CallLog
import android.provider.ContactsContract
import android.provider.Telephony
import android.util.Log
import com.example.mhealth.models.PersonalityVector
import java.util.*

class DataCollector(private val context: Context) {

    private val TAG = "MHealthDataCollector"

    fun collectDailyData(): PersonalityVector {
        val calendar = Calendar.getInstance()
        val endTime = calendar.timeInMillis
        
        calendar.set(Calendar.HOUR_OF_DAY, 0)
        calendar.set(Calendar.MINUTE, 0)
        calendar.set(Calendar.SECOND, 0)
        calendar.set(Calendar.MILLISECOND, 0)
        val startTime = calendar.timeInMillis

        Log.i(TAG, "--- START COLLECTION: ${Date(startTime)} to ${Date(endTime)} ---")

        val usageStats = collectAppUsageStats(startTime, endTime)
        val communicationStats = collectCommunicationStats(startTime, endTime)
        val totalContacts = countTotalContacts()
        
        Log.i(TAG, "--- COLLECTION SUMMARY ---")
        Log.i(TAG, "Screen Time: ${usageStats.totalMinutes}m")
        Log.i(TAG, "Unlocks: ${usageStats.unlockCount}")
        Log.i(TAG, "Contacts: $totalContacts")

        return PersonalityVector(
            screenTimeHours = usageStats.totalMinutes / 60f,
            unlockCount = usageStats.unlockCount.toFloat(),
            socialAppRatio = usageStats.socialRatio,
            callsPerDay = communicationStats.callCount.toFloat(),
            textsPerDay = communicationStats.smsCount.toFloat(),
            uniqueContacts = totalContacts.toFloat(), 
            
            dailyDisplacementKm = 4.2f, 
            locationEntropy = 1.6f,
            homeTimeRatio = 0.7f,
            placesVisited = 4.0f,
            
            wakeTimeHour = usageStats.firstEventHour,
            sleepTimeHour = usageStats.lastEventHour,
            sleepDurationHours = 8.0f, 
            darkDurationHours = 9.0f,
            chargeDurationHours = 6.0f,
            conversationFrequency = communicationStats.callCount.toFloat(),
            appBreakdown = usageStats.breakdown
        )
    }

    private data class UsageResult(
        val totalMinutes: Long, 
        val unlockCount: Int, 
        val socialRatio: Float,
        val firstEventHour: Float,
        val lastEventHour: Float,
        val breakdown: Map<String, Long>
    )

    private fun collectAppUsageStats(startTime: Long, endTime: Long): UsageResult {
        val usm = context.getSystemService(Context.USAGE_STATS_SERVICE) as UsageStatsManager
        
        var unlocks = 0
        var firstEventMs = Long.MAX_VALUE
        var lastEventMs = Long.MIN_VALUE
        
        val events = usm.queryEvents(startTime, endTime)
        val event = UsageEvents.Event()
        
        Log.d(TAG, "--- Analyzing Usage Events ---")
        while (events.hasNextEvent()) {
            events.getNextEvent(event)
            val ts = event.timeStamp
            if (ts < firstEventMs) firstEventMs = ts
            if (ts > lastEventMs) lastEventMs = ts

            // Type 15 (SCREEN_INTERACTIVE) - Proxy for unlock/screen on
            if (event.eventType == 15) {
                unlocks++
                Log.d(TAG, "Unlock/Interactive detected at ${Date(ts)}")
            }
        }

        val stats = usm.queryUsageStats(UsageStatsManager.INTERVAL_DAILY, startTime, endTime)
        var totalTimeMs = 0L
        var socialTimeMs = 0L
        val socialPkgs = listOf("facebook", "whatsapp", "instagram", "twitter", "tiktok", "snapchat", "messenger", "social", "youtube")
        val breakdown = mutableMapOf<String, Long>()

        Log.d(TAG, "--- App Usage Breakdown ---")
        stats.filter { it.totalTimeInForeground > 0 }.sortedByDescending { it.totalTimeInForeground }.forEach {
            val pkg = it.packageName.lowercase()
            
            val isExcluded = pkg.contains("launcher") || 
                             pkg.contains("systemui") || 
                             pkg.contains("google.android.gms") || 
                             pkg.contains("inputmethod") ||
                             pkg.contains("wallpaper") ||
                             pkg == "android" ||
                             pkg == context.packageName

            if (!isExcluded) {
                totalTimeMs += it.totalTimeInForeground
                val minutes = it.totalTimeInForeground / 60000
                breakdown[it.packageName] = minutes
                
                if (socialPkgs.any { sp -> pkg.contains(sp) }) {
                    socialTimeMs += it.totalTimeInForeground
                }
                Log.d(TAG, "  $pkg: ${minutes}m")
            }
        }

        val cal = Calendar.getInstance()
        if (firstEventMs == Long.MAX_VALUE) firstEventMs = startTime
        cal.timeInMillis = firstEventMs
        val firstHour = cal.get(Calendar.HOUR_OF_DAY) + cal.get(Calendar.MINUTE) / 60f
        
        if (lastEventMs == Long.MIN_VALUE) lastEventMs = endTime
        cal.timeInMillis = lastEventMs
        val lastHour = cal.get(Calendar.HOUR_OF_DAY) + cal.get(Calendar.MINUTE) / 60f

        val ratio = if (totalTimeMs > 0) socialTimeMs.toFloat() / totalTimeMs else 0f
        
        return UsageResult(totalTimeMs / 60000, unlocks, ratio, firstHour, lastHour, breakdown)
    }

    private fun collectCommunicationStats(startTime: Long, endTime: Long): CommStats {
        var calls = 0
        var sms = 0
        val interactedContacts = mutableSetOf<String>()
        try {
            context.contentResolver.query(CallLog.Calls.CONTENT_URI, arrayOf(CallLog.Calls.NUMBER), "${CallLog.Calls.DATE} >= ?", arrayOf(startTime.toString()), null)?.use {
                calls = it.count
                val idx = it.getColumnIndex(CallLog.Calls.NUMBER)
                while (it.moveToNext()) {
                    it.getString(idx)?.let { n -> interactedContacts.add(n.filter { it.isDigit() }.takeLast(10)) }
                }
            }
            context.contentResolver.query(Telephony.Sms.CONTENT_URI, arrayOf(Telephony.Sms.ADDRESS), "${Telephony.Sms.DATE} >= ?", arrayOf(startTime.toString()), null)?.use {
                sms = it.count
                val idx = it.getColumnIndex(Telephony.Sms.ADDRESS)
                while (it.moveToNext()) {
                    it.getString(idx)?.let { a -> interactedContacts.add(a.filter { it.isDigit() }.takeLast(10)) }
                }
            }
        } catch (e: Exception) { Log.e(TAG, "Comm error", e) }
        return CommStats(calls, sms, interactedContacts.size)
    }

    private fun countTotalContacts(): Int {
        return try {
            val cursor = context.contentResolver.query(
                ContactsContract.Contacts.CONTENT_URI,
                arrayOf(ContactsContract.Contacts._ID),
                "${ContactsContract.Contacts.HAS_PHONE_NUMBER} = 1",
                null, null
            )
            val count = cursor?.count ?: 0
            cursor?.close()
            count
        } catch (e: Exception) { 
            Log.e(TAG, "Contacts query failed", e)
            0 
        }
    }

    private data class CommStats(val callCount: Int, val smsCount: Int, val uniqueContactsToday: Int)
}
