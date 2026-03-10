package com.anonymous.MentalHealthApp

import android.app.usage.UsageEvents
import android.app.usage.UsageStatsManager
import android.content.Context
import android.content.Intent
import android.provider.Settings
import com.facebook.react.bridge.*
import java.util.*

class UsageStatsModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {

    override fun getName(): String {
        return "UsageStatsModule"
    }

    @ReactMethod
    fun checkPermission(promise: Promise) {
        val usageStatsManager = reactApplicationContext.getSystemService(Context.USAGE_STATS_SERVICE) as UsageStatsManager
        val currentTime = System.currentTimeMillis()
        val stats = usageStatsManager.queryUsageStats(UsageStatsManager.INTERVAL_DAILY, currentTime - 1000 * 60, currentTime)
        promise.resolve(stats != null && stats.isNotEmpty())
    }

    @ReactMethod
    fun showUsageAccessSettings() {
        val intent = Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS)
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        reactApplicationContext.startActivity(intent)
    }

    @ReactMethod
    fun getDailyStats(startTime: Double, endTime: Double, promise: Promise) {
        val usageStatsManager = reactApplicationContext.getSystemService(Context.USAGE_STATS_SERVICE) as UsageStatsManager
        val stats = usageStatsManager.queryAndAggregateUsageStats(startTime.toLong(), endTime.toLong())

        val result = Arguments.createArray()
        for ((packageName, usageStats) in stats) {
            val appMap = Arguments.createMap()
            appMap.putString("packageName", packageName)
            appMap.putDouble("totalTimeInForeground", usageStats.totalTimeInForeground.toDouble())
            appMap.putDouble("lastTimeUsed", usageStats.lastTimeUsed.toDouble())
            result.pushMap(appMap)
        }
        promise.resolve(result)
    }

    @ReactMethod
    fun getScreenEvents(startTime: Double, endTime: Double, promise: Promise) {
        val usageStatsManager = reactApplicationContext.getSystemService(Context.USAGE_STATS_SERVICE) as UsageStatsManager
        val events = usageStatsManager.queryEvents(startTime.toLong(), endTime.toLong())
        
        val result = Arguments.createArray()
        val event = UsageEvents.Event()
        
        while (events.hasNextEvent()) {
            events.getNextEvent(event)
            if (event.eventType == UsageEvents.Event.SCREEN_INTERACTIVE || 
                event.eventType == UsageEvents.Event.SCREEN_NON_INTERACTIVE ||
                event.eventType == UsageEvents.Event.USER_INTERACTION ||
                event.eventType == 26 || // KEYGUARD_INTERACTIVE
                event.eventType == 27 || // KEYGUARD_GONE
                event.eventType == 12 // UsageEvents.Event.NOTIFICATION_INTERRUPTION
            ) {
                
                val eventMap = Arguments.createMap()
                eventMap.putInt("type", event.eventType)
                eventMap.putDouble("timestamp", event.timeStamp.toDouble())
                eventMap.putString("packageName", event.packageName)
                result.pushMap(eventMap)
            }
        }
        promise.resolve(result)
    }
}
