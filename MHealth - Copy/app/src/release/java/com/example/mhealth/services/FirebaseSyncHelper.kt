package com.example.mhealth.services

import android.content.Context
import com.example.mhealth.models.PersonalityVector

/**
 * Offline release stub for FirebaseSyncHelper.
 * Since release builds of Lumen are 100% offline local sandboxes,
 * all cloud sync operations are completely compiled out and disabled.
 */
object FirebaseSyncHelper {
    suspend fun syncBaseline(context: Context, baseline: PersonalityVector, today: Int) {
        // Offline release stub — no-op
    }

    suspend fun syncUnstagedDailyFeatures(context: Context, email: String) {
        // Offline release stub — no-op
    }
}
