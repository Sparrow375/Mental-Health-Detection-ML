package com.example.mhealth.services

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters

class CloudSyncWorker(appContext: Context, workerParams: WorkerParameters) :
    CoroutineWorker(appContext, workerParams) {

    companion object {
        fun schedulePeriodic(context: Context) {
            // Offline stub — periodic sync is completely disabled in release builds
        }
    }

    override suspend fun doWork(): Result {
        // Offline stub — immediate success with zero network operations
        return Result.success()
    }
}
