package com.example.mhealth.logic.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

/**
 * MHealthDatabase — Room database singleton.
 * Version 1: initial schema with 4 tables.
 *
 * Access via MHealthDatabase.getInstance(context)
 */
@Database(
    entities = [
        DailyFeaturesEntity::class,
        BaselineEntity::class,
        AnalysisResultEntity::class,
        UserProfileEntity::class,
        UserCredentialsEntity::class
    ],
    version = 3,
    exportSchema = false
)
abstract class MHealthDatabase : RoomDatabase() {

    abstract fun dailyFeaturesDao(): DailyFeaturesDao
    abstract fun baselineDao(): BaselineDao
    abstract fun analysisResultDao(): AnalysisResultDao
    abstract fun userProfileDao(): UserProfileDao
    abstract fun userCredentialsDao(): UserCredentialsDao

    companion object {
        @Volatile private var INSTANCE: MHealthDatabase? = null

        fun getInstance(context: Context): MHealthDatabase =
            INSTANCE ?: synchronized(this) {
                INSTANCE ?: Room.databaseBuilder(
                    context.applicationContext,
                    MHealthDatabase::class.java,
                    "mhealth_database"
                )
                    .fallbackToDestructiveMigration()
                    .build()
                    .also { INSTANCE = it }
            }
    }
}
