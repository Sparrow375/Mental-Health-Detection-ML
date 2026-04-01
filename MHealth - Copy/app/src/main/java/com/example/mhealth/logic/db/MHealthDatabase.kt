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
    version = 6,
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

        private val MIGRATION_3_4 = object : androidx.room.migration.Migration(3, 4) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE daily_features ADD COLUMN isSimulated INTEGER NOT NULL DEFAULT 0")
            }
        }

        private val MIGRATION_4_5 = object : androidx.room.migration.Migration(4, 5) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE daily_features ADD COLUMN backgroundAudioHours REAL NOT NULL DEFAULT 0.0")
                db.execSQL("ALTER TABLE daily_features ADD COLUMN mediaCountToday REAL NOT NULL DEFAULT 0.0")
                db.execSQL("ALTER TABLE daily_features ADD COLUMN appInstallsToday REAL NOT NULL DEFAULT 0.0")
            }
        }

        private val MIGRATION_5_6 = object : androidx.room.migration.Migration(5, 6) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE daily_features ADD COLUMN calendarEventsToday REAL NOT NULL DEFAULT 0.0")
            }
        }

        fun getInstance(context: Context): MHealthDatabase =
            INSTANCE ?: synchronized(this) {
                INSTANCE ?: Room.databaseBuilder(
                    context.applicationContext,
                    MHealthDatabase::class.java,
                    "mhealth_database"
                )
                    .addMigrations(MIGRATION_3_4, MIGRATION_4_5, MIGRATION_5_6)
                    .fallbackToDestructiveMigration()
                    .build()
                    .also { INSTANCE = it }
            }
    }
}
