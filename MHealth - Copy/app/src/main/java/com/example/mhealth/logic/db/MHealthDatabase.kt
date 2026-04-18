package com.example.mhealth.logic.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

/**
 * MHealthDatabase — Room database singleton.
 * Version 9: added app_sessions and person_dna tables for Level 2 Behavioral DNA.
 * Version 10: added notification_events table for L2 Digital DNA.
 * Version 11: added L2 scoring fields to analysis_results table.
 *
 * Access via MHealthDatabase.getInstance(context)
 */
@Database(
    entities = [
        DailyFeaturesEntity::class,
        BaselineEntity::class,
        AnalysisResultEntity::class,
        UserProfileEntity::class,
        UserCredentialsEntity::class,
        AppSessionEntity::class,
        PersonDnaEntity::class,
        NotificationEventEntity::class
    ],
    version = 12,
    exportSchema = false
)
abstract class MHealthDatabase : RoomDatabase() {

    abstract fun dailyFeaturesDao(): DailyFeaturesDao
    abstract fun baselineDao(): BaselineDao
    abstract fun analysisResultDao(): AnalysisResultDao
    abstract fun userProfileDao(): UserProfileDao
    abstract fun userCredentialsDao(): UserCredentialsDao
    abstract fun appSessionDao(): AppSessionDao
    abstract fun personDnaDao(): PersonDnaDao
    abstract fun notificationEventDao(): NotificationEventDao

    companion object {
        @Volatile private var INSTANCE: MHealthDatabase? = null

        private val MIGRATION_11_12 = object : androidx.room.migration.Migration(11, 12) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE user_profile_db ADD COLUMN dnaReady INTEGER NOT NULL DEFAULT 0")
            }
        }
        
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

        private val MIGRATION_6_7 = object : androidx.room.migration.Migration(6, 7) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL(
                    """DELETE FROM daily_features
                       WHERE id NOT IN (
                           SELECT MAX(id) FROM daily_features GROUP BY userId, date
                       )"""
                )
                db.execSQL(
                    """CREATE UNIQUE INDEX IF NOT EXISTS index_daily_features_userId_date
                       ON daily_features(userId, date)"""
                )
            }
        }

        private val MIGRATION_7_8 = object : androidx.room.migration.Migration(7, 8) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE daily_features ADD COLUMN bgAudioBreakdownJson TEXT NOT NULL DEFAULT '{}'")
            }
        }

        private val MIGRATION_8_9 = object : androidx.room.migration.Migration(8, 9) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                // Level 2 Behavioral DNA: app_sessions table
                db.execSQL("""
                    CREATE TABLE IF NOT EXISTS app_sessions (
                        session_id TEXT NOT NULL,
                        app_package TEXT NOT NULL,
                        open_timestamp INTEGER NOT NULL,
                        close_timestamp INTEGER NOT NULL,
                        trigger TEXT NOT NULL,
                        interaction_count INTEGER NOT NULL,
                        date TEXT NOT NULL,
                        PRIMARY KEY(session_id)
                    )
                """)
                db.execSQL("CREATE INDEX IF NOT EXISTS index_app_sessions_app_package_open_timestamp ON app_sessions(app_package, open_timestamp)")
                db.execSQL("CREATE INDEX IF NOT EXISTS index_app_sessions_date ON app_sessions(date)")

                // Level 2 Behavioral DNA: person_dna table
                db.execSQL("""
                    CREATE TABLE IF NOT EXISTS person_dna (
                        id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                        person_id TEXT NOT NULL,
                        dna_json TEXT NOT NULL,
                        created_at INTEGER NOT NULL,
                        last_updated INTEGER NOT NULL
                    )
                """)
                db.execSQL("CREATE UNIQUE INDEX IF NOT EXISTS index_person_dna_person_id ON person_dna(person_id)")
            }
        }

        private val MIGRATION_9_10 = object : androidx.room.migration.Migration(9, 10) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                // Level 2 Digital DNA: notification_events table
                db.execSQL("""
                    CREATE TABLE IF NOT EXISTS notification_events (
                        event_id TEXT NOT NULL,
                        app_package TEXT NOT NULL,
                        arrival_timestamp INTEGER NOT NULL,
                        action TEXT NOT NULL,
                        tap_latency_min REAL,
                        date TEXT NOT NULL,
                        PRIMARY KEY(event_id)
                    )
                """)
                db.execSQL("CREATE INDEX IF NOT EXISTS index_notification_events_app_package_arrival_timestamp ON notification_events(app_package, arrival_timestamp)")
                db.execSQL("CREATE INDEX IF NOT EXISTS index_notification_events_date ON notification_events(date)")
            }
        }

        private val MIGRATION_10_11 = object : androidx.room.migration.Migration(10, 11) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                // L2 Digital DNA scoring fields on analysis_results
                db.execSQL("ALTER TABLE analysis_results ADD COLUMN l2Modifier REAL NOT NULL DEFAULT 1.0")
                db.execSQL("ALTER TABLE analysis_results ADD COLUMN coherence REAL NOT NULL DEFAULT 0.0")
                db.execSQL("ALTER TABLE analysis_results ADD COLUMN rhythmDissolution REAL NOT NULL DEFAULT 0.0")
                db.execSQL("ALTER TABLE analysis_results ADD COLUMN sessionIncoherence REAL NOT NULL DEFAULT 0.0")
                db.execSQL("ALTER TABLE analysis_results ADD COLUMN effectiveScore REAL NOT NULL DEFAULT 0.0")
                db.execSQL("ALTER TABLE analysis_results ADD COLUMN evidenceAccumulated REAL NOT NULL DEFAULT 0.0")
                db.execSQL("ALTER TABLE analysis_results ADD COLUMN patternType TEXT NOT NULL DEFAULT 'stable'")
                db.execSQL("ALTER TABLE analysis_results ADD COLUMN flaggedFeatures TEXT NOT NULL DEFAULT '[]'")
            }
        }

        fun getInstance(context: Context): MHealthDatabase =
            INSTANCE ?: synchronized(this) {
                INSTANCE ?: Room.databaseBuilder(
                    context.applicationContext,
                    MHealthDatabase::class.java,
                    "mhealth_database"
                )
                    .addMigrations(
                        MIGRATION_3_4, MIGRATION_4_5, MIGRATION_5_6, MIGRATION_6_7,
                        MIGRATION_7_8, MIGRATION_8_9, MIGRATION_9_10, MIGRATION_10_11, MIGRATION_11_12
                    )
                    .setJournalMode(RoomDatabase.JournalMode.WRITE_AHEAD_LOGGING)
                    .fallbackToDestructiveMigration()
                    .build()
                    .also { INSTANCE = it }
            }
    }
}