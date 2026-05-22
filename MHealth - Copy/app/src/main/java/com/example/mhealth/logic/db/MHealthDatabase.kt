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
 * Version 12: added dnaReady column to user_profile_db.
 * Version 13: added daily_dna_snapshot table with nightChecks.
 * Version 14: schema hash realignment after nightChecks added (no structural change).
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
        NotificationEventEntity::class,
        DailyDnaSnapshotEntity::class
    ],
    version = 15,
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
    abstract fun dailyDnaSnapshotDao(): DailyDnaSnapshotDao

    companion object {
        @Volatile private var INSTANCE: MHealthDatabase? = null

        private val MIGRATION_11_12 = object : androidx.room.migration.Migration(11, 12) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE user_profile_db ADD COLUMN dnaReady INTEGER NOT NULL DEFAULT 0")
            }
        }

        private val MIGRATION_12_13 = object : androidx.room.migration.Migration(12, 13) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("""
                    CREATE TABLE IF NOT EXISTS daily_dna_snapshot (
                        userId TEXT NOT NULL,
                        date TEXT NOT NULL,
                        totalSessions INTEGER NOT NULL,
                        totalScreenTimeHours REAL NOT NULL,
                        firstPickupHour REAL,
                        lastActivityHour REAL,
                        activeWindowHours REAL,
                        avgSessionMinutes REAL NOT NULL,
                        microSessionPct REAL NOT NULL,
                        shortSessionPct REAL NOT NULL,
                        mediumSessionPct REAL NOT NULL,
                        deepSessionPct REAL NOT NULL,
                        marathonSessionPct REAL NOT NULL,
                        selfOpenPct REAL NOT NULL,
                        notificationOpenPct REAL NOT NULL,
                        totalNotifications INTEGER NOT NULL,
                        notificationTapRate REAL NOT NULL,
                        notificationDismissRate REAL NOT NULL,
                        notificationIgnoreRate REAL NOT NULL,
                        uniqueAppsUsed INTEGER NOT NULL,
                        topAppPackage TEXT,
                        nightChecks INTEGER NOT NULL,
                        appDnaJson TEXT NOT NULL,
                        createdAt INTEGER NOT NULL,
                        PRIMARY KEY(userId, date)
                    )
                """)
            }
        }

        // Version 14: recreate daily_dna_snapshot to fix schema mismatch.
        // The v13 table may have been created without nightChecks or with DEFAULT 0,
        // which doesn't match Room's entity definition (no @ColumnInfo default).
        // Drop-and-recreate is safe since this table holds computed aggregates.
        private val MIGRATION_13_14 = object : androidx.room.migration.Migration(13, 14) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("DROP TABLE IF EXISTS daily_dna_snapshot")
                db.execSQL("""
                    CREATE TABLE IF NOT EXISTS daily_dna_snapshot (
                        userId TEXT NOT NULL,
                        date TEXT NOT NULL,
                        totalSessions INTEGER NOT NULL,
                        totalScreenTimeHours REAL NOT NULL,
                        firstPickupHour REAL,
                        lastActivityHour REAL,
                        activeWindowHours REAL,
                        avgSessionMinutes REAL NOT NULL,
                        microSessionPct REAL NOT NULL,
                        shortSessionPct REAL NOT NULL,
                        mediumSessionPct REAL NOT NULL,
                        deepSessionPct REAL NOT NULL,
                        marathonSessionPct REAL NOT NULL,
                        selfOpenPct REAL NOT NULL,
                        notificationOpenPct REAL NOT NULL,
                        totalNotifications INTEGER NOT NULL,
                        notificationTapRate REAL NOT NULL,
                        notificationDismissRate REAL NOT NULL,
                        notificationIgnoreRate REAL NOT NULL,
                        uniqueAppsUsed INTEGER NOT NULL,
                        topAppPackage TEXT,
                        nightChecks INTEGER NOT NULL,
                        appDnaJson TEXT NOT NULL,
                        createdAt INTEGER NOT NULL,
                        PRIMARY KEY(userId, date)
                    )
                """)
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

        // Version 15: Replace backgroundAudioHours with musicTimeMinutes.
        // Must recreate daily_features table because:
        //  (a) SQLite on older Android lacks DROP COLUMN support
        //  (b) ALTER TABLE ADD COLUMN bakes DEFAULT values into the schema;
        //      Room expects 'undefined' for columns without @ColumnInfo(defaultValue),
        //      so any column originally added via migration has a permanent mismatch.
        // Table recreation fixes both issues in one atomic step.
        private val MIGRATION_14_15 = object : androidx.room.migration.Migration(14, 15) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                // 1. Create new table with exact schema Room expects
                db.execSQL("""
                    CREATE TABLE IF NOT EXISTS daily_features_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                        userId TEXT NOT NULL,
                        date TEXT NOT NULL,
                        screenTimeHours REAL NOT NULL,
                        unlockCount REAL NOT NULL,
                        appLaunchCount REAL NOT NULL,
                        notificationsToday REAL NOT NULL,
                        socialAppRatio REAL NOT NULL,
                        callsPerDay REAL NOT NULL,
                        callDurationMinutes REAL NOT NULL,
                        uniqueContacts REAL NOT NULL,
                        conversationFrequency REAL NOT NULL,
                        dailyDisplacementKm REAL NOT NULL,
                        locationEntropy REAL NOT NULL,
                        homeTimeRatio REAL NOT NULL,
                        placesVisited REAL NOT NULL,
                        wakeTimeHour REAL NOT NULL,
                        sleepTimeHour REAL NOT NULL,
                        sleepDurationHours REAL NOT NULL,
                        darkDurationHours REAL NOT NULL,
                        chargeDurationHours REAL NOT NULL,
                        memoryUsagePercent REAL NOT NULL,
                        networkWifiMB REAL NOT NULL,
                        networkMobileMB REAL NOT NULL,
                        downloadsToday REAL NOT NULL,
                        storageUsedGB REAL NOT NULL,
                        appUninstallsToday REAL NOT NULL,
                        upiTransactionsToday REAL NOT NULL,
                        totalAppsCount REAL NOT NULL,
                        musicTimeMinutes REAL NOT NULL DEFAULT 0.0,
                        mediaCountToday REAL NOT NULL,
                        appInstallsToday REAL NOT NULL,
                        calendarEventsToday REAL NOT NULL,
                        dailySteps REAL NOT NULL,
                        appBreakdownJson TEXT NOT NULL,
                        notificationBreakdownJson TEXT NOT NULL,
                        appLaunchesBreakdownJson TEXT NOT NULL,
                        bgAudioBreakdownJson TEXT NOT NULL,
                        syncedToCloud INTEGER NOT NULL,
                        isSimulated INTEGER NOT NULL
                    )
                """)

                // 2. Copy data, converting backgroundAudioHours → musicTimeMinutes
                db.execSQL("""
                    INSERT INTO daily_features_new (
                        id, userId, date, screenTimeHours, unlockCount, appLaunchCount,
                        notificationsToday, socialAppRatio, callsPerDay, callDurationMinutes,
                        uniqueContacts, conversationFrequency, dailyDisplacementKm, locationEntropy,
                        homeTimeRatio, placesVisited, wakeTimeHour, sleepTimeHour, sleepDurationHours,
                        darkDurationHours, chargeDurationHours, memoryUsagePercent, networkWifiMB,
                        networkMobileMB, downloadsToday, storageUsedGB, appUninstallsToday,
                        upiTransactionsToday, totalAppsCount, musicTimeMinutes, mediaCountToday,
                        appInstallsToday, calendarEventsToday, dailySteps, appBreakdownJson,
                        notificationBreakdownJson, appLaunchesBreakdownJson, bgAudioBreakdownJson,
                        syncedToCloud, isSimulated
                    )
                    SELECT
                        id, userId, date, screenTimeHours, unlockCount, appLaunchCount,
                        notificationsToday, socialAppRatio, callsPerDay, callDurationMinutes,
                        uniqueContacts, conversationFrequency, dailyDisplacementKm, locationEntropy,
                        homeTimeRatio, placesVisited, wakeTimeHour, sleepTimeHour, sleepDurationHours,
                        darkDurationHours, chargeDurationHours, memoryUsagePercent, networkWifiMB,
                        networkMobileMB, downloadsToday, storageUsedGB, appUninstallsToday,
                        upiTransactionsToday, totalAppsCount, backgroundAudioHours * 60.0, mediaCountToday,
                        appInstallsToday, calendarEventsToday, dailySteps, appBreakdownJson,
                        notificationBreakdownJson, appLaunchesBreakdownJson, bgAudioBreakdownJson,
                        syncedToCloud, isSimulated
                    FROM daily_features
                """)

                // 3. Drop old table and rename new one
                db.execSQL("DROP TABLE daily_features")
                db.execSQL("ALTER TABLE daily_features_new RENAME TO daily_features")

                // 4. Recreate unique index
                db.execSQL("CREATE UNIQUE INDEX IF NOT EXISTS index_daily_features_userId_date ON daily_features(userId, date)")

                // 5. Migrate baseline table
                db.execSQL("""
                    UPDATE baseline 
                    SET featureName = 'musicTimeMinutes', 
                        baselineValue = baselineValue * 60.0, 
                        stdDeviation = stdDeviation * 60.0 
                    WHERE featureName = 'backgroundAudioHours'
                """)
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
                    MIGRATION_7_8, MIGRATION_8_9, MIGRATION_9_10, MIGRATION_10_11, MIGRATION_11_12,
                    MIGRATION_12_13, MIGRATION_13_14, MIGRATION_14_15
                )
                    .setJournalMode(RoomDatabase.JournalMode.WRITE_AHEAD_LOGGING)
                    .fallbackToDestructiveMigration()
                    .build()
                    .also { INSTANCE = it }
            }
    }
}
