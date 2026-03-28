# MHealth — Complete Data Extraction Reference

> End-to-end specification of every data point collected, how it is sourced,
> computed, stored, and used in the ML pipeline.

---

## Architecture Overview

```
[ Android Hardware / OS APIs ]
          │  every 15 min
          ▼
[ DataCollector.kt ]  ← collectSnapshot()
          │
          ▼
[ PersonalityVector ]  ← in-memory model snapshot
          │
    ┌─────┴───────────────┐
    │                     │
    ▼                     ▼
[ DataRepository ]   [ JsonConverter ]
 (StateFlow / UI)         │
                          ▼
                  [ DailyFeaturesEntity ]
                   (Room SQLite — daily row)
                          │
               ┌──────────┴──────────┐
               │                     │
               ▼                     ▼
        [ BaselineEntity ]    [ NightlyAnalysisWorker ]
         (28-day μ ± σ)              │
                                     ▼
                              [ PythonEngine.kt ]
                               (anomaly detection)
                                     │
                                     ▼
                           [ AnalysisResultEntity ]
                            (alert level, score, pattern)
                                     │
                                     ▼
                            [ Firebase Firestore ]
                              (cloud backup)
```

---

## Collection Schedule

| Trigger | Interval | Action |
|---|---|---|
| `MonitoringService.scheduleMonitoring()` | Every 15 min | `collectSnapshot()` → update live UI |
| `MonitoringService` midnight rollover | Daily at 00:05 | Save `DailyFeaturesEntity` to Room |
| `NightlyAnalysisWorker` (WorkManager) | Daily at 00:05 | Run Python engine, save results |
| `DataCollector.captureLocationSnapshot()` | Every 15 min | Append GPS fix to daily track |

---

## Feature-by-Feature Extraction

---

### GROUP 1 — Screen & App Usage (Digital Wellbeing Equivalent)

---

#### 1. `screenTimeHours`
| Field | Value |
|---|---|
| **What** | Total time all non-system apps were in the foreground today |
| **Android API** | `UsageStatsManager.queryEvents(startOfDay, now)` |
| **Events used** | `MOVE_TO_FOREGROUND (1)` / `MOVE_TO_BACKGROUND (2)` pairs |
| **Formula** | `Σ (backgroundTime - foregroundTime)` for each app session |
| **Method** | `parseUsageEvents()` in `DataCollector.kt` |
| **Unit** | Hours (Float) |
| **Accuracy note** | Same pair-replay algorithm used internally by Android's Digital Wellbeing — identical to Settings → Digital Wellbeing → Screen Time |
| **DB column** | `daily_features.screenTimeHours` |
| **Used in model** | ✅ Baseline + Anomaly detection |

---

#### 2. `unlockCount`
| Field | Value |
|---|---|
| **What** | Number of times the device was unlocked today |
| **Android API** | `UsageStatsManager.queryEvents()` |
| **Events used** | `KEYGUARD_HIDDEN (18)` — fired on each successful unlock |
| **Formula** | Count of event type 18 since `startOfDay` |
| **Unit** | Integer count (Float) |
| **DB column** | `daily_features.unlockCount` |
| **Used in model** | ✅ |

---

#### 3. `appLaunchCount`
| Field | Value |
|---|---|
| **What** | Total distinct app launches today (scientific debounce applied) |
| **Android API** | `UsageStatsManager.queryEvents()` |
| **Events used** | `MOVE_TO_FOREGROUND (1)` |
| **Debounce** | Only counted as new launch if app was in background > 1,500 ms (eliminates system flickers) |
| **Formula** | `Σ debounced MOVE_TO_FOREGROUND events` per non-excluded package |
| **Method** | `parseUsageEvents()` → `appLaunches` map |
| **Unit** | Integer count (Float) |
| **DB column** | `daily_features.appLaunchCount` |
| **Used in model** | ✅ |

---

#### 4. `notificationsToday`
| Field | Value |
|---|---|
| **What** | Total notification interruptions today (all apps combined) |
| **Android API** | `UsageStatsManager.queryEvents()` |
| **Events used** | `NOTIFICATION_INTERRUPTION (12)` |
| **Formula** | Count of type-12 events from non-system packages |
| **Unit** | Integer count (Float) |
| **DB column** | `daily_features.notificationsToday` |
| **Used in model** | ✅ |

---

#### 5. `socialAppRatio`
| Field | Value |
|---|---|
| **What** | Fraction of total screen time spent in social media apps (0–1) |
| **Android API** | `UsageStatsManager` + `PackageManager.getApplicationInfo()` |
| **Method** | Filter `appMs` map by `ApplicationInfo.CATEGORY_SOCIAL` (API 26+) + name-based fallback (WhatsApp, Instagram, etc.) |
| **Formula** | `socialMs / totalScreenMs` |
| **Display** | Multiplied by 100 for % display in UI/table |
| **Unit** | Ratio 0–1 stored; displayed as % |
| **DB column** | `daily_features.socialAppRatio` |
| **Used in model** | ✅ Critical feature for depression detection |

---

#### 6. `appBreakdown` *(per-app, not in scalar model)*
| Field | Value |
|---|---|
| **What** | Map of `packageName → foreground minutes` for every non-system app |
| **Source** | `parseUsageEvents()` → `appMs` map |
| **Storage** | NOT stored in `DailyFeaturesEntity` (too large for a column) — live in `PersonalityVector` |
| **Used for** | Per-App Breakdown card in Monitor & HomeScreen bar charts |

---

#### 7. `appLaunchesBreakdown` *(per-app)*
| Field | Value |
|---|---|
| **What** | Map of `packageName → launch count today` |
| **Source** | `parseUsageEvents()` → `appLaunches` map |
| **Storage** | Live in `PersonalityVector` only |
| **Used for** | Per-App Breakdown table in Monitor screen |

---

#### 8. `notificationBreakdown` *(per-app)*
| Field | Value |
|---|---|
| **What** | Map of `packageName → notification count today` |
| **Source** | `parseUsageEvents()` — event type 12, keyed by package |
| **Storage** | Live in `PersonalityVector` only |
| **Used for** | Per-App Breakdown table; flagged orange if > 30 notifs |

---

### GROUP 2 — Communication

---

#### 9. `callsPerDay`
| Field | Value |
|---|---|
| **What** | Total calls made or received today |
| **Android API** | `ContentResolver.query(CallLog.Calls.CONTENT_URI)` |
| **Filter** | `CallLog.Calls.DATE >= startOfDay` |
| **Formula** | `cursor.count` |
| **Permission** | `READ_CALL_LOG` |
| **Unit** | Integer count (Float) |
| **DB column** | `daily_features.callsPerDay` |
| **Used in model** | ✅ |

---

#### 10. `callDurationMinutes`
| Field | Value |
|---|---|
| **What** | Total speak time across all calls today (minutes) |
| **Android API** | `CallLog.Calls.DURATION` column |
| **Formula** | `Σ duration_seconds / 60` |
| **Unit** | Minutes (Float) |
| **DB column** | `daily_features.callDurationMinutes` |
| **Used in model** | ✅ |

---

#### 11. `uniqueContacts`
| Field | Value |
|---|---|
| **What** | Count of unique phone numbers in today's call log (callers + callees) |
| **Android API** | `ContentResolver.query(CallLog.Calls.CONTENT_URI)` filtered by today |
| **Formula** | `Set<String>.size` of normalised phone numbers from call log |
| **Previous bug** | Was previously querying starred contacts — completely wrong; fixed |
| **Unit** | Integer count (Float) |
| **DB column** | `daily_features.uniqueContacts` |
| **Used in model** | ✅ Social withdrawal indicator |

---

#### 12. `conversationFrequency`
| Field | Value |
|---|---|
| **What** | Average number of calls per unique contact today (interaction density) |
| **Formula** | `callsPerDay / uniqueContacts` (or `callsPerDay` if 0 contacts) |
| **Previous bug** | Was identical to `callsPerDay` — wasted model dimension; fixed |
| **Unit** | Float ratio |
| **DB column** | `daily_features.conversationFrequency` |
| **Used in model** | ✅ |

---

### GROUP 3 — Location & Movement

---

#### 13. `dailyDisplacementKm`
| Field | Value |
|---|---|
| **What** | Total geographic distance travelled today (polyline length) |
| **Android API** | `FusedLocationProviderClient.getCurrentLocation()` every 15 min |
| **Formula** | Haversine distance between consecutive GPS fixes; `Σ all segments` |
| **Storage** | GPS points stored in `DataRepository._locationSnapshots` (max 96 = 24h @ 15min), persisted to SharedPrefs |
| **Unit** | Kilometres (Float) |
| **DB column** | `daily_features.dailyDisplacementKm` |
| **Used in model** | ✅ Agoraphobia & depression indicator |

---

#### 14. `locationEntropy`
| Field | Value |
|---|---|
| **What** | Shannon entropy of location diversity (how many distinct cells visited) |
| **Formula** | Cell grid: 0.001° ≈ 110m. `H = −Σ (p × ln p)` where p = fraction of fixes in each cell |
| **Range** | 0 (stayed in one place all day) → higher = more diverse movement |
| **Unit** | Float (nats) |
| **DB column** | `daily_features.locationEntropy` |
| **Used in model** | ✅ Complements displacement for detecting social isolation |

---

#### 15. `homeTimeRatio`
| Field | Value |
|---|---|
| **What** | Fraction of the day spent at home (most-visited cell = home proxy) |
| **Formula** | `fixes_in_most_visited_cell / total_fixes` |
| **Display** | Multiplied by 100 for % display in UI/table |
| **Unit** | Ratio 0–1 stored; displayed as % |
| **DB column** | `daily_features.homeTimeRatio` |
| **Used in model** | ✅ High home time = social withdrawal indicator |

---

#### 16. `placesVisited`
| Field | Value |
|---|---|
| **What** | Number of distinct geographic cells (≈110m grid) visited today |
| **Formula** | `cells.size` from the same grid used by `locationEntropy` |
| **Unit** | Integer count (Float) |
| **DB column** | `daily_features.placesVisited` |
| **Used in model** | ✅ |

---

### GROUP 4 — Sleep Proxy

---

#### 17. `sleepDurationHours`
#### 18. `sleepTimeHour`
#### 19. `wakeTimeHour`
| Field | Value |
|---|---|
| **What** | Estimated sleep onset, wake time, and sleep duration; no wearable needed |
| **Android API** | `UsageStatsManager.queryEvents()` — screen on/off + app events |
| **Method** | `calculateSleepProxy()` in `DataCollector.kt` |
| **Algorithm** | Scans event stream in the night window (18:00 yesterday → 14:00 today). Tracks longest continuous gap between any 2 interaction events. The longest gap = sleep episode. |
| **Formula** | `longestGapMs / 3_600_000f` = sleep duration |
| **Resolution** | 15-min accuracy (tied to event granularity) |
| **Unit** | Hours float (duration); 24h clock float (times) |
| **DB columns** | `sleepDurationHours`, `sleepTimeHour`, `wakeTimeHour` |
| **Used in model** | ✅ Sleep disruption is the strongest leading indicator in the model |

---

#### 20. `darkDurationHours`
| Field | Value |
|---|---|
| **What** | Total cumulative time the screen was off today |
| **Android API** | `SCREEN_NON_INTERACTIVE (16)` → `SCREEN_INTERACTIVE (15)` pairs |
| **Formula** | `Σ (screenOnAt - screenOffAt)` for all off-periods today |
| **Unit** | Hours (Float) |
| **DB column** | `daily_features.darkDurationHours` |
| **Used in model** | ✅ High dark time = more passive/inactive periods |

---

### GROUP 5 — System Metrics

---

#### 21. `chargeDurationHours`
| Field | Value |
|---|---|
| **What** | Total hours the phone was plugged in and charging today |
| **Android API** | `Intent.ACTION_BATTERY_CHANGED` (battery broadcast) polled every 15 min |
| **Formula** | `accumulatedChargeHours` incremented by `15/60f` each tick if `isCharging == true` |
| **Storage** | Accumulated in `DataRepository._accumulatedChargeHours`, persisted to SharedPrefs |
| **DB column** | `daily_features.chargeDurationHours` |
| **Used in model** | ✅ Spending many hours charging = sedentary, often correlated with mood episodes |

---

#### 22. `memoryUsagePercent`
| Field | Value |
|---|---|
| **What** | Percentage of total RAM currently in use |
| **Android API** | `ActivityManager.getMemoryInfo()` |
| **Formula** | `(totalMem - availMem) * 100 / totalMem` |
| **Unit** | % (Float) |
| **DB column** | `daily_features.memoryUsagePercent` |
| **Used in model** | ✅ Proxy for background app activity level |

---

#### 23. `networkWifiMB`
#### 24. `networkMobileMB`
| Field | Value |
|---|---|
| **What** | Wi-Fi and mobile data consumed today (MB) |
| **Android API** | `NetworkStatsManager.querySummaryForDevice()` |
| **Formula** | `(rxBytes + txBytes) / (1024 × 1024)` |
| **Granularity** | Since `startOfDay` UTC |
| **Unit** | Megabytes (Float) |
| **DB columns** | `networkWifiMB`, `networkMobileMB` |
| **Used in model** | ✅ High mobile data = out-of-home activity or streaming binges |

---

#### 25. `storageUsedGB`
| Field | Value |
|---|---|
| **What** | Total internal storage currently used |
| **Android API** | `StatFs(Environment.getDataDirectory())` |
| **Formula** | `(totalBlocks - availBlocks) × blockSize / 1_073_741_824f` |
| **Unit** | Gigabytes (Float) |
| **DB column** | `daily_features.storageUsedGB` |
| **Used in model** | ✅ Baseline change in storage = media hoarding / app churn indicator |

---

#### 26. `mediaCountToday`
| Field | Value |
|---|---|
| **What** | Number of media files (photos, videos, audio) added to device today |
| **Android API** | `MediaStore.Files.getContentUri("external")` |
| **Filter** | `DATE_ADDED >= startOfDay / 1000` (Unix seconds) |
| **Formula** | `cursor.count` |
| **Unit** | Integer count (Float) |
| **DB column** | `daily_features.storageUsedGB` *(scalar in DailyFeaturesEntity not yet added — see note)* |
| **Used in model** | ✅ Now included via `toMap()` |

---

### GROUP 6 — New Expanded Features

---

#### 27. `downloadsToday`
| Field | Value |
|---|---|
| **What** | Files downloaded today (any type via system or browser) |
| **Android API** | `MediaStore.Downloads.EXTERNAL_CONTENT_URI` (API 29+) |
| **Filter** | `DATE_ADDED >= startOfDay / 1000` |
| **Formula** | `cursor.count` |
| **Unit** | Integer count (Float) |
| **DB column** | `daily_features.downloadsToday` |
| **Used in model** | ✅ Sudden download spike = impulsive browsing; absence = shutdown signal |

---

#### 28. `appInstallsToday`
#### 29. `appUninstallsToday`
| Field | Value |
|---|---|
| **What** | Apps newly installed / removed today |
| **Android API** | `PackageManager.getInstalledPackages(0)` |
| **Installs** | `firstInstallTime >= startOfDay` count |
| **Uninstalls** | Delta: `prev_pkg_count` (SharedPrefs) − `currentCount`; reset daily at midnight |
| **Daily reset** | `DataRepository.resetDailyState()` clears `prev_pkg_count` |
| **Unit** | Integer count (Float) |
| **DB columns** | `appInstallsToday`, `appUninstallsToday` |
| **Used in model** | ✅ App churn = behavioural novelty-seeking or digital purge |

---

#### 30. `upiTransactionsToday`
| Field | Value |
|---|---|
| **What** | Launches of UPI / payment apps today (proxy for financial transactions) |
| **Source** | `appLaunchesBreakdown` map filtered by known UPI package prefixes |
| **Detected apps** | Google Pay, PhonePe, Paytm, Amazon Pay, MobiKwik, FreeCharge, Airtel Thanks, SBI Pay, Axis Mobile, BOI Mobile |
| **Formula** | `Σ launches[pkg]` for matching packages |
| **Unit** | Integer count (Float) |
| **DB column** | `daily_features.upiTransactionsToday` |
| **Used in model** | ✅ Impulsive spending / financial stress indicator |

---

#### 31. `nightInterruptions`
| Field | Value |
|---|---|
| **What** | Number of phone unlocks during last night's sleep window (previous night 22:00 → today 05:00) |
| **Android API** | `UsageStatsManager.queryEvents()` |
| **Events used** | `KEYGUARD_HIDDEN (18)` |
| **Window** | `startOfDay − 2h` (22:00 yesterday) → `startOfDay + 5h` (05:00 today), capped at `now` |
| **Previous bug** | Was querying 00:00–05:00 today → always 0 during daytime; fixed to previous night |
| **Unit** | Integer count (Float) |
| **DB column** | `daily_features.nightInterruptions` |
| **Used in model** | ✅ Critical sleep quality indicator; insomnia / anxiety marker |

---

## Storage Flow

```
collectSnapshot()
    │
    ├─ PersonalityVector (in-memory, updated every 15 min)
    │       → DataRepository.latestVector (StateFlow → live UI)
    │
    ├─ At day rollover (midnight):
    │       ↓
    │   DailyFeaturesEntity (Room: daily_features table)
    │       ↓
    │   BaselineEntity (Room: baseline table) — after 28 days
    │       ↓
    │   Firebase Firestore (cloud backup)
    │
    └─ NightlyAnalysisWorker (00:05 daily):
            ↓
        JsonConverter.toEngineJson()
            ↓
        PythonEngine.runAnalysis()
            ↓
        AnalysisResultEntity (Room: analysis_results table)
```

---

## Baseline Building (28-Day Period)

- One `DailyFeaturesEntity` saved per calendar day
- After 28 days: `MonitoringService.persistBaselineToRoom()` computes:
  - `μ (mean)` = average over 28 days for each feature
  - `σ (std dev)` = standard deviation over 28 days
- Both stored as `BaselineEntity` rows (one per feature)
- Baseline also uploaded to Firestore for cloud recovery

---

## ML Model Feature Dimensions

| Category | Features | Count |
|---|---|---|
| Screen/App | screenTime, unlocks, launches, notifications, socialRatio | 5 |
| Communication | calls, callDuration, uniqueContacts, convFrequency | 4 |
| Location | displacement, entropy, homeRatio, places | 4 |
| Sleep | sleepDuration, sleepTime, wakeTime, darkHours | 4 |
| System | charge, memory, wifi, mobile, storage | 5 |
| New Expanded | downloads, appInstalls, appUninstalls, UPI, nightChecks | 5 |
| Previously missing | mediaCount, calendarEvents | 2 |
| **Total** | | **29** |

---

## Required Android Permissions

| Permission | Features Using It |
|---|---|
| `PACKAGE_USAGE_STATS` | All screen/app/sleep features via UsageStatsManager |
| `READ_CALL_LOG` | callsPerDay, callDuration, uniqueContacts |
| `READ_CONTACTS` | uniqueContacts fallback |
| `ACCESS_FINE_LOCATION` | displacement, entropy, homeRatio, places |
| `ACTIVITY_RECOGNITION` | Step counter (TYPE_STEP_COUNTER sensor) |
| `READ_MEDIA_IMAGES` | mediaCountToday, downloadsToday |
| `READ_CALENDAR` | calendarEventsToday |
| `FOREGROUND_SERVICE` | MonitoringService (keeps collection alive) |
| `NETWORK_STATS` | networkWifiMB, networkMobileMB |

