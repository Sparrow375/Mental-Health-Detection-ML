# Lumen v2.0 Release Task List

## Sprint: Lumen v2.0 Major UI Revamp & Backend Alignment (Completed)

### Phase 1: Code Architecture & Release UI Modularization
- [x] **T73**: Create `BadgeEntity` and `BadgeDao` in Room DB for quest achievement persistence.
- [x] **T74**: Create `SharedComponents.kt` and `Charts.kt` under release `ui/components/` to extract reusable UI widgets, dialogs, and custom graphics.
- [x] **T75**: Extract `HomeScreen.kt`, `ActivitiesScreen.kt`, `InsightsScreen.kt`, `CheckInScreen.kt`, `SettingsScreen.kt` under release `ui/screens/`.
- [x] **T76**: Refactor release `MainActivity.kt` to act strictly as scaffold & 5-tab navigation host (`Home`, `Activities`, `Insights`, `Check In`, `Settings`).

### Phase 2: Feature & Permission Vector Cleanup (29 → 28 Features)
- [x] **T77**: Remove `calendarEventsToday` from Python System 1 & System 2 engines (`feature_meta.py`, `data_structures.py`, `l1_scorer.py`, `s1_profile.py`, `bayesian_baseline.py`, `baseline_builder.py`, `s1_s2_adapter.py`, `config.py`).
- [x] **T78**: Remove `calendarEventsToday` from Kotlin models, `DataCollector`, `MonitoringService`, and `JsonConverter`.
- [x] **T79**: Implement in-place baseline JSON migration to strip `calendarEventsToday` without resetting user baselines.
- [x] **T80**: Remove `READ_CALENDAR` permission from `AndroidManifest.xml`.

### Phase 3: Telemetry & Notification Engine Enhancements
- [x] **T81**: Implement soft non-alarming push notification trigger on initial `anomalyDetected` state transition (`sustained_deviation_days >= 3`) with 3-day cooldown in `MonitoringService.kt`.
- [x] **T82**: Add gap detection logging and data integrity checks in `MonitoringService.kt` to catch missing telemetry days.

### Phase 4: App-Wide Design System & Soft-Quantitative Language Polish
- [x] **T83**: Remove all emojis app-wide; replace with styled Material icons (mood faces, highlight cards, toast messages).
- [x] **T84**: Enforce soft-quantitative language policy across `generateBehavioralSummary()`, `WeeklyDigestDialog`, and qualitative cards (relative % instead of raw numeric hours/steps).

### Phase 5: Home Screen Redesign
- [x] **T85**: Implement uncluttered landing page in `HomeScreen.kt`: greeting, permission banner, ambient background breathing glow, Wellness Pulse hero card with subtle visual indicators, Quick Check-in prompt, and core navigation cards.

### Phase 6: Insights Tab Redesign & Sector Curation
- [x] **T86**: Implement Weekly Story narrative card with mini visual status dots/bars.
- [x] **T87**: Implement multi-dimensional Rhythm Trends chart (Rest, Social, Movement, Digital) with labeled Y-axis ("% of Baseline"), date X-axis, and 2W/4W toggle.
- [x] **T88**: Fix Behavioral Fingerprint 6-axis filled-area radar chart comparing current week vs baseline.
- [x] **T89**: Curate insight sectors: remove Charging/Circadian Boundary, add new "Routine & Places" sector (home time ratio, location variety).
- [x] **T90**: Revamp Sector Drill-Down screens with qualitative narrative summaries, relative baseline comparisons, outlier day callouts, and 7-day sparklines.

### Phase 7: Quest Gamification System (Activities Tab)
- [x] **T91**: Implement `ActivitiesScreen.kt` featuring Habit Quests, Wind Down Companion, Digital Detox, and Breathing Exercise.
- [x] **T92**: Build animated streak counters with 1-day grace recovery.
- [x] **T93**: Implement 8 milestone badges in Room DB (`BadgeDao`), badge unlock logic, and Badge Gallery modal.
- [x] **T94**: Add optional Streak Reminder notification setting in `SettingsScreen.kt`.

### Phase 8: Verification & Compilation
- [x] **T95**: Verify build compilation for both Release (`assembleRelease`) and Debug (`assembleDebug`) variants.
- [x] **T96**: Perform end-to-end verification of migration, navigation, notifications, and telemetry reporting.
