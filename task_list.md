# Lumen Insights Enhancement Task List

## Sprint: Pilot Readiness Adjustments (In Progress)
- [x] **T60**: Fix interaction tempo graph missing mappings in `getFeatureValueFromEntity()` for keystroke speed, backspace ratio, scroll velocity.
- [x] **T61**: Fix research share button action: resolve coroutine cancellation bug, launch ACTION_VIEW intent opening Google Form URL.
- [x] **T62**: Fix digital detox interrupted warning persistence: add close/dismiss icon to the warning.
- [x] **T63**: Fix rhythm score drop: clamp individual feature deviations in `safeDev()` to `2.0f` and raise the callsPerDay scale floor to `3f`.
- [x] **T64**: Remove manual coordinate setting options for home location in both Settings and Onboarding step 6.
- [x] **T65**: Add navigation bar padding to `OnboardingWizard` to prevent system navigation bar interference.
- [x] **T66**: Add a lightweight WebView-based map picker with geocoding address search to the debug build for developer evaluation.

## Sprint: Pivot & Release Readiness (Completed)

### Phase 1: P0 Critical Release Fixes
- [x] **T31**: Implement call log proxy signals using dialer app launches (`com.samsung.android.dialer`, `com.samsung.android.incallui`, `com.google.android.dialer`, `com.truecaller`) from `UsageEvents` (unblocking Relational/Communication metrics).
- [x] **T32**: Add a client-side force app update on launch using Google Play In-App Updates API (`AppUpdateManager` in `MainActivity.onCreate`).
- [x] **T33**: Verify end-to-end permission flows in release build, ensuring that if usage stats or location isn't granted, a persistent non-dismissible banner is displayed on the dashboard.

### Phase 2: P1 Core Wellness & ML Showcase
- [x] **T34**: Enrich the guided box breathing card to launch a full-screen interactive breathing screen with Canvas animations, phase indicators, and haptic feedback.
- [x] **T35**: Add Interaction Tempo insight sector displaying Typing Speed, Edit Frequency, and Scroll Pace (qualitative baseline comparisons + 7-day sparklines).
- [x] **T36**: Add Charging Patterns + Daylight Exposure + Daily Engagement insight sectors.
- [x] **T37**: Evolve raw stats on all insight cards to reflective qualitative narratives comparing current status to personal baseline.
- [x] **T38**: Implement Event Annotation system ("Add a note") triggered by elevated anomaly scores or notable pattern shifts.
- [x] **T39**: Surface "Rhythm Consistency %" gauge (derived from ML anomaly score) on the dashboard with coherence indicators.
- [x] **T40**: Add a Behavioral Fingerprint 6-axis Radar/Spider chart normalized to personal baseline.
- [x] **T41**: Expand the Daily Quotes system to include diverse wisdom categories (mindfulness, stoicism, self-compassion, science).
- [x] **T42**: Enhance Monthly Screeners with a larger UI layout and a qualitative monthly reflection note field.

### Phase 3: P2 Engagement & Anonymization (Completed)
- [x] **T43**: Implement Passive-Verified Habit Goals (Digital Sunset, Circadian Anchor, Movement Boost, Focus Mode) with streak counters and Home Screen cards.
- [x] **T44**: Implement Pattern Stories Narration Engine based on anomaly deviations and cluster coherence changes.
- [x] **T45**: Implement Weekly Digest Report Card displaying weekly telemetry highlights, watch items, and mood correlation trends.
- [x] **T46**: Implement Anonymous Research Contribution export system triggered 30 days post-installation with differential privacy noise and PII scrubbing.

### Phase 4: P3 Polish & Delight (Completed)
- [x] **T47**: Implement Wind-Down Companion (bedtime reminders, dark UI state).
- [x] **T48**: Implement Digital Detox Timer with screen-off tracking and records.
- [x] **T49**: Implement Mood × Behavior Correlation Cards.

## Sprint: Adaptive Wellness & Bug Fixes (Completed)

- [x] **T-A**: Fix nav-bar button overlap on all 5 full-screen dialogs (`WindDownOverlay`, `DigitalDetoxTimerOverlay`, `FullScreenBreathingScreen` x2, `WeeklyDigestDialog`) — added `rememberNavBarPadding()` helper that reads from the host Activity's `rootWindowInsets`, applied as explicit bottom padding in each dialog since `navigationBarsPadding()` returns 0 inside Dialog windows.
- [x] **T-B**: Added functional "Share Feedback" card (`Intent.ACTION_SENDTO` → `support@lumenapp.health`) to `SettingsScreen` in the release build, placed above Privacy Policy card.
- [x] **T-C**: Added permission-aware status note under Relational Frequency and Interaction Tempo insight cards — shows ⚠ if service is off (with instructions) or 📡 if service is on but data is still accumulating.
- [x] **T-D**: Replaced hard-coded 11PM wind-down target with adaptive circadian baseline — reads last 14 days of checkin `sleep_time` history, averages it (≥3 entries required), falls back to 23:00 for new users, respects Habit Goals override if set. Card subtitle text adapts to tell user whether it's personal or default.
- [x] **T-E**: Updated `context.md` and `task_list.md`. Build verified: `assembleRelease` → **BUILD SUCCESSFUL**.

## Sprint: UX Polish & Phase 3 Extensions (Completed)
- [x] **T50**: Fix guided box breathing state machine (resolve exhale->hold loop and duplicate volume adjustments).
- [x] **T51**: Fix system back button navigation handler for inner tabs in Insights (activeDetailSector).
- [x] **T52**: Fix UI elements going under the navigation bar (add `decorFitsSystemWindows = false` to full-screen dialogs like `WeeklyDigestDialog` and `FullScreenBreathingScreen`).
- [x] **T53**: Redesign Weekly Digest to be highly analytical (add weekly average consistency score gauge, daily consistency mini bar chart/sparkline, habit quest completion rate, and detailed metric vs baseline comparisons).
- [x] **T54**: Fix x-axis overlaps in Rhythm Detail Screen trend chart (implement step spacing for dates).
- [x] **T55**: snappier StaggeredFadeIn animations (decrease delay to 30ms/index, cap at 150ms, decrease duration to 300ms).
- [x] **T56**: Daily Quotes cleanup and refresh behavior (regex strip category tag, random quote selection on app start).
- [x] **T57**: Implement Wind-Down Companion details (bedtime approach reminders, dim-screen overlay theme).
- [x] **T58**: Implement Digital Detox Timer details (configurable timer, screen-off tracking with broadcast receiver, success/interrupted notification & streak).
- [x] **T59**: Expand Mood x Behavior Correlation Cards (physical vs mood, sleep vs mood, digital vs mood).


## Sprint: Google Play Store Compliance (Completed)

- [x] **T26**: Fix inadequate prominent disclosure text for Location permissions in release `LocationDisclosureDialog` to explicitly mention background location usage and the exact required phrase "even when the app is closed or not in use".
- [x] **T27**: Fix onboarding step 6 (Home Location Capture) and Settings screen to show the `LocationDisclosureDialog` immediately before prompting for location permissions, ensuring that consent requests are always immediately preceded by an in-app disclosure.
- [x] **T28**: Remove automatic permission request on startup in release `MainLumenDashboard()` to prevent launching runtime permission dialogs without user context or disclosure.
- [x] **T29**: Update debug `MainActivity.kt` to also implement proper prominent disclosure before requesting location permissions, or avoid automatic startup permission prompts.
- [x] **T30**: Update `context.md` with compliance changes.

---

## Sprint: Home Screen Polish & Daily Check-in Fixes (Completed)

- [x] **T19**: Correct daily check-in calmness/anxiety rating inversion (value 5 is "Calm / Relaxed", value 1 is "Severe Stress").
- [x] **T20**: Dismiss check-in prompt on Home tab once completed today (hiding it completely).
- [x] **T21**: Restore rhythm consistency preview chart (`QualitativeTrendChart`) inside the clickable card on the Insights tab (renamed from Circadian Rhythm Consistency).
- [x] **T22**: Add "Mindful Breathing Pause" interactive box breathing card on the Home Screen.
- [x] **T23**: Add "Daily Focus & Wisdom" rotating quote card on the Home Screen.
- [x] **T24**: Add "Today's Routine Snapshot" telemetry comparisons card on the Home Screen (without showing targets).
- [x] **T25**: Update `context.md` with the latest changes and structure.


### Previous Tasks (Completed)
- [x] **T1**: Uncap check-in history (remove 7-entry trim from `saveCheckinToHistory`, keep unlimited)
- [x] **T2**: Add journal note field to `DailyCheckinTab` + store in check-in history JSON
- [x] **T3**: GPS display fix — reverse geocode home location, hide raw coordinates in Settings & Home
- [x] **T4**: Behavioral summary prompt on Home screen (sector-aware contextual observation replacing generic status)
- [x] **T5**: Fix Rhythm Consistency chart (composite adherence score + gradient fill + day-of-week labels)
- [x] **T6**: Insights behavioral timeline with integrated check-in entries + journal note markers
- [x] **T7**: Per-sector detail screens (navigate from insight cards, multi-line trend charts, baseline reference lines)
- [x] **T8**: Add Daylight Exposure + Charging Routine insight cards to Insights tab
- [x] **T9**: Mood × Behavior correlation cards (cross-reference check-in data with telemetry)
- [x] **T10**: Personal milestones / celebration card on Home screen
- [x] **T11**: Automatic daily backup to Downloads via MediaStore API (no extra permissions)
- [x] **T12**: Weekly qualitative summary notification (Sunday evening)
- [x] **T13**: Settings toggles for auto-backup and weekly summary notifications
- [x] **T14**: Update `context.md` with all changes
- [x] **Compiler Fix**: Added missing Compose `rememberTextMeasurer` and `drawText` imports, and defined `formatValue` helper function to fix release build compilation
- [x] **iOS Task 1**: Formulate iOS migration strategy and technical feasibility analysis
- [x] **iOS Task 2**: Update `context.md` with iOS feasibility guidelines

