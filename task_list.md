# Lumen v2.0 Release Task List

## Sprint: UI Fixes & Feature Improvements

### Phase A: Brute Fixes
- [x] **T97**: Fix Home Screen `NavTile` header title horizontal alignment (`HomeScreen.kt`).
- [x] **T98**: Add `.navigationBarsPadding()` to `FullScreenBreathingScreen` in `ActivitiesScreen.kt` so Stop button is not cut off by navigation bar.
- [x] **T99**: Wire up `showDetoxOverlay` and `showWindDownOverlay` in `ActivitiesScreen.kt` to open full-screen timer / wind-down overlays.
- [x] **T100**: Fix edge padding in `CheckInScreen.kt` for balanced side margins.
- [x] **T101**: Fix "Highest Shift Day" callout in `InsightsScreen.kt` to show exact day name and deviation metrics.

### Phase B: Improvements & Overhauls
- [x] **T102**: Revert and restore full Settings Screen (`SettingsScreen.kt`) to `9edd60a` parity (metadata profile, theme toggle, system permissions cards, daily auto backup, import/export JSON backup, notifications preferences, country-specific crisis helplines, anonymized research contribution, soft reset & hard reset with auto-backup).
- [x] **T103**: Overhaul `InsightsScreen.kt`: Add 0-100 Daily Rhythm Score Gauge at top, qualitative status badge, and concise narrative (sanitizing all user-facing copy to remove the word "baseline").
- [x] **T104**: Overhaul Sector Detail screens in `InsightsScreen.kt`: replace progress bar with plain text comparison ("vs usual norm"), add 7D/14D/30D history tabs, render raw value line chart with usual norm dashed line, and display daily raw value list.
- [x] **T105**: Build Comprehensive Daily & Reflection History in `InsightsScreen.kt`: daily cards with Rhythm Score, check-in ratings, journal notes, top feature shifts, and tap-to-open Full Day Detail Modal displaying complete daily stats.
- [x] **T106**: Overhaul Weekly Trends in `Charts.kt`: convert to normalized stacked area chart with translucent fills, 80%-120% Usual Norm Y-axis, complete color legend (including purple = Digital Focus), and move into expandable `WeeklyTrendsScreen`.
- [x] **T107**: Expand Quests into dedicated full-screen `QuestsScreen` in `ActivitiesScreen.kt`: Active habit quests, custom habit creation dialog, badge gallery with share action, and streak stats (without leaderboard).

### Verification
- [x] **T108**: Compile release APK via `./gradlew :app:compileReleaseKotlin` and verify all 9 issues.

## Sprint: Insights UI & Visual Fixes
- [x] **T109**: Fix Daily Rhythm Score gauge at top of Insights tab (fallback from zero effectiveScore to anomalyScore or live telemetry calculation).
- [x] **T110**: Fix history scores (eliminate hardcoded 85 for current day and 100 for clean previous days).
- [x] **T111**: Fix calendar tap data binding (load full Room daily features history so tapping any day displays authentic day metrics and reflections).
- [x] **T112**: Overhaul History Calendar into true Month-Grid Calendar with month picker, navigation arrows, and ordinal date suffixes ("14th", "1st").
- [x] **T113**: Fix sleep & bedtime circadian circular averaging, clean AM/PM formatting, and human-readable time comparison narrative.
- [x] **T114**: Verify compilation and test with exported dataset.
- [x] **T115**: Sync top Insights hero score to directly display today's live history score.
- [x] **T116**: Fix text cutoff on "Challenging" in Daily Check-In rating selector with maxLines=1, letter spacing, and card padding adjustments.

