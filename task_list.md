# Lumen Insights Enhancement Task List

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

