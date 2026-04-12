# MHealth Behavioral Feature Logic (29-Vector)

This document details the implemented logic used by the Android Kotlin `DataCollector.kt` pipeline to translate raw device telemetry into a rich 29-feature behavioral vector.

## 1. Location & Spatial Movement

1. **Daily Displacement (km)**
   - Uses a Grid-Cell Transition Method (~110-meter cell boundaries) mapped from Android's FusedLocationProvider. Drops GPS drift by only counting absolute cell-to-cell traversal distances.
2. **Location Entropy**
   - Shannon entropy calculated based on the *fraction of time* spent per geographic grid cell, not raw ping counts. High entropy = erratic/highly varied routine. 
3. **Home Time Ratio**
   - Percentage of the 24-hour day spent within a defined 500-meter geospatial radius of the anchored "Home" coordinate. Bridges overnight GPS sleep hours intelligently.
4. **Places Visited**
   - Absolute discrete count of unique ~110-meter grid cells populated by the user's GPS signature today.

## 2. Activity & Health

5. **Daily Steps**
   - Read from the hardware strictly via Android's `TYPE_STEP_COUNTER`. Computes the delta from the midnight cache mark to capture distinct daily volume.
6. **Sleep Duration (Hours)**
   - Operates on a 3-Signal Fusion heuristic across a strict 6:00 PM to 12:00 PM (noon) temporal window. Locates the largest continuous screen-off gap, smoothing over any micro-wakes (< 5 mins). Time-shifted dynamically by Do-Not-Disturb configurations.
7. **Wake Time Hour**
   - The timestamp concluding the derived primary Sleep Duration gap.
8. **Sleep Time Hour**
   - The timestamp initiating the derived primary Sleep Duration gap.

## 3. Communication & Sociability

9. **Calls Per Day**
   - Raw volume of telephony events (inbound/outbound/missed) retrieved from Android `CallLog`.
10. **Call Duration (Minutes)**
    - Total summation of connected active voice minutes.
11. **Unique Contacts**
    - Distinct phone numbers engaged traversing the daily call log payload.
12. **Conversation Frequency**
    - Calculated as `Calls Per Day / Unique Contacts`. Differentiates between expansive social networks and hyper-fixated conversational loops.

## 4. Digital Engagement & Media

13. **Screen Time (Hours)**
    - Derived natively by pairing `ACTIVITY_RESUMED` and `ACTIVITY_PAUSED` UsageEvents exactly, ignoring generic system launchers.
14. **Unlock Count**
    - Evaluated purely via `KEYGUARD_HIDDEN` system broadcasts.
15. **App Launch Count**
    - Count of distinct foregrounding app events, applying a strict 1.5s debounce filter to deter UI flickering noise.
16. **Social App Ratio**
    - Evaluates total time elapsed inside heuristically defined or Play Store categorized social bundles (Meta, X, TikTok, Telegram) divided by aggregate daily Screen Time.
17. **Notifications Today**
    - Derived via Notification Listener parsing `NOTIFICATION_SEEN` impacts.
18. **Background Audio (Hours)**
    - Detects invisible consumption (Spotify, Podcasts) by intercepting the system `AudioManager` while the screen is extinguished.
19. **Media Count Today**
    - Incremental daily volume of external media (photos/videos) authored and synced to the local gallery repository.

## 5. Erratic Behavior & Instability Indicators

20. **UPI Transactions Today**
    - Maps financial volatility. Filters exact application package launches for specified regionally active payment gateways (GPay, PhonePe, Cred).
21. **Total Apps Count**
    - Absolute volume of third-party software packages resident on the internal storage.
22. **App Installs Today**
    - Scans Package Manager for unique software footprints with an initial "first installed" stamp matching the current calendar day.
23. **App Uninstalls Today**
    - Evaluated retroactively by calculating baseline decay from the trailing 24-hr aggregate `Total Apps Count`.
24. **Downloads Today**
    - Parses the `/Downloads` filesystem directory for uniquely created or heavily modified datestamps spanning the current 24-hour block.

## 6. System & Infrastructure Tethers

25. **Charge Duration (Hours)**
    - Accumulator strictly indexing intervals where the physical `BatteryManager.EXTRA_PLUGGED` status registers as active.
26. **Dark Duration (Hours)**
    - Unfiltered additive aggregate of raw screen-extinguished hours across the total 24-period. Distinct from the filtered Sleep Duration.
27. **Memory Usage (Percent)**
    - Instantaneous RAM load proxy parsed from `ActivityManager.MemoryInfo`.
28. **Storage Used (GB)**
    - Checks absolute byte saturation on `Environment.getDataDirectory()` via `StatFs`.
29. **Network Bandwidth (Wi-Fi & Mobile MB)**
    - Extracts distinct payloads downloaded/uploaded utilizing `NetworkStatsManager` querying, mapping digital consumption intensity. 
