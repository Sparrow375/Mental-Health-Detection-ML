# MHealth Behavioral Feature Logic (31-Vector)

This document details the implemented logic used by the Android Kotlin `DataCollector.kt` pipeline to translate raw device telemetry into a rich 31-feature behavioral vector.

## 1. Location & Spatial Movement

1. **Daily Displacement (km)**
   - Uses a **Grid-Cell Transition Method** with `%.4f` precision (~11.1-meter cell boundaries) mapped from Android's FusedLocationProvider. Drops GPS drift by only counting absolute cell-to-cell traversal distances.
2. **Location Entropy**
   - **Time-Weighted Shannon entropy** calculated based on the *actual wall-clock duration* spent per geographic grid cell. High entropy = highly varied routine.
3. **Home Time Ratio**
   - **Dynamic Divisor Calculation**: Percentage of the **currently elapsed day** (time since midnight) spent within a 500-meter geospatial radius of the anchored "Home" coordinate. 
   - **Overnight Bridge**: Utilizes yesterday's final GPS fix to intelligently account for hours spent at home during sleep before the first daily GPS ping.
4. **Places Visited**
   - Absolute discrete count of unique ~11.1-meter grid cells populated by the user's GPS signature today.

**Adaptive GPS Accuracy thresholds:** To minimize noise, fixes are filtered by state:
- **Stationary**: Accuracy $\leq 200m$
- **Walking**: Accuracy $\leq 100m$
- **Vehicle**: Accuracy $\leq 50m$
- **Home Detection**: Accuracy $\leq 800m$ (relaxed threshold to ensure coverage on budget devices).

## 2. Activity & Health

5. **Daily Steps**
   - Read via Android's `TYPE_STEP_COUNTER` sensor. Computes the delta from the midnight cache mark to capture distinct daily volume.
6. **Sleep Duration (Hours)**
   - Operates on a **3-Signal Fusion** heuristic across a temporal window from 6:00 PM yesterday to 2:00 PM today.
   - Detects the longest continuous gap between any interaction events (Screen ON, App Launch, Unlock). 
7. **Wake Time Hour**
   - The fractional timestamp concluding the primary detected sleep gap.
8. **Sleep Time Hour**
   - The fractional timestamp initiating the primary detected sleep gap.

## 3. Communication & Sociability

9. **Calls Per Day**
   - Raw volume of telephony events (inbound/outbound/missed) retrieved from the Android `CallLog`.
10. **Call Duration (Minutes)**
    - Total summation of connected active voice minutes.
11. **Unique Contacts**
    - Distinct phone numbers engaged traversing the daily call log payload.
12. **Conversation Frequency**
    - Calculated as `Calls / Unique Contacts`. Differentiates between expansive social networks and hyper-fixated conversational loops.

## 4. Digital Engagement & Media

13. **Screen Time (Hours)**
    - Derived by exactly replaying `MOVE_TO_FOREGROUND` and `MOVE_TO_BACKGROUND` `UsageEvents` pairs.
14. **Unlock Count**
    - Evaluated purely via `KEYGUARD_HIDDEN` system broadcasts (Type 18 events).
15. **App Launch Count**
    - Count of distinct foregrounding app events, applying a strict 1.5s debounce filter to deter UI flicker noise.
16. **Social App Ratio**
    - Evaluates total time elapsed inside social-categorized bundles (Instagram, WhatsApp, X, etc.) divided by aggregate daily Screen Time.
17. **Notifications Today**
    - Total count of `NOTIFICATION_INTERRUPTION` (Type 12) events.
18. **Background Audio (Hours)**
    - Tracks intervals where media-capable apps are active while the system `AudioManager.isMusicActive()` and the screen is extinguished.
19. **Media Count Today**
    - Volume of new media files (photos/videos) detected in the local gallery via `MediaStore` addition timestamps.

## 5. Behavioural & Financial Stability

20. **UPI Transactions Today**
    - Detection of financial activity by filtering launches of payment gateway packages (GPay, PhonePe, Paytm, etc.).
21. **Total Apps Count**
    - Absolute volume of third-party software packages resident on the device.
22. **App Installs Today**
    - Scans Package Manager for packages with a `firstInstallTime` matching the current day.
23. **App Uninstalls Today**
    - Computed by comparing the previous day's registered package count against current volume.
24. **Downloads Today**
    - Parses `MediaStore.Downloads` for uniquely created datestamps spanning the current 24-hour block.
25. **Calendar Events Today**
    - Count of active events in the `CalendarContract` instances matching the current day.

## 6. System & Infrastructure Tethers

26. **Charge Duration (Hours)**
    - Accumulator strictly indexing intervals where `BatteryManager.EXTRA_PLUGGED` is active.
27. **Dark Duration (Hours)**
    - Unfiltered additive aggregate of all screen-extinguished hours.
28. **Memory Usage (Percent)**
    - RAM load proxy parsed from `ActivityManager.MemoryInfo`.
29. **Storage Used (GB)**
    - Absolute byte saturation on internal storage via `StatFs`.
30. **Network Bandwidth (Wi-Fi MB)**
31. **Network Bandwidth (Mobile MB)**
    - Extracts distinct payloads downloaded/uploaded utilizing `NetworkStatsManager` querying.
