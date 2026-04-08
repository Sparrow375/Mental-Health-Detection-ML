# MHealth Behavioral Feature Calculation Logic

This document details the logical methodologies used by the MHealth data collection pipeline to translate raw device telemetry into meaningful behavioral indicators. This focuses exclusively on the logic and heuristics without dwelling on technical implementation code.

---

## 1. Sleep & Wake Patterns

### **Sleep Time, Wake Time, & Sleep Duration**
- **Logic**: We estimate sleep using a "3-Signal Fusion" approach over a **fixed 18-hour overnight window (6:00 PM to 12:00 PM the next day)**.
- **Why 18 hours?** A rolling 24-hour window could mistake daytime phone inactivity (e.g., the phone sitting on a desk from 9 AM to 5 PM) for the primary sleep episode. Constraining the window to the evening-to-midday range forces the system to only find sleep when it realistically occurs. Night-shift workers are intentionally excluded from this heuristic for now.
- **Step 1 (Screen Gaps)**: The system groups raw screen "On" and "Off" events specifically within the 6 PM–12 PM overnight window into continuous screen sessions, then identifies all the "gaps" between them.
- **Step 2 (Micro-wake Filtering)**: Brief screen usages of less than 5 minutes (micro-wakes, e.g. briefly checking the time at 3 AM) are ignored. The gaps bordering these micro-wakes are merged together so they don't fragment the primary sleep episode.
- **Step 3 (Longest Episode)**: The longest continuous screen-off gap within the 18-hour window is designated as the primary sleep episode. The start of this gap is tentatively "Sleep Time" and the end is "Wake Time".
- **Step 4 (Do Not Disturb Fusion)**: 
  - If Do Not Disturb (DND) turned on near the start of the gap, the Sleep Time is adjusted to an average of the two, locking in intent.
  - If DND turned off during the gap, Wake Time is adjusted to match the DND off time, under the assumption that the user woke up to turn it off.

### **Dark Duration**
- **Logic**: The total absolute hours the device screen was turned off throughout the entire day. Unlike sleep duration, this is purely additive and ignores fragmentation.

---

## 2. Location & Movement Metrics

### **Daily Displacement (Distance Traveled)**
- **Logic**: Calculated using a **Grid-Cell Transition Method** to eliminate GPS drift noise.
- The phone's GPS snapshots are mapped to ~110-meter grid cells (matching the same cells used for Entropy and Places Visited).
- Distance is only added to the running total when a GPS ping lands in a **completely different grid cell** from the previous one. If the phone is stationary on a desk and the GPS drifts by 15 meters, it stays in the same cell, so **0 km is added**.
- When a genuine cell-to-cell transition happens (the user physically moved), the Haversine distance between the center-points of those two cells is added to the total.
- **Why this is better**: Eliminates the silent "phantom distance" bug where stationary GPS noise accumulated kilometers over the course of a day.

### **Location Entropy**
- **Logic**: Entropy represents how "predictable" or "scattered" a user's geographical footprint is, measured in **time spent** — not ping count.
- Using the GPS snapshot timestamps, the system calculates the actual **wall-clock hours** spent at each ~110-meter grid cell (capped at 12 hours per gap to prevent bridging overnight absences). This is the same methodology used by Home Time Ratio.
- Mathematical Shannon entropy is then computed from the distribution of **time fractions** across all visited cells.
- **Meaning**: A high entropy value indicates the user's time was spread across many different locations (variable routine). A low entropy value means the majority of their time was spent in one or two places (e.g., Home and Work).
- **Why time vs. pings**: Using ping counts was inaccurate — the GPS checks in more frequently when moving, which would unfairly inflate the entropy score for people who just commuted vs. people who truly visited many places.

### **Home Time Ratio**
- **Logic**: Determines the percentage of the 24-hour day a user spent physically at home.
- A static "Home" location is set in the system. As the GPS pings throughout the day, the system sums up the time gap between any two sequential pings that occurred within a 500-meter radius of Home. 
- It also intelligently bridges the overnight gap (from midnight to the first morning ping) so sleep hours properly credit to "home time" even if the GPS goes to sleep. 

### **Places Visited**
- **Logic**: The absolute, discrete count of unique ~110-meter grid cells the user was physically present in today.

---

## 3. Screen Time & App Usage

### **Screen Time**
- **Logic**: Measured identically to Android's built-in Digital Wellbeing. Instead of polling estimates, the system chronologically pairs "App Moved to Foreground" and "App Moved to Background" events for the entire day. The exact durations between these events are summed up. System components (like default launchers and system UI components) are deliberately ignored so simply looking at the home screen does not distort active usage stats.

### **App Launch Count**
- **Logic**: The number of times user applications are opened/brought to the screen. 
- **Debouncing**: To prevent rapid toggling between apps from artificially inflating the number, an app must have been in the background for at least 1.5 seconds before returning to the screen to count as a "new launch".

### **Unlock Count**
- **Logic**: Direct count of how many times the device screen passed the lock screen barrier.

### **Social App Ratio**
- **Logic**: The fraction of total screen time dedicated to social interactions.
- The system groups total screen time by app. It then flags social apps either by checking if the OS officially categorizes them as "Social", or by triggering text heuristics (e.g., the app name includes "instagram", "whatsapp", "tiktok", "facebook", "telegram", etc.).
- It divides the time spent in these grouped apps by the total screen time.

---

## 4. Communication Activities

### **Calls Per Day & Call Duration**
- **Logic**: Reads the native phone call log to count the total number of incoming and outgoing calls that took place since midnight, as well as summing their total talk duration in minutes.

### **Unique Contacts & Conversation Frequency**
- **Logic**:
  - **Unique Contacts**: Looks at the call log and counts how many *distinct phone numbers* the user called or received calls from today.
  - **Conversation Frequency**: Calculated by dividing the Total Calls by Unique Contacts. This indicates whether a user is repeatedly talking to a small inner circle, or having one-off calls with many different people.

---

## 5. Device Interaction & Media

### **Notifications Today**
- **Logic**: Total count of push notifications and alerts received by the device that actually interrupted the user/system.

### **Charge Duration**
- **Logic**: A continuously accumulating timer that ticks upward only while the device is physically connected to a power source.

### **Background Audio Hours**
- **Logic**: Dedicated to tracking passive media consumption (music, podcasts). Given music apps run invisibly, the system natively interrogates the device's Audio Manager. During routine checks, if an active media session belongs to a recognized music/podcast package (Spotify, YouTube Music, Audible, local players, etc.), it increments the total background audio playtime.

### **Memory Usage & Storage Limits**
- **Logic**: Captures device strain. 
  - **Memory (RAM)**: Percentage of immediate system RAM presently occupied.
  - **Storage**: Absolute volume (in Gigabytes) of internal device storage that is currently full.

### **Network Data Consumption**
- **Logic**: Queries the system's exact bandwidth monitor to retrieve the total Megabytes of data downloaded and uploaded today, distinctly segmented into Wi-Fi traffic versus Cellular (Mobile) traffic.

---

## 6. Daily Micro-behaviors & Events

### **Media Added Today**
- **Logic**: Evaluates the device's external storage gallery and counts exactly how many new photos or videos were generated or saved onto the device since midnight.

### **Downloads Today**
- **Logic**: Inspects the device's physical `Downloads` folder to count exactly how many files have a creation/modification timestamp occurring today.

### **UPI / Financial Transactions**
- **Logic**: Uses the app launch data to specifically filter for common digital wallet and UPI apps (Google Pay, PhonePe, Paytm, CRED). Launching these apps serves as a proxy metric for conducting financial transactions or online shopping behaviors.

### **App Installs & Uninstalls**
- **Logic**: 
  - **Installs**: Scans all applications and counts how many have an initial "First Installed" timestamp from today.
  - **Uninstalls**: Assessed by keeping a running tally of the total app count. If today's total app count is lower than yesterday's mathematically, the difference is recorded as uninstalls.
  - **Total Apps**: Absolute volume of non-system applications living on the phone.

### **Calendar Events**
- **Logic**: Queries the on-device Calendar providers to count the number of meetings, reminders, or scheduled events that intersect with today's 24-hour block.

### **Daily Steps**
- **Logic**: Reads the hardware pedometer sensor inside the phone. Since the sensor yields a lifetime cumulative number, the system records a baseline snapshot every morning. The day's total steps are logic-derived by subtracting the morning baseline from the current live number.
