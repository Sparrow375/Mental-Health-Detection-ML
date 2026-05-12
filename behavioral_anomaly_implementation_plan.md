**Behavioral Anomaly Detection System**

Detailed Implementation Plan

**Level 1 \+ Level 2 (Digital DNA) Pipeline**

# **1\. System Overview**

This system passively monitors smartphone behavior to detect sustained behavioral deterioration — specifically depressive and anxiety-related episodes — without requiring active user input. It operates in two complementary layers.

| Layer | What it measures |
| :---- | :---- |
| Level 1 (L1) | 29 daily aggregate features — screen time, displacement, sleep, calls, etc. Detects deviation from personal baseline in magnitude and velocity. |
| Level 2 (L2 / Digital DNA) | Per-session micro-patterns — when apps are used, how long, what triggers the open, session texture quality, rhythm integrity. Distinguishes context (exam, holiday) from deterioration. |

The two layers are not independent scoring systems. L1 detects that something has changed. L2 computes a modifier that either suppresses L1 evidence (known or coherent new context) or amplifies it (unfamiliar pattern with degraded texture). Only the modifier-adjusted score feeds the evidence accumulator.

| Core design principle: A single bad day never triggers an alert. Evidence must accumulate across days, and that accumulation is gated by both temporal persistence and L2 texture quality. Context changes suppress — texture dissolution amplifies. |
| :---- |

# **2\. Module Architecture**

The system consists of four layers: Android data collection, Python backend processing, persistent storage, and output/reporting.

## **2.1 Android Modules**

| Module | Responsibility |
| :---- | :---- |
| PersonalityDataCollector | Midnight batch job. Reads sensors and usage stats APIs. Writes one 29-feature daily row to local SQLite. |
| SessionEventLogger | Foreground/background listener on AccessibilityService or UsageStatsManager. Logs every app session: app\_id, open\_ts, close\_ts, trigger type, interaction count. |
| NotificationEventLogger | NotificationListenerService. Logs notification arrival, and whether it was tapped, dismissed, or ignored, with tap latency in minutes. |
| DataSyncService | Nightly sync of local SQLite rows to backend over encrypted channel. Handles offline buffering. |

## **2.2 Python Backend Modules**

| Module | Responsibility |
| :---- | :---- |
| baseline\_builder.py | Runs after sufficient baseline days. Builds PersonalityVector, AppDNA per app, PhoneDNA, L1 anchor clusters (DBSCAN), L2 contextual texture profiles (K-means per archetype). Saves to person profile store. |
| l1\_scorer.py | Daily. Computes 29 weighted z-scores (deviation magnitude) and 29 EWMA slopes (deviation velocity). Produces composite L1 score \[0,1\]. |
| l2\_scorer.py | Daily. Computes context coherence, rhythm dissolution (KL divergence), session incoherence, and L2 modifier \[0.15, 2.0\]. |
| evidence\_engine.py | Stateful. Applies L2 modifier to L1 score. Updates sustained\_deviation\_days and evidence\_accumulated with exponential compounding on anomalous days and 8% decay on normal days. |
| candidate\_cluster.py | Triggered when L1 is anomalous but L2 texture is coherent. Manages 7-day candidate window. Promotes to anchor cluster or retroactively releases held evidence. |
| alert\_engine.py | Determines alert level (green/yellow/orange/red) based on sustained gate, composite score, and critical feature deviations. Classifies pattern type. |
| prediction\_engine.py | End-of-period retrospective analysis. Checks peak evidence and peak sustained days against stricter thresholds. Produces FinalPrediction with recommendation tier. |
| reporter.py | Assembles AnomalyReport and DailyReport from all upstream outputs. Writes to output DB for UI and System 2 consumption. |

## **2.3 Storage Schema (SQLite per person)**

| Table | Contents |
| :---- | :---- |
| daily\_features | One row per day: date \+ 29 L1 feature values. |
| session\_events | One row per app session: app\_id, open\_ts, close\_ts, trigger, interaction\_count. |
| notification\_events | One row per notification: app\_id, arrival\_ts, action (tap/dismiss/ignore), tap\_latency\_min. |
| person\_profile | Serialised PersonalityVector, AppDNA dict, PhoneDNA, L1 cluster centroids/radii, L2 texture profiles per archetype. Rebuilt at 28/60/90 days. |
| evidence\_state | Persistent: sustained\_deviation\_days, evidence\_accumulated, max\_evidence, max\_sustained\_days, max\_anomaly\_score. Updated nightly. |
| candidate\_state | Nullable: candidate open/close timestamps, buffered daily vectors, texture quality scores, promotion/rejection status. |
| daily\_reports | One row per monitoring day: all fields from DailyReport dataclass. |
| anomaly\_reports | One row per monitoring day: all fields from AnomalyReport dataclass. |

# **3\. Core Data Structures**

## **3.1 PersonalityVector (L1 baseline)**

29 float fields, one per feature, storing the baseline mean for that person. Companion variances dict stores per-feature standard deviations. Frozen after baseline establishment.

| Group | Features | Clinical weight |
| :---- | :---- | :---- |
| Screen & App | screenTimeHours, unlockCount, appLaunchCount, notificationsToday, socialAppRatio | 0.8–1.4 |
| Communication | callsPerDay, callDurationMinutes, uniqueContacts, conversationFrequency | 0.9–1.3 |
| Location & Movement | dailyDisplacementKm, locationEntropy, homeTimeRatio, placesVisited | 1.1–1.5 |
| Sleep & Circadian | wakeTimeHour, sleepTimeHour, sleepDurationHours, darkDurationHours | 1.0–1.6 |
| System Usage | chargeDurationHours, memoryUsagePercent, networkWifiMB, networkMobileMB, storageUsedGB | 0.4–0.8 |
| Behavioural Signals | totalAppsCount, upiTransactionsToday, appUninstallsToday, appInstallsToday | 0.8–1.1 |
| Calendar & Engagement | calendarEventsToday, mediaCountToday, downloadsToday, backgroundAudioHours | 0.6–0.9 |

## **3.2 AppDNA (per-app L2 baseline)**

Built for every app used 3+ times during baseline. Contains temporal, session, trigger, and sequence DNA.

| Field | Description |
| :---- | :---- |
| usage\_heatmap | np.ndarray shape (7, 24\) — mean minutes used per day-of-week × hour slot across baseline days. |
| primary\_time\_range | Tuple (hour\_start, hour\_end) representing the window containing 80% of usage. |
| time\_concentration\_ratio | Fraction of total app time that falls within the primary time range. Stable indicator of temporal anchoring. |
| time\_concentration\_std | Day-to-day variance of that ratio. High std \= person's timing is naturally loose; deviation threshold widens accordingly. |
| avg\_session\_minutes / std / p10 / p90 | Distribution of session duration. p10 \= baseline for 'short', p90 \= baseline for 'deep' sessions. |
| abandon\_rate / abandon\_rate\_std | Fraction of sessions under 45s with fewer than 5 interaction events. Variance tells us how consistent this is. |
| self\_open\_ratio / notification\_open\_ratio / shortcut\_open\_ratio | Trigger DNA: how the person habitually opens this app. Sum \= 1.0. |
| notification\_response\_latency\_median / std | Minutes between notification arrival and app open. Rising latency \= withdrawal; collapsing latency \= anxious reactivity. |
| pre\_open\_apps / post\_open\_apps | Dicts of app\_id → frequency. Captures the behavioral grammar: which apps typically precede or follow this one. |
| interactions\_per\_minute\_mean / std | Engagement density within sessions. Low \= passive scrolling, high \= purposeful active use. |
| weekday\_sessions\_per\_day / weekend\_sessions\_per\_day | Split usage rates. Collapse of natural weekday/weekend difference is itself a signal. |
| daily\_use\_consistency / max\_gap\_days | How reliably the app appears day-to-day. An app with max\_gap\_days \= 7 in baseline should not be flagged for a 5-day absence. |

## **3.3 PhoneDNA (device-level L2 baseline)**

| Field | Description |
| :---- | :---- |
| first\_pickup\_hour\_mean / std | When phone is first touched each day. Low std \= consistent morning routine. |
| active\_window\_duration\_mean / std | Hours between first and last pickup. Depression compresses or collapses this window. |
| pickups\_per\_hour\_by\_hour | np.ndarray (24,) — mean pickups per hour. Captures 3am restlessness and midday activity patterns. |
| pickup\_burst\_rate | Fraction of pickups within 5 minutes of a previous pickup. High \= anxious/compulsive checking. |
| inter\_pickup\_interval\_mean / std | Mean and variance of gap between pickups. Depression creates irregular gaps — long dead zones punctuated by short bursts. |
| session\_duration\_distribution | 5-bin histogram: \<2min, 2–15min, 15–30min, 30–60min, 60+min. Healthy: bimodal. Depressed: collapses toward short end. |
| deep\_session\_ratio / micro\_session\_ratio | Fraction of sessions above 20 min (purposeful) and below 2 min (checking behavior). |
| app\_cooccurrence\_matrix | np.ndarray (N\_apps × N\_apps) — how often pairs of apps appear in same 30-minute window. Captures purposeful behavioral clusters. |
| notification\_open\_rate / dismiss\_rate / ignore\_rate | How the person relates to incoming notifications. ignore\_rate rising \= behavioral paralysis. |
| daily\_rhythm\_regularity | Mean autocorrelation of hourly pickup pattern across days. 1.0 \= identical rhythm every day. Depression dissolves this. |
| weekday\_weekend\_delta | L1 norm of mean feature difference between weekday and weekend days. Depression collapses this delta — every day looks the same. |
| historically\_active\_hours | List of hours where baseline pickups exceed threshold. Dead zones during monitoring \= those hours going silent. |

## **3.4 ContextualTextureProfile (L2 per archetype)**

One profile per L1 DBSCAN archetype. Texture is always evaluated relative to the matched L1 context, not globally.

| Field | Description |
| :---- | :---- |
| archetype\_id | Which L1 cluster this profile belongs to. |
| member\_days | Number of baseline days that contributed to this profile. |
| texture\_centroids / radii | Used when member\_days \>= 10\. K-means centroids (K chosen by silhouette score, max 3\) of the 22-feature L2 texture vector for days in this archetype. |
| texture\_mean / std | Fallback when member\_days \< 10\. Mean and std of each texture feature across member days. Anomaly scored as deviation in std units. |
| tolerance\_factor | Scalar multiplier on anomaly thresholds. Computed as mean intra-archetype texture variance. High-variance archetypes (exam, travel) get wider tolerance automatically. |

## **3.5 The 22-Feature L2 Texture Vector**

Built daily during monitoring. Normalized against the matched archetype's texture profile, not a global baseline.

| Group | Features | What degrades in depression |
| :---- | :---- | :---- |
| Temporal anchoring (4) | time\_in\_primary\_window\_ratio, temporal\_anchor\_deviation, first\_pickup\_hour\_deviation, rhythm\_dissolution\_score | Rhythm scatters — apps used at random hours with no structure |
| Session quality (5) | weighted\_abandon\_rate, deep\_session\_ratio, micro\_session\_ratio, session\_duration\_collapse, interaction\_density\_ratio | Sessions shorten and abort — nothing holds attention |
| Agency & initiation (4) | self\_open\_ratio, notification\_open\_rate, notification\_ignore\_rate, pickup\_burst\_rate | Initiation collapses — only notification-reactive or compulsive |
| Attention coherence (4) | app\_switching\_rate, app\_cooccurrence\_consistency, distinct\_apps\_ratio, session\_context\_match | Purposeful app sequences dissolve into fragmented random browsing |
| Rhythm & structure (3) | daily\_rhythm\_regularity, weekday\_weekend\_alignment, dead\_zone\_count | Structure flattens — every day becomes identical and featureless |
| Notification relationship (2) | notification\_response\_latency\_shift, notification\_to\_session\_ratio | Reactive ratio rises or latency spikes — loss of agency over phone use |

# **4\. Pipeline Phases**

| 0 | Android Data Collection Runs continuously on device |
| :---: | :---- |

### **What runs and when**

* L1 aggregate collector — midnight batch job. Reads UsageStatsManager, GPS/activity APIs, call log, calendar. Computes 29 feature values for the completed day. Writes one row to local SQLite daily\_features table.

* L2 session event logger — fires on every app foreground/background transition via AccessibilityService. Records: app\_package, open\_timestamp\_ms, close\_timestamp\_ms, trigger (SELF | NOTIFICATION | SHORTCUT | WIDGET), interaction\_count.

* Notification event logger — NotificationListenerService. Records: app\_package, arrival\_timestamp\_ms, action (TAP | DISMISS | IGNORE), tap\_latency\_minutes (null if not tapped).

* DataSyncService — runs nightly after L1 collection. Syncs all new rows to backend via encrypted POST. Buffers locally if offline; syncs on next connectivity.

### **New permissions required vs existing**

| Permission / API | Status |
| :---- | :---- |
| UsageStatsManager | Existing — already used by L1 PersonalityDataCollector |
| AccessibilityService (foreground/background events) | New — required for session event logging. User must grant in Accessibility settings. |
| NotificationListenerService | New — required for notification DNA. User must grant in Notification access settings. |
| PACKAGE\_USAGE\_STATS | Existing |
| ACCESS\_FINE\_LOCATION | Existing |

| The only new Android instrumentation is the session event logger and notification listener. Everything else is already captured. The entire richness of L2 rests on these two additions. |
| :---- |

**Output tables written:**

| daily\_features (29 cols) | session\_events | notification\_events |
| :---: | :---: | :---: |

| 1 | Baseline Establishment Days 1–28 (low confidence) → days 28–90 (full confidence) |
| :---: | :---- |

### **Confidence tiers**

| Days elapsed | Action |
| :---- | :---- |
| Day 28 | Build preliminary baseline. Flag all profiles as LOW\_CONFIDENCE. Use wider tolerance bands. Already better than nothing. |
| Day 60 | Rebuild all profiles from scratch using 60-day history. Upgrade to MEDIUM\_CONFIDENCE. DBSCAN clusters are more reliable. |
| Day 90 | Final rebuild. Freeze anchor clusters permanently. Upgrade to HIGH\_CONFIDENCE. Switch to rolling candidate cluster logic for anything new. |

### **Step 1.1 — PersonalityVector construction**

* Compute mean and std-dev of each of 29 features across all baseline days.

* Store as PersonalityVector (means) \+ variances dict (std-devs).

* These are frozen as the anchor reference. Never updated after baseline.

### **Step 1.2 — Per-app AppDNA construction**

* For each app appearing in session\_events 3+ times during baseline:

  * Build (7, 24\) usage heatmap: for each day-of-week × hour combination, compute mean minutes from session events.

  * Compute primary\_time\_range: find the smallest contiguous hour window containing 80% of total usage.

  * Compute time\_concentration\_ratio and its day-to-day std.

  * Compute session signature: mean, std, p10, p90 of session duration; abandon\_rate and its std.

  * Compute trigger DNA: fraction of sessions with each trigger type; notification response latency distribution.

  * Compute pre/post transition dicts from session sequence logs.

  * Compute weekday vs weekend split for session count and duration.

  * Compute daily\_use\_consistency and max\_gap\_days.

### **Step 1.3 — PhoneDNA construction**

* Aggregate across all session events (not per-app):

  * First and last pickup hour distribution from session\_events timestamps.

  * Pickup burst rate from inter-event timing.

  * Session duration 5-bin histogram.

  * Daily rhythm regularity via autocorrelation of hourly pickup vector.

  * Notification relationship metrics from notification\_events.

  * App co-occurrence matrix from sessions within same 30-minute windows.

  * Weekday/weekend delta: L1 norm of mean feature difference.

  * Historically active hours: hours with mean pickups \> threshold.

### **Step 1.4 — L1 context clustering (DBSCAN)**

* Build 12-feature L1 daily vector for each baseline day: sleep\_duration, wake\_time, sleep\_time, daily\_displacement\_km, location\_entropy, places\_visited, calls\_per\_day, conversation\_duration, screen\_time, unlock\_count, social\_app\_ratio, dark\_duration\_hours.

* Normalize each feature to \[0,1\] using person's own baseline min/max.

* Compute covariance matrix for Mahalanobis distance metric.

* Determine DBSCAN epsilon per person: sort distances to k-th nearest neighbor (k=3), find elbow of the k-distance graph.

* Run DBSCAN with computed epsilon, min\_samples=3, metric='mahalanobis'.

* Extract K cluster centroids (mean of member days) and radii (max intra-cluster Mahalanobis distance).

* Label outlier days (DBSCAN noise points) — days that fit no archetype. Store their vectors separately; they provide no clustering power but may later become seed points for new clusters.

### **Step 1.5 — L2 contextual texture profiles (K-means per archetype)**

* For each L1 archetype from DBSCAN:

  * Collect the subset of baseline days assigned to this archetype.

  * Build 22-feature L2 texture vector for each member day.

  * If member\_days \>= 10: run K-means with K=2 and K=3. Choose K with silhouette score. If improvement \< 0.05, stay K=2. Store texture\_centroids and texture\_radii.

  * If member\_days \< 10: compute texture\_mean and texture\_std as fallback. No clustering.

  * Compute tolerance\_factor as mean intra-archetype texture variance. High-variance archetypes get proportionally wider thresholds.

### **Step 1.6 — Detector calibration**

* Run L1 anomaly scoring retroactively against all baseline days.

* Compute mean and std of baseline anomaly scores.

* If baseline mean \> 0.30 (noisy user): raise PEAK\_EVIDENCE\_THRESHOLD and PEAK\_SUSTAINED\_THRESHOLD\_DAYS proportionally. Cap at \[2.71, 6.0\] and \[5, 12\] respectively.

* ANOMALY\_SCORE\_THRESHOLD (0.38) is never changed.

**Outputs stored to person\_profile table:**

| PersonalityVector | AppDNA\[\] per app | PhoneDNA | K anchor clusters |
| :---: | :---: | :---: | :---: |
| **L2 texture profiles** | **calibrated thresholds** |  |  |

| 2 | Daily Scoring — L1 Pipeline Runs every night after midnight aggregate is written |
| :---: | :---- |

### **Step 2.1 — Deviation magnitude**

For each of the 29 features:

z\_raw \= (today\_value \- baseline\_mean) / baseline\_std

z\_weighted \= z\_raw \* FEATURE\_META\[feature\]\['weight'\]

* baseline\_std defaults to 1.0 if baseline has no stored variance for a feature.

* Weights range from 0.4 (storageUsedGB, quasi-static) to 1.6 (sleepDurationHours, highest clinical weight).

* Output: dict of 29 weighted z-scores.

### **Step 2.2 — Deviation velocity (EWMA)**

For each feature, using 7-day rolling window:

ewma\_t \= 0.4 \* current\_value \+ 0.6 \* previous\_ewma

slope \= (ewma\_last \- ewma\_first) / window\_length

velocity \= slope / baseline\_mean  (normalized by baseline)

* alpha \= 0.4: recent days weighted more; detects accelerating trends.

* Output: dict of 29 normalized velocity slopes.

### **Step 2.3 — Composite L1 score**

magnitude\_score \= mean(|all weighted z-scores|) / 3.0   → capped at 1.0

velocity\_score  \= mean(|all velocities|) \* 10.0         → capped at 1.0

L1\_score \= 0.7 \* magnitude\_score \+ 0.3 \* velocity\_score

* Division by 3.0: a 3-sigma event across all features maps to magnitude\_score \= 1.0.

* Multiply by 10: normalizes small slope values into comparable range with magnitude.

* 70/30 split: current state weighted more than trend, but trend contributes meaningfully (catches slow drift).

**Outputs:**

| 29 weighted z-scores | 29 velocity slopes | L1 composite score \[0,1\] |
| :---: | :---: | :---: |

| 3 | Daily Scoring — L2 Pipeline Runs in parallel with L1, same nightly batch |
| :---: | :---- |

### **Step 3.1 — Context coherence**

* Build today's L1 daily vector (same 12 features as used in DBSCAN).

* Compute Mahalanobis distance from today's vector to all K anchor centroids.

  coherence \= max(0, 1.0 \- (nearest\_distance / (context\_radius \* 1.5)))

* If within 1.5× the nearest cluster radius: matched context, coherence \> 0\.

* If all centroids are beyond 1.5× their radii: coherence ≈ 0, potential new pattern.

* matched\_context\_id \= index of nearest centroid, or \-1 if no match.

### **Step 3.2 — Rhythm dissolution (KL divergence)**

* For each app active today with a baseline AppDNA:

  * Build today's hourly usage distribution (24-bin histogram, normalized to sum 1).

  * Retrieve the baseline heatmap row for today's day-of-week from AppDNA.

  * Add epsilon (1e-9) to both distributions to avoid log(0). Renormalize baseline.

      kl \= scipy.stats.entropy(today\_dist \+ 1e-9, baseline\_dist \+ 1e-9)

  * Aggregate weighted by each app's historical importance (avg\_session\_minutes × sessions\_per\_active\_day).

  rhythm\_dissolution \= clip(weighted\_mean\_kl / 3.0, 0, 1\)

* Conditioning: if matched\_context\_id \>= 0, compare against that archetype's heatmap, not the global one.

### **Step 3.3 — Session incoherence**

Three sub-signals, averaged:

* Abandon spike: for each active app, compute today's abandon\_rate. Delta vs baseline abandon\_rate, floored at 0\.

* Duration collapse: for apps with baseline avg\_session\_minutes \> 5, compute today\_avg / baseline\_avg. score \= max(0, 1 \- ratio). Captures lost capacity for sustained engagement.

* Trigger shift: for each active app, compute today's self\_open\_ratio. score \= max(0, baseline\_self\_ratio \- today\_self\_ratio). Rising \= shifting from intentional to reactive use.

  session\_incoherence \= mean(\[mean(abandon\_deltas), mean(duration\_collapses), mean(trigger\_drops)\])

### **Step 3.4 — New DNA pattern check**

* If coherence \< 0.25 (no matching L1 archetype) AND session\_incoherence \< 0.3 (texture still healthy):

  * Flag as candidate\_new\_pattern \= True.

  * This day will be handled by candidate cluster evaluation (Phase 5\) rather than normal evidence accumulation.

  * If coherence \< 0.25 AND session\_incoherence \>= 0.3: unfamiliar AND degraded — strongest clinical signal. No candidate evaluation needed.

### **Step 3.5 — L2 modifier computation**

suppression   \= coherence \* 0.85

amplification \= (rhythm\_dissolution \* 0.6 \+ session\_incoherence \* 0.4) \* 1.5

modifier      \= clip(1.0 \- suppression \+ amplification, 0.15, 2.0)

| Modifier range | Interpretation |
| :---- | :---- |
| 0.15 – 0.5 | Strongly suppress. High coherence to known context, healthy texture. L1 anomaly is context-driven, not clinical. |
| 0.5 – 0.9 | Moderate suppression. Mostly matches a known context with some texture degradation. |
| 0.9 – 1.1 | Neutral. Mixed signals. L1 runs at face value. |
| 1.1 – 1.5 | Moderate amplification. Unfamiliar pattern or rhythm dissolution beginning. |
| 1.5 – 2.0 | Strong amplification. Unknown context with degraded texture — strongest clinical signal. |

**Outputs:**

| coherence \[0,1\] | matched\_context\_id | rhythm\_dissolution \[0,1\] | session\_incoherence \[0,1\] |
| :---: | :---: | :---: | :---: |
| **L2 modifier \[0.15–2.0\]** | **candidate\_flag bool** |  |  |

| 4 | Evidence Engine Stateful — persists across days in evidence\_state table |
| :---: | :---- |

### **Step 4.1 — Effective score**

effective\_score \= L1\_score \* L2\_modifier

* This is the only value that enters the accumulator. L1 and L2 are combined here and not separately tracked in the accumulator.

### **Step 4.2 — Accumulation (anomalous day)**

If effective\_score \> ANOMALY\_SCORE\_THRESHOLD (0.38):

sustained\_deviation\_days \+= 1

evidence\_accumulated \+= effective\_score \* (1.0 \+ sustained\_deviation\_days \* 0.1)

* The multiplier (1 \+ days × 0.1) creates exponential compounding. Day 1 adds \~1×, day 5 adds \~1.5×, day 10 adds \~2×.

* A 7-day sustained episode accumulates substantially more evidence than 7 scattered anomalous days — even with identical daily scores.

### **Step 4.3 — Decay (normal day)**

If effective\_score \<= ANOMALY\_SCORE\_THRESHOLD:

sustained\_deviation\_days \= max(0, sustained\_deviation\_days \- 1\)

evidence\_accumulated     \= evidence\_accumulated \* 0.92

* 8% daily decay: evidence does not vanish instantly. Takes \~9 normal days to halve accumulated evidence.

* This means a genuine recovery curve is distinguishable from a normal fluctuation mid-episode.

### **Step 4.4 — Peak tracking**

if evidence\_accumulated \> max\_evidence: max\_evidence \= evidence\_accumulated

if sustained\_deviation\_days \> max\_sustained\_days: max\_sustained\_days \= sustained\_deviation\_days

if L1\_score \> max\_anomaly\_score: max\_anomaly\_score \= L1\_score

* Peak values are used exclusively for retrospective clinical prediction (Phase 8). They are never reset during monitoring.

**Persistent state written to evidence\_state table:**

| effective\_score | sustained\_deviation\_days | evidence\_accumulated | max\_evidence |
| :---: | :---: | :---: | :---: |
| **max\_sustained\_days** | **max\_anomaly\_score** |  |  |

| 5 | Candidate Cluster Evaluation Triggered when candidate\_flag \= True from L2 pipeline |
| :---: | :---- |

This phase handles the case where L1 is anomalous but L2 texture is coherent — indicating a potentially new behavioral archetype (exam period, new job, travel) rather than a clinical episode.

### **Step 5.1 — Open candidate window**

* Pause evidence accumulation for this person.

* Create candidate\_state record: open\_timestamp, status \= EVALUATING.

* Buffer today's L1 daily vector and L2 texture vector.

* Do not generate alert above green. Do not write evidence updates.

### **Step 5.2 — Days 1–3: Hold and observe**

* Each subsequent day: append L1 and L2 vectors to candidate buffer.

* Track daily texture quality: abandon\_rate, self\_open\_ratio, rhythm\_dissolution per day.

* Do not accumulate evidence. Do not alert.

### **Step 5.3 — Days 4–7: Evaluate texture quality**

| Condition | Outcome |
| :---- | :---- |
| Texture quality holds: session\_incoherence \< 0.35 on majority of candidate days AND no monotonic degradation trend | Promote to real cluster. Compute new centroid and radius from candidate vectors. Add to anchor clusters. Retroactively CLEAR all held evidence. Suppress L1 going forward when matching this cluster. |
| Texture degrades: session\_incoherence trending upward OR \> 0.35 on majority of days | Classify as clinical onset. Retroactively RELEASE all held evidence at full weight. Resume normal accumulation at full rate. The paused days hit the accumulator simultaneously. |
| Texture monotonically degrades (even if starting low) | Tiebreaker rule: any monotonic degradation during the window defaults to clinical, regardless of starting value. Prevents slow-onset depression from masquerading as a new healthy archetype. |

| The retroactive release is critical. Detection sensitivity is not lost during the evaluation window — the decision is deferred, not discarded. If the window turns clinical, all those days of evidence arrive at once. |
| :---- |

| 6 | Alert Determination Runs after evidence engine, same nightly batch |
| :---: | :---- |

### **Step 6.1 — Sustained gate (absolute)**

* No escalation above green unless:

  * sustained\_deviation\_days \>= SUSTAINED\_THRESHOLD\_DAYS (5), OR

  * evidence\_accumulated \>= EVIDENCE\_THRESHOLD (2.0)

* A single day with effective\_score \= 0.95 stays green. This gate is absolute and cannot be overridden by score magnitude alone.

### **Step 6.2 — Critical feature deviation**

critical\_deviation \= max(|z\_score| for f in CRITICAL\_FEATURES)

Critical features: sleepDurationHours, dailyDisplacementKm, screenTimeHours, socialAppRatio, wakeTimeHour, callsPerDay, upiTransactionsToday, totalAppsCount.

### **Step 6.3 — Alert level assignment**

| Level | Condition (after sustained gate is cleared) |
| :---- | :---- |
| Green | score \< 0.35 AND critical\_deviation \< 2.0 SD |
| Yellow | score \< 0.50 AND critical\_deviation \< 2.5 SD |
| Orange | score \< 0.65 OR critical\_deviation \< 3.0 SD |
| Red | score \>= 0.65 AND critical\_deviation \>= 3.0 SD |

### **Step 6.4 — Pattern type detection**

Look at last 7 days of deviation history. Compute mean (m) and std (s) of daily mean-absolute-deviation values.

| Pattern | Condition |
| :---- | :---- |
| stable | m \< 0.5 |
| rapid\_cycling | s \> 1.0 AND m \> 0.5 — BPD signature |
| acute\_spike | m \> 1.5 AND s \< 0.8 — elevated and holding |
| gradual\_drift | |slope of avg\_devs over 7 days| \> 0.1 — depression signature |
| mixed\_pattern | Elevated but no clear shape |

### **Step 6.5 — Flagged features and top deviations**

* Flagged features: list all features with |weighted z-score| \> 1.5, formatted as human-readable strings: 'sleepDurationHours (2.41 SD)'.

* Top 5 deviations: dict of the 5 features with largest absolute weighted z-scores.

**Outputs:**

| alert\_level | pattern\_type | flagged\_features\[\] | top\_5\_deviations |
| :---: | :---: | :---: | :---: |

| 7 | Daily Output Reports Written to DB each night, consumed by UI and System 2 |
| :---: | :---- |

### **AnomalyReport (raw, for downstream systems)**

| Field | Type / value |
| :---- | :---- |
| timestamp | datetime — time of analysis |
| overall\_anomaly\_score | float — L1 composite score (pre-modifier) |
| effective\_score | float — L1 × L2 modifier |
| feature\_deviations | Dict\[str, float\] — all 29 weighted z-scores |
| deviation\_velocity | Dict\[str, float\] — all 29 EWMA slopes |
| l2\_modifier | float — the full L2 modifier value |
| matched\_context\_id | int — which L1 archetype matched, or \-1 |
| coherence\_score | float |
| rhythm\_dissolution | float |
| session\_incoherence | float |
| alert\_level | str — green / yellow / orange / red |
| flagged\_features | List\[str\] |
| pattern\_type | str |
| sustained\_deviation\_days | int |
| evidence\_accumulated | float |

### **DailyReport (human-readable, for UI and clinicians)**

| Field | Type / value |
| :---- | :---- |
| day\_number | int — day since monitoring began |
| date | datetime |
| anomaly\_score | float — effective score |
| alert\_level | str |
| flagged\_features | List\[str\] — human-readable with SD values |
| pattern\_type | str |
| sustained\_deviation\_days | int |
| evidence\_accumulated | float |
| top\_deviations | Dict\[str, float\] — top 5 features |
| notes | str — auto-generated: 'Sustained deviation (7 days) | Evidence: 3.42 | Pattern: gradual\_drift | HIGH ALERT: ORANGE' |

| 8 | Final Prediction Generated at end of monitoring period or on clinical request |
| :---: | :---- |

### **Step 8.1 — had\_episode() check (retrospective, strict)**

had\_episode \= (max\_evidence \>= PEAK\_EVIDENCE\_THRESHOLD) OR

             (max\_sustained\_days \>= PEAK\_SUSTAINED\_THRESHOLD\_DAYS)

* Uses peak values, not current state. An episode that partially recovered still registers.

* PEAK thresholds are stricter than real-time thresholds and are calibrated per person during baseline.

### **Step 8.2 — Pattern analysis**

* Look at last 7 days of full\_anomaly\_history.

* If std of recent scores \> 0.15: pattern \= 'unstable/cycling'

* If mean of recent scores \> 0.5: pattern \= 'persistent\_elevation'

* Otherwise: pattern \= 'stable'

### **Step 8.3 — Confidence score**

confidence \= min(0.95, monitoring\_days / 30 \* 0.8 \+ 0.15)

* Grows with monitoring duration. Starts at 0.15 at day 1, reaches 0.95 at day 35+. Caps at 95% — the system is never fully certain.

### **Step 8.4 — Recommendation tiers**

| Tier | Condition and meaning |
| :---- | :---- |
| NORMAL | No sustained anomaly. No significant evidence. No clinical action needed. |
| WATCH | max\_evidence \> WATCH\_EVIDENCE\_THRESHOLD (1.5) but below PEAK\_EVIDENCE\_THRESHOLD. Some periodic evidence of deviation. Suggest extending monitoring or additional check-ins. |
| MONITOR | had\_episode() \= True via sustained days gate. Significant sustained deviation detected. Clinical follow-up recommended. |
| REFER | had\_episode() \= True AND max\_evidence \>= 4.0. Very strong evidence of sustained behavioral deviation. Immediate clinical evaluation recommended. |

**FinalPrediction output:**

| patient\_id | sustained\_anomaly bool | confidence 0–0.95 | pattern\_identified |
| :---: | :---: | :---: | :---: |
| **evidence\_summary dict** | **recommendation string** |  |  |

# **5\. Threshold Reference**

All thresholds, their default values, which are calibrated per-person, and which are fixed.

| Threshold | Default value | Calibrated? |
| :---- | :---- | :---- |
| ANOMALY\_SCORE\_THRESHOLD | 0.38 | Never — fixed for all users |
| SUSTAINED\_THRESHOLD\_DAYS | 5 | No |
| EVIDENCE\_THRESHOLD (real-time) | 2.0 | No |
| PEAK\_EVIDENCE\_THRESHOLD (retrospective) | 7.0 | Yes — raised if baseline mean \> 0.30 |
| PEAK\_SUSTAINED\_THRESHOLD\_DAYS | 10 | Yes — raised if baseline mean \> 0.30 |
| WATCH\_EVIDENCE\_THRESHOLD | 1.5 | No |
| L2 candidate texture threshold | 0.35 session\_incoherence | No |
| L2 coherence match radius | 1.5 × cluster radius | Implicitly yes — radius comes from DBSCAN per person |
| Minimum app baseline appearances | 3 sessions | No |
| Minimum archetype member days for K-means | 10 days | No |
| DBSCAN min\_samples | 3 | No |
| Candidate window length | 7 days | No |
| Evidence decay rate | 0.92 per normal day | No |

# **6\. Dependencies**

## **6.1 Android (Java / Kotlin)**

| Dependency | Purpose |
| :---- | :---- |
| AccessibilityService API | Session event logging — foreground/background transitions |
| NotificationListenerService | Notification DNA — arrival, action, latency |
| UsageStatsManager | Daily aggregate feature collection (existing) |
| Room (SQLite ORM) | Local persistence of all event tables |
| WorkManager | Scheduled nightly L1 collection and sync jobs |

## **6.2 Python backend**

| Package | Purpose |
| :---- | :---- |
| numpy | Array math throughout — heatmaps, distance computation, EWMA |
| scipy | KL divergence (scipy.stats.entropy), Mahalanobis distance |
| scikit-learn | DBSCAN, KMeans, silhouette\_score, pairwise\_distances |
| pandas | DataFrame manipulation for baseline and monitoring data |
| dataclasses | PersonalityVector, AppDNA, PhoneDNA, all report structures |
| collections.deque | Rolling windows in evidence engine and velocity computation |
| json | Serialization of DNA structures to person\_profile SQLite column |

# **7\. Suggested File Structure**

## **Android**

app/

  collectors/

    PersonalityDataCollector.kt     ← existing L1 nightly batch

    SessionEventLogger.kt           ← new: AccessibilityService listener

    NotificationEventLogger.kt      ← new: NotificationListenerService

  sync/

    DataSyncService.kt              ← nightly upload to backend

  db/

    AppDatabase.kt                  ← Room DB with all 4 tables

    DailyFeaturesDao.kt

    SessionEventDao.kt              ← new

    NotificationEventDao.kt         ← new

## **Python backend**

system1/

  data\_structures.py               ← PersonalityVector, AppDNA, PhoneDNA,

                                      ContextualTextureProfile, all report classes

  feature\_meta.py                  ← FEATURE\_META weights, CRITICAL\_FEATURES, ALL\_FEATURES

  baseline/

    baseline\_builder.py            ← orchestrates all baseline construction steps

    l1\_clusterer.py                ← DBSCAN with Mahalanobis distance

    l2\_texture\_builder.py          ← K-means per archetype, fallback mean/std

    app\_dna\_builder.py             ← per-app DNA construction

    phone\_dna\_builder.py           ← device-level DNA construction

    detector\_calibration.py        ← retroactive threshold calibration

  scoring/

    l1\_scorer.py                   ← deviation magnitude \+ velocity \+ composite

    l2\_scorer.py                   ← coherence \+ rhythm\_dissolution \+ session\_incoherence \+ modifier

  engine/

    evidence\_engine.py             ← stateful accumulation, decay, peak tracking

    candidate\_cluster.py           ← 7-day evaluation window, promote/reject logic

    alert\_engine.py                ← sustained gate, level assignment, pattern detection

    prediction\_engine.py           ← final retrospective prediction

  output/

    reporter.py                    ← assembles AnomalyReport \+ DailyReport

  simulation/

    synthetic\_data\_generator.py    ← existing scenario simulator

    system1.py                     ← existing end-to-end simulation runner

# **8\. Recommended Implementation Order**

Ordered to allow partial validation at each step before the next is built.

1. data\_structures.py — define all dataclasses first. Everything else imports from here.

2. feature\_meta.py — weights, critical features list.

3. app\_dna\_builder.py \+ phone\_dna\_builder.py — build from session event data. Validate against synthetic sessions.

4. l1\_clusterer.py — DBSCAN with Mahalanobis. Validate cluster shapes on known synthetic scenarios.

5. l2\_texture\_builder.py — K-means per archetype. Validate silhouette scores.

6. baseline\_builder.py — orchestrates steps 3–5. Validate full baseline profile on 28-day synthetic data.

7. detector\_calibration.py — retroactive threshold calibration. Unit test with noisy vs clean synthetic baselines.

8. l1\_scorer.py — z-scores \+ velocity \+ composite. This already exists in system1.py, extract and refactor.

9. l2\_scorer.py — coherence \+ rhythm\_dissolution \+ session\_incoherence \+ modifier. Validate modifier values on known-exam and known-depression synthetic days.

10. evidence\_engine.py — stateful accumulation. Unit test with known episode sequences.

11. candidate\_cluster.py — 7-day evaluation window. Test with synthetic exam-onset and depression-onset sequences.

12. alert\_engine.py — sustained gate \+ levels \+ pattern detection.

13. prediction\_engine.py — retrospective prediction. Validate all 4 recommendation tiers.

14. reporter.py — assemble final output structures.

15. Android SessionEventLogger \+ NotificationEventLogger — final step, hardware-dependent. Mock with synthetic event sequences during all prior development.

| Steps 1–14 can be developed and validated entirely against synthetic data. The Android instrumentation (step 15\) only needs to be real for production — mock session events are sufficient for building and testing the entire Python pipeline. |
| :---- |

# **9\. Key Design Decisions and Rationale**

| Decision | Rationale |
| :---- | :---- |
| L2 is a modifier, not a separate score | Keeps a single evidence accumulator. L1 and L2 are not competing — L2 tells L1 what to do with what it found. |
| DBSCAN for L1 clustering, K-means for L2 | L1 needs variable K (person's life has unknown number of behavioral modes). L2 has knowable K (healthy vs degraded texture). Different problems, different algorithms. |
| Mahalanobis distance in DBSCAN | Corrects for feature correlation without collapsing features into abstract PCA components. Interpretability preserved. |
| L2 texture conditioned on L1 archetype | Prevents weekday/weekend variance from masking healthy vs degraded texture signal. Each archetype has its own texture baseline. |
| Exponential evidence compounding | A sustained episode should accumulate disproportionately more evidence than scattered bad days, because sustained change is clinically distinct from daily noise. |
| 8% evidence decay on normal days | Prevents false positives from erasing quickly, while ensuring genuine recovery is distinguishable from noise. |
| Two detection modes: real-time vs retrospective | Real-time is lenient (alerts early, catches things). Retrospective is strict (committed clinical claim). Different error tolerance for different audiences. |
| Candidate cluster evaluation window | The core mechanism that makes the system tolerable in real life. New healthy archetypes are not pathologized. Depression masquerading as new archetype is caught by texture degradation gate. |
| Anchor clusters never overwritten | System always remembers who the person was at baseline. Drift away from all anchors is itself a signal worth preserving over months. |
| No dimensionality reduction | Feature dimensionality is low (12 for L1 clustering, 22 for L2). PCA would harm interpretability without improving clustering. Mahalanobis handles correlation. |
| Confidence capped at 95% | The system is never fully certain. A 95% cap reflects the irreducible uncertainty in passive behavioral monitoring for clinical purposes. |

