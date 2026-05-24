# MHealth Project Context & Architecture

This document serves as the canonical central reference and context for the Mental Health Detection ML (MHealth) codebase. It maps out the active state of the project, its core systems, mathematical heuristics, features, data flow, and file structures.

---

## 1. Project Philosophy & System Architecture

The MHealth system is designed as a **passive, risk-averse decision-support tool** to detect early warning signs of mental health disorders (depression, bipolar/BPD, and schizophrenia) using smartphone telemetry. It does not provide diagnoses; instead, it triggers **caregiver check-ins** and **in-app reflection prompts** based on robust anomaly detection and clinical prototype matching.

The codebase is split into two distinct execution systems linked via a coupling adapter:
*   **System 1 (On-Device Anomaly Detector)**: Parallel dual-layer monitoring. Runs a surface-level layer (Layer 1, 29D vector) and a micro-level layer (Layer 2, app session and notification telemetry) to compute a daily anomaly score, calibrate personal thresholds, and discover new healthy lifestyle contexts via rolling candidate windows.
*   **System 2 (Clinical Disorder Matcher)**: Diagnostic characterization pipeline. Triggered when System 1 alarms. Pre-filters transient stress via a Life Event Filter, matches deviation profiles geometrically against clinical disorder prototypes, applies expert clinical overrides, and validates classifications using temporal shape timeseries analysis.

```
                       [Device Telemetry Sensors]
                                    │
                                    ▼
                        [Data Collection & Fusion]
                         /                      \
       Onboarding: Days 1-28                  Monitoring: Day 28+
                     /                              \
        [3-Gate Baseline Screener]            [System 1: Anomaly Detector]
        ├── Gate 1: Pop. Anchor (Day 7)       ├── Layer 1: Surface Scorer (Z-Score + EWMA)
        ├── Gate 2: WoW Stability (Day 14-21) ├── Layer 2: Micro Scorer (DBSCAN + Heatmaps)
        └── Gate 3: Proto Proximity (Day 28)  ├── Candidate Window Context Discovery (7-day)
                     │                        └── Compounding Evidence Engine (Comp. alpha=0.15)
         Passes      │ Fails                        │
         ┌───────────┴──────────┐                   ▼
         ▼                      ▼             S1 Alert Threshold Triggered
     [Frame 2]              [Frame 1]               │
     Personal               Synthetic               ▼
     Baseline               Population         [System 2: Disorder Matcher]
     Anchored               Anchored          ├── Life Event Filter (Stage 0)
         │                      │             ├── Frame Selector (Gate Outcome)
         └──────────┬───────────┘             ├── Prototype Matcher (Euclidean/Cosine)
                    │                         ├── Clinical Guardrails Overrides (Persistence Gated)
                    ▼                         ├── Temporal Shape Validator (Autocorr/Drift)
              Disorder Match                  └── Explainability Narrative & Radar
```

---

## 2. Core Telemetry: The 29-Feature Personality Vector

Telemetry features are gathered on-device and mapped directly to a 29-dimensional vector, grouped as follows:

| Group | Feature Key | Source & Notes |
|---|---|---|
| **A: Screen & App Activity** | `screenTimeHours` | Foreground app durations via `UsageStatsManager` |
| | `unlockCount` | Lock screen unlock count |
| | `appLaunchCount` | App launches debounced with a 1.5s delay |
| | `notificationsToday` | Notification interruptions from non-system apps |
| | `socialAppRatio` | Social app time / total screen time |
| **B: Communication** | `callsPerDay` | Total calls (incoming, outgoing, missed) via `CallLog` |
| | `callDurationMinutes`| Total call duration in minutes |
| | `uniqueContacts` | Count of unique phone numbers called/received |
| | `conversationFrequency`| Calls per unique contact (depth vs. breadth) |
| **C: Location & Movement** | `dailyDisplacementKm` | Distance traveled via **Grid-Cell Transition Method** |
| | `locationEntropy` | Shannon entropy of **wall-clock hours spent in grid cells** |
| | `homeTimeRatio` | Fraction of 24h day spent in home grid cell |
| | `placesVisited` | Unique discrete ~110m grid cells visited today |
| **D: Sleep & Circadian** | `wakeTimeHour` | Hour of first meaningful usage post screen-off gap |
| | `sleepTimeHour` | Hour phone went silent before main sleep gap |
| | `sleepDurationHours` | Longest contiguous screen-off gap (overnight 18h window) |
| | `darkDurationHours` | Total absolute screen-off hours across the day |
| **E: System Usage** | `chargeDurationHours` | Cumulative daily charging time in hours |
| | `memoryUsagePercent` | RAM occupancy percent |
| | `networkWifiMB` | MegaBytes consumed over Wi-Fi today |
| | `networkMobileMB` | MegaBytes consumed over Cellular networks today |
| | `storageUsedGB` | Internal device storage full (in GB) |
| **F: Behavioural Signals** | `totalAppsCount` | Absolute count of installed non-system apps |
| | `upiTransactionsToday`| Launch count of UPI apps (GPay, PhonePe, Paytm, etc.) |
| | `appUninstallsToday` | Count of apps uninstalled today |
| | `appInstallsToday` | Count of apps newly installed today |
| **G: Calendar & Engagement**| `calendarEventsToday` | Meetings or reminders intersecting today |
| | `mediaCountToday` | New photos/videos added to gallery |
| | `downloadsToday` | Direct scan of the `Downloads` directory |
| | `musicTimeMinutes` | Debounced background music/podcast audio playing time |

### Sophisticated Calculation Heuristics
*   **3-Signal Sleep Fusion Heuristic**: Sleep duration is extracted in a restricted overnight 18-hour window (6:00 PM to 12:00 PM next day) to avoid confusing daytime stationary desk time with sleep. It detects the longest screen-off gap, merges screen micro-wakes (<5 min, e.g. checking time at 3:00 AM), and fuses Do Not Disturb (DND) events (adjusting Sleep Time to DND onset and Wake Time to DND offset).
*   **Grid-Cell Transition Method**: GPS positions are mapped into ~110m square grid cells. Distance is added *only* when the user genuinely transitions to a new cell, resolving the "phantom distance" bug where stationary GPS drift accumulated kilometers on a desk.
*   **Location Entropy (Time-based)**: Calculated using the actual **wall-clock hours spent in each grid cell** (rather than raw GPS ping counts, which artificially inflate when moving).
*   **Circular Time Normalizer**: Clocks wake/sleep hours continuously (e.g. 11:00 PM and 1:00 AM are mapped as 2 hours apart rather than 22 hours apart).

---

## 3. Onboarding: Establishing the Baseline (Days 1–28)

During the onboarding period, the system gathers daily surface features alongside micro-level session data to build the user's "Personal Normal" profile.

1.  **PersonalityVector (Step 1.1)**: Computes the mean and standard deviation of each L1 feature over the 28-day window. Standard deviation floors (e.g. 0.05) are enforced to prevent division-by-zero explosions during monitoring.
2.  **AppDNA & PhoneDNA (Steps 1.2–1.3)**:
    *   **AppDNA**: Builds hourly usage probability distributions (24-bin usage heatmaps) for each app, grouped by day-of-week, alongside baseline metrics: `abandon_rate` (sessions < 45s and < 5 interactions), `avg_session_minutes`, and `self_open_ratio`.
    *   **PhoneDNA**: Extracts macro rhythm structures: active windows, historically active hours, pickup burst rates, daily rhythm regularity, and weekday-weekend delta.
3.  **L1 Anchor Clustering (Step 1.4)**: Runs **DBSCAN clustering** using a Mahalanobis distance metric on the 12-feature L1 clustering subspace. It establishes "behavioral archetypes" (centroids, radii, and covariance matrices). Epsilon is auto-determined via a k-distance elbow graph.
4.  **L2 Texture Profiles (Step 1.5)**: For each discovered DBSCAN cluster, it groups baseline days belonging to that archetype and builds a 22-dimensional micro-texture profile. If an archetype has $\ge 10$ member days, it fits **K-Means clustering** (with $K \in \{2, 3\}$, optimized by silhouette score) to capture sub-structures. Otherwise, it falls back to mean/std profile bounds.
5.  **3-Gate Baseline Screener**: Evaluates the onboarding period to detect if a user has pre-existing symptoms or an active disorder:
    *   *Gate 1 (Day 7 — Population Anchor)*: Flags if $\ge 3$ features exceed $2.5$ SD (uses $4.0$ SD for features with highly stable baselines to avoid demographic false flags).
    *   *Gate 2 (Days 14–21 — Stability Check)*: Measures week-over-week variance. Flags if observed drift $> 1.5\times$ population drift for $\ge 3$ features (signals bipolar/cycling).
    *   *Gate 3 (Day 28 — Proximity Check)*: Runs population-anchored prototype matching. Flags as contaminated if top match $\neq$ healthy and confidence $> 0.65$.
    *   *Self-Report Gate*: If self-report questionnaires (PHQ-9 or GAD-7) score $\ge 10$, triggers baseline contamination override (takes priority as gold standard).
    *   *Outcome*: Passed $\rightarrow$ Locks baseline, uses **Frame 2** (deviation from own baseline) for monitoring. Failed $\rightarrow$ Discards baseline, falls back to **Frame 1** (synthetic population healthy baseline) for monitoring.

---

## 4. Continuous Monitoring: System 1 (Day 28+)

System 1 operates daily as a parallel dual-layer scoring and evidence tracking engine.

### Layer 1: Surface Scorer (`L1Scorer`)
Calculates deviations from the established baseline vector $P_0$.
*   **Weighted Z-Score Deviation**: Uses Bayesian posterior updates (with a rolling learning rate) to update the baseline dynamically. Standard deviation is capped at a ceiling (default 4.0) to prevent outlier domination. Enforces asymmetric directionality dampening (e.g. positive improvements in sleep/steps are dampened by 0.1 so only degradation drives anomaly scores).
*   **EWMA Velocity**: Tracks the rate of change using an Exponentially Weighted Moving Average ($\alpha = 0.4$) over a 7-day window.
*   **Composite Anomaly Score**:
    $$\text{L1\_score} = 0.7 \times \text{Magnitude\_score} + 0.3 \times \text{Velocity\_score}$$
    *Magnitude* is the RMS of weighted Z-scores divided by 3.0 (capped at 1.0). *Velocity* is the RMS of normalized slopes multiplied by 10 (capped at 1.0).

### Layer 2: Micro Scorer (`L2Scorer`)
Acts as a context filter and signal modifier, suppressing scores on familiar days and amplifying them on degraded, unfamiliar days.
1.  **Context Coherence**: Normalizes today's 12-feature L1 vector to $[0,1]$ using baseline min/max, then calculates the Mahalanobis distance to all DBSCAN centroids.
    $$\text{coherence} = \max\left(0, 1.0 - \frac{\text{nearest\_distance}}{\text{radius} \times 1.5}\right)$$
    If outside all cluster radii ($1.5\times$), `matched_context_id = -1` and `coherence = 0.0`.
2.  **Rhythm Dissolution**: Computes KL-divergence between today's 24-bin hourly app usage distributions and baseline heatmaps for today's day-of-week, weighted by app importance.
3.  **Session Incoherence**: Average of three sub-signals:
    *   *Abandon Spike*: today's abandon rate minus baseline.
    *   *Duration Collapse*: $1.0 - (\text{today\_avg} / \text{baseline\_avg})$ for long-session apps.
    *   *Trigger Shift*: baseline self-open ratio minus today's self-open ratio.
4.  **L2 Modifier & Effective Score**:
    $$\text{suppression} = \text{coherence} \times 0.85$$
    $$\text{amplification} = (\text{rhythm\_dissolution} \times 0.6 + \text{session\_incoherence} \times 0.4) \times 1.5$$
    $$\text{L2\_modifier} = \text{clip}(1.0 - \text{suppression} + \text{amplification},\ 0.15,\ 2.0)$$
    $$\text{effective\_score} = \text{L1\_score} \times \text{L2\_modifier}$$

### Context Discovery: Candidate Cluster Evaluator
Allows baseline adaptation without losing clinical safety. If today's behavior is unfamiliar but structured (`coherence < 0.25` and `session_incoherence < 0.3`), the system opens a **7-day Candidate Window**:
*   **Hold & Observe (Days 1–3)**: Buffers daily vectors and pauses evidence accumulation.
*   **Evaluate Texture (Days 4–7)**:
    *   *Promote*: If session incoherence remains healthy ($< 0.35$ on majority) and no monotonic degradation, the window closes, a new anchor cluster centroid is appended to the DBSCAN state, and held evidence is permanently discarded.
    *   *Reject*: If session incoherence degrades ($> 0.35$ on majority or monotonic worsening), the window closes as a clinical onset, and all held evidence is retroactively released to the evidence engine!
*   **Grace Period**: Post-promotion of a new cluster, a 7-day grace period applies, suppressing new evidence accumulation to allow the user's profile to stabilize.

### Compounding Evidence Engine (`EvidenceEngine`)
Maintains a state machine to track cumulative deviation:
*   If `effective_score > threshold` (0.38):
    $$\text{evidence} += \text{effective\_score} \times (1.0 + \text{days\_sustained} \times 0.15) \times \text{trend\_factor}$$
    *   `trend_factor` is $1.0$ if worsening or flat; $0.5$ if stabilizing (lifestyle adaptation).
*   If `effective_score <= threshold`: consecutive days decay, and evidence decays by 8% per day ($\text{evidence} \times= 0.92$).
*   **Alert Threshold**: Triggered when `evidence_accumulated >= 2.0` (orange/red alert levels).

---

## 5. Diagnostic Characterization: System 2

Triggered when System 1 alarms, System 2 characterizes the clinical nature of the episode.

### 1. Stage 0: Life Event Filter
Prevents false-alarm clinical matches from transient events:
*   Dismisses the anomaly if $\le 3$ features co-deviate AND max deviation is $\le 3.0$ SD.
*   Dismisses if the anomaly resolves within 10 days, or if no feature exceeds a floor of 1.0 SD.

### 2. Prototype Matcher (Geometric Distance Classifier)
Matches user Z-score deviation vectors ($[-5.0, +5.0]$ clamped) against expert clinical disorder prototypes (Depression Type 1/2, Bipolar Depressive/Manic, Schizophrenia/Psychosis, Anxiety, and Healthy).
*   **Weighted Cosine Similarity**: Captures the directional shape of the deviation.
*   **Weighted Euclidean Distance**: Captures the magnitude, adding a **5.0x penalty for sign mismatches** on significant deviations ($> 0.3$ SD).
*   **Combined Match Score**:
    $$\text{Match\_score} = 0.6 \times \text{Cosine\_similarity} + 0.4 \times \frac{1.0}{1.0 + \text{Euclidean\_distance}}$$
*   **Confidence Tiers**: $\ge 0.75 \rightarrow$ HIGH; $0.55\text{–}0.75 \rightarrow$ LOW; $< 0.55 \rightarrow$ UNCLASSIFIED.

### 3. Clinical Override Guardrails
Explicit clinical rules overlaying the geometric matching for critical sparse-signal cases. Requires a temporal persistence gate (System 1 evidence $\ge 0.4$ or days sustained $\ge 3$) to prevent single-day false alarms:
*   *Social Withdrawal Override*: If $\ge 2$ communication features drop below -1.2 SD, or one drops below -1.2 SD and displacement drops below -1.5 SD, overrides normal matching to force a **Depression** match.
*   *Psychosis/Disorganization Override*: If $\ge 2$ psychosis-specific features exceed 1.5 SD and show inconsistent directions (mixed positive and negative signs), overrides normal matching to force a **Bipolar Depressive / Cycling** match.

### 4. Temporal Shape Validator
Analyzes the 60-day sliding window of anomaly scores to validate the classification:
*   *Monotonic Drift* ($R^2 > 0.6$, negative slope): Confirms **Depression**, downgrades Cycling.
*   *Oscillating* (autocorrelation lag 3-10 days $> 0.4$): Confirms **Cycling/Bipolar**, downgrades Depression.
*   *Chaotic* (high variance, low lag-1 autocorrelation): Confirms **Schizophrenia**, downgrades Depression.
*   *Episodic Spike* (recovers in 14 days): Confirms **Anxiety / Life Event**.
*   *Phase Flip* (sudden shift $> 3.0$ SD between weekly means): Confirms **Bipolar Manic**.
*   *Confidence Boost*: Boosts score by $1.2\times$ if compatible, downgrades by $0.6\times$ if contradictory.

---

## 6. Codebase Structure & Registry

The python intelligence engine resides in the Android app source directory:
`MHealth - Copy\app\src\main\python\`

```
├── app/src/main/python/
│   ├── system1/                    # SYSTEM 1: Anomaly Detection Engine
│   │   ├── baseline/               # Baseline profile & DNA builders
│   │   │   ├── app_dna_builder.py       # Tracks app usage profiles & heatmaps
│   │   │   ├── phone_dna_builder.py     # Tracks active hours, regularities
│   │   │   ├── baseline_builder.py      # Orchestrator for baseline profiling
│   │   │   ├── bayesian_baseline.py     # Rolling Bayesian updates during monitoring
│   │   │   ├── l1_clusterer.py          # Mahalanobis DBSCAN clustering
│   │   │   ├── l2_texture_builder.py    # Per-archetype K-Means micro-textures
│   │   │   └── detector_calibration.py  # Adaptive weight & ceiling calibration
│   │   ├── scoring/                # Layer 1 & Layer 2 Scorer pipelines
│   │   │   ├── l1_scorer.py             # Weighted z-scores, EWMA velocity & L1 score
│   │   │   └── l2_scorer.py             # Context Mahalanobis, KL rhythm, incoherence
│   │   ├── engine/                 # State machines and accumulators
│   │   │   ├── alert_engine.py          # Alert levels assignment & patterns
│   │   │   ├── evidence_engine.py       # Evidence accumulation, compounding & decay
│   │   │   ├── candidate_cluster.py     # 7-day candidate window context discovery
│   │   │   └── prediction_engine.py     # Retrospective prediction generator
│   │   ├── output/
│   │   │   └── reporter.py              # Formatting of daily & anomaly reports
│   │   ├── detector.py             # Facade orchestrator for System 1
│   │   ├── feature_meta.py         # Static weights, directionalities & metadata
│   │   ├── data_structures.py      # Structured dataclasses and type definitions
│   │   └── user_profile.py         # User demographics & self-reported scores
│   │
│   ├── system2/                    # SYSTEM 2: Clinical Disorder Matcher
│   │   ├── pipeline.py             # Orchestrates Screener -> Filter -> Matcher -> Validator
│   │   ├── baseline_screener.py    # 3-Gate Baseline Screener for contamination
│   │   ├── life_event_filter.py    # Pre-filter for transient stressors
│   │   ├── prototype_matcher.py    # Cosine/Euclidean matching with sign penalties
│   │   ├── temporal_validator.py   # Timeseries validation (drift, cycle, chaos)
│   │   ├── explainability.py       # Spider radar chart & patient narrative generator
│   │   ├── s1_s2_adapter.py        # camelCase ↔ snake_case translation & S1Input wrapper
│   │   └── config.py               # Literature norms, weights, clinical prototypes
│   │
│   ├── dna.py                      # Core sensor data-binding models
│   ├── dna_engine.py               # Extraction rules & data pipelines
│   ├── engine.py                   # On-device processing task loops
│   └── s1_profile.py               # Baseline manager & loader
```

---

### Update Log
*   **2026-05-22**: Rewrote context.md to match the modular layout of the updated codebase. Documented Mahalanobis DBSCAN, L2 Scorer mechanics (KL hourly divergence, incoherence sub-signals, modifier formula), 7-day Candidate Window lifecycle (Hold & Observe, Evaluate Texture, Promote vs. Reject), and the updated System 2 diagnostic pipeline.
*   **2026-05-23**: Conducted telemetry feature audit. Documented redundant features (system metrics, raw bytes, dark duration overlap) and proposed high-value clinical additions (steps, keystroke/touch dynamics, light exposure, circadian entropy, active vocal prosody check-ins).
*   **2026-05-23**: Completed `/grill-me` architectural design review. Documented the new symmetrical parallel L1/L2 PCA + Mean Shift pipeline and the "Variance-Bounded Compactness Test" post-filter for dynamic baseline adaptation and alert suppression.
*   **2026-05-24**: Resolved team build compatibility issues. Removed machine-specific `org.gradle.java.home` from project `gradle.properties`, resolved the Kotlin compiler `NULL` reference error in `AppDnaComputer.kt` by migrating to standard `JSONObject.NULL`, and optimized memory limits (Gradle to 2 GB, Kotlin to 1 GB) to protect developer PCs from RAM starvation freezes. Implemented a fully automated Cloud Build CI/CD pipeline using GitHub Actions (`.github/workflows/android.yml`) to compile and distribute debug APKs on push/PR and manual trigger, bypassing local hardware constraints.
*   **2026-05-24**: Successfully implemented the approved clinical telemetry vector and symmetrical PCA + Mean Shift pipeline. Solved the critical L2 Digital DNA baseline construction bugs: fixed the previous day's midnight raw session purge inside `MonitoringService.kt` by converting it into a rolling 60-day cleanup window using `deleteOlderThan()`, retaining raw history for proper baseline DNA building. Fixed the Python custom Mean Shift clustering bug in `s1_profile.py` by removing the post-PCA Z-score re-normalization that destroyed variance ratios, and tuned the quantile density bandwidth to `0.3` to match the successful branch. Fixed a raw slicing index alignment bug in `baseline_builder.py` by grouping sessions and notifications by actual calendar dates. Implemented the dynamic weight multipliers with a Weight Freezing Guardrail in `s1_profile.py` to lock clinical baseline weights post-onboarding and prevent clinical normalization bias. Staged, committed, and pushed changes, automatically triggering a cloud build and pre-release of the compiled `app-debug.apk`.
*   **2026-05-24**: Finalized the 30-feature telemetry refactoring by overhauling the remaining Python analytics scripts (`l1_scorer.py`, `s1_profile.py`, `baseline_builder.py`, `bayesian_baseline.py`) and the clinical matcher system (`config.py`, `life_event_filter.py`, `s1_s2_adapter.py`). Aligned all diagnostic weights, population norms, expected drifts, and disorder prototypes to the new clinical features (Physical Activity, Interaction Dynamics, Circadian Charging Regularity, and Daylight Exposure), eliminating all references to defunct metrics. Committed and pushed final changes to the remote repository, triggering the cloud build.
*   **2026-05-24**: Fixed final compilation blocker in `DataRepository.kt` by adding the missing `import java.util.Calendar` statement to resolve the unresolved reference error under task `:app:compileDebugKotlin`. Staged, committed, and pushed the resolution to `origin/Mhealth-app`, initiating a clean automated build in the cloud.
*   **2026-05-24**: Completed full Hard Reset DB tables wipe (L1/L2 data, snapshots, baselines, and analysis history) synchronized with automated pre-reset JSON data preservation backups. Built robust PCA + Mean Shift degenerate fallbacks for <2 days of sparse data in `s1_profile.py` to decouple the digital DNA clustering engine from PCA/SVD convergence crashes. Unified and synchronized Layer 1 and Layer 2 baseline progress indicators throughout the Kotlin/Room and background monitoring services.
*   **2026-05-24**: Synchronized both Layer 1 and Layer 2 baseline days under `baselineDaysRequired` to eliminate settings clutter. Configured a shared keystore signing profile in `build.gradle.kts` for all build types to ensure seamless package upgrades without signature mismatch prompts. Capped progress bars at 100% on small test baselines and added the visually rich `AnomalyScoreFlowCard` diagnostics breakdown panel detailing the dual-layer calculations.
*   **2026-05-24**: Implemented purely idiographic variance-floored anomaly scorer by bypassing population norm priors blending in `bayesian_baseline.py` and using customized standard deviation floors for all 30 features, resolving the `0.485` anomaly score on zero-deviation baseline days. Integrated the "Tracking Day Reset" UX in `MonitorScreen.kt` so active monitoring restarts at "Day 1 of Active Tracking" upon baseline completion. Restructured step-by-step diagnostic headers in `AnomalyScoreFlowCard` using premium, right-aligned, non-wrapping `Surface` score badges.
*   **2026-05-24**: Integrated real-time provisional anomaly scoring and dynamic Bayesian baseline updates into `MainActivity.kt` and `MonitorScreen.kt`. Collected reactive `provisionalAnalysis` and `provisionalBaseline` flows in both screens. Mapped their values dynamically to the primary Anomaly Score Gauge, the step-by-step diagnostic `AnomalyScoreFlowCard`, the `ComparisonCard`, the `FeatureTableCard`, and the 30-feature `Feature Deviation Radar` chart, ensuring today's live/provisional telemetry and Bayesian baseline updates adjust in real-time.

---

## 7. Telemetry Feature Audit & Recommendations (May 2026)

Conducted a thorough clinical and engineering audit of the 29-feature L1 and 22-feature L2 arrays.

### 7.1 Redundant / High-Noise Features
*   **System Telemetry (`storageUsedGB`, `memoryUsagePercent`)**: High noise, zero clinical relevance, managed by Android system/caching logic rather than user agency.
*   **Raw Network Bytes (`networkWifiMB`, `networkMobileMB`)**: Dominated by high-variance entertainment streaming or background updates, highly collinear with direct app telemetry.
*   **`totalAppsCount`**: Quasi-static feature leading to near-zero variance issues in daily z-scoring. Already captured dynamically by inst/uninstalls.
*   **`darkDurationHours`**: Highly redundant with `sleepDurationHours` which is calculated via the superior 3-Signal Sleep Fusion Heuristic.

### 7.2 High-Value Clinical Features to Add
*   **Physical Activity (`dailyStepCount` / `activeMinutes`)**: Clinically isolates pacing/agitation (manic/anxious states) from homebound immobility (depressive states), which GPS displacement alone cannot differentiate.
*   **Interaction Dynamics (`keystrokeDynamics` & `screenTouchDynamics`)**: Typing speed, backspace rate, and scroll velocity are golden biomarkers for psychomotor agitation and retardation.
*   **Daylight Exposure (`daylightExposureMinutes`)**: Passive lux level tracking (Sensor.TYPE_LIGHT) detects room isolation/darkness, a classic vegetative symptom.
*   **Circadian Charging habits (`chargeRegularityEntropy`)**: Erratic, middle-of-night charging indicates sleep hygiene decay and lifestyle disorganization.
*   **Vocal Prosody Check-ins (`voiceProsodyCheckin`)**: Acoustic pitch and speech rate extracted from voluntary daily check-ins bypass Android passive mic privacy blocks while retrieving valuable vocal markers.

---

## 8. Symmetrical Dual-Layer & Cluster-Non-Formation Pipeline (May 2026 Refactoring Blueprint)

Following an architectural deep dive, the finalized blueprints for the core processing engines of System 1 have been established:

### 8.1 Onboarding & Archetype Discovery
*   **Parallel Dimensionality Reduction**: L1 (29D) and L2 (22D) are reduced independently to 3D/4D principal component spaces via **PCA** to capture maximum behavioral variance while resolving the curse of dimensionality.
*   **Parallel Archetype Discovery**: Run **Mean Shift Clustering** independently on L1 and L2 PCA projections. The density bandwidth is set adaptively based on the baseline variance of the principal components, discovering unique anchor clusters for macro (L1) and micro (L2) behaviors separately.

### 8.2 Two-Pass Daily Scoring Pipeline
1.  **Pass 1: Layer 1 (Surface)**:
    *   Compute magnitude and EWMA velocity of today's L1 vector against the Golden L1 Baseline to output a raw L1 anomaly score.
    *   Evaluate distance to nearest L1 anchor cluster centroid. If within $1.5\times$ radius, apply **Radial Proximity Decay** (closer to centroid = larger reduction, down to near 0).
2.  **Pass 2: Layer 2 (Micro)**:
    *   Compute magnitude and EWMA velocity of today's L2 vector against the Golden L2 Baseline to output a raw L2 anomaly score.
    *   Evaluate distance to nearest L2 anchor cluster centroid. If within $1.5\times$ radius, apply **Radial Proximity Decay** to smoothly suppress micro-anomaly.
3.  **Fusion & Thresholding**:
    *   Fuse L1 and L2 adjusted scores using a **Weighted Geometric Mean** ($\text{Score} = S_1^{0.6} \times S_2^{0.4}$).
    *   If the fused score exceeds `ANOMALY_SCORE_THRESHOLD` (e.g., 0.38), increment `sustained_days`.

### 8.3 Post-Filter: Cluster-Non-Formation (Sustained Anomaly Check)
*   **Variance-Bounded Compactness Test**: When consecutive anomalous days reach $N$ days (e.g., 5–10 days), evaluate their compactness in the L1 and L2 PCA spaces:
    *   If maximum pairwise distance and standard deviations are **below the average baseline cluster radii**, the anomalous period forms a tight, stable group ──► **Promote as New Context** (suppress alert, merge these days into the Golden Baseline as a new healthy lifestyle context).
    *   If the anomalous period is **scattered and chaotic** (fails compactness test), it indicates a disorganized clinical onset ──► **Flag user/caregiver** as a high-probability depressive episode.

---

## 7. Telemetry Feature Audit & Recommendations (May 2026)

Conducted a thorough clinical and engineering audit of the 29-feature L1 and 22-feature L2 arrays.

### 7.1 Redundant / High-Noise Features
*   **System Telemetry (`storageUsedGB`, `memoryUsagePercent`)**: High noise, zero clinical relevance, managed by Android system/caching logic rather than user agency.
*   **Raw Network Bytes (`networkWifiMB`, `networkMobileMB`)**: Dominated by high-variance entertainment streaming or background updates, highly collinear with direct app telemetry.
*   **`totalAppsCount`**: Quasi-static feature leading to near-zero variance issues in daily z-scoring. Already captured dynamically by inst/uninstalls.
*   **`darkDurationHours`**: Highly redundant with `sleepDurationHours` which is calculated via the superior 3-Signal Sleep Fusion Heuristic.

### 7.2 High-Value Clinical Features to Add
*   **Physical Activity (`dailyStepCount` / `activeMinutes`)**: Clinically isolates pacing/agitation (manic/anxious states) from homebound immobility (depressive states), which GPS displacement alone cannot differentiate.
*   **Interaction Dynamics (`keystrokeDynamics` & `screenTouchDynamics`)**: Typing speed, backspace rate, and scroll velocity are golden biomarkers for psychomotor agitation and retardation.
*   **Daylight Exposure (`daylightExposureMinutes`)**: Passive lux level tracking (Sensor.TYPE_LIGHT) detects room isolation/darkness, a classic vegetative symptom.
*   **Circadian Charging habits (`chargeRegularityEntropy`)**: Erratic, middle-of-night charging indicates sleep hygiene decay and lifestyle disorganization.
*   **Vocal Prosody Check-ins (`voiceProsodyCheckin`)**: Acoustic pitch and speech rate extracted from voluntary daily check-ins bypass Android passive mic privacy blocks while retrieving valuable vocal markers.
