# Swasthiti System Architecture

> **See Also:** For comprehensive clinical-grade specifications and dataset requirements, refer to the [Clinical Architecture Specification](./Clinical_Architecture_Specification.md).

This document outlines the robust, edge-computed Machine Learning architecture for the Mental Health Detection (Swasthiti) application. The system operates on a highly secure **Edge-ML** paradigm, meaning all core analysis happens directly on the user's Android phone via embedded Python (Chaquopy), and no raw sensor data is ever streamed to the cloud.

---

## 1. Top-Level Data Flow

The architecture is woven together via an orchestration engine (`engine.py`) that executes nightly on the user's phone.

```text
Sensors (Mobile Android App)
        ↓
Layer 1: Data Collection & Feature Extraction (29 Features)
        ↓
[ Nightly Android WorkManager Trigger ]
        ↓
Layer 2: Edge-ML Orchestrator (engine.py via Chaquopy)
        |
        ├─▶ Level 2: Behavioral DNA System (Context & Clustering)
        |       └─▶ Computes L2_Modifier
        |
        ├─▶ System 1: Improved Anomaly Detector
        |       └─▶ Computes Magnitude, EWMA Velocity & Evidence Accumulation
        |
        └─▶ System 2: 6-Phase Diagnostic Pipeline
                └─▶ Filters, matches prototypes, and tests clinical guardrails
        ↓
Output Phase: Sync to Cloud Firebase
        ↓
Layer 5: Clinical Admin Dashboard 
```

---

## 2. Layer 1: Mobile Data Collection (The Sensors)

The Kotlin-based Android App passively collects multimodal data. Currently, it generates a **29-feature vector** categorized into 7 semantic behavioral groups:

| Group | Features Measured | Clinical Relevance |
| :--- | :--- | :--- |
| **A: Screen & App** | Screen Time, Unlocks, App Launches, Notifications, Social App Ratio | Indicators for doom-scrolling, withdrawal, or manic engagement. |
| **B: Communication** | Calls per day, Call Duration, Unique Contacts, Conversation Frequency | Identifies acute social withdrawal or sudden surges in sociability. |
| **C: Movement** | Daily Displacement (km), Location Entropy, Home Time Ratio, Places Visited | High sensitivity for severe lethargy (Depression) or erratic wandering (Psychosis/Mania). |
| **D: Sleep (Proxies)** | Wake Time, Sleep Time, Sleep Duration, Dark Duration | Crucial for capturing circadian rhythm shifts and insomnia/hypersomnia. |
| **E: System Usage** | Charge Duration, Network MB (Wi-Fi/Mobile) | Additional proxies for generalized device attachment vs detachment. |
| **F: Behavioural** | Total Apps Count, UPI Transactions, App Installs/Uninstalls | Measures financial impulsivity and erratic digital behavior. |
| **G: Engagement** | Calendar Events, Media Count, Background Audio | Highlights passive vs. active states (e.g. listening to background podcasts vs interacting). |

---

## 3. Level 2: The Behavioral DNA System

Rather than just comparing a user against an "average" static baseline, the ML pipeline utilizes a highly sophisticated mathematical mapping of a user's routines called **PersonDNA**.

### 3.1 Baseline Phase (K-Means Anchoring)
During the first 28 days, `dna.py` runs a dynamic K-Means clustering algorithm (figuring out the optimal $K$ based on silhouette scores) on the user's app usage data. It groups their behaviors into mathematical "Anchor Clusters." For example, it naturally discovers the mathematical difference between a user's typical "Workday," "Weekend," and "Off-Day".

### 3.2 Monitoring Phase & Context Metrics
Every night, `dna_engine.py` evaluates the current day's behavior against the PersonDNA anchors to formulate three metrics:
1. **Context Coherence:** How mathematically close today is to a known baseline cluster.
2. **Rhythm Integrity:** The KL-Divergence of the hourly usage heat-map (discovers if a user is using identical apps, but at 3 AM instead of 3 PM).
3. **Session Incoherence:** Checks if app sessions are unusually short, suddenly abandoned, or lacking self-triggers.

### 3.3 Dynamic Rolling Discovery
If behavior deviates, but stabilizes into a *new, healthy routine* with high texture quality (e.g., the user got a new job with new hours), the system learns it over a temporary observation window and mathematically "promotes" it to a permanent new cluster, preventing eternal false-positive alerts.

Finally, these metrics output an **`l2_modifier`** (ranging from $0.15$ to $2.0$), which suppresses or amplifies System 1's detection algorithms based on textual health.

---

## 4. System 1: Improved Anomaly Detector

**Core Responsibility:** Answer "Is something mathematically wrong?"

Unlike baseline comparisons that cry wolf on a single bad day, System 1 (`system1.py`) utilizes a **Sustained Evidence Accumulation Tracker**.

### 4.1 Scoring Mechanism
System 1 evaluates the 29-feature vector and outputs an Anomaly Score ($0-1$) based on:
* **Magnitude ($70\%$):** Clinically weighted Z-Score deviations from the baseline vectors. Variables like `sleepDurationHours` have massive impact weights, while `memoryUsagePercent` has very low impact.
* **Velocity ($30\%$):** Uses Exponentially Weighted Moving Averages (EWMA; $\alpha=0.4$) to track *how fast* variables are changing.

### 4.2 L2-Modulated Exponential Accumulation
System 1 multiplies its raw anomaly score by the Behavioral DNA's **`l2_modifier`**. If the result crosses a threshold ($> 0.38$), it accumulates evidence. If the anomaly happens on multiple consecutive days, the evidence snowballs *exponentially* (e.g., `day * 0.1` modifiers), guaranteeing severe alerts for sustained, pathological behavioral drifts while dismissing single-day hiccups.

---

## 5. System 2: 6-Phase Diagnostic Pipeline

**Core Responsibility:** Answer "What clinical pattern does this resemble?"

If System 1 pushes a comprehensive anomaly report, it passes into `pipeline.py`. System 2 does not attempt to "diagnose," but rather classify mathematical overlap with recognized psychiatric phenomenologies. It executes linearly through 6 logical phases:

1. **Phase 1 — Baseline Screener:** Evaluates if the original 28-day baseline is contaminated (e.g., the user downloaded the app while already mid-depressive episode) and flags it for early intervention.
2. **Phase 2 — Life Event Filter:** Identifies acute event signatures (e.g. abrupt drops in all tech usage mimicking camping/vacations) and immediately dismisses them as normal situational disruptions. 
3. **Phase 3 — Distance Scoring (Prototype Matcher):** Translates current Z-Score deviations and maps them to generalized clinical prototypes (Depression, Mania, Anxiety, Schizophrenia). Uses **Weighted Euclidean Distance** alongside massive sign-mismatch penalties to rank the closest mathematical overlaps.
4. **Phase 4 — Clinical Guardrails:** Overrules the pure geometric mathematical matcher if explicit, literature-backed criteria are met. 
   * *Rule 1 (Psychosis Cluster):* If location entropy drops simultaneously with profound sleep disruption and social cutoff, forces a Schizophrenia-Type flag.
   * *Rule 2 (Withdrawal Cluster):* If calls, unique contacts, and conversation duration plummet together, forces a Depression-Type flag.
5. **Phase 5 — Temporal Validator:** Refined check preventing "rapid cycling" alerts unless the behavior historically fits a bipolar/cyclothymic frame.
6. **Phase 6 — Explainability Engine:** Deconstructs the winning statistical distance and generates human-readable clinical narratives (e.g., "Alert triggered primarily driven by severe $-2.1$ SD drop in daily movement and $-1.8$ SD drop in conversation frequency").

---

## 6. Output & Dashboard Layer 

Because the ML pipeline entirely executes within Android's `NightlyAnalysisWorker.kt` background thread, user privacy is permanently maintained.

1. **Local Room DB:** The massive JSON bundle output by `engine.py` (containing System 1 evidence, System 2 classification, and DNA metrics) is saved to the local SQLite database. 
2. **Cloud Sync:** `CloudSyncWorker.kt` synchronizes *only the mathematical result output* to Google Firebase Cloud Firestore (and never raw microphone or app data).
3. **Admin Dashboard:** A React + Vite web dashboard queries Firebase to graphically map user anomalies across the population, rendering the temporally validated data into actionable clinician/researcher trendlines.

---

## 7. Web App Architecture (Admin Dashboard)

The presentation layer is an enterprise-grade administrative dashboard designed for clinical researchers and mental health professionals to safely interpret the mathematical diagnostic outputs from the mobile edge-compute pipeline.

### 7.1 Core Frontend Architecture
- **Framework**: Built on **React 18** and **Vite** using **TypeScript** for strict, compilable type safety.
- **Routing & State**: Modular, page-based architecture (`Dashboard`, `PatientList`, `PatientDetail`, `Reports`) driven by centralized context providers (`PrivacyContext` / `usePrivacy` hooks) to restrict data views to appropriate authorization levels.

### 7.2 Cloud Data Integration (Firebase)
- The frontend integrates directly with **Google Firebase Cloud Firestore** via `src/firebase/config.ts` and dedicated abstraction bridges (`dataHelper.ts`).
- It acts purely as a consumer, subscribing strictly to the abstracted mathematical outputs pushed by the Android apps. It handles zero raw media or identifiable local telemetry.

### 7.3 Telemetry Visualization
- **Dynamic Charting Data Flow**: Custom React components (`BaselineLineGraph`, `BaselineComparison`) unpack the large JSON payload stored in Firestore.
- Plots the user's daily variations in System 1 (Magnitude and EWMA Velocity) against the immutable baseline thresholds configured during the PersonDNA clustering phase.

### 7.4 Voice Assessment Gateway
- Contains a dedicated proxy interface (`VoiceAssessment.tsx`) pointing to a containerized Python/HuggingFace microservice API (`Swasthiti-voice-api`).
- Evaluates discrete vocal recordings (when explicitly authorized/submitted) against a `wavlm_lora_v10` fine-tuned acoustic model designed to detect subtle phonation and articulation alterations mapped to depressive and psychotic states.

