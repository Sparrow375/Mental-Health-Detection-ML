# Clinical Architecture Specification
**Project:** Mental Health Early Risk Detection
**Architecture Style:** Vertical-Horizontal Matrix (Domain-Driven Clinical System)
**Version:** 5.0 (Edge-ML Chaquopy Architecture)
**Status:** IMPLEMENTED

## 1. Executive Summary

This document defines the professional clinical-grade architecture for the Mental Health Early Risk Detection platform. The system operates entirely on an **Edge-ML Paradigm**, ensuring maximum privacy by running all algorithmic inference directly on the Android device via an embedded Python engine (Chaquopy).

## 2. The Vertical-Horizontal Matrix Layout

### 2.A. Horizontal Layers: The Infrastructure Foundation

1. **[Layer 1] Edge Sensing & Feature Extraction (Top):**
   - The Kotlin Android app silently derives **29 passive multimodal behavioral features** (e.g., location entropy, sleep duration, social app ratio, displacement) using Android OS hardware sensors and APIs.
2. **[Layer 2] Secure Transact & Edge-Execution Middleware:**
   - Synchronizes raw data extraction with the **Chaquopy Python Engine**. Driven by the `NightlyAnalysisWorker`, ensuring Python ML logic runs locally. The core entry point is `engine.py`.
3. **[Layer 3] Encrypted Datastore (Bottom):**
   - An encrypted local **Android Room Database** (SQLite) manages zero-latency edge caching of Adaptive Behavioral Signatures and System 1/2 outputs.
4. **[Layer 4] Cloud Synchronization:**
   - Evaluated, secure, and privacy-stripped JSON inference results are synced to **Cloud Firestore** using `CloudSyncWorker` for the Clinician Dashboard.

---

### 2.B. Vertical Slices: The Functional ML Pipelines

#### ❖ Vertical Slice 1: PersonDNA & L2 Diagnostics (Context Orchestration)
*Extracting physiological baseline routines and behavioral archetypes.*
- **Top (Sensing):** Ingests daily app-session heuristics and 28-day history.
- **Middle (Processing - `s1_profile.py` & `dna_engine.py`):** 
  - Runs **Clinical-Weighted PCA (2D) + Mean-Shift** clustering to discover "Anchor Clusters" (e.g., Workdays, Weekends). 
  - Performs **K-Means clustering** for AppDNA fingerprints.
  - Implements **Rolling Cluster Discovery** for candidate behavioral shifts.
  - Calculates *Context Coherence*, *Rhythm Integrity*, and *Session Incoherence*.
- **Output:** Passes an `l2_modifier` (amplified by a **Cluster-Mismatch Penalty**) to dial the amplitude of System 1 Anomaly Scoring.

#### ❖ Vertical Slice 2: System 1 (Anomaly Detection)
*The mathematical core answering "Is something wrong?"*
- **Middle (Processing - `system1/detector.py`):** Evaluates the 29-feature vector against the Signature-anchored baseline. Combines **Z-Score Magnitude**, **EWMA Velocity**, and **Mahalanobis Distance**.
- **Output:** Features an exponential **Sustained Evidence Accumulator**. Brief deviations are ignored; sustained downward drifts (3+ days) trigger alert levels.

#### ❖ Vertical Slice 3: System 2 (Diagnostic Pipeline)
*The clinical mapping answering "What does this look like?"*
- **Middle (Processing - `system2/pipeline.py`):** Orchestrates a linear 6-phase triage:
  1. **Baseline Contamination Screener**: Detects if the 28-day baseline is pre-compromised.
  2. **Life Event Filter**: Rejects noise from situational stress (e.g., vacations, exams).
  3. **Geometric Prototype Matcher**: Compares vectors against clinical norms (Depression, Schizophrenia, Bipolar, BPD).
  4. **Clinical Guardrails**: Hard overrides for explicit risk clusters (e.g., Psychosis withdrawal).
  5. **Temporal Validator**: Validates the shape of the anomaly over time.
  6. **Explainability Engine**: Identifies top-3 contributing behavioral features.
- **Bottom:** Generates a human-readable clinical flag packed to Firestore.

#### ❖ Vertical Slice 4: Clinician CDSS (Decision Support) Dashboard
*The interpretability engine serving actionable data.*
- **Top:** A decoupled **React + Vite** SPA protected by Firebase Auth. Maps complex longitudinal edge-computed metrics into **Radar charts** and historical timelines. Features integrated **Audio-Biomarker evaluation** (Hugging Face API).

---

## 3. Visualization Prompts
> *"A professional cloud computing matrix block diagram. Blue horizontal strata represent 'Sensing', 'Chaquopy Engine', 'Room DB', and 'Firestore'. Translucent vertical pillars cut through the strata labeled 'PersonDNA & L2 Diagnostics', 'System 1 Anomaly Array', 'System 2 Diagnostics', and 'Clinician CDSS Dashboard'. Sterile medical aesthetic. Minimalist."*
