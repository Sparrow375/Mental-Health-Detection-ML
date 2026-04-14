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
   - The Kotlin Android app silently derives 29 passive multimodal behavioral features (e.g., location entropy, sleep duration, conversation frequency) using Android OS hardware sensors and APIs.
2. **[Layer 2] Secure Transact & Edge-Execution Middleware:**
   - Synchronizes raw data extraction with the **Chaquopy Python Engine**. Driven by the `NightlyAnalysisWorker`, ensuring Python ML logic runs locally.
3. **[Layer 3] Encrypted Datastore (Bottom):**
   - An encrypted local **Android Room Database** (SQLite) manages zero-latency edge caching of Adaptive Behavioral Signatures and System 1/2 outputs.
4. **[Layer 4] Cloud Synchronization:**
   - Evaluated, secure, and privacy-stripped JSON inference results are synced to **Cloud Firestore** using `CloudSyncWorker` for the Clinician Dashboard.

---

### 2.B. Vertical Slices: The Functional ML Pipelines

#### ❖ Vertical Slice 1: Adaptive Behavioral Signatures (Context Orchestration)
*Extracting physiological baseline routines.*
- **Top (Sensing):** Ingests daily app-session heuristics.
- **Middle (Processing - `abs_engine.py`):** Runs Unsupervised K-Means clustering to discover "Baseline Clusters" (e.g., Workdays, Weekends). Calculates *Context Coherence*, *Rhythm Integrity*, and *Session Incoherence* to dynamically understand behavioral shifts.
- **Output:** Passes an `l2_modifier` to dial the amplitude of Anomaly Scoring.

#### ❖ Vertical Slice 2: System 1 (Anomaly Detection)
*The mathematical core answering "Is something wrong?"*
- **Middle (Processing - `system1.py`):** Evaluates the 29-feature vector against the Signature-anchored baseline. Combines *Z-Score Magnitude* with *EWMA Velocity*.
- **Output:** Features an exponential **Sustained Evidence Accumulator**. Brief deviations are ignored; sustained downward drifts trigger exponential alert scores.

#### ❖ Vertical Slice 3: System 2 (Diagnostic Pipeline)
*The clinical mapping answering "What does this look like?"*
- **Middle (Processing - `pipeline.py`):** Orchestrates a linear 6-phase triage:
  1. Baseline Contamination Screener.
  2. Life Event Filter (rejects situational stress like vacations).
  3. Geometric Prototype Matcher (compares mathematically against Depression, Schizophrenia, Bipolar, BPD norms).
  4. Clinical Guardrails (hard overrides for explicit Psychosis or Withdrawal clusters).
  5. Temporal Validator.
  6. Explainability Engine.
- **Bottom:** Generates a human-readable clinical flag packed to Firestore.

#### ❖ Vertical Slice 4: Clinician CDSS (Decision Support) Dashboard
*The interpretability engine serving actionable data.*
- **Top:** A decoupled React + Vite SPA (Single Page Application) protected by Firebase Auth. Maps complex longitudinal edge-computed metrics into highly readable Radar charts and historical timelines. Features integrated Audio-Biomarker evaluation (Hugging Face API).

---

## 3. Visualization Prompts
> *"A professional cloud computing matrix block diagram. Blue horizontal strata represent 'Sensing', 'Chaquopy Engine', 'Room DB', and 'Firestore'. Translucent vertical pillars cut through the strata labeled 'Adaptive Behavioral Signatures (ABS)', 'System 1 Anomaly Array', 'System 2 Diagnostics', and 'Clinician CDSS Dashboard'. Sterile medical aesthetic. Minimalist."*
