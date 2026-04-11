# Clinical Architecture Specification
**Project:** Mental Health Early Risk Detection
**Architecture Style:** Vertical-Horizontal Matrix (Domain-Driven Clinical System)
**Version:** 4.0 (Professional Matrix Layout)
**Status:** APPROVED

## 1. Executive Summary

This document defines the professional clinical-grade architecture for the Mental Health Early Risk Detection platform. To effectively communicate the depth of the pipeline to stakeholders and regulatory bodies, the system is modeled as a **Vertical-Horizontal Architecture Matrix**. 

This paradigm breaks down the apparent simplicity of the tech stack by separating infrastructural foundations (Horizontal Layers) from the complex functional machine-learning workflows (Vertical Slices) that parse patient behavior into clinical insights.

---

## 2. The Vertical-Horizontal Matrix Layout

### 2.A. Horizontal Layers: The Infrastructure Foundation
These layers span across the entire system from left to right, providing the standardized technical capabilities that all features rely upon.

1. **[Layer 1] Edge Presentation & Sensing Layer (Top):**
   - The user-facing interfaces. Includes the Android OS hardware sensors (background logic) and the clinician's Web Browser rendering the React DOM, which serves as the frontend for direct interaction.
2. **[Layer 2] Secure Transport & Synchronization Layer:**
   - The networking mesh. Utilizes **Android WorkManager** for intelligent payload batching, while integrating dedicated **Voice Analysis REST APIs** (e.g., HuggingFace endpoint integrations) for real-time model inference. Enforces secure HTTPS / TLS 1.3 natively.
3. **[Layer 3] Identity & Authentication Middleware:**
   - Powered by **Firebase Authentication**. Issues cryptographically secure JWT tokens and enforces Role-Based Access Control (RBAC), strictly separating 'Patient' write-access from 'Clinician' read-access.
4. **[Layer 4] Clinical Data Persistence Layer (Bottom):**
   - The underlying storage foundation. Combines an encrypted local **Android Room Database** (SQLite) for zero-latency edge caching with **Cloud Firestore** for centralized, highly available NoSQL storage of clinical vectors.

---

### 2.B. Vertical Slices: The Functional Pipelines
These columns cut vertically *through* the horizontal infrastructure, representing the chronological journey of data from raw behavior to clinical decision.

#### ❖ Vertical Slice 1: Active Patient Monitoring System
*The continuous, passive collection of behavioral markers.*
- **Top (Interaction):** The Kotlin **MHealth App** silently collects voice heuristics, accelerometer/location data, and app-usage timestamps (`SleepEstimator.kt`).
- **Middle (Transport & Auth):** Raw data is structured into user-day JSON vectors. WorkManager utilizes the patient's Firebase JWT to authenticate the upload.
- **Bottom (Persistence):** Data is written to secure Firestore documents specifically scoped to the protected UUID of the patient, isolating PII (Personally Identifiable Information).

#### ❖ Vertical Slice 2: ML Inference & Baseline Computation
*The offline mathematical brain of the operation.*
- **Bottom (Persistence):** Standalone Python ML workers stream the new historical data trajectories from Firestore.
- **Middle (Processing):** Executes `system2` and `Personality Vector` logic. 
   - Establishes a rolling 4-week "Personal Normal" baseline.
   - Calculates Standard Deviations (pitch, jitter, shimmer from voice; interaction frequency).
   - Flags anomalies using pessimistic thresholds designed for high clinical recall.
- **Top (Output):** Generated "Anomaly Scores" and feature-specific deviation metrics are written back to Firestore.

#### ❖ Vertical Slice 3: Voice Detection of Depression Pipeline
*The specialized acoustic anomaly intelligence.*
- **Top & Middle (Interaction & Transport):** Triggered actively or passively to capture audio. Routes acoustic payloads out to a dedicated **Voice Analysis API** (Hugging Face endpoint).
- **Middle (Processing):** Deep learning acoustic models analyze exact vocal biomarkers (MFCCs, spectral roll-off) uniquely correlated with clinical depression.
- **Bottom (Output):** Ingests the highly specialized acoustic depression score directly back into the Admin Dashboard and the baseline storage layer.

#### ❖ Vertical Slice 4: Clinical CDSS (Decision Support) Dashboard
*The interpretability engine serving actionable data to doctors.*
- **Middle (Transport & Auth):** Clinician logs via Firebase Auth to retrieve decrypted anomaly records.
- **Top (Interaction):** The **React + Vite Admin Dashboard** maps high-dimensional mathematical flags into visual intelligence.
   - Leverages `Recharts` to draw comparative radar charts, showing baseline vs. current psychological state.
   - Structures output as a formal Clinical Report, explicitly noting missing data and model confidence to comply with FDA "Software as a Medical Device" transparency guidelines.

---

## 3. High-Fidelity Image Generation Prompts

If you are using an AI text-to-image generator (such as Midjourney, DALL-E) or diagram software (like Draw.io / Lucidchart) to visualize this system for presentations, use the following highly descriptive prompt:

> *"A professional, highly complex enterprise cloud architecture diagram for a clinical healthcare platform. The diagram must clearly display a Vertical-Horizontal matrix. Horizontally, visually define 4 stacked layers: 'Edge Sensing', 'Secure Sync & APIs', 'Firebase Authentication', and 'Encrypted Datastore' at the base. Vertically, slice these layers with four glowing bounded columns: 'Patient Active Monitoring' on the left, 'Offline ML Baseline Inference' in the middle-left, 'Voice Detection of Depression API' (featuring a waveform AI icon and HuggingFace logo) in the middle-right, and 'Clinical Decision Support Dashboard' on the far right (showing a web UI with complex radar data charts). Do not use neon or cyberpunk themes. Use a pristine, sterile, and highly professional corporate medical aesthetic: sharp vector lines, flat isometric blocks, against a clean white or light gray background. Color palette: Navy blue, steel gray, medical teal, and sage green."*
