# Model Integration Plan: Voice-based Mental Health Detection

This document outlines the finalized technical implementation to integrate the `Early-Mental-Health-Disorder-Detection-via-voice` model into your existing `admin-dashboard` web app using a containerized API architecture, designed with a strict $0 budget and optimized for maximum speed (lowest latency).

## Goal

To build a standalone module that serves two distinct workflows:
1.  **Public Users:** Anonymous voice assessment feature.
2.  **Clinicians (Admins):** Secure dashboard to log standalone voice assessments, logging only the AI scores, patient names, timestamps, and clinician IDs without duplicating large audio files.

## Proposed Architecture (High-Speed $0 Setup)

1.  **React Frontend (Firebase Hosting)**
    *   **Direct-to-API Upload:** The React frontend sends the raw audio file directly to the Python API.
    *   **Cold-Start Wake-Up Protocol:** Because Hugging Face puts unused instances to sleep, the React frontend actively pings the API status before upload. If the server is asleep, the UI displays a specialized "Booting up AI... (2 minutes)" countdown, preventing accidental `503 Service Unavailable` crashes.
2.  **AI Backend (FastAPI - Hugging Face Spaces)**
    *   **ML-Optimized Free Hosting:** The Python API is deployed to **Hugging Face Spaces (Docker Tier)**.
    *   **Anti-DDoS Shield (Rate Limiting):** To prevent hackers from spoofing CORS and crashing the free instance, the FastAPI backend implements strict IP-based Rate Limiting (e.g., max 5 requests per minute per IP).
    *   **Latency & Tensor Optimization:** The API executes `FFmpeg`. First, it normalizes all incoming audio strictly to **16kHz, Mono channel** to prevent Machine Learning Tensor mismatches. Then it uses `silenceremove` to strip out all non-speaking gaps, and crops a dense **60-90 second representative slice**. This guarantees highly accurate acoustic biomarkers while dropping inference time to mere seconds.
    *   **Instant Return:** The API calculates the result and immediately fires the JSON score back to the frontend.
3.  **Persistence (Firebase Firestore - Optimized Schema)**
    *   Once the React frontend instantly gets the result back from Hugging Face, it saves exactly four fields to Firestore: **Score(float), Patient Name(string), Timestamp(Date), and Clinician ID(string)**. The original audio file is deliberately discarded to prevent double-archiving, as clinicians already maintain local access to their patients' files.

---

## Proposed Changes

### 1. Web App Routing & Public Pages
#### [NEW] [src/pages/Home.tsx](file:///C:/Users/embar/OneDrive-N/D0cuments/GitHub/Mental-Health-Detection-ML/admin-dashboard/src/pages/Home.tsx)
The public landing page featuring the autonomous voice assessment module.

#### [NEW] [src/pages/Login.tsx](file:///C:/Users/embar/OneDrive-N/D0cuments/GitHub/Mental-Health-Detection-ML/admin-dashboard/src/pages/Login.tsx)
Gateway for clinicians.

#### [MODIFY] [src/App.tsx](file:///C:/Users/embar/OneDrive-N/D0cuments/GitHub/Mental-Health-Detection-ML/admin-dashboard/src/App.tsx)
Multi-route app setup.

### 2. Voice Assessment Components
#### [NEW] [src/components/VoiceAssessment.tsx](file:///C:/Users/embar/OneDrive-N/D0cuments/GitHub/Mental-Health-Detection-ML/admin-dashboard/src/components/VoiceAssessment.tsx)
The core component managing two input modes: Upload `.mp3` or Live Record (dynamically capped at 4 to 5 minutes).

*   **Security Prop Switch (`isAdmin={boolean}`)**: If rendered on the Home Page (`isAdmin={false}`), the "Name" input box is completely hidden, making accidental database writes impossible. If `isAdmin={true}`, it fully enables clinical logging.
*   **Hardware Fallback**: The React module actively catches `NotAllowedError` or `NotFoundError` if a user has no microphone or clicks "Deny", instantly switching to a friendly *"Microphone disabled. Please use the Upload tab"* UI limit instead of silently freezing.
*   **Medical Data Integrity**: The "Analyze" button is strictly grayed out (`disabled={!patientName}`) on the admin side until a valid string is entered, preventing the creation of anonymous, orphaned medical records.
*   **Workflow**: Component "wakes up" Hugging Face -> POSTs the file -> retrieves fast response -> saves `{name, score, timestamp, clinician_id}` directly to the Firestore Database if `isAdmin=true`.

### 3. AI API Service & Dockerization
#### [NEW] [api/main.py](file:///C:/Users/embar/OneDrive-N/D0cuments/GitHub/Mental-Health-Detection-ML/api/main.py)
Creates the FastAPI endpoint `/api/predict`.
*   **Rate Limiting**: Integrates `slowapi` to enforce strict IP request throttling.
*   **Audio Math Logic**: Uses `FFmpeg` to rigorously enforce `ar 16000` (16kHz), `ac 1` (Mono channel), and `silenceremove` against incoming blobs preventing model crashes.
*   Returns `{"prediction": "Depressed", "probability": 0.88}` instantly.

#### [NEW] [api/Dockerfile](file:///C:/Users/embar/OneDrive-N/D0cuments/GitHub/Mental-Health-Detection-ML/api/Dockerfile)
Constructs the deployment environment for Hugging Face Spaces. Installs necessary dependencies including `ffmpeg`.

### 4. Security & Data Decoupling
*   **Hugging Face Defenses**: Protected by CORS rules and backend IP Rate Limiting.
*   **Firestore Database Rules**: Strict Firebase Security Rules ensure that only logged-in Admins can log medical scores into the database. Data collisions are handled natively via Firebase Auto-ID document creation.

---

## Verification Plan
1. Send dummy `curl` spam to the Hugging Face endpoint to verify IP Rate Limiting triggers successfully.
2. Upload heavily corrupted 48kHz Stereo audio; verify FFmpeg correctly standardizes to 16kHz Mono before hitting the model.
3. Validate "Empty Patient Name" state securely disables API requests on the React frontend.
