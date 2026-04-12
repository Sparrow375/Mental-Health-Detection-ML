# Voice Assessment API Integration Specification

**Status:** IMPLEMENTED
**Target:** Admin Dashboard

This document details the completed technical implementation of the `Early-Mental-Health-Disorder-Detection-via-voice` model inside the `admin-dashboard`. Built to run securely and autonomously with minimal overhead.

## 1. Goal
To provide a standalone module in the Clinician Dashboard that securely assesses patient acoustic biomarkers for symptoms of depression without storing raw PII audio in the cloud.

## 2. Implemented Architecture (Hugging Face Spaces)

1. **React Frontend (Firebase Hosting)**
    *   **Direct-to-API:** The React UI (`VoiceAssessment.tsx`) routes raw audio (microphone blob or `.mp3` upload) directly to the Hugging Face Docker API, skipping intermediatory cloud servers.
    *   **Cold-Start Wake-Up:** Because Hugging Face pauses inactive instances, the UI gracefully pings the endpoint to wake the container, displaying a "Booting" state rather than failing silently.

2. **AI Backend (FastAPI)**
    *   **Hosting:** Hosted continuously on Hugging Face Spaces (Docker Tier).
    *   **Rate Limiting:** Guarded by strict IP-based throttling to prevent API abuse.
    *   **Pre-Processing Math:** The API leverages `FFmpeg` to rigorously clean incoming audio:
        *   Resamples to **16kHz, Mono channel** to prevent PyTorch tensor dimensional mismatches.
        *   Engages variable `silenceremove` to crop out dead air.
        *   Slices an optimal dense acoustic window to achieve 2-second low-latency inference.
    *   Calculates the MFCC / WavLM acoustic biomarker probabilities and returns JSON (`{"prediction": "Depressed", "probability": 0.88}`).

3. **Data Logging (Firestore)**
    *   **Zero-Retention Audio:** Once the React frontend parses the result, the raw audio file is completely discarded from RAM. 
    *   **Persistence:** Only the deterministic health scores (`Score`, `Patient Name`, `Timestamp`, `Clinician ID`) are securely committed to Cloud Firestore using the authenticated user's Firebase token. 

## 3. UI Component Security Checks

The `VoiceAssessment.tsx` component is dynamically scoped:
*   **Disabled Anonymous Logs:** When an Admin uses the tool, the "Analyze" button enforces a strict `disabled={!patientName}` lock. Unnamed, orphaned medical records cannot be created.
*   **Hardware Fallback:** Native JS hooks catch `NotAllowedError` for dead microphones, seamlessly transitioning the clinician to the file-browser upload fallback flow without throwing a Javascript exception.
