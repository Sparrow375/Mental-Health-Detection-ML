"""
api_server.py  —  Lumen · WavLM Depression-Detection API Proxy
==============================================================
Replaces the former Modal.com deployment (modal_app.py).

Instead of running GPU inference via the Modal Python SDK, this
lightweight FastAPI server forwards every /api/predict request to the
Hugging Face Space that now hosts the WavLM inference pipeline.

All response shapes are identical to the original Modal endpoint so the
React frontend (VoiceAssessment.tsx) and any other callers need zero changes.

Environment variables (set in .env.local or your hosting platform):
  HF_SPACE_URL — base URL of the deployed HF Space, e.g.
                 https://nitish1018-mhealth-voice-api.hf.space
                 NEVER hardcode this value.

Run locally:
  pip install fastapi[standard] httpx python-multipart uvicorn
  HF_SPACE_URL=https://... uvicorn api_server:app --reload --port 8000
"""

import os

import httpx
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Config ──────────────────────────────────────────────────────────

# Read HF Space base URL from the environment — NEVER hardcoded.
HF_SPACE_URL: str = os.environ.get("HF_SPACE_URL", "").rstrip("/")

# HTTP timeout (seconds) for requests forwarded to the HF Space.
# HF Spaces can take up to ~60 s on a cold start; 120 s gives headroom.
HF_REQUEST_TIMEOUT = 120.0
HF_HEALTH_TIMEOUT = 10.0

# ── FastAPI app ──────────────────────────────────────────────────────

app = FastAPI(title="Lumen WavLM API", version="2.0.0")

# Mirror the permissive CORS policy of the original Modal endpoint so
# the React dashboard can call this proxy from any origin in dev/prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# ── Internal helper ──────────────────────────────────────────────────

def _hf_url(path: str) -> str:
    """Construct a full HF Space URL, raising if the env var is unset."""
    if not HF_SPACE_URL:
        raise RuntimeError(
            "HF_SPACE_URL environment variable is not set. "
            "Add it to your .env.local file or hosting environment."
        )
    return f"{HF_SPACE_URL}{path}"


# ── Health check ─────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """
    Proxy the health check to the HF Space so callers can verify
    the upstream model container is alive before submitting audio.
    Response shape: {"status": "ok", "device": "...", "model_loaded": bool}
    """
    try:
        async with httpx.AsyncClient(timeout=HF_HEALTH_TIMEOUT) as client:
            r = await client.get(_hf_url("/api/health"))
        return JSONResponse(r.json(), status_code=r.status_code)
    except RuntimeError as exc:
        return JSONResponse({"status": "error", "detail": str(exc)}, status_code=500)
    except httpx.RequestError as exc:
        return JSONResponse(
            {"status": "error", "detail": f"HF Space unreachable: {exc}"},
            status_code=502,
        )


# ── Inference endpoint ───────────────────────────────────────────────

@app.post("/api/predict")
async def predict(
    request: Request,
    file: UploadFile | None = File(default=None),
):
    """
    Forward an audio file to the HF Space /api/predict endpoint and
    return its JSON prediction unchanged.

    Accepts the same two request styles as the original Modal endpoint
    so the React VoiceAssessment component needs zero modifications:
      1. multipart/form-data  — "file" field  (browser blob upload)
      2. raw body bytes       — application/octet-stream / audio/*

    Response shape (from HF Space, passed through unchanged):
      {
        "prediction":  "Depressed" | "Not Depressed",
        "probability": float,
        "confidence":  float,
        "n_chunks":    int
      }
    All downstream variable names in the frontend remain identical.
    """
    try:
        # ── 1. Read incoming audio ────────────────────────────────────
        if file is not None:
            # Style 1: multipart upload (existing frontend sends this)
            audio_bytes  = await file.read()
            content_type = file.content_type or "audio/wav"
            filename     = file.filename or "audio.wav"
        else:
            # Style 2: raw body bytes (curl / programmatic callers)
            audio_bytes  = await request.body()
            content_type = request.headers.get("content-type", "audio/wav")
            filename     = "audio.wav"

        if not audio_bytes:
            return JSONResponse({"detail": "No audio data received."}, status_code=422)

        # ── 2. Forward to HF Space via HTTP POST ─────────────────────
        # The HF Space /api/predict endpoint accepts multipart/form-data
        # with a single "file" field — identical to the original Modal API.
        async with httpx.AsyncClient(timeout=HF_REQUEST_TIMEOUT) as client:
            response = await client.post(
                _hf_url("/api/predict"),
                files={"file": (filename, audio_bytes, content_type)},
            )

        # ── 3. Return HF Space JSON unchanged ────────────────────────
        # Variable names in the response dict are kept identical to the
        # original Modal endpoint so all downstream code continues to work
        # without modification (prediction, probability, confidence, n_chunks).
        return JSONResponse(response.json(), status_code=response.status_code)

    except RuntimeError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=500)
    except httpx.RequestError as exc:
        return JSONResponse(
            {"detail": f"HF Space request failed: {exc}"},
            status_code=502,
        )
    except Exception as exc:
        return JSONResponse({"detail": str(exc)}, status_code=500)
