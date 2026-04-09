"""
main.py — MHealth Voice Depression Detection API
Deployed on Hugging Face Spaces (Docker tier).
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from torch import nn
from transformers import Wav2Vec2FeatureExtractor, WavLMForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

# ── Rate Limiter ──────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="MHealth Voice Depression API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ──────────────────────────────────────────────────────
ALLOWED_ORIGINS = [
    "https://mhealth-a0812.web.app",
    "https://mhealth-a0812.firebaseapp.com",
    "http://localhost:5173",
    "http://localhost:4173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "wavlm_lora_v10" / "best_model"
SR = 16_000
CHUNK_SECONDS = 10
CHUNK_LENGTH = SR * CHUNK_SECONDS
OVERLAP_SECONDS = 2
OVERLAP_LENGTH = SR * OVERLAP_SECONDS
STRIDE = CHUNK_LENGTH - OVERLAP_LENGTH
SILENCE_THRESHOLD_DB = -40.0
MIN_AUDIO_SECONDS = 3
DECISION_THRESHOLD = 0.4
MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model Architecture (must match train.py v10) ──────────────

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0

        self.query = nn.Parameter(torch.randn(num_heads, 1, self.head_dim) * 0.02)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, H = hidden_states.shape
        keys = self.key_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        values = self.value_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)
        attn = torch.matmul(self.query, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attended = torch.matmul(attn, values).squeeze(2).reshape(B, H)
        return self.norm(self.out_proj(attended))


class WavLMAttentionPool(WavLMForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        h = config.hidden_size
        self.attn_pool = AttentionPooling(h, num_heads=2)
        self.cls_drop = nn.Dropout(0.15)
        self.cls_proj = nn.Linear(h, 256)
        self.cls_norm = nn.LayerNorm(256)
        self.cls_out = nn.Linear(256, config.num_labels)

    def forward(self, input_values, attention_mask=None, labels=None, **kwargs):
        h = self.wavlm(input_values, attention_mask=attention_mask).last_hidden_state
        pooled = self.attn_pool(h)
        x = torch.relu(self.cls_proj(self.cls_drop(pooled)))
        return SequenceClassifierOutput(logits=self.cls_out(self.cls_drop(self.cls_norm(x))))


# ── Lazy model singleton ──────────────────────────────────────
_model = None
_feature_extractor = None


def get_model():
    global _model, _feature_extractor
    if _model is None:
        _model = WavLMAttentionPool.from_pretrained(str(MODEL_PATH))
        _model = _model.to(DEVICE).eval()
        _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(MODEL_PATH))
    return _model, _feature_extractor


# ── Audio helpers ─────────────────────────────────────────────

def load_audio(raw: bytes, suffix: str) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(raw)
        tmp = f.name
    try:
        waveform, sr = torchaudio.load(tmp)
        if sr != SR:
            waveform = torchaudio.transforms.Resample(sr, SR)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0).numpy()
    finally:
        os.unlink(tmp)


def strip_silence(audio: np.ndarray) -> np.ndarray:
    threshold = 10 ** (SILENCE_THRESHOLD_DB / 20.0)
    win = int(SR * 0.05)
    energy = np.array([
        np.sqrt(np.mean(audio[i: i + win] ** 2))
        for i in range(0, len(audio) - win, win)
    ])
    active = energy > threshold
    if not np.any(active):
        return audio
    first = np.argmax(active) * win
    last = (len(active) - 1 - np.argmax(active[::-1])) * win + win
    buf = int(SR * 0.2)
    return audio[max(0, first - buf): min(len(audio), last + buf)]


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    std = np.std(audio)
    return (audio - np.mean(audio)) / std if std > 1e-8 else audio


def chunk_audio(audio: np.ndarray) -> list:
    total = len(audio)
    if total < CHUNK_LENGTH:
        return [np.pad(audio, (0, CHUNK_LENGTH - total))]
    chunks, start = [], 0
    while start < total:
        end = start + CHUNK_LENGTH
        chunk = audio[start:end]
        if len(chunk) < CHUNK_LENGTH:
            chunk = np.pad(chunk, (0, CHUNK_LENGTH - len(chunk)))
        chunks.append(chunk)
        start += STRIDE
        if end >= total:
            break
    return chunks


@torch.no_grad()
def infer_chunk(model, extractor, chunk: np.ndarray) -> np.ndarray:
    inputs = extractor(chunk, sampling_rate=SR, return_tensors="pt", padding=False)
    logits = model(inputs.input_values.to(DEVICE)).logits
    return torch.softmax(logits, dim=-1).cpu().numpy()[0]


def pool_probs(chunk_probs: np.ndarray) -> dict:
    if len(chunk_probs) == 0:
        return {"prediction": "Not Depressed", "probability": 0.0, "confidence": 0.0, "n_chunks": 0}
    mean_dep = chunk_probs.mean(axis=0)[1]
    confs = np.max(chunk_probs, axis=1)
    w_dep = (chunk_probs.T * confs).T.sum(axis=0)[1] / confs.sum()
    final = float(0.5 * mean_dep + 0.5 * w_dep)
    prediction = "Depressed" if final > DECISION_THRESHOLD else "Not Depressed"
    confidence = float(min(abs(final - DECISION_THRESHOLD) / max(DECISION_THRESHOLD, 1 - DECISION_THRESHOLD), 1.0))
    return {"prediction": prediction, "probability": final, "confidence": confidence, "n_chunks": len(chunk_probs)}


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "MHealth Voice API running", "model": "WavLM-LoRA-v10"}


@app.get("/api/health")
async def health():
    return {"status": "ok", "device": DEVICE, "model_loaded": _model is not None}


@app.post("/api/predict")
@limiter.limit("5/minute")
async def predict(request: Request, file: UploadFile = File(...)):
    ALLOWED_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".opus"}
    suffix = Path(file.filename or "audio.wav").suffix.lower() or ".wav"
    if suffix not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported format: {suffix}")

    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(413, "File too large (max 50 MB)")

    try:
        model, extractor = get_model()

        audio = load_audio(content, suffix)
        if len(audio) / SR < 1.0:
            raise HTTPException(400, "Audio too short (min 1 second)")

        stripped = strip_silence(audio)
        audio = stripped if len(stripped) / SR >= MIN_AUDIO_SECONDS else audio
        audio = normalize_audio(audio)

        chunks = chunk_audio(audio)
        probs = np.array([infer_chunk(model, extractor, c) for c in chunks])

        return JSONResponse(pool_probs(probs))

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Inference failed: {exc}") from exc
