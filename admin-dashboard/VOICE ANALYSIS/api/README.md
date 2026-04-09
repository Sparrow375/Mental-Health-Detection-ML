---
title: MHealth Voice API
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# MHealth Voice Depression Detection API

FastAPI backend powered by WavLM-LoRA fine-tuned for depression detection from voice.

## Endpoints

- `GET /` — Health check
- `GET /api/health` — Model status
- `POST /api/predict` — Upload audio file, receive prediction JSON

## Response Format

```json
{
  "prediction": "Depressed" | "Not Depressed",
  "probability": 0.73,
  "confidence": 0.55,
  "n_chunks": 6
}
```

## Rate Limiting

Max **5 requests per minute** per IP address.
