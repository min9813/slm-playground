from __future__ import annotations

from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from backend.tts_engine import OUTPUT_DIR, PROJECT_ROOT, get_engine, read_wave
from backend.rtc import mount_rtc
from backend.vl_engine import get_vl_manager, read_image


FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(title="SLM Playground", version="0.3.0")
engine = get_engine()
vl = get_vl_manager()

# Realtime speech-to-speech (WebRTC via fastrtc). No-op if fastrtc is missing.
RTC_ENABLED = mount_rtc(app)


def gpu_status() -> dict | None:
    """Shared GPU memory snapshot, independent of which engine owns the card."""
    if not torch.cuda.is_available():
        return None
    device = torch.device("cuda:0")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "name": torch.cuda.get_device_name(device),
        "memory_free_mb": round(free_bytes / 1024 / 1024, 1),
        "memory_total_mb": round(total_bytes / 1024 / 1024, 1),
        "memory_used_mb": round((total_bytes - free_bytes) / 1024 / 1024, 1),
    }


def model_registry() -> list[dict]:
    """All toggleable models (audio + VL) with their current load state."""
    audio = {
        "key": "audio",
        "label": "LFM2.5-Audio-1.5B-JP",
        "kind": "audio",
        "note": "TTS · ASR · speech-to-speech",
        "grounding": False,
        "loaded": engine.is_loaded,
        "loaded_at": engine.loaded_at.isoformat() if engine.loaded_at else None,
    }
    vl_models = [{**d, "kind": "vl"} for d in vl.descriptors()]
    return [audio, *vl_models]


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    max_new_tokens: int = Field(default=192, ge=16, le=1024)


@app.get("/api/status")
def status() -> dict:
    return {**engine.status(), "realtime": RTC_ENABLED}


@app.get("/api/models")
def list_models() -> dict:
    return {"models": model_registry(), "gpu": gpu_status()}


class ModelRequest(BaseModel):
    key: str


def _set_model(key: str, *, load: bool) -> dict:
    if key == "audio":
        if load:
            engine.load()
        else:
            engine.unload()
    elif vl.has(key):
        if load:
            vl.load(key)
        else:
            vl.unload(key)
    else:
        raise HTTPException(status_code=404, detail=f"Unknown model '{key}'.")
    return {"models": model_registry(), "gpu": gpu_status()}


@app.post("/api/models/load")
def load_model(payload: ModelRequest) -> dict:
    try:
        return _set_model(payload.key, load=True)
    except RuntimeError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc


@app.post("/api/models/unload")
def unload_model(payload: ModelRequest) -> dict:
    return _set_model(payload.key, load=False)


@app.post("/api/vl/infer")
def vl_infer(
    file: UploadFile = File(...),
    key: str = Form(...),
    prompt: str = Form(""),
    max_new_tokens: int = Form(256),
    grounding: bool = Form(False),
) -> dict:
    if not vl.has(key):
        raise HTTPException(status_code=404, detail=f"Unknown VL model '{key}'.")
    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image upload.")
    try:
        image = read_image(raw)
    except Exception as exc:  # noqa: BLE001 - surface decode errors to the client
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}") from exc

    max_new_tokens = max(16, min(1024, max_new_tokens))
    try:
        result = vl.infer(
            key, image, prompt, max_new_tokens=max_new_tokens, grounding=grounding
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {**result, "gpu": gpu_status()}


@app.post("/api/tts")
def create_tts(payload: TTSRequest) -> dict:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")

    try:
        return engine.generate(text, max_new_tokens=payload.max_new_tokens)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _read_upload(file: UploadFile):
    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio upload.")
    try:
        return read_wave(raw)
    except Exception as exc:  # noqa: BLE001 - surface decode errors to the client
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode audio (send a WAV/FLAC/OGG file): {exc}",
        ) from exc


@app.post("/api/asr")
def create_asr(
    file: UploadFile = File(...),
    max_new_tokens: int = Form(256),
) -> dict:
    wave, sample_rate = _read_upload(file)
    try:
        return engine.transcribe(wave, sample_rate, max_new_tokens=max_new_tokens)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/chat")
def create_chat(
    file: UploadFile = File(...),
    max_new_tokens: int = Form(512),
) -> dict:
    wave, sample_rate = _read_upload(file)
    try:
        return engine.converse(wave, sample_rate, max_new_tokens=max_new_tokens)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the Liquid AI JP TTS web server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload.")
    args = parser.parse_args()

    uvicorn.run(
        "backend.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
