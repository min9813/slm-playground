from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from backend.tts_engine import OUTPUT_DIR, PROJECT_ROOT, get_engine, read_wave
from backend.rtc import mount_rtc


FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(title="Liquid AI JP Voice", version="0.2.0")
engine = get_engine()

# Realtime speech-to-speech (WebRTC via fastrtc). No-op if fastrtc is missing.
RTC_ENABLED = mount_rtc(app)


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    max_new_tokens: int = Field(default=192, ge=16, le=1024)


@app.get("/api/status")
def status() -> dict:
    return {**engine.status(), "realtime": RTC_ENABLED}


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
