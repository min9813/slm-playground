# SLM Playground

A playground for experimenting with small / local LLMs — Japanese text chat,
voice conversation, and TTS. Currently centered on Liquid AI's models; other
lightweight models will be added over time.

The demos here are built on `LiquidAI/LFM2.5-Audio-1.5B-JP`
(an omni audio model that handles TTS, ASR, and speech-to-speech in one):

- `run_text.py`: `LiquidAI/LFM2.5-1.2B-JP-202606` text generation
- `run_audio_tts.py`: Japanese **TTS** (text → speech)
- `run_asr.py`: Japanese **ASR** (speech → text)
- `backend/` + `frontend/`: browser UI with three modes — 音声合成 (TTS),
  音声認識 (ASR), and 音声対話 (speech-to-speech chat)

## Environment (uv)

A single `uv`-managed environment (`pyproject.toml` + `uv.lock`) covers
everything. `requires-python = >=3.12` (needed by `liquid-audio`).

```bash
uv sync --extra all      # text + audio + web server
# or selectively: uv sync --extra audio   /   uv sync (text only)
```

`torch`/`torchaudio` are pinned to the CUDA 12.6 wheels (tested on the local
NVIDIA GeForce GTX 1080 Ti, driver `535.309.01`) via `[tool.uv.index]`. For a
CPU-only host, change that index URL to `https://download.pytorch.org/whl/cpu`.

## Run (CLI)

```bash
# text generation
uv run python run_text.py '日本語で一文だけ挨拶してください。' --max-new-tokens 24 --device cuda

# TTS: text -> speech
uv run python run_audio_tts.py 'こんにちは。' --output outputs/tts_jp.wav --device cuda

# ASR: speech -> text (accepts WAV/FLAC/OGG)
uv run python run_asr.py outputs/tts_jp.wav --device cuda
```

All scripts also support CPU fallback (`--device cpu`). Generated files stay
under `outputs/`.

## Server

```bash
uv run serve                       # http://127.0.0.1:8000  (--host / --port / --reload)
# equivalent: uv run python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The model is lazy-loaded on the first inference
request. The UI has three tabs:

| Tab | Endpoint | What it does |
| --- | --- | --- |
| 音声合成 (TTS) | `POST /api/tts` (JSON) | Japanese text → speech |
| 音声認識 (ASR) | `POST /api/asr` (multipart audio) | record/upload audio → transcript |
| 音声対話 (Chat) | `POST /api/chat` (multipart audio) | speak → spoken + text reply |

The browser records mic audio and encodes it to mono 16-bit PCM WAV client-side,
so the server needs no `ffmpeg`. Generated WAVs are written to `outputs/` and
served from `/outputs/`. Each chat turn is processed independently (no history).

Optional task prompts can be overridden with env vars: `LIQUID_TTS_PROMPT`,
`LIQUID_ASR_PROMPT`, `LIQUID_CHAT_PROMPT`.

### Realtime speech-to-speech (WebRTC)

If the `realtime` extra is installed (`uv sync --extra all`, pulls in `fastrtc`),
the 音声対話 tab also offers a **リアルタイム会話** button. It opens a WebRTC
connection (`POST /webrtc/offer`); fastrtc's VAD auto-detects when you stop
talking and the model streams a spoken reply back (reply text via SSE at
`/api/rtc/outputs`). Mimi streaming decode is used for low-latency audio chunks.
`/api/status` reports `realtime: true` when it is available; the button stays
hidden otherwise. Uses the same single model instance as the other endpoints.

> Microphone capture (both record buttons and realtime) needs a **secure
> context** — open the UI via `https://` or `http://localhost` (e.g. an SSH
> tunnel). A plain `http://<remote-ip>` origin will have the mic blocked by the
> browser.

## Frontend Design Skill

The external `frontend-design` skill from Anthropic Claude Code was installed for both environments:

- Codex: `/home/min9813/.codex/skills/frontend-design`
- Hermes: `/home/min9813/.hermes/skills/creative/frontend-design`

Restart Codex/Hermes sessions to have the new skill discovered automatically.

On this GTX 1080 Ti, `--dtype auto` resolves to `float32` because Pascal GPUs do
not have fast native `float16` or `bfloat16` tensor-core inference.
