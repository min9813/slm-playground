# SLM Playground

A playground for experimenting with small / local LLMs — Japanese text chat,
voice conversation, and TTS. Currently centered on Liquid AI's models; other
lightweight models will be added over time.

The demos here are built on `LiquidAI/LFM2.5-Audio-1.5B-JP`
(an omni audio model that handles TTS, ASR, and speech-to-speech in one):

- `run_text.py`: `LiquidAI/LFM2.5-1.2B-JP-202606` text generation
- `run_audio_tts.py`: Japanese **TTS** (text → speech)
- `run_asr.py`: Japanese **ASR** (speech → text)
- `backend/` + `frontend/`: browser UI with four modes — 音声合成 (TTS),
  音声認識 (ASR), 音声対話 (speech-to-speech chat), and 画像理解 (**VL**
  image understanding via `LiquidAI/LFM2.5-VL-1.6B` / `LFM2.5-VL-450M`)

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
| 画像理解 (VL) | `POST /api/vl/infer` (multipart image) | image + prompt → text (+ bbox) |

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

### 画像理解 (Vision-Language)

The 画像理解 (VL) tab runs Liquid AI's LFM2.5-VL image models:

| Model | Notes |
| --- | --- |
| `LiquidAI/LFM2.5-VL-1.6B` | general VL · OCR · document QA |
| `LiquidAI/LFM2.5-VL-450M` | tiny; adds **visual grounding** (bounding-box prediction) |

Upload an image, type a prompt, and press 推論. Tick **グラウンディング (bbox)**
on the 450M to ask for object boxes — the reply's JSON boxes are parsed
best-effort and drawn on the image canvas (RefCOCO-M ≈ 81 on the 450M).

**GPU on/off toggles.** The card here (GTX 1080 Ti, 11 GB) cannot hold the
audio model and the VL models at the same time, so the VL tab has a **モデル管理**
rack: each model (audio, VL-1.6B, VL-450M) has an ON/OFF switch backed by
`POST /api/models/load` / `/unload`. Turn the audio model OFF to free VRAM
before switching a VL model ON. Live GPU memory is shown in the rack header and
the metrics panel. Unlike the audio model (lazy-loaded on first request), VL
models load only when you flip them ON. Install the extra with
`uv sync --extra all` (or `--extra vision` for just image decoding).

## Acceleration (llama.cpp / GGUF, INT8)

On the local GTX 1080 Ti (Pascal, SM 6.1), INT8 GGUF via a locally-built
llama.cpp gives **3.0–6.3× faster generation** and **2–7× less VRAM** than
transformers fp32, for text, VL, and audio-ASR (TTS/S2S audio-output stays on
`liquid-audio`). Build with `bash scripts/build_llamacpp_cuda.sh` (no system
CUDA / no sudo needed), then `source scripts/llamacpp_env.sh`. Full method,
per-model benchmark numbers, and what was ruled out on Pascal (TensorRT, vLLM,
AWQ/Marlin, ONNX-CUDA INT8, …) are in [docs/acceleration.md](docs/acceleration.md).

## Model Survey / Benchmarks

Small VLM, on-device multimodal, WebGPU, and realtime audio candidates are
tracked in [docs/model_survey.md](docs/model_survey.md). Use
`uv run python scripts/hf_model_inventory.py` to refresh Hugging Face metadata,
and `uv run python scripts/bench_vlm_transformers.py --model <model-id>` for a
first-pass local VLM latency/VRAM test.

## Frontend Design Skill

The external `frontend-design` skill from Anthropic Claude Code was installed for both environments:

- Codex: `/home/min9813/.codex/skills/frontend-design`
- Hermes: `/home/min9813/.hermes/skills/creative/frontend-design`

Restart Codex/Hermes sessions to have the new skill discovered automatically.

On this GTX 1080 Ti, `--dtype auto` resolves to `float32` because Pascal GPUs do
not have fast native `float16` or `bfloat16` tensor-core inference.
