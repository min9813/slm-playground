from __future__ import annotations

from datetime import datetime
import hashlib
import io
import os
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import soundfile as sf
import torch
from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor
from liquid_audio.detokenizer import LFM2AudioDetokenizer
from safetensors.torch import load_file
from transformers import Lfm2Config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_ID = os.environ.get("LIQUID_TTS_MODEL", "LiquidAI/LFM2.5-Audio-1.5B-JP")

# System prompts that select the model's task in sequential / interleaved mode.
TTS_SYSTEM_PROMPT = os.environ.get("LIQUID_TTS_PROMPT", "Perform TTS in japanese.")
ASR_SYSTEM_PROMPT = os.environ.get("LIQUID_ASR_PROMPT", "Perform ASR.")
CHAT_SYSTEM_PROMPT = os.environ.get(
    "LIQUID_CHAT_PROMPT", "Respond with interleaved text and audio."
)

# Output sample rate of the LFM2 audio detokenizer.
OUTPUT_SAMPLE_RATE = 24_000


def read_wave(source: str | Path | bytes | bytearray) -> tuple[torch.Tensor, int]:
    """Read a WAV/FLAC/OGG source into a ``(1, T)`` float32 mono tensor + sample rate.

    ``source`` may be a path or the raw bytes of an uploaded file.
    """
    handle: Any = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    data, sample_rate = sf.read(handle, dtype="float32", always_2d=True)  # (T, channels)
    mono = data.mean(axis=1)  # downmix to mono -> (T,)
    wave = torch.from_numpy(np.ascontiguousarray(mono)).unsqueeze(0)  # (1, T)
    return wave, int(sample_rate)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested != "auto":
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[requested]

    if device.type != "cuda":
        return torch.float32

    major, _ = torch.cuda.get_device_capability(device)
    if major >= 8:
        return torch.bfloat16
    if major >= 7:
        return torch.float16
    return torch.float32


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def preload_detokenizer(
    processor: LFM2AudioProcessor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Avoid liquid-audio 1.3.0's hard-coded `.cuda()` path and control dtype."""
    if processor.detokenizer_path is None:
        return

    detok_path = Path(processor.detokenizer_path)
    detok_config = Lfm2Config.from_pretrained(detok_path / "config.json")
    if isinstance(detok_config.layer_types, list):
        detok_config.layer_types = [
            "full_attention" if layer == "sliding_attention" else layer
            for layer in detok_config.layer_types
        ]

    detok = LFM2AudioDetokenizer(detok_config).eval()
    detok_weights = load_file(detok_path / "model.safetensors", device="cpu")
    detok.load_state_dict(detok_weights)
    detok.to(device=device, dtype=dtype)
    processor._audio_detokenizer = detok.eval()


class TTSEngine:
    def __init__(self, *, device: str | None = None, dtype: str | None = None) -> None:
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        self.device = resolve_device(device or os.environ.get("LIQUID_TTS_DEVICE", "auto"))
        self.dtype = resolve_dtype(
            dtype or os.environ.get("LIQUID_TTS_DTYPE", "auto"), self.device
        )
        self.processor: LFM2AudioProcessor | None = None
        self.model: LFM2AudioModel | None = None
        self.loaded_at: datetime | None = None
        self.load_seconds: float = 0.0
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self.processor is not None and self.model is not None

    def load(self) -> float:
        """Eagerly load the audio model (used by the model on/off rack)."""
        with self._lock:
            return self._ensure_loaded()

    def unload(self) -> bool:
        """Free the audio model from VRAM (used to make room for VL models)."""
        with self._lock:
            if not self.is_loaded:
                return False
            self.processor = None
            self.model = None
            self.loaded_at = None
            import gc

            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return True

    def status(self) -> dict[str, Any]:
        return {
            "model_id": MODEL_ID,
            "loaded": self.is_loaded,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "device": str(self.device),
            "dtype": dtype_name(self.dtype),
            "cuda": self._cuda_status(),
        }

    def generate(self, text: str, *, max_new_tokens: int) -> dict[str, Any]:
        request_started = time.perf_counter()
        lock_started = time.perf_counter()

        with self._lock:
            queue_seconds = time.perf_counter() - lock_started
            model_load_seconds = self._ensure_loaded()
            assert self.processor is not None
            assert self.model is not None

            prompt_started = time.perf_counter()
            chat = self._build_chat(text)
            prompt_seconds = time.perf_counter() - prompt_started

            generation_started = time.perf_counter()
            audio_out: list[torch.Tensor] = []
            with torch.inference_mode():
                for token in self.model.generate_sequential(
                    **chat,
                    max_new_tokens=max_new_tokens,
                    audio_temperature=0.8,
                    audio_top_k=64,
                ):
                    if token.numel() > 1:
                        audio_out.append(token.detach().to("cpu"))
            generation_seconds = time.perf_counter() - generation_started

            if len(audio_out) <= 1:
                raise RuntimeError("No audio tokens were generated; try increasing max_new_tokens.")

            decode_started = time.perf_counter()
            audio_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0).to(self.device)
            waveform = self.processor.decode(audio_codes).cpu()[0]
            decode_seconds = time.perf_counter() - decode_started

            write_started = time.perf_counter()
            output_path = self._next_output_path(text)
            sf.write(output_path, waveform, 24_000)
            write_seconds = time.perf_counter() - write_started

            total_seconds = time.perf_counter() - request_started
            return {
                "audio_url": f"/outputs/{output_path.name}",
                "filename": output_path.name,
                "file_bytes": output_path.stat().st_size,
                "text_chars": len(text),
                "max_new_tokens": max_new_tokens,
                "audio_tokens": max(0, len(audio_out) - 1),
                "sample_rate": 24_000,
                "audio_duration_seconds": waveform.numel() / 24_000,
                "timings": {
                    "queue_seconds": queue_seconds,
                    "model_load_seconds": model_load_seconds,
                    "prompt_seconds": prompt_seconds,
                    "generation_seconds": generation_seconds,
                    "decode_seconds": decode_seconds,
                    "write_seconds": write_seconds,
                    "total_seconds": total_seconds,
                },
                "runtime": self._runtime_info(),
            }

    def transcribe(
        self, wave: torch.Tensor, sample_rate: int, *, max_new_tokens: int = 256
    ) -> dict[str, Any]:
        """Speech-to-text (ASR) via sequential text generation."""
        request_started = time.perf_counter()
        lock_started = time.perf_counter()

        with self._lock:
            queue_seconds = time.perf_counter() - lock_started
            model_load_seconds = self._ensure_loaded()
            assert self.processor is not None
            assert self.model is not None

            prompt_started = time.perf_counter()
            chat = self._build_asr_chat(wave, sample_rate)
            prompt_seconds = time.perf_counter() - prompt_started

            generation_started = time.perf_counter()
            text_tokens: list[torch.Tensor] = []
            with torch.inference_mode():
                for token in self.model.generate_sequential(
                    **chat,
                    max_new_tokens=max_new_tokens,
                ):
                    if token.numel() != 1:
                        break  # ASR is text-only; ignore any audio frames
                    if int(token) == 128:  # <|audio_start|> — not expected for ASR
                        break
                    text_tokens.append(token.detach().to("cpu"))
            generation_seconds = time.perf_counter() - generation_started

            if not text_tokens:
                raise RuntimeError("No text was produced; the audio may be silent or too short.")

            transcript = self.processor.text.decode(
                torch.cat(text_tokens), skip_special_tokens=True
            ).strip()

            total_seconds = time.perf_counter() - request_started
            return {
                "text": transcript,
                "text_tokens": len(text_tokens),
                "max_new_tokens": max_new_tokens,
                "audio_input_seconds": wave.shape[1] / sample_rate,
                "timings": {
                    "queue_seconds": queue_seconds,
                    "model_load_seconds": model_load_seconds,
                    "prompt_seconds": prompt_seconds,
                    "generation_seconds": generation_seconds,
                    "total_seconds": total_seconds,
                },
                "runtime": self._runtime_info(),
            }

    def converse(
        self, wave: torch.Tensor, sample_rate: int, *, max_new_tokens: int = 512
    ) -> dict[str, Any]:
        """Speech-to-speech: reply to spoken input with interleaved text + audio."""
        request_started = time.perf_counter()
        lock_started = time.perf_counter()

        with self._lock:
            queue_seconds = time.perf_counter() - lock_started
            model_load_seconds = self._ensure_loaded()
            assert self.processor is not None
            assert self.model is not None

            prompt_started = time.perf_counter()
            chat = self._build_conversation_chat(wave, sample_rate)
            prompt_seconds = time.perf_counter() - prompt_started

            generation_started = time.perf_counter()
            text_tokens: list[torch.Tensor] = []
            audio_frames: list[torch.Tensor] = []
            with torch.inference_mode():
                for token in self.model.generate_interleaved(
                    **chat,
                    max_new_tokens=max_new_tokens,
                    audio_temperature=0.8,
                    audio_top_k=64,
                ):
                    if token.numel() == 1:
                        text_tokens.append(token.detach().to("cpu"))
                    elif not bool((token == 2048).any()):  # skip <|audio_end|> frame
                        audio_frames.append(token.detach().to("cpu"))
            generation_seconds = time.perf_counter() - generation_started

            reply_text = ""
            if text_tokens:
                reply_text = self.processor.text.decode(
                    torch.cat(text_tokens), skip_special_tokens=True
                ).replace("<|text_end|>", "").strip()

            decode_started = time.perf_counter()
            audio_url = filename = None
            file_bytes = 0
            audio_duration = 0.0
            if audio_frames:
                audio_codes = torch.stack(audio_frames, 1).unsqueeze(0).to(self.device)
                waveform = self.processor.decode(audio_codes).cpu()[0]
                output_path = self._next_output_path(reply_text or "chat", prefix="chat")
                sf.write(output_path, waveform, OUTPUT_SAMPLE_RATE)
                audio_url = f"/outputs/{output_path.name}"
                filename = output_path.name
                file_bytes = output_path.stat().st_size
                audio_duration = waveform.numel() / OUTPUT_SAMPLE_RATE
            decode_seconds = time.perf_counter() - decode_started

            if not reply_text and audio_url is None:
                raise RuntimeError("The model returned no response; try speaking again.")

            total_seconds = time.perf_counter() - request_started
            return {
                "text": reply_text,
                "audio_url": audio_url,
                "filename": filename,
                "file_bytes": file_bytes,
                "audio_duration_seconds": audio_duration,
                "audio_tokens": len(audio_frames),
                "sample_rate": OUTPUT_SAMPLE_RATE,
                "max_new_tokens": max_new_tokens,
                "timings": {
                    "queue_seconds": queue_seconds,
                    "model_load_seconds": model_load_seconds,
                    "prompt_seconds": prompt_seconds,
                    "generation_seconds": generation_seconds,
                    "decode_seconds": decode_seconds,
                    "total_seconds": total_seconds,
                },
                "runtime": self._runtime_info(),
            }

    def stream_reply(self, audio: tuple[int, "np.ndarray"], *, max_new_tokens: int = 512):
        """Streaming speech-to-speech for fastrtc / WebRTC.

        Takes one turn of input audio ``(sample_rate, ndarray)`` and yields, as they
        are produced: ``("audio", (24000, int16[1, N]))`` chunks (Mimi streaming
        decode) and ``("text", partial_transcript)`` updates.
        """
        sample_rate, array = audio
        array = np.asarray(array).reshape(-1)
        if np.issubdtype(array.dtype, np.integer):
            wave = torch.from_numpy(array.astype("float32") / 32768.0)
        else:
            wave = torch.from_numpy(array.astype("float32"))
        wave = wave.unsqueeze(0)  # (1, N)

        with self._lock:
            self._ensure_loaded()
            assert self.processor is not None
            assert self.model is not None
            mimi = self.processor.mimi

            chat = self._build_conversation_chat(wave.to(self.device), sample_rate)
            text_tokens: list[torch.Tensor] = []
            with torch.inference_mode(), mimi.streaming(1):
                for token in self.model.generate_interleaved(
                    **chat,
                    max_new_tokens=max_new_tokens,
                    audio_temperature=0.8,
                    audio_top_k=64,
                ):
                    if token.numel() == 1:
                        text_tokens.append(token.detach())
                        text = (
                            self.processor.text.decode(
                                torch.cat(text_tokens), skip_special_tokens=True
                            )
                            .replace("<|text_end|>", "")
                            .strip()
                        )
                        yield ("text", text)
                    elif not bool((token == 2048).any()):
                        waveform = mimi.decode(token[None, :, None])[0]
                        pcm = (
                            waveform.detach().float().cpu().numpy().reshape(1, -1) * 32767.0
                        )
                        pcm = np.clip(pcm, -32768, 32767).astype(np.int16)
                        yield ("audio", (OUTPUT_SAMPLE_RATE, pcm))

    def _runtime_info(self) -> dict[str, Any]:
        return {
            "model_id": MODEL_ID,
            "device": str(self.device),
            "dtype": dtype_name(self.dtype),
            "cuda": self._cuda_status(),
        }

    def _ensure_loaded(self) -> float:
        if self.is_loaded:
            return 0.0

        started = time.perf_counter()
        processor = LFM2AudioProcessor.from_pretrained(MODEL_ID, device=self.device).eval()
        processor.to(device=self.device, dtype=self.dtype)
        preload_detokenizer(processor, device=self.device, dtype=self.dtype)
        model = LFM2AudioModel.from_pretrained(
            MODEL_ID,
            device=self.device,
            dtype=self.dtype,
        ).eval()

        self.processor = processor
        self.model = model
        self.loaded_at = datetime.now()
        self.load_seconds = time.perf_counter() - started
        return self.load_seconds

    def _build_chat(self, text: str) -> ChatState:
        assert self.processor is not None

        chat = ChatState(self.processor, dtype=self.dtype)
        chat.new_turn("system")
        chat.add_text(TTS_SYSTEM_PROMPT)
        chat.end_turn()

        chat.new_turn("user")
        chat.add_text(text)
        chat.end_turn()
        chat.new_turn("assistant")
        return chat

    def _build_asr_chat(self, wave: torch.Tensor, sample_rate: int) -> ChatState:
        assert self.processor is not None

        chat = ChatState(self.processor, dtype=self.dtype)
        chat.new_turn("system")
        chat.add_text(ASR_SYSTEM_PROMPT)
        chat.end_turn()

        chat.new_turn("user")
        chat.add_audio(wave, sample_rate)
        chat.end_turn()
        chat.new_turn("assistant")
        return chat

    def _build_conversation_chat(self, wave: torch.Tensor, sample_rate: int) -> ChatState:
        assert self.processor is not None

        chat = ChatState(self.processor, dtype=self.dtype)
        chat.new_turn("system")
        chat.add_text(CHAT_SYSTEM_PROMPT)
        chat.end_turn()

        chat.new_turn("user")
        chat.add_audio(wave, sample_rate)
        chat.end_turn()
        chat.new_turn("assistant")
        return chat

    def _next_output_path(self, text: str, *, prefix: str = "tts") -> Path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        return OUTPUT_DIR / f"{prefix}_{timestamp}_{digest}.wav"

    def _cuda_status(self) -> dict[str, Any] | None:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return None

        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        return {
            "name": torch.cuda.get_device_name(self.device),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(self.device))),
            "memory_free_mb": round(free_bytes / 1024 / 1024, 1),
            "memory_total_mb": round(total_bytes / 1024 / 1024, 1),
            "memory_used_mb": round((total_bytes - free_bytes) / 1024 / 1024, 1),
        }


_ENGINE: TTSEngine | None = None


def get_engine() -> TTSEngine:
    """Return the process-wide engine singleton (one model on the GPU, shared by
    the HTTP endpoints and the fastrtc handler)."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = TTSEngine()
    return _ENGINE
