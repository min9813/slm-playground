"""Vision-language demo engine for LiquidAI LFM2.5-VL models.

Two checkpoints are exposed and can be loaded / unloaded independently so the
GPU memory budget (an 11 GB GTX 1080 Ti here, often already occupied by the
LFM2.5-Audio model) stays under control:

- ``vl-1.6b`` — ``LiquidAI/LFM2.5-VL-1.6B`` (general VL / OCR / document QA)
- ``vl-450m`` — ``LiquidAI/LFM2.5-VL-450M`` (tiny; adds visual grounding /
  bounding-box prediction)

The models are *not* lazy-loaded: the UI toggles them on and off explicitly
(``load`` / ``unload``) so the user controls what sits in VRAM at any moment.
"""

from __future__ import annotations

from datetime import datetime
import gc
import io
import json
import os
import re
import threading
import time
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from backend.tts_engine import dtype_name, resolve_device, resolve_dtype


# Selectable checkpoints. ``grounding`` flags the model that supports
# bounding-box / object-detection prompts (the 450M).
VL_MODELS: dict[str, dict[str, Any]] = {
    "vl-1.6b": {
        "model_id": os.environ.get("LIQUID_VL16_MODEL", "LiquidAI/LFM2.5-VL-1.6B"),
        "label": "LFM2.5-VL-1.6B",
        "grounding": False,
        "note": "general VL · OCR · document QA",
    },
    "vl-450m": {
        "model_id": os.environ.get("LIQUID_VL450_MODEL", "LiquidAI/LFM2.5-VL-450M"),
        "label": "LFM2.5-VL-450M",
        "grounding": True,
        "note": "tiny · visual grounding / bbox",
    },
}

# A grounding prompt the 450M understands well: ask for JSON boxes. Used when the
# UI's "grounding" checkbox is ticked and the user's prompt is empty / generic.
GROUNDING_HINT = (
    " Return every matching object as a JSON list of "
    '{"label": str, "bbox_2d": [x1, y1, x2, y2]} with pixel coordinates.'
)


def read_image(source: bytes | bytearray) -> Image.Image:
    """Decode uploaded image bytes into an RGB ``PIL.Image``."""
    image = Image.open(io.BytesIO(source))
    return image.convert("RGB")


def _parse_boxes(text: str, width: int, height: int) -> list[dict[str, Any]]:
    """Best-effort extraction of grounding boxes from a model reply.

    LFM2.5-VL-450M emits bounding boxes as JSON. The exact schema is not pinned
    down in the docs, so we tolerate a few shapes (``bbox_2d`` / ``bbox`` /
    ``box`` keys, dict-or-list coords) and normalise everything to absolute
    pixel ``[x1, y1, x2, y2]``. Coordinates that look normalised (<= 1) or
    0–1000 scaled are rescaled to the real image size. Returns ``[]`` when no
    boxes can be recovered — the caller just shows the raw text in that case.
    """
    boxes: list[dict[str, Any]] = []
    # Grab the first JSON array or object in the text.
    candidates = re.findall(r"\[.*\]|\{.*\}", text, flags=re.DOTALL)
    for blob in candidates:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            coords = item.get("bbox_2d") or item.get("bbox") or item.get("box")
            if isinstance(coords, dict):
                coords = [
                    coords.get("x1") or coords.get("xmin"),
                    coords.get("y1") or coords.get("ymin"),
                    coords.get("x2") or coords.get("xmax"),
                    coords.get("y2") or coords.get("ymax"),
                ]
            if not isinstance(coords, (list, tuple)) or len(coords) != 4:
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in coords)
            except (TypeError, ValueError):
                continue
            boxes.append(
                {
                    "label": str(item.get("label") or item.get("name") or ""),
                    "bbox": [x1, y1, x2, y2],
                }
            )
        if boxes:
            break  # first parseable JSON blob wins
    # Rescale to absolute pixels: normalised (<= 1) or a 0–1000 grid both occur.
    for box in boxes:
        biggest = max(abs(v) for v in box["bbox"])
        if biggest <= 1.0:
            sx, sy = width, height
        elif biggest <= 1000.0 and (width > 1000 or height > 1000):
            sx, sy = width / 1000.0, height / 1000.0
        else:
            sx = sy = 1.0
        x1, y1, x2, y2 = box["bbox"]
        box["bbox"] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
    return boxes


class _VLModel:
    """One loadable LFM2.5-VL checkpoint."""

    def __init__(self, key: str, *, device: torch.device, dtype: torch.dtype) -> None:
        self.key = key
        self.spec = VL_MODELS[key]
        self.model_id = self.spec["model_id"]
        self.device = device
        self.dtype = dtype
        self.model: Any = None
        self.processor: Any = None
        self.loaded_at: datetime | None = None
        self.load_seconds = 0.0

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def load(self) -> float:
        if self.is_loaded:
            return 0.0
        started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=self.dtype,
        ).to(self.device)
        model.eval()
        self.processor = processor
        self.model = model
        self.loaded_at = datetime.now()
        self.load_seconds = time.perf_counter() - started
        return self.load_seconds

    def unload(self) -> bool:
        if not self.is_loaded:
            return False
        self.model = None
        self.processor = None
        self.loaded_at = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return True

    def infer(
        self, image: Image.Image, prompt: str, *, max_new_tokens: int, grounding: bool
    ) -> dict[str, Any]:
        assert self.model is not None and self.processor is not None
        text_prompt = prompt.strip() or ("Detect the main objects." if grounding else "Describe this image.")
        if grounding and "json" not in text_prompt.lower():
            text_prompt += GROUNDING_HINT

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        prompt_started = time.perf_counter()
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(self.device)
        prompt_seconds = time.perf_counter() - prompt_started
        input_tokens = int(inputs["input_ids"].shape[1])

        generation_started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generation_seconds = time.perf_counter() - generation_started

        generated = outputs[:, input_tokens:]
        reply = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        output_tokens = int(generated.shape[1])

        boxes: list[dict[str, Any]] = []
        if grounding:
            boxes = _parse_boxes(reply, image.width, image.height)

        return {
            "text": reply,
            "boxes": boxes,
            "image_size": [image.width, image.height],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "max_new_tokens": max_new_tokens,
            "timings": {
                "prompt_seconds": prompt_seconds,
                "generation_seconds": generation_seconds,
                "total_seconds": prompt_seconds + generation_seconds,
            },
        }


class VLManager:
    """Holds the selectable VL checkpoints and serialises GPU access."""

    def __init__(self, *, device: str | None = None, dtype: str | None = None) -> None:
        self.device = resolve_device(device or os.environ.get("LIQUID_VL_DEVICE", "auto"))
        self.dtype = resolve_dtype(dtype or os.environ.get("LIQUID_VL_DTYPE", "auto"), self.device)
        self._models = {
            key: _VLModel(key, device=self.device, dtype=self.dtype) for key in VL_MODELS
        }
        self._lock = threading.Lock()

    def keys(self) -> list[str]:
        return list(self._models)

    def has(self, key: str) -> bool:
        return key in self._models

    def is_loaded(self, key: str) -> bool:
        return self._models[key].is_loaded

    def load(self, key: str) -> float:
        with self._lock:
            try:
                return self._models[key].load()
            except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
                self._models[key].unload()
                raise RuntimeError(
                    "GPU out of memory while loading this model. Turn another "
                    "model off first (the GTX 1080 Ti can hold only one large "
                    "model at a time)."
                ) from exc

    def unload(self, key: str) -> bool:
        with self._lock:
            return self._models[key].unload()

    def unload_all(self) -> None:
        with self._lock:
            for model in self._models.values():
                model.unload()

    def infer(
        self,
        key: str,
        image: Image.Image,
        prompt: str,
        *,
        max_new_tokens: int,
        grounding: bool,
    ) -> dict[str, Any]:
        with self._lock:
            model = self._models[key]
            if not model.is_loaded:
                raise RuntimeError(
                    f"Model '{key}' is not loaded. Switch it on first."
                )
            try:
                result = model.infer(
                    image, prompt, max_new_tokens=max_new_tokens, grounding=grounding
                )
            except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "GPU out of memory during inference. Try a smaller image, "
                    "fewer max tokens, or unload another model."
                ) from exc
            result["runtime"] = self.runtime_info(key)
            return result

    def runtime_info(self, key: str) -> dict[str, Any]:
        return {
            "model_id": self._models[key].model_id,
            "device": str(self.device),
            "dtype": dtype_name(self.dtype),
        }

    def descriptors(self) -> list[dict[str, Any]]:
        out = []
        for key, spec in VL_MODELS.items():
            model = self._models[key]
            out.append(
                {
                    "key": key,
                    "label": spec["label"],
                    "model_id": spec["model_id"],
                    "grounding": spec["grounding"],
                    "note": spec["note"],
                    "loaded": model.is_loaded,
                    "loaded_at": model.loaded_at.isoformat() if model.loaded_at else None,
                }
            )
        return out


_MANAGER: VLManager | None = None


def get_vl_manager() -> VLManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = VLManager()
    return _MANAGER
