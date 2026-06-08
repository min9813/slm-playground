#!/usr/bin/env python3
"""List local / on-device multimodal model candidates from Hugging Face.

The list is intentionally curated: it covers small VLMs, on-device multimodal
models, and realtime audio candidates that are relevant to this repo's goals.
It does not download weights.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

from huggingface_hub import HfApi


@dataclass(frozen=True)
class Candidate:
    model_id: str
    group: str
    note: str


DEFAULT_CANDIDATES: tuple[Candidate, ...] = (
    Candidate("LiquidAI/LFM2.5-VL-450M", "vlm-fast", "current tiny Liquid VL baseline"),
    Candidate("LiquidAI/LFM2.5-VL-1.6B", "vlm-fast", "current larger Liquid VL baseline"),
    Candidate("LiquidAI/LFM2-VL-450M-GGUF", "vlm-gguf", "llama.cpp path for Liquid VL"),
    Candidate("LiquidAI/LFM2-VL-1.6B-GGUF", "vlm-gguf", "llama.cpp path for Liquid VL"),
    Candidate("HuggingFaceTB/SmolVLM-256M-Instruct", "vlm-webgpu", "smallest SmolVLM; ONNX/WebGPU candidate"),
    Candidate("HuggingFaceTB/SmolVLM-500M-Instruct", "vlm-webgpu", "small SmolVLM; ONNX/WebGPU candidate"),
    Candidate("HuggingFaceTB/SmolVLM2-256M-Video-Instruct", "video-vlm-webgpu", "tiny video-aware SmolVLM2"),
    Candidate("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "video-vlm-webgpu", "small video-aware SmolVLM2"),
    Candidate("HuggingFaceTB/SmolVLM2-2.2B-Instruct", "video-vlm-quality", "upper SmolVLM2 quality point"),
    Candidate("vikhyatk/moondream2", "vlm-edge", "tiny custom-code VLM; strong local baseline"),
    Candidate("Qwen/Qwen2-VL-2B-Instruct", "vlm-quality", "widely used 2B VLM baseline"),
    Candidate("Qwen/Qwen2.5-VL-3B-Instruct", "vlm-quality", "stronger 3B VLM baseline"),
    Candidate("Qwen/Qwen3-VL-2B-Instruct", "vlm-quality", "new Qwen3-VL small baseline"),
    Candidate("Qwen/Qwen3-VL-4B-Instruct", "vlm-quality", "quality upper bound for 11 GB VRAM tests"),
    Candidate("unsloth/Qwen3.5-0.8B-GGUF", "native-vlm-gguf-fast", "preferred Qwen3.5 tiny GGUF if available"),
    Candidate("unsloth/Qwen3.5-2B-GGUF", "native-vlm-gguf-fast", "preferred Qwen3.5 first test on this GPU"),
    Candidate("unsloth/Qwen3.5-4B-GGUF", "native-vlm-gguf-quality", "preferred Qwen3.5 4B quantized lead"),
    Candidate("unsloth/Qwen3.5-9B-GGUF", "native-vlm-gguf-quality", "upper Qwen3.5 target for GTX 1080 Ti; start Q4"),
    Candidate("unsloth/Qwen3.5-9B-MTP-GGUF", "native-vlm-gguf-quality", "MTP variant; benchmark after plain 9B"),
    Candidate("allenai/MolmoE-1B-0924", "vlm-edge", "small Molmo MoE; old but important edge baseline"),
    Candidate("allenai/Molmo-7B-D-0924", "vlm-quality-heavy", "original Molmo open VLM baseline"),
    Candidate("allenai/Molmo2-4B", "video-vlm-quality", "Molmo2 image/video/multi-image + grounding; first Molmo2 test"),
    Candidate("allenai/Molmo2-8B", "video-vlm-quality-heavy", "Molmo2 quality point; likely heavy for 1080 Ti"),
    Candidate("allenai/MolmoWeb-4B", "vlm-web", "Molmo2 variant aimed at web / GUI tasks"),
    Candidate("prism-ml/Bonsai-1.7B-gguf", "slm-gguf", "1-bit on-device text SLM; not VLM"),
    Candidate("prism-ml/Bonsai-4B-gguf", "slm-gguf", "1-bit on-device text SLM; not VLM"),
    Candidate("prism-ml/Ternary-Bonsai-1.7B-gguf", "slm-gguf", "1.58-bit on-device text SLM; not VLM"),
    Candidate("onnx-community/Bonsai-1.7B-ONNX", "slm-webgpu", "Bonsai ONNX / Transformers.js lead; text-only"),
    Candidate("prism-ml/bonsai-image-binary-4B-gemlite-1bit", "imagegen-on-device", "Bonsai Image 4B binary CUDA/GemLite text-to-image"),
    Candidate("prism-ml/bonsai-image-ternary-4B-gemlite-2bit", "imagegen-on-device", "Bonsai Image 4B ternary CUDA/GemLite text-to-image"),
    Candidate("prism-ml/bonsai-image-binary-4B-mlx-1bit", "imagegen-apple", "Bonsai Image 4B binary MLX text-to-image"),
    Candidate("prism-ml/bonsai-image-ternary-4B-mlx-2bit", "imagegen-apple", "Bonsai Image 4B ternary MLX text-to-image"),
    Candidate("Green-Sky/bonsai-image-binary-4B-GGUF", "imagegen-gguf", "community Bonsai Image GGUF lead; verify runtime"),
    Candidate("google/gemma-4-E2B-it", "omni-on-device", "Gemma 4 efficient any-to-any; first Gemma 4 local test"),
    Candidate("google/gemma-4-E4B-it", "omni-on-device", "Gemma 4 larger efficient any-to-any"),
    Candidate("google/gemma-4-12B-it", "omni-quality-heavy", "Gemma 4 unified 12B; likely too heavy for 1080 Ti fp32"),
    Candidate("google/gemma-4-26B-A4B-it", "vlm-quality-heavy", "Gemma 4 MoE image-text quality point; not this GPU first"),
    Candidate("google/gemma-4-31B-it", "vlm-quality-heavy", "Gemma 4 dense image-text upper bound; not this GPU first"),
    Candidate("unsloth/gemma-4-E2B-it-GGUF", "omni-gguf", "Gemma 4 E2B GGUF lead"),
    Candidate("unsloth/gemma-4-E4B-it-GGUF", "omni-gguf", "Gemma 4 E4B GGUF lead"),
    Candidate("onnx-community/gemma-4-E2B-it-ONNX", "omni-webgpu", "Gemma 4 E2B ONNX / WebGPU lead"),
    Candidate("google/gemma-3n-E2B-it", "omni-on-device", "on-device image/audio/video/text candidate"),
    Candidate("google/gemma-3n-E4B-it", "omni-on-device", "larger Gemma 3n quality point"),
    Candidate("microsoft/Phi-3.5-vision-instruct", "vlm-quality", "4B-class Microsoft VLM"),
    Candidate("microsoft/Phi-4-multimodal-instruct", "omni-quality", "audio+vision multimodal, likely heavy"),
    Candidate("Qwen/Qwen2.5-Omni-7B", "realtime-omni-heavy", "speech/image/video/text in, speech/text out"),
    Candidate("kyutai/moshiko-pytorch-bf16", "realtime-audio-heavy", "full-duplex speech dialogue candidate"),
    Candidate("gpt-omni/mini-omni", "realtime-audio-small", "0.5B experimental speech-to-speech"),
    Candidate("microsoft/VibeVoice-Realtime-0.5B", "streaming-tts", "small realtime streaming TTS"),
)


def escape_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def print_markdown(rows: Iterable[tuple[Candidate, object]]) -> None:
    print("| group | model | downloads | likes | pipeline | tags | note |")
    print("| --- | --- | ---: | ---: | --- | --- | --- |")
    for cand, info in rows:
        if isinstance(info, Exception):
            print(
                f"| {escape_cell(cand.group)} | `{escape_cell(cand.model_id)}` | "
                f"ERR | ERR | ERR | ERR | {escape_cell(info)} |"
            )
            continue
        tags = ", ".join((info.tags or [])[:8])
        print(
            f"| {escape_cell(cand.group)} | `{escape_cell(cand.model_id)}` | "
            f"{info.downloads or 0} | {info.likes or 0} | "
            f"{escape_cell(info.pipeline_tag)} | {escape_cell(tags)} | {escape_cell(cand.note)} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", help="Optional extra model IDs to query.")
    args = parser.parse_args()

    candidates = list(DEFAULT_CANDIDATES)
    candidates.extend(Candidate(model_id, "custom", "user-supplied") for model_id in args.models)

    api = HfApi()
    rows = []
    for cand in candidates:
        try:
            rows.append((cand, api.model_info(cand.model_id)))
        except Exception as exc:  # network / gated / deleted model
            rows.append((cand, exc))
    print_markdown(rows)


if __name__ == "__main__":
    main()
