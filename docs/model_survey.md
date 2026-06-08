# Local multimodal model survey

Updated: 2026-06-08

Goal: find small / local models that are useful for on-device realtime
interaction, browser/WebGPU experiments, and local vision-language quality
tests. X/Twitter can be useful for discovery, but it is noisy and hard to
reproduce; treat it as a lead source only. This survey uses Hugging Face model
metadata plus official docs / GitHub pages as the durable source of truth.

Status note: the first pass missed several current families. Added in later
revisions: Qwen3.5 native multimodal models, Molmo2 / MolmoE, Bonsai text SLMs,
Bonsai Image 4B, and Gemma 4. Bonsai Image is a low-bit text-to-image diffusion
model, not a vision-language understanding model.

## What to test first

### Tier 1: likely to fit and answer quickly on the GTX 1080 Ti

| Model | Why it matters | Runtime path | First test |
| --- | --- | --- | --- |
| `LiquidAI/LFM2.5-VL-450M` | Current tiny VL baseline; fast edge target; bbox support. | transformers now; GGUF when available for 2.5. | latency, Japanese caption, OCR, bbox |
| `LiquidAI/LFM2.5-VL-1.6B` | Current quality point already integrated. | transformers now; GGUF when available for 2.5. | compare quality vs 450M |
| `HuggingFaceTB/SmolVLM-256M-Instruct` | Very small VLM; ONNX tag makes it a WebGPU candidate. | transformers / Transformers.js. | browser-style q4/q8 and webcam cadence |
| `HuggingFaceTB/SmolVLM-500M-Instruct` | Better small SmolVLM point. | transformers / Transformers.js. | same as 256M |
| `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | Small video-aware model. | transformers / Transformers.js. | frame sampling + temporal QA |
| `vikhyatk/moondream2` | Proven tiny local VLM baseline with strong community use. | custom transformers code / Ollama-style runtimes. | caption, OCR, object query |
| `Qwen/Qwen2-VL-2B-Instruct` | Widely used 2B VLM reference. | transformers; possible GGUF ecosystem. | quality/latency reference |
| `Qwen/Qwen3-VL-2B-Instruct` | Newer Qwen small VLM. | transformers. | compare against Qwen2-VL and Liquid |
| `unsloth/Qwen3.5-0.8B-GGUF` | Preferred Qwen3.5 tiny quantized lead if available. | llama.cpp + mmproj. | OCR, caption, WebGPU / GGUF feasibility |
| `unsloth/Qwen3.5-2B-GGUF` | Best first Qwen3.5 test for this box. | llama.cpp + mmproj. | compare against Qwen3-VL-2B |
| `allenai/MolmoE-1B-0924` | Small Molmo MoE baseline; old but relevant for edge. | transformers, custom code. | caption, pointing/grounding |
| `google/gemma-4-E2B-it` | Gemma 4 efficient any-to-any model; likely first Gemma 4 local test. | transformers; GGUF / ONNX leads exist. | image QA, audio/text path feasibility |

### Tier 2: quality upper bounds for this machine

| Model | Why it matters | Risk |
| --- | --- | --- |
| `Qwen/Qwen2.5-VL-3B-Instruct` | Strong 3B VLM baseline. | May be slow / memory-heavy in fp32 on Pascal. |
| `Qwen/Qwen3-VL-4B-Instruct` | Stronger small Qwen3-VL point. | 11 GB VRAM is tight without quantized runtime. |
| `unsloth/Qwen3.5-4B-GGUF` | Preferred Qwen3.5 4B quality point. | Start with Q4; fp32 official checkpoint is not the target on Pascal. |
| `unsloth/Qwen3.5-9B-GGUF` | Upper Qwen3.5 target for this GPU. | Start with `Q4_K_M`, `Q4_K_S`, `IQ4_XS`, or `UD-Q4_K_XL`; low ctx/image tokens. |
| `unsloth/Qwen3.5-9B-MTP-GGUF` | MTP variant to compare after plain 9B. | Treat as second pass; runtime behavior needs measurement. |
| `allenai/Molmo2-4B` | New Molmo2 image/video/multi-image + grounding model. | Requires `trust_remote_code`; may be heavy but worth testing. |
| `allenai/MolmoWeb-4B` | Molmo2 family variant for web / GUI tasks. | Model-specific behavior; benchmark separately. |
| `google/gemma-4-E4B-it` | Larger Gemma 4 efficient any-to-any model. | Likely tighter on 11 GB; quantized runtime preferred. |
| `google/gemma-4-12B-it` | New Gemma 4 unified 12B any-to-any checkpoint. | Too heavy for fp32 on this GPU; only test quantized GGUF / cloud. |
| `google/gemma-3n-E2B-it` | On-device design; image/audio/video/text tags. | Requires checking local runtime support and dtype behavior. |
| `microsoft/Phi-3.5-vision-instruct` | 4B-class Microsoft VLM reference. | Likely slower; may need model-specific code. |

### Tier 2b: text-only on-device SLMs to pair with vision frontends

These are not VLMs, but they matter for cascaded local agents where vision is
handled by a small VLM and reasoning / tool use is handled by a fast text SLM.

| Model | Why it matters | Runtime path |
| --- | --- | --- |
| `prism-ml/Bonsai-1.7B-gguf` | 1-bit on-device text SLM; small enough for browser/local experiments. | llama.cpp / GGUF |
| `prism-ml/Bonsai-4B-gguf` | Larger Bonsai text SLM. | llama.cpp / GGUF |
| `prism-ml/Ternary-Bonsai-1.7B-gguf` | 1.58-bit text SLM variant. | llama.cpp / GGUF |
| `onnx-community/Bonsai-1.7B-ONNX` | WebGPU / Transformers.js lead. | ONNX / Transformers.js |

### Runs on the GTX 1080 Ti

This is the practical shortlist for the local GPU. Qwen rows prefer Unsloth GGUF
when a matching Unsloth repo exists.

| Status | Models | Notes |
| --- | --- | --- |
| Runs now / first tests | `LiquidAI/LFM2.5-VL-450M`, `LiquidAI/LFM2.5-VL-1.6B`, `HuggingFaceTB/SmolVLM-256M-Instruct`, `HuggingFaceTB/SmolVLM-500M-Instruct`, `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, `vikhyatk/moondream2`, `unsloth/Qwen3.5-0.8B-GGUF`, `unsloth/Qwen3.5-2B-GGUF`, `prism-ml/Bonsai-1.7B-gguf`, `prism-ml/Ternary-Bonsai-1.7B-gguf` | These should fit comfortably or have a small-enough runtime path. |
| Runs with quant / measure carefully | `unsloth/Qwen3.5-4B-GGUF`, `unsloth/Qwen3.5-9B-GGUF`, `unsloth/gemma-4-E2B-it-GGUF`, `unsloth/gemma-4-E4B-it-GGUF`, `prism-ml/Bonsai-4B-gguf`, `Green-Sky/bonsai-image-binary-4B-GGUF` | Use Q4-class GGUF, short context, and low image resolution/tokens. |
| Borderline | `unsloth/Qwen3.5-9B-MTP-GGUF`, `google/gemma-4-12B-it` via GGUF, `HuggingFaceTB/SmolVLM2-2.2B-Instruct`, `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen3-VL-4B-Instruct` | May fit but not a realtime target on this Pascal GPU. |
| Not a good GPU target | `allenai/Molmo2-4B`, `allenai/Molmo2-8B`, `allenai/MolmoWeb-4B`, `google/gemma-4-26B-A4B-it`, `google/gemma-4-31B-it`, `Qwen/Qwen2.5-Omni-7B` | Too heavy in fp32 or too much setup for the first local benchmark pass. |

### Tier 2c: local image generation / "vision output"

These are not VLMs. They are useful if the broader goal includes on-device image
generation or image-to-image experiments.

| Model | Why it matters | Runtime path |
| --- | --- | --- |
| `prism-ml/bonsai-image-binary-4B-gemlite-1bit` | 1-bit Bonsai Image 4B; CUDA/GemLite text-to-image lead. | diffusers / GemLite |
| `prism-ml/bonsai-image-ternary-4B-gemlite-2bit` | 1.58-bit Bonsai Image 4B; CUDA/GemLite text-to-image lead. | diffusers / GemLite |
| `prism-ml/bonsai-image-binary-4B-mlx-1bit` | Apple Silicon path. | MLX |
| `prism-ml/bonsai-image-ternary-4B-mlx-2bit` | Apple Silicon ternary path. | MLX |
| `Green-Sky/bonsai-image-binary-4B-GGUF` | Community GGUF / mobile-on-device lead. | verify stable-diffusion.cpp / GGUF runtime |

### Tier 3: realtime conversation / audio leads

| Model | Role | Expected local outcome |
| --- | --- | --- |
| `LiquidAI/LFM2.5-Audio-1.5B-JP` | Current local realtime speech baseline. | Already works; optimize app latency, not model runtime. |
| `Qwen/Qwen2.5-Omni-7B` | Any-to-any: text/image/video/audio in, text/speech out. | Interesting, but probably too heavy for 1080 Ti realtime. |
| `Qwen/Qwen3.5-35B-A3B` / Qwen3.5 Omni leads | Newer native multimodal / omni direction. | Too heavy for this GPU; track for cloud or future hardware. |
| `google/gemma-4-E2B-it` | Any-to-any local candidate, including image/text and possibly audio paths. | First Gemma 4 realtime feasibility check. |
| `google/gemma-4-12B-it` | New unified Gemma 4 12B checkpoint. | Needs quantized path / larger GPU. |
| `kyutai/moshiko-pytorch-bf16` | Full-duplex speech dialogue. | Research lead; likely too heavy and English-oriented. |
| `gpt-omni/mini-omni` | Small experimental speech-to-speech. | Worth a quick quality check if setup is simple. |
| `microsoft/VibeVoice-Realtime-0.5B` | Streaming TTS, not full dialogue. | Useful for cascaded STT -> LLM -> TTS latency tests. |

## Benchmark protocol

### Vision-language

Measure:

- model load seconds
- peak VRAM and steady VRAM
- prompt/image preprocessing seconds
- generation seconds and output tokens/sec
- end-to-end seconds
- answer quality on a fixed local image set

Use a compact test set:

| Set | Images | Prompts |
| --- | --- | --- |
| caption | 5 natural / UI / indoor images | `Describe this image in Japanese.` |
| OCR | 5 screenshots / labels / receipts | `Read all visible text.` |
| grounding | 5 object images | `Return JSON boxes for ...` |
| document | 5 forms / tables | `Answer this question from the document.` |
| realtime camera | 30 sampled webcam frames | short caption every N frames |

Start with:

```bash
uv run python scripts/hf_model_inventory.py

uv run python scripts/bench_vlm_transformers.py \
  --model LiquidAI/LFM2.5-VL-450M \
  --max-new-tokens 64 --runs 3 --dtype float32

cmake --build .llamacpp/build --target llama-server -j
source scripts/llamacpp_env.sh
llama-server \
  -hf unsloth/Qwen3.5-2B-GGUF:Qwen3.5-2B-Q4_K_M.gguf \
  --mmproj-auto -ngl 99 -c 4096

uv run python scripts/bench_vlm_transformers.py \
  --model allenai/Molmo2-4B \
  --max-new-tokens 64 --runs 3 --dtype float32 --trust-remote-code

uv run python scripts/bench_vlm_transformers.py \
  --model google/gemma-4-E2B-it \
  --max-new-tokens 64 --runs 3 --dtype float32
```

For this GTX 1080 Ti, keep `float32` as the default for transformers. Pascal
does not have Tensor Cores, and previous fp16 tests were slower or failed.
Quantized GGUF / ONNX / WebGPU paths should be tested separately.

### Realtime target metrics

For voice / camera interaction, tokens/sec alone is not enough. Track:

- time to first text token
- time to first audio chunk
- total turn latency
- realtime factor for TTS (`generation_seconds / audio_seconds`)
- frame cadence for camera (`processed_frames_per_second`)
- interruption behavior and VAD delay

For practical conversation, prefer a cascaded baseline:

```text
streaming ASR -> small local LLM/VLM -> streaming TTS
```

Native omni models are worth testing, but local realtime success depends on
time-to-first-audio, not just whether the model can generate speech.

## Current known local result

Generic transformers runner sanity check:

```text
LiquidAI/LFM2.5-VL-450M
runtime: cuda / float32
load: 23.86s
prompt: 0.052s
generation: 0.990s
output: 20 tok, 20.19 tok/s
memory: used 2075 MB, peak allocated 1763 MB
sample: この画像には、赤い長方形と青い円が描かれています。
```

Existing Liquid-specific benchmark results remain in `docs/acceleration.md`.

## Current gaps / corrections

- The previous survey was not latest: Qwen3.5, Molmo2, MolmoE, and Bonsai were
  missing.
- There is no need to search specifically for `Qwen3.5-VL` as the official
  Qwen3.5 line is already tagged `image-text-to-text`; the HF Transformers docs
  describe Qwen3.5 as natively multimodal with image/video token support.
- `Molmo2-4B` is now a high-priority test because it supports image, video,
  multi-image understanding, and grounding, and has Transformers examples.
- `Bonsai` has two relevant branches:
  - Bonsai text SLMs (`Bonsai-1.7B/4B/8B`) for fast local reasoning.
  - Bonsai Image 4B for low-bit text-to-image generation. This is "vision" in
    the image generation sense, not VLM perception.
- `Gemma 4` supersedes the older Gemma 3n row for new testing. Start with
  `google/gemma-4-E2B-it`; treat 12B/26B/31B as quantized/cloud targets.

## Sources

- Liquid LFM2.5-VL-450M model card: https://huggingface.co/LiquidAI/LFM2.5-VL-450M
- Liquid vision model docs: https://docs.liquid.ai/docs/models/vision-models
- SmolVLM2 blog / model collection: https://huggingface.co/blog/smolvlm2
- Qwen3-VL-2B model card: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- Qwen3.5 Transformers docs: https://huggingface.co/docs/transformers/en/model_doc/qwen3_5
- Unsloth Qwen3.5-2B GGUF: https://huggingface.co/unsloth/Qwen3.5-2B-GGUF
- Unsloth Qwen3.5-9B GGUF: https://huggingface.co/unsloth/Qwen3.5-9B-GGUF
- Molmo2-4B model card: https://huggingface.co/allenai/Molmo2-4B
- MolmoE-1B model card: https://huggingface.co/allenai/MolmoE-1B-0924
- Bonsai GGUF: https://huggingface.co/prism-ml/Bonsai-1.7B-gguf
- Bonsai ONNX: https://huggingface.co/onnx-community/Bonsai-1.7B-ONNX
- Bonsai Image 4B announcement: https://prismml.com/news/bonsai-image-4b
- Bonsai Image binary MLX: https://huggingface.co/prism-ml/bonsai-image-binary-4B-mlx-1bit
- Bonsai Image ternary GemLite: https://huggingface.co/prism-ml/bonsai-image-ternary-4B-gemlite-2bit
- Gemma 4 HF blog: https://huggingface.co/blog/gemma4
- Gemma 4 E2B: https://huggingface.co/google/gemma-4-E2B-it
- Gemma 4 12B: https://huggingface.co/google/gemma-4-12B-it
- Gemma 3n overview: https://ai.google.dev/gemma/docs/gemma-3n
- Qwen2.5-Omni GitHub: https://github.com/QwenLM/Qwen2.5-Omni
- Moshi GitHub: https://github.com/kyutai-labs/moshi
- Transformers.js WebGPU / quantization notes: https://github.com/huggingface/transformers.js
