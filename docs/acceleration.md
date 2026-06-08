# Inference acceleration on a GTX 1080 Ti (Pascal, SM 6.1)

We accelerate the three model families in this repo with **llama.cpp + GGUF
(INT8 / Q8_0 and Q4)**, run on CUDA. On this Pascal card the only relevant
hardware accelerator is the `dp4a` INT8 instruction, which llama.cpp's MMQ
kernels exploit — so integer-quantized GGUF is the one path that actually
speeds things up. fp16/Tensor-core paths (TensorRT, vLLM, AWQ/Marlin,
FlashAttention-2, bitsandbytes-int8, torch.compile/Triton) require SM 7.5+ and
do **not** run here; ONNX-Runtime CUDA does not accelerate INT4/INT8 on Pascal
(falls back to CPU). See the research notes at the bottom.

## Setup (no system CUDA, no sudo)

```bash
bash scripts/build_llamacpp_cuda.sh   # assembles CUDA 12.6 toolkit + builds llama.cpp (sm_61)
source scripts/llamacpp_env.sh        # puts llama-* on PATH with the CUDA libs
```

GGUF weights live under `gguf/` (download with `hf download LiquidAI/<repo> <file> --local-dir gguf/<name>`).

## Measured results (GTX 1080 Ti, 11 GB; greedy decode)

### ① Text — `LFM2.5-1.2B-JP-202606`
| Runtime | gen tok/s | prefill tok/s | VRAM | vs fp32 |
| --- | ---: | ---: | ---: | ---: |
| transformers fp32 (baseline) | 61.9 | — | 5.0 GB | 1.0× |
| **llama.cpp Q8_0 (int8)** | **187.4** | 6521 | ~1.2 GB | **3.0×** |
| llama.cpp Q4_K_M | 258.9 | 6249 | ~0.7 GB | 4.2× |

```bash
llama-bench -m gguf/text/LFM2.5-1.2B-JP-202606-Q8_0.gguf -ngl 99
llama-cli   -m gguf/text/LFM2.5-1.2B-JP-202606-Q8_0.gguf -ngl 99 -p "日本語で挨拶して。"
```

### ② Vision-Language — `LFM2.5-VL-1.6B` / `LFM2.5-VL-450M`
Decode speed is set by the LM backbone (mmproj/vision only affects image prefill).
Image captioning verified correct in Japanese.

| Model | Runtime | gen tok/s | vs fp32 |
| --- | --- | ---: | ---: |
| VL-1.6B | transformers fp32 | 38.7 | 1.0× |
| VL-1.6B | **llama.cpp Q8_0 (int8)** | **185.9** | **4.8×** |
| VL-450M | transformers fp32 | 64.5 | 1.0× |
| VL-450M | **llama.cpp Q8_0 (int8)** | **409.3** | **6.3×** |

```bash
llama-mtmd-cli -m gguf/vl16/LFM2.5-VL-1.6B-Q8_0.gguf \
  --mmproj gguf/vl16/mmproj-LFM2.5-VL-1.6b-Q8_0.gguf \
  --image outputs/vl_test.png -p "この画像を説明して。" -ngl 99
```
Further prefill savings: reduce image tokens (fewer tiles / smaller
`max_image_tokens`); keep tokens higher for 450M bbox grounding accuracy.

### ③ Audio — `LFM2.5-Audio-1.5B-JP` (the hard one)
| Capability | llama.cpp on GPU | Notes |
| --- | --- | --- |
| **ASR (speech→text)** | ✅ **works** | `-sys "Perform ASR."`; transcribed `こんにちは。` correctly. Decode 186 tok/s (Q8_0). |
| Audio understanding (speech→text reply) | ✅ works | audio-in, text-out |
| **TTS / speech-out / full S2S** | ❌ not available for Liquid audio yet | upstream `llama-tts` exists, but it is currently OuteTTS/WavTokenizer-oriented and does not consume Liquid's 4-part audio stack (`model` + `mmproj` + `tokenizer` + `vocoder`). Keep using `liquid-audio` (fp32) for TTS/S2S. |

```bash
llama-mtmd-cli -m gguf/audio/LFM2.5-Audio-1.5B-JP-Q8_0.gguf \
  --mmproj gguf/audio/mmproj-LFM2.5-Audio-1.5B-JP-Q8_0.gguf \
  -sys "Perform ASR." --audio outputs/tts_jp.wav -p " " -ngl 99
```

## Summary
INT8 GGUF via llama.cpp gives **3.0–6.3× faster generation** and **2–7× less
VRAM** across text, VL, and audio-ASR on this Pascal GPU, with quality intact.
TTS/S2S audio *output* stays on `liquid-audio` until llama.cpp's Liquid audio
runner lands upstream.

## Tracking Liquid audio output in llama.cpp

Current upstream check:

- llama.cpp HEAD checked locally: `f0156d1401500512ad85042ccf38970568b12253`
- `tools/tts` / `llama-tts` builds and runs for OuteTTS-style TTS.
- Liquid audio GGUF files are present locally:
  - `gguf/audio/LFM2.5-Audio-1.5B-JP-Q8_0.gguf`
  - `gguf/audio/mmproj-LFM2.5-Audio-1.5B-JP-Q8_0.gguf`
  - `gguf/audio/tokenizer-LFM2.5-Audio-1.5B-JP-Q8_0.gguf`
  - `gguf/audio/vocoder-LFM2.5-Audio-1.5B-JP-Q8_0.gguf`
- The current `llama-tts` CLI accepts `-m` and `-mv`, but has no Liquid-specific
  `mmproj` / audio tokenizer path. A direct probe fails while loading the Liquid
  vocoder:

```bash
source scripts/llamacpp_env.sh
llama-tts \
  -m gguf/audio/LFM2.5-Audio-1.5B-JP-Q8_0.gguf \
  -mv gguf/audio/vocoder-LFM2.5-Audio-1.5B-JP-Q8_0.gguf \
  -p "こんにちは。" -ngl 99
```

Expected current failure:

```text
unknown model architecture: 'this model cannot be used as LLM, use it via --model-vocoder in TTS examples'
```

So the next useful action is not TensorRT/ONNX work, but periodically updating
llama.cpp and checking whether Liquid's 4-part speech-output runner lands.

## What was evaluated and skipped (Pascal SM 6.1)
- **TensorRT 10 / vLLM** — require SM 7.5+ → cannot run.
- **AWQ / Marlin / FP8 / FlashAttention-2** — require SM 8.0/7.5 → excluded.
- **bitsandbytes int8 (LLM.int8())** — requires SM 7.5; 4-bit runs but memory-only/slow.
- **torch.compile (Triton)** — Triton needs SM 7.0 → eager fallback only.
- **ONNX Runtime CUDA** — fp32 ≈ parity; INT4/INT8 (MatMulNBits) falls back to CPU on Pascal (WebGPU is the accelerated quant path, not CUDA).
- **fp16/bf16 in transformers** — *slower* on Pascal (fp16 ≈ 1/64 fp32); keep fp32.
