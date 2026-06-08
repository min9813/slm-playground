"""CLI for LiquidAI LFM2.5-VL image understanding + a fp16/fp32 speed benchmark.

Examples:
    # describe an image with the 1.6B model
    uv run python run_vl.py photo.jpg --model 1.6b --prompt "What is this?"

    # benchmark fp16 vs fp32 on both models (synthetic image if none given)
    uv run python run_vl.py --benchmark --model both --dtype both --runs 3

Set HF_HOME to reuse the repo cache, e.g. `HF_HOME=$(pwd)/hf-cache`.
"""

from __future__ import annotations

import argparse
import time

import torch
from PIL import Image, ImageDraw

from backend.vl_engine import VL_MODELS, VLManager, read_image

MODEL_KEYS = {"1.6b": "vl-1.6b", "450m": "vl-450m"}
DTYPES = {"fp16": "float16", "fp32": "float32"}


def synthetic_image() -> Image.Image:
    """A deterministic test image so benchmark runs are comparable."""
    img = Image.new("RGB", (640, 426), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([60, 80, 260, 300], fill="#c0392b")
    d.ellipse([320, 120, 520, 320], fill="#2980b9")
    d.text((70, 330), "LFM2.5-VL benchmark image", fill="black")
    return img


def vram_used_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / 1024 / 1024


def run_one(key: str, dtype_name: str, image: Image.Image, prompt: str,
            max_new_tokens: int, runs: int) -> dict:
    mgr = VLManager(dtype=dtype_name)
    load_s = mgr.load(key)
    vram = vram_used_mb(mgr.device)

    # warmup (kernel autotuning / cache) — not counted
    mgr.infer(key, image, prompt, max_new_tokens=max_new_tokens, grounding=False)

    samples = []
    last_text = ""
    for _ in range(runs):
        r = mgr.infer(key, image, prompt, max_new_tokens=max_new_tokens, grounding=False)
        samples.append((r["timings"]["generation_seconds"], r["output_tokens"]))
        last_text = r["text"]

    mgr.unload(key)
    if mgr.device.type == "cuda":
        torch.cuda.empty_cache()

    gen = sum(s[0] for s in samples) / len(samples)
    toks = sum(s[1] for s in samples) / len(samples)
    return {
        "model": VL_MODELS[key]["label"],
        "dtype": dtype_name,
        "device": str(mgr.device),
        "load_s": load_s,
        "vram_mb": vram,
        "gen_s": gen,
        "out_tokens": toks,
        "tok_per_s": toks / gen if gen else 0.0,
        "sample_text": last_text,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="LFM2.5-VL CLI + fp16/fp32 benchmark.")
    p.add_argument("image", nargs="?", help="image path (synthetic if omitted)")
    p.add_argument("--prompt", default="Describe this image in one sentence.")
    p.add_argument("--model", choices=["1.6b", "450m", "both"], default="1.6b")
    p.add_argument("--dtype", choices=["fp16", "fp32", "both"], default="fp32")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--runs", type=int, default=3, help="timed runs (benchmark mode)")
    p.add_argument("--benchmark", action="store_true", help="report timing table")
    args = p.parse_args()

    image = read_image(open(args.image, "rb").read()) if args.image else synthetic_image()
    keys = list(MODEL_KEYS.values()) if args.model == "both" else [MODEL_KEYS[args.model]]
    dtypes = list(DTYPES.values()) if args.dtype == "both" else [DTYPES[args.dtype]]

    if not args.benchmark:
        key, dtype = keys[0], dtypes[0]
        mgr = VLManager(dtype=dtype)
        mgr.load(key)
        r = mgr.infer(key, image, args.prompt, max_new_tokens=args.max_new_tokens, grounding=False)
        print(r["text"])
        print(f"[{VL_MODELS[key]['label']} · {dtype} · "
              f"gen {r['timings']['generation_seconds']:.2f}s · {r['output_tokens']} tok]")
        return

    results = []
    for key in keys:
        for dtype in dtypes:
            print(f"running {VL_MODELS[key]['label']} / {dtype} ...", flush=True)
            t0 = time.perf_counter()
            results.append(run_one(key, dtype, image, args.prompt,
                                   args.max_new_tokens, args.runs))
            print(f"  done in {time.perf_counter() - t0:.1f}s")

    print("\n" + "=" * 92)
    print(f"{'model':<20}{'dtype':<9}{'load s':>8}{'vram MB':>10}"
          f"{'gen s':>9}{'out tok':>9}{'tok/s':>9}")
    print("-" * 92)
    for r in results:
        print(f"{r['model']:<20}{r['dtype']:<9}{r['load_s']:>8.1f}{r['vram_mb']:>10.0f}"
              f"{r['gen_s']:>9.2f}{r['out_tokens']:>9.0f}{r['tok_per_s']:>9.2f}")
    print("=" * 92)
    print(f"device: {results[0]['device']} · runs/avg: {args.runs} · "
          f"max_new_tokens: {args.max_new_tokens}")


if __name__ == "__main__":
    main()
