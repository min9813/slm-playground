#!/usr/bin/env python3
"""Benchmark Hugging Face transformers image-to-text models.

This is a practical first-pass harness for small VLMs that implement the common
chat-template image-text interface. It records load time, peak VRAM, prompt
processing time, generation time, and output tokens. Models with custom APIs
(for example Moondream2) should get a dedicated runner once selected.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor


def synthetic_image() -> Image.Image:
    image = Image.new("RGB", (640, 426), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([60, 80, 260, 300], fill="#c0392b")
    draw.ellipse([320, 120, 520, 320], fill="#2980b9")
    draw.text((70, 330), "local VLM benchmark", fill="black")
    return image


def cuda_mem_mb(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"used": 0.0, "peak": 0.0}
    free, total = torch.cuda.mem_get_info(device)
    return {
        "used": (total - free) / 1024 / 1024,
        "peak": torch.cuda.max_memory_allocated(device) / 1024 / 1024,
    }


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 7:
            return torch.float16
        return torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def build_inputs(processor: Any, image: Image.Image, prompt: str, device: torch.device) -> Any:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    try:
        return processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(device)
    except TypeError:
        # Some processors expect image placeholders in the text and images separately.
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        return processor(text=text, images=image, return_tensors="pt").to(device)


def run_model(args: argparse.Namespace, image: Image.Image) -> dict[str, Any]:
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()
    load_seconds = time.perf_counter() - load_started

    rows = []
    last_text = ""
    for run_idx in range(args.runs + args.warmups):
        prompt_started = time.perf_counter()
        inputs = build_inputs(processor, image, args.prompt, device)
        prompt_seconds = time.perf_counter() - prompt_started
        input_tokens = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0

        gen_started = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        generation_seconds = time.perf_counter() - gen_started

        generated = outputs[:, input_tokens:] if input_tokens else outputs
        try:
            text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        except Exception:
            text = ""
        output_tokens = int(generated.shape[1])

        if run_idx >= args.warmups:
            rows.append(
                {
                    "prompt_seconds": prompt_seconds,
                    "generation_seconds": generation_seconds,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "tok_per_s": output_tokens / generation_seconds if generation_seconds else 0.0,
                }
            )
            last_text = text

    mem = cuda_mem_mb(device)
    del model, processor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    avg = {
        key: sum(row[key] for row in rows) / len(rows)
        for key in ("prompt_seconds", "generation_seconds", "input_tokens", "output_tokens", "tok_per_s")
    }
    return {
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "image_size": [image.width, image.height],
        "load_seconds": load_seconds,
        "memory_mb": mem,
        "runs": rows,
        "average": avg,
        "sample_text": last_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local path.")
    parser.add_argument("--image", help="Image path. Uses a synthetic image when omitted.")
    parser.add_argument("--prompt", default="Describe this image in one Japanese sentence.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB") if args.image else synthetic_image()
    result = run_model(args, image)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    avg = result["average"]
    print(f"model: {result['model']}")
    print(f"runtime: {result['device']} / {result['dtype']}")
    print(f"load: {result['load_seconds']:.2f}s")
    print(
        f"avg: prompt {avg['prompt_seconds']:.3f}s · gen {avg['generation_seconds']:.3f}s · "
        f"{avg['output_tokens']:.1f} tok · {avg['tok_per_s']:.2f} tok/s"
    )
    print(
        f"memory: used {result['memory_mb']['used']:.0f} MB · "
        f"peak allocated {result['memory_mb']['peak']:.0f} MB"
    )
    print(f"sample: {result['sample_text']}")


if __name__ == "__main__":
    main()
