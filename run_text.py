import argparse
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "LiquidAI/LFM2.5-1.2B-JP-202606"


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


def move_tensor_values(values: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in values.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Liquid AIのLFM2.5-1.2B-JP-202606の特徴を日本語で3点に要約してください。",
    )
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    args = parser.parse_args()

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    if device.type == "cuda":
        print(
            f"Using {torch.cuda.get_device_name(device)} "
            f"(capability {'.'.join(map(str, torch.cuda.get_device_capability(device)))})",
            flush=True,
        )
    print(f"Loading {MODEL_ID} on {device} with {dtype} ...", flush=True)
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    print(f"Loaded in {time.time() - started:.1f}s", flush=True)

    messages = [{"role": "user", "content": args.prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    )
    if isinstance(encoded, torch.Tensor):
        input_ids = encoded.to(device)
        generate_kwargs = {}
    else:
        encoded = move_tensor_values(encoded, device)
        input_ids = encoded["input_ids"]
        generate_kwargs = {key: value for key, value in encoded.items() if key != "input_ids"}

    print("\n--- prompt ---")
    print(args.prompt)
    print("\n--- output ---", flush=True)
    started = time.time()
    with torch.inference_mode():
        output = model.generate(
            input_ids,
            **generate_kwargs,
            do_sample=True,
            temperature=0.1,
            top_k=50,
            repetition_penalty=1.05,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output[0, input_ids.shape[-1] :].detach().cpu()
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(text.strip())
    print(f"\nGenerated {generated.numel()} tokens in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
