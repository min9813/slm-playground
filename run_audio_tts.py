import argparse
import os
import time
from pathlib import Path

import soundfile as sf
import torch
from safetensors.torch import load_file
from transformers import Lfm2Config
from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor
from liquid_audio.detokenizer import LFM2AudioDetokenizer


MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B-JP"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "text",
        nargs="?",
        default="こんにちは。Liquid AIの日本語音声モデルをテストしています。",
    )
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--output", default="outputs/tts_jp.wav")
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
    processor = LFM2AudioProcessor.from_pretrained(MODEL_ID, device=device).eval()
    processor.to(device=device, dtype=dtype)
    preload_detokenizer(processor, device=device, dtype=dtype)
    model = LFM2AudioModel.from_pretrained(
        MODEL_ID,
        device=device,
        dtype=dtype,
    ).eval()
    print(f"Loaded in {time.time() - started:.1f}s", flush=True)

    chat = ChatState(processor, dtype=dtype)
    chat.new_turn("system")
    chat.add_text("Perform TTS in japanese.")
    chat.end_turn()

    chat.new_turn("user")
    chat.add_text(args.text)
    chat.end_turn()

    print("\n--- text ---")
    print(args.text)
    print("\nGenerating audio tokens ...", flush=True)

    chat.new_turn("assistant")
    started = time.time()
    audio_out = []
    with torch.inference_mode():
        for token in model.generate_sequential(
            **chat,
            max_new_tokens=args.max_new_tokens,
            audio_temperature=0.8,
            audio_top_k=64,
        ):
            if token.numel() > 1:
                audio_out.append(token.detach().to("cpu"))

    if len(audio_out) <= 1:
        raise RuntimeError("No audio tokens were generated; try increasing --max-new-tokens.")

    audio_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0).to(device)
    waveform = processor.decode(audio_codes).cpu()[0]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, waveform, 24_000)
    print(
        f"Wrote {output_path} ({waveform.numel() / 24_000:.2f}s audio) "
        f"in {time.time() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
