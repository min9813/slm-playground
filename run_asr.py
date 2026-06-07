import argparse
import time

from backend.tts_engine import TTSEngine, read_wave


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe Japanese speech with LiquidAI/LFM2.5-Audio (ASR)."
    )
    parser.add_argument("audio", help="Path to a WAV/FLAC/OGG file to transcribe.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    args = parser.parse_args()

    engine = TTSEngine(device=args.device, dtype=args.dtype)
    print(f"Loading {engine.status()['model_id']} on {engine.device} with {engine.dtype} ...", flush=True)

    wave, sample_rate = read_wave(args.audio)
    print(f"Input: {args.audio} ({wave.shape[1] / sample_rate:.2f}s @ {sample_rate} Hz)", flush=True)

    started = time.time()
    result = engine.transcribe(wave, sample_rate, max_new_tokens=args.max_new_tokens)

    print("\n--- transcript ---")
    print(result["text"])
    print(
        f"\n({result['text_tokens']} tokens in {time.time() - started:.1f}s, "
        f"load {result['timings']['model_load_seconds']:.1f}s, "
        f"generation {result['timings']['generation_seconds']:.1f}s)"
    )


if __name__ == "__main__":
    main()
