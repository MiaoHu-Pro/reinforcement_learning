#!/usr/bin/env python3
"""Load local GPT-2 and compare generation with and without a KV cache."""

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = "~/scratch/llms_model/gpt2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text with a pretrained GPT-2 model stored locally."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt", default="Hello, I am")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=("compare", "cache", "no-cache"),
        default="compare",
        help="Benchmark both methods (default), or run just one method.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use temperature/top-k sampling; the default is greedy decoding.",
    )
    return parser.parse_args()


def reset_seed(seed: int, device: torch.device) -> None:
    """Make cache and no-cache sampling directly comparable."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def generate_once(
    model,
    inputs: dict[str, torch.Tensor],
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    use_cache: bool,
) -> tuple[torch.Tensor, float, int, int | None]:
    """Generate once and return IDs, seconds, token count and peak GPU bytes."""
    reset_seed(args.seed, device)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "use_cache": use_cache,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": args.sample,
    }
    if args.sample:
        generation_kwargs.update(
            temperature=args.temperature,
            top_k=args.top_k,
        )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]
    peak_bytes = (
        torch.cuda.max_memory_allocated() if device.type == "cuda" else None
    )
    return output_ids, elapsed, generated_tokens, peak_bytes


def print_result(name: str, elapsed: float, tokens: int, peak_bytes: int | None) -> None:
    tokens_per_second = tokens / elapsed if elapsed else float("inf")
    print(f"\n{name}")
    print(f"  Generated tokens: {tokens}")
    print(f"  Time:             {elapsed:.4f} seconds")
    print(f"  Speed:            {tokens_per_second:.2f} tokens/second")
    if peak_bytes is not None:
        print(f"  Peak GPU memory:  {peak_bytes / 1024**3:.3f} GiB")


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"GPT-2 directory does not exist: {model_path}\n"
            "Pass the correct location with --model-path."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reset_seed(args.seed, device)

    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")

    # local_files_only=True guarantees that Transformers will not download files.
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    # GPT-2 has no native padding token. EOS is safe for single-prompt generation.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    prompt_tokens = inputs["input_ids"].shape[1]
    max_positions = model.config.max_position_embeddings
    if prompt_tokens + args.max_new_tokens > max_positions:
        raise ValueError(
            f"Prompt ({prompt_tokens} tokens) + requested output "
            f"({args.max_new_tokens} tokens) exceeds GPT-2's "
            f"{max_positions}-token context window."
        )

    methods = []
    if args.mode in ("compare", "cache"):
        methods.append(("WITH KV cache", True))
    if args.mode in ("compare", "no-cache"):
        methods.append(("WITHOUT KV cache", False))

    results = {}
    for name, use_cache in methods:
        result = generate_once(
            model, inputs, tokenizer, args, device, use_cache=use_cache
        )
        results[use_cache] = result
        output_ids, elapsed, generated_tokens, peak_bytes = result
        print_result(name, elapsed, generated_tokens, peak_bytes)

    if args.mode == "compare":
        cache_ids, cache_time, _, _ = results[True]
        no_cache_ids, no_cache_time, _, _ = results[False]
        identical = torch.equal(cache_ids, no_cache_ids)
        print("\nComparison")
        print(f"  Outputs identical: {identical}")
        print(f"  KV-cache speedup:  {no_cache_time / cache_time:.2f}x")

    # Both comparison runs should produce the same IDs with greedy decoding.
    output_ids = results[True if True in results else False][0]
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\nGenerated text:\n")
    print(generated_text)


if __name__ == "__main__":
    main()




# Run with sampling:
# python load_local_gpt2.py \
#   --prompt "Artificial intelligence will" \
#   --max-new-tokens 100
#
# Or greedy decoding:
#
# python load_local_gpt2.py \
#   --prompt "Artificial intelligence will" \
#   --max-new-tokens 100 \
#   --greedy
#
# The script:
#
# Loads from ~/scratch/llms_model/gpt2
# Prevents internet downloads with local_files_only=True
# Automatically uses CUDA when available
# Uses FP16 on your A100
# Enables GPT-2’s built-in KV cache
# Supports sampling and greedy decoding
#
# The local directory should contain Hugging Face-format files such as config.json, tokenizer files, and model.safetensors or pytorch_model.bin.
# Run the comparison:
#
# python load_local_gpt2.py \
#   --prompt "Artificial intelligence will" \
#   --max-new-tokens 200
#
# It now reports:
#
# Generation time with and without KV cache
# Tokens per second
# KV-cache speedup ratio
# Peak GPU memory
# Whether both runs produced identical token IDs
# Generated text
#
# Run only one method:
# python load_local_gpt2.py --mode cache
# python load_local_gpt2.py --mode no-cache
#
# Greedy decoding is now the default, matching the original code. For sampling:
#
# python load_local_gpt2.py --sample --temperature 0.8 --top-k 50
#
# For a meaningful A100 comparison, use at least --max-new-tokens 200; KV-cache benefits become clearer as the sequence grows.