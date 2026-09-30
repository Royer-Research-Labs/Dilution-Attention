"""Inference benchmark: prefill latency and decode throughput per attention path.

Builds a model from an experiment config (fresh weights) or a checkpoint and
measures, for each requested path:

  - prefill: one full forward over a random prompt of --prompt-length
  - decode:  greedy continuation for --decode-tokens, measured both as a
    full-prefix recompute per token and via the cached decode path
    (docs/kernels.md v4).

Paths: "eager" (tensor path), "compiled" (torch.compile of the model), and
"kernel" (fused Triton forward on dilution layers; softmax layers are
unaffected by the knob).

    python -m dilution.benchmark --config configs/... [--checkpoint runs/.../latest.pt]
        [--paths eager kernel] [--prompt-length 4096] [--decode-tokens 64]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig, build_model
from .runtime import load_checkpoint, load_experiment_config, resolve_runtime


def _build(args) -> tuple[torch.nn.Module, int, int]:
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
        model_config = ModelConfig.from_dict(dict(checkpoint["model_config"]))
        model = build_model(model_config)
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        resolved = load_experiment_config(args.config)
        model_config = ModelConfig.from_dict(resolved["model"])
        torch.manual_seed(args.seed)
        model = build_model(model_config)
    return model, model_config.vocab_size, model_config.context_length


@torch.no_grad()
def _bench_prefill(model, prompt, autocast, warmup, iterations) -> float:
    for _ in range(warmup):
        with autocast:
            model(prompt)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        with autocast:
            model(prompt)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iterations * 1000.0


@torch.no_grad()
def _bench_decode(model, prompt, decode_tokens, block_size, autocast) -> float:
    tokens = prompt.clone()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(decode_tokens):
        window = tokens[:, -block_size:]
        with autocast:
            logits, _ = model(window)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return decode_tokens * prompt.shape[0] / elapsed  # tokens/second across batch


@torch.no_grad()
def _bench_decode_cached(model, prompt, decode_tokens, block_size, autocast) -> float:
    """v4 cached decode: prefill once, then O(T*d)-per-token steps."""

    if prompt.shape[1] + decode_tokens > block_size:
        prompt = prompt[:, : block_size - decode_tokens]
    with autocast:
        logits, caches = model.prefill(prompt)
    position = prompt.shape[1]
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(decode_tokens):
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        with autocast:
            logits = model.decode_step(next_token, position, caches)
        position += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return decode_tokens * prompt.shape[0] / elapsed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="experiment YAML (fresh weights)")
    source.add_argument("--checkpoint", help="latest.pt from dilution.train")
    parser.add_argument("--paths", nargs="+", default=["eager", "compiled", "kernel"],
                        choices=("eager", "compiled", "kernel"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-length", type=int, default=1024)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="JSON output path")
    args = parser.parse_args(argv)

    runtime = resolve_runtime("cuda", "auto")
    model, vocab_size, block_size = _build(args)
    model = model.to(runtime.device).eval()
    prompt_length = min(args.prompt_length, block_size)
    torch.manual_seed(args.seed)
    prompt = torch.randint(0, vocab_size, (args.batch_size, prompt_length), device=runtime.device)
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if args.dtype == "bfloat16"
        else torch.autocast("cuda", enabled=False)
    )

    results: dict[str, Any] = {
        "batch_size": args.batch_size,
        "prompt_length": prompt_length,
        "decode_tokens": args.decode_tokens,
        "dtype": args.dtype,
        "parameters": sum(p.numel() for p in model.parameters()),
        "paths": {},
    }
    for path in args.paths:
        model.enable_dilution_kernel(path == "kernel")
        run_model: torch.nn.Module = torch.compile(model) if path == "compiled" else model
        torch.cuda.reset_peak_memory_stats()
        prefill_ms = _bench_prefill(run_model, prompt, autocast, args.warmup, args.iterations)
        decode_tps = _bench_decode(run_model, prompt, args.decode_tokens, block_size, autocast)
        cached_tps = None
        if path != "compiled" and hasattr(model, "prefill"):
            try:
                cached_tps = _bench_decode_cached(
                    model, prompt, args.decode_tokens, block_size, autocast
                )
            except NotImplementedError:
                cached_tps = None
        entry = {
            "prefill_ms": round(prefill_ms, 2),
            "prefill_tokens_per_s": round(prompt_length * args.batch_size / prefill_ms * 1000.0),
            "decode_tokens_per_s": round(decode_tps, 2),
            "decode_cached_tokens_per_s": round(cached_tps, 2) if cached_tps else None,
            "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        results["paths"][path] = entry
        print(
            f"{path:<9} prefill {entry['prefill_ms']:>9.2f} ms "
            f"({entry['prefill_tokens_per_s']:>9,} tok/s)  "
            f"decode {entry['decode_tokens_per_s']:>8.2f} tok/s  "
            f"cached {entry['decode_cached_tokens_per_s'] or float('nan'):>8.2f} tok/s  "
            f"peak {entry['peak_memory_gib']:>6.2f} GiB",
            flush=True,
        )
    model.enable_dilution_kernel(False)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
