"""Sampling evaluation: continue the same validation prompts with several models and compare the
samples (see src/dilution/sampling.py for the prompt design and every metric).

    PYTHONPATH=src python scripts/sample_eval.py \
        --model dilution_nope_s2357=runs/scale_16384/all_dilution_nope_seed2357/latest_1x.pt \
        --model softmax_rope_s1337=runs/scale_16384/all_softmax_rope_seed1337/latest.pt \
        --baseline softmax_rope_s1337 --lengths 256 1024 4096 16128 --extrapolate 32768 \
        --output results/samples/scale209m_1x.json

Three passes, one model in memory at a time: (1) every model samples every prompt under every
decoding; (2) every model scores every sample and the human continuation (cross-scoring, for
prompts up to --cross-max-length); (3) the judge scores every sample. Writes <output>.json
(aggregates, paired differences against each --baseline and against the human text, provenance),
<output>.metrics.jsonl.gz (every sample's metrics, without token ids or text: what the summary is
computed from; scripts/verify_samples.py recomputes the summary from it), <output>.samples.jsonl
(every sample in full) and <output>.md (a few prompts with each model's samples, for reading).
The judge is pinned to --judge-revision, the commit used for the published results.

Prompt lengths past a model's trained context (--extrapolate) need a model without a learned
position table; RoPE layers then use dynamic NTK base scaling (as in the NIAH extrapolation
ladders) unless --no-ntk.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from dilution.extend import ExtendedContext as Extended
from dilution.niah_eval import checkpoint_identity, load_model_for_niah
from dilution.runtime import create_token_stream
from dilution.sampling import (EOT, GPT2_VOCAB, Judge, continuation_nll, sample_stats, select_end_positions,
                               summarize, truncate_at_eot, write_metrics)

DECODINGS = {
    "ancestral": dict(temperature=1.0),               # the model's own distribution: calibration
    "nucleus": dict(temperature=1.0, top_p=0.9),      # the usual practical setting
    "greedy": dict(temperature=0.0),                  # degeneration / looping
}
HUMAN = "human"


def log(msg: str) -> None:
    print(f"[sample_eval {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def batches(n: int, size: int):
    for start in range(0, n, size):
        yield start, min(n, start + size)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=CHECKPOINT")
    ap.add_argument("--baseline", action="append",
                    help="model the others are compared against, prompt by prompt; repeatable "
                         "(default: the first --model)")
    ap.add_argument("--lengths", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--extrapolate", type=int, nargs="*", default=[],
                    help="prompt lengths beyond the trained context (models without learned positions)")
    ap.add_argument("--prompts", type=int, default=64)
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--decodings", nargs="+", default=list(DECODINGS), choices=list(DECODINGS))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--token-budget", type=int, default=262144, help="prompt+new tokens per batch")
    ap.add_argument("--judge", default="HuggingFaceTB/SmolLM2-1.7B")
    ap.add_argument("--judge-revision", default="effd688a12921b4cc83e3312b6feb579f70f9c71",
                    help="Hugging Face commit of the judge (default: the one behind the published results)")
    ap.add_argument("--judge-context", type=int, default=4096)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--no-cross", action="store_true", help="skip scoring samples with our models")
    ap.add_argument("--cross-max-length", type=int, default=4096,
                    help="cross-score only prompts up to this length (each sample re-reads its whole "
                         "prompt under every model, so the cost grows with models^2 x length)")
    ap.add_argument("--no-ntk", action="store_true")
    ap.add_argument("--show", type=int, default=2, help="prompts written out in the .md file")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    for spec in a.model:
        name, _, ckpt = spec.partition("=")
        if not ckpt or name == HUMAN or name in models:
            ap.error(f"bad or duplicate --model {spec!r}")
        models[name] = Path(ckpt)
    baselines = a.baseline or [next(iter(models))]
    for b in baselines:
        if b not in models:
            ap.error(f"--baseline {b} is not a --model name")

    # ---- prompts: one set of end positions, shared by every length and model
    cfgs = {n: yaml.safe_load((p.parent / "resolved_config.yaml").read_text(encoding="utf-8"))
            for n, p in models.items()}
    val_paths = {str(c["data"]["val_path"]) for c in cfgs.values()}
    if len(val_paths) > 1:
        log(f"WARNING: models were trained on different validation streams {sorted(val_paths)}; "
            "prompts come from the first")
    stream = create_token_stream(cfgs[next(iter(models))]["data"], "val")
    lengths = sorted(set(a.lengths) | set(a.extrapolate))
    ends = select_end_positions(lambda s, n: stream.read_tokens(s, n).tolist(), stream.num_tokens,
                                count=a.prompts, max_prompt=max(lengths), new_tokens=a.new_tokens,
                                seed=a.seed)
    prompts = {L: torch.stack([stream.read_tokens(e - L, L) for e in ends]).long() for L in lengths}
    human = [stream.read_tokens(e, a.new_tokens).tolist() for e in ends]
    log(f"{len(ends)} prompts at lengths {lengths}, {a.new_tokens} new tokens, decodings {a.decodings}")

    # samples[(gen, L, dec)] = list of continuation id lists (human: dec "reference", same text)
    samples: dict[tuple[str, int, str], list[list[int]]] = {}
    for L in lengths:
        samples[(HUMAN, L, "reference")] = [list(h) for h in human]
    skipped: dict[str, list[int]] = {}
    identities = {}
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.no_grad()

    # ---- pass 1: generation
    for name, ckpt in models.items():
        model = load_model_for_niah(ckpt, device)
        identities[name] = checkpoint_identity(ckpt, getattr(model, "checkpoint_step", None))
        identities[name].update(context_length=model.config.context_length,
                                positional=getattr(model.config, "positional", None),
                                layer_attention=getattr(model.config, "layer_attention", None),
                                attention=getattr(model.config, "attention", None))
        for L in lengths:
            need = L + a.new_tokens
            if need > model.config.context_length and (
                    L not in a.extrapolate or getattr(model.config, "positional", "learned") == "learned"):
                skipped.setdefault(name, []).append(L)
                continue
            size = max(1, min(len(ends), a.token_budget // need))
            with Extended(model, need, not a.no_ntk), autocast, torch.no_grad():
                for dec in a.decodings:
                    t0, out = time.time(), []
                    for b, (s, e) in enumerate(batches(len(ends), size)):
                        gen = torch.Generator(device=device).manual_seed(
                            a.seed * 1_000_003 + L * 101 + b * 7 + a.decodings.index(dec))
                        ids = model.generate(prompts[L][s:e].to(device), a.new_tokens, generator=gen,
                                             vocab_limit=GPT2_VOCAB, **DECODINGS[dec])
                        out.extend(ids[:, L:].tolist())
                    samples[(name, L, dec)] = out
                    log(f"{name} L={L} {dec}: {len(out)} samples in {time.time() - t0:.0f}s")
        del model
        gc.collect()
        torch.cuda.empty_cache() if device == "cuda" else None

    # per-sample records
    records = {}
    for (gen, L, dec), conts in samples.items():
        for i, c in enumerate(conts):
            ids, _ = truncate_at_eot(c)
            rec = {"gen": gen, "L": L, "dec": dec, "prompt": i, "end": ends[i], "ids": ids}
            rec.update(sample_stats(c, prompts[L][i].tolist()))
            rec["xnll"] = {}
            records[(gen, L, dec, i)] = rec

    # ---- pass 2: cross-scoring with our own models
    if not a.no_cross:
        for name, ckpt in models.items():
            model = load_model_for_niah(ckpt, device)
            for L in lengths:
                need = L + a.new_tokens
                if L > a.cross_max_length or need > model.config.context_length and (
                        L not in a.extrapolate or getattr(model.config, "positional", "learned") == "learned"):
                    continue
                size = max(1, min(len(ends), a.token_budget // need))
                t0 = time.time()
                with Extended(model, need, not a.no_ntk), autocast, torch.no_grad():
                    for (gen, L2, dec), conts in samples.items():
                        if L2 != L:
                            continue
                        for s, e in batches(len(ends), size):
                            cut = [records[(gen, L, dec, i)]["ids"] or [EOT] for i in range(s, e)]
                            nll = continuation_nll(model, prompts[L][s:e].to(device), cut)
                            for i, v in zip(range(s, e), nll):
                                records[(gen, L, dec, i)]["xnll"][name] = v
                log(f"cross-scored L={L} with {name} in {time.time() - t0:.0f}s")
            del model
            gc.collect()
            torch.cuda.empty_cache() if device == "cuda" else None

    # ---- pass 3: the judge
    judge_info = None
    if not a.no_judge:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        judge = Judge(a.judge, device=device, context=a.judge_context, revision=a.judge_revision)
        judge_info = {"name": judge.name, "revision": judge.revision, "context": judge.context}
        for L in lengths:
            # the judge keeps at most judge.context of its own tokens (about as many GPT-2 tokens);
            # decode 1.5x that so the cut happens in the judge's tokenization
            tail = int(1.5 * judge.context)
            ptext = [enc.decode(prompts[L][i, -tail:].tolist()) for i in range(len(ends))]
            t0 = time.time()
            for (gen, L2, dec), conts in samples.items():
                if L2 != L:
                    continue
                keys = [(gen, L, dec, i) for i in range(len(ends))]
                texts = [enc.decode([t for t in records[k]["ids"] if t != EOT]) for k in keys]
                for k, text, sc in zip(keys, texts, judge.score(ptext, texts)):
                    records[k].update(judge_bpb=sc.bpb, judge_tokens=sc.n_tokens, bytes=sc.n_bytes, text=text)
            log(f"judged L={L} in {time.time() - t0:.0f}s")
        del judge
        gc.collect()
    else:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        for k, rec in records.items():
            rec["text"] = enc.decode([t for t in rec["ids"] if t != EOT])

    # ---- aggregate (the same function scripts/verify_samples.py reruns on the metrics file)
    table, paired = summarize(records.values(), models=list(models), lengths=lengths,
                              decodings=a.decodings, baselines=baselines, human=HUMAN)

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "task": "sample_eval", "argv": sys.argv[1:], "baselines": baselines, "lengths": lengths,
        "extrapolate": a.extrapolate, "prompts": len(ends), "new_tokens": a.new_tokens,
        "decodings": {d: DECODINGS[d] for d in a.decodings}, "seed": a.seed, "ntk": not a.no_ntk,
        "judge": judge_info, "validation_stream": sorted(val_paths)[0], "prompt_end_positions": ends,
        "models": identities, "skipped_lengths": skipped, "table": table, "paired": paired,
    }
    out.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    with out.with_suffix(".samples.jsonl").open("w", encoding="utf-8") as fh:
        for rec in records.values():
            fh.write(json.dumps(rec, default=str) + "\n")
    write_metrics(out.with_suffix(".metrics.jsonl.gz"), records.values())
    write_markdown(out.with_suffix(".md"), a, lengths, ends, prompts, records, models, enc)

    # console summary
    print(f"\n{'L':>6} {'decoding':9} {'model':28} {'judge bpb':>10} {'vs ' + baselines[0]:>24} {'rep-4':>6} "
          f"{'loop':>5} {'copy-8':>6} {'eot':>5}")
    lookup = {(p["model"], p["L"], p["dec"]): p for p in paired}
    for row in table:
        p = lookup.get((row["gen"], row["L"], row["dec"]), {})
        d = p.get("vs", {}).get(baselines[0], {}).get("judge_bpb_minus_baseline")
        vs = f"{d['mean']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]" if d and d["n"] else ""
        print(f"{row['L']:>6} {row['dec']:9} {row['gen']:28} {row['judge_bpb']:>10.3f} {vs:>24} "
              f"{row['rep_4']:>6.3f} {row['looping']:>5.2f} {row['copy_8']:>6.3f} {row['eot']:>5.2f}")
    log(f"wrote {out}, {out.with_suffix('.metrics.jsonl.gz')}, {out.with_suffix('.samples.jsonl')}, "
        f"{out.with_suffix('.md')}")
    return 0


def write_markdown(path, a, lengths, ends, prompts, records, models, enc):
    lines = [f"# Samples ({path.stem})", "",
             "Prompt tail, the human continuation, then each model's sample per decoding. "
             "Continuations stop at the model's end-of-text token, shown as ⏎EOT.", ""]
    gens = [HUMAN] + list(models)
    for i in range(min(a.show, len(ends))):
        for L in lengths:
            tail = enc.decode(prompts[L][i, -120:].tolist()).replace("<|endoftext|>", "⏎EOT")
            lines += [f"## prompt {i}, length {L}", "", "> …" + tail.replace("\n", "\n> "), ""]
            for gen in gens:
                for dec in (["reference"] if gen == HUMAN else a.decodings):
                    rec = records.get((gen, L, dec, i))
                    if rec is None:
                        continue
                    text = rec.get("text", "") + (" ⏎EOT" if rec.get("eot") else "")
                    bpb = rec.get("judge_bpb")
                    meta = f"judge {bpb:.3f} bpb, " if bpb is not None else ""
                    meta += f"rep-4 {rec['rep_4']:.2f}"
                    lines += [f"**{gen} / {dec}** ({meta})", "", "```text", text.strip(), "```", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
