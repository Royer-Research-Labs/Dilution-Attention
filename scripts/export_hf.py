"""Build the published Hugging Face models from training checkpoints.

    PYTHONPATH=src python scripts/export_hf.py [--out ../hf-export] [--only NAME ...] [--org ORG]

For every model in MODELS: load the training checkpoint, save its weights (fp32 safetensors, the
tied embedding stored once) and config with `dilution.hub.save_pretrained`, reload the result with
`dilution.hub.from_pretrained` and check that it reproduces the checkpoint's logits exactly, then
write a model card (README.md) whose numbers come from the published result files: loss on the
common validation slice, NIAH retrieval in and beyond the trained context, and, when present, the
PG-19 perplexity by position. Nothing is uploaded; the script prints the upload commands.

The weights are released under Apache-2.0; the training data is FineWeb-Edu (ODC-By 1.0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from dilution.config import ModelConfig, build_model
from dilution.hub import from_pretrained, save_pretrained
from dilution.runtime import load_checkpoint

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/Royer-Research-Labs/Dilution-Attention"
S, K = "runs/scale_16384", "runs/scale_1024"

# name -> (checkpoint, common-slice eval stem, in-context NIAH stem, NIAH extrapolation stems,
#          PG-19 (result file, model key), arm description, budget, seed)
_EXT16 = ("64k", "128k", "256k")
MODELS = {}


def _add(name, ckpt, eval_stem, niah_stem, ext, pg19, arm, budget, seed):
    MODELS[name] = dict(checkpoint=ckpt, eval=eval_stem, niah=niah_stem, ext=ext, pg19=pg19, arm=arm,
                        budget=budget, seed=seed)


for s in (2357,):
    _add(f"dilution-209m-16k-nope-1x-s{s}", f"{S}/all_dilution_nope_seed{s}/latest_1x.pt",
         f"scale1x__all_dilution_nope_seed{s}", f"scale1x__all_dilution_nope_seed{s}",
         [f"extrap{k}__scale_16384_all_dilution_nope_seed{s}" for k in _EXT16],
         ("pg19_209m_16k_1x", f"dilution_nope_s{s}"),
         "dilution attention in every layer, no position encoding", "1x", s)
    _add(f"softmax-209m-16k-nope-1x-s{s}", f"{S}/all_softmax_nope_lr6e-4_seed{s}/latest_1x.pt",
         f"scale1x__all_softmax_nope_lr6e-4_seed{s}", f"scale1x__all_softmax_nope_lr6e-4_seed{s}",
         [f"extrap{k}__scale_16384_all_softmax_nope_lr6e-4_seed{s}" for k in _EXT16],
         ("pg19_209m_16k_1x", f"softmax_nope_s{s}"),
         "softmax attention in every layer, no position encoding (control; this seed hit gradient spikes "
         "near step 13,400 and never recovered the loss)", "1x", s)
for s in (1337, 2357):
    _add(f"softmax-209m-16k-rope-1x-s{s}", f"{S}/all_softmax_rope_seed{s}/latest.pt",
         f"scale1x__all_softmax_rope_seed{s}", f"scale1x__all_softmax_rope_seed{s}",
         [f"extrap64k_ntk__scale_16384_all_softmax_rope_seed{s}"],
         ("pg19_209m_16k_1x", f"softmax_rope_s{s}"),
         "softmax attention in every layer with RoPE (base 5e5); the control", "1x", s)
    _add(f"alt6-209m-16k-1x-s{s}", f"{S}/alt6_softmax_rope_dilution_seed{s}/latest.pt",
         f"scale1x__alt6_softmax_rope_dilution_seed{s}", f"scale1x__alt6_softmax_rope_dilution_seed{s}",
         [f"extrap{k}_ntk__scale_16384_alt6_softmax_rope_dilution_seed{s}" for k in _EXT16],
         ("pg19_209m_16k_1x", f"alt6_s{s}"),
         "hybrid: softmax with RoPE (base 5e5) and dilution without position encoding in alternating layers",
         "1x", s)
    _add(f"dilution-209m-16k-nope-2x-s{s}", f"{S}/all_dilution_nope_seed{s}/latest.pt",
         f"scale2x__all_dilution_nope_seed{s}", f"scale2x__all_dilution_nope_seed{s}",
         [f"extrap{k}__scale2x__all_dilution_nope_seed{s}" for k in _EXT16],
         ("pg19_209m_16k_2x", f"dilution_nope_s{s}"),
         "dilution attention in every layer, no position encoding", "2x", s)
    _add(f"softmax-209m-16k-nope-2x-s{s}", f"{S}/all_softmax_nope_lr6e-4_seed{s}/latest.pt",
         f"scale2x__all_softmax_nope_seed{s}", f"scale2x__all_softmax_nope_seed{s}",
         [f"extrap{k}__scale2x__all_softmax_nope_seed{s}" for k in _EXT16],
         ("pg19_209m_16k_2x", f"softmax_nope_s{s}"),
         "softmax attention in every layer, no position encoding (control)", "2x", s)
    for arm, stem, desc in (("dilution-209m-1k-nope", "all_dilution_nope_1k",
                             "dilution attention in every layer, no position encoding"),
                            ("softmax-209m-1k-rope", "all_softmax_rope_1k",
                             "softmax attention in every layer with RoPE (base 5e5); the control"),
                            ("softmax-209m-1k-nope", "all_softmax_nope_1k",
                             "softmax attention in every layer, no position encoding (control)")):
        ntk = "_ntk" if "rope" in stem else ""
        _add(f"{arm}-s{s}", f"{K}/{stem}_seed{s}/latest.pt", f"scale1k__{stem}_seed{s}",
             f"scale1k__{stem}_seed{s}", [f"extrap32k{ntk}__{stem}_seed{s}"],
             ("pg19_209m_1k", f"{arm.split('-')[0]}_{arm.split('-')[-1]}_s{s}"), desc, "1x", s)


def niah(stems):
    """{length: accuracy} merged over result files, preferring the niah-v4 re-score."""
    out = {}
    for stem in stems:
        for d in ("niah_v4", "niah"):
            p = ROOT / "results" / d / f"{stem}.json"
            if p.exists():
                for r in json.loads(p.read_text(encoding="utf-8")):
                    out.setdefault(int(r["context_length"]), float(r["accuracy"]))
                break
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def card(name, spec, org, model, ckpt_meta):
    cfg = model.config
    ctx = cfg.context_length
    loss_file = ROOT / "results/eval" / f"{spec['eval']}.json"
    loss = json.loads(loss_file.read_text(encoding="utf-8"))["loss"] if loss_file.exists() else None
    ladder = niah([spec["niah"]] + spec["ext"])
    in_ctx = max((L for L in ladder if L <= ctx * 1.06), default=None)
    ext_rows = [(L, a) for L, a in sorted(ladder.items()) if L > ctx * 1.06]
    pg = None
    pg_file = ROOT / "results/longppl" / f"{spec['pg19'][0]}.json"
    if pg_file.exists():
        res = json.loads(pg_file.read_text(encoding="utf-8"))
        e = res["models"].get(spec["pg19"][1])
        if e:
            pg = (res["ranges"], e["curve"])
    rope = cfg.positional in ("rope", "rope_alt") or "rope" in name or "alt6" in name
    tokens = ckpt_meta["tokens_seen"]
    lines = [
        "---", "license: apache-2.0", "language: [en]", "library_name: dilution-attention",
        "pipeline_tag: text-generation", "datasets: [HuggingFaceFW/fineweb-edu]",
        "tags: [dilution-attention, attention, long-context, length-extrapolation, research]", "---", "",
        f"# {name}", "",
        f"A 209M-parameter research language model from the [DilutionAttention release]({REPO}): "
        f"{spec['arm']}. Trained at a {ctx:,}-token context on {tokens / 1e9:.2f}B tokens "
        f"({spec['budget']} Chinchilla), seed {spec['seed']}. It is a base model for studying the "
        "attention operator, not an assistant: it is not instruction-tuned or safety-tuned.", "",
        "## Results", "",
        "| measure | value |", "|---|---|",
    ]
    if loss is not None:
        lines.append(f"| validation loss (common 1M-token FineWeb-Edu slice) | {loss:.4f} nats/token |")
    if in_ctx:
        lines.append(f"| NIAH retrieval at {in_ctx:,} tokens (in context; chance 1/7) | {ladder[in_ctx]:.2f} |")
    for L, acc in ext_rows:
        lines.append(f"| NIAH retrieval at {L:,} tokens ({L / ctx:.1f}x the trained context"
                     f"{', NTK scaling' if rope else ''}) | {acc:.2f} |")
    if pg:
        rs, curve = pg
        picks = [(lo, hi, v) for (lo, hi), v in zip(rs, curve) if hi in (ctx, 2 * ctx, 4 * ctx, 8 * ctx, 32 * ctx)]
        for lo, hi, v in picks:
            where = " (in context)" if hi <= ctx else f" ({hi // ctx}x{', NTK scaling' if rope else ''})"
            lines.append(f"| PG-19 loss at positions {lo:,}-{hi:,}{where} | {v:.3f} nats/token |")
    lines += [
        "",
        f"NIAH is likelihood-scored keyword retrieval from repetitive filler (see the release's "
        f"[reproducing.md]({REPO}/blob/main/docs/reproducing.md)), not long-context reasoning. "
        f"Every number here, and the matched controls, are in the release's "
        f"[evidence ledger]({REPO}/blob/main/docs/signals.md).", "",
        "## Use", "",
        "```bash",
        f'pip install "dilution-attention[hub] @ git+{REPO}" tiktoken',
        "```", "",
        "```python",
        "import tiktoken, torch",
        "from dilution.hub import from_pretrained",
        "",
        f'model = from_pretrained("{org}/{name}", device="cuda")   # fused Triton kernels on CUDA',
        'enc = tiktoken.get_encoding("gpt2")',
        'ids = torch.tensor([enc.encode("The history of the printing press")], device="cuda")',
        'with torch.autocast("cuda", dtype=torch.bfloat16):',
        "    out = model.generate(ids, 64, temperature=1.0, top_p=0.9, vocab_limit=50257)",
        "print(enc.decode(out[0].tolist()))",
        "```", "",
        f"Inputs longer than {ctx:,} tokens need `dilution.extend.ExtendedContext(model, length"
        f"{', ntk=True' if rope else ''})` around the call.", "",
        "## Training", "",
        f"- Architecture: {cfg.n_layers} layers, width {cfg.d_model}, {cfg.n_heads} heads, SwiGLU, no biases, "
        f"tied embeddings, GPT-2 vocabulary; position encoding: {cfg.positional}.",
        "- Data: FineWeb-Edu `sample-100BT` (revision fc9850d), a 40B-token GPT-2-tokenized corpus rebuilt "
        f"with `dilution-prepare-data` ([reproducing.md]({REPO}/blob/main/docs/reproducing.md)).",
        f"- Schedule: 65,536 tokens per step, LR 6e-4 cosine over a 2x-Chinchilla (8.4B-token) schedule; "
        + ("this is the step-64,000 (1x) checkpoint, where the learning rate is 55.7% of peak."
           if spec["budget"] == "1x" else "this is the end of that schedule."),
        f"- Run record: `{spec['checkpoint'].rsplit('/', 1)[0]}/` in the release repository (metrics, "
        "resolved config, data provenance). Source checkpoint SHA-256: "
        f"`{ckpt_meta['sha256']}`, step {ckpt_meta['step']:,}.", "",
        "## License and data", "",
        "Weights: Apache-2.0. Training data: FineWeb-Edu (ODC-By 1.0), used with attribution. The model "
        "can produce incorrect, biased or offensive text; it is a research artifact.", "",
        "## Citation", "",
        f"See [CITATION.cff]({REPO}/blob/main/CITATION.cff) in the release repository.", "",
    ]
    return "\n".join(lines)


@torch.no_grad()
def check_roundtrip(model, out_dir: Path) -> float:
    loaded = from_pretrained(out_dir, device="cpu")
    ids = torch.randint(0, 50257, (1, 64), generator=torch.Generator().manual_seed(0))
    a, _ = model(ids, ids)
    b, _ = loaded(ids, ids)
    return float((a - b).abs().max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT.parent / "hf-export"))
    ap.add_argument("--org", default="Royer-Research-Labs")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--cards-only", action="store_true", help="rewrite model cards for already-exported models")
    a = ap.parse_args()
    names = a.only or list(MODELS)
    for name in names:
        spec = MODELS[name]
        ckpt_path = ROOT / spec["checkpoint"]
        out = Path(a.out) / name
        meta_file = out / "config.json"
        if a.cards_only and meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))["metadata"]
            model = from_pretrained(out, device="cpu")
        else:
            ckpt = load_checkpoint(ckpt_path, map_location="cpu")
            model = build_model(ModelConfig.from_dict(dict(ckpt["model_config"])))
            model.load_state_dict(ckpt["model"], strict=True)
            model.eval()
            meta = {"name": name, "source_checkpoint": spec["checkpoint"], "sha256": sha256(ckpt_path),
                    "step": int(ckpt["step"]), "tokens_seen": int(ckpt["tokens_seen"]), "seed": spec["seed"],
                    "arm": spec["arm"], "chinchilla": spec["budget"], "repository": REPO,
                    "license": "apache-2.0"}
            save_pretrained(model, out, metadata=meta)
            diff = check_roundtrip(model, out)
            if diff != 0.0:
                print(f"FAILED {name}: reloaded logits differ by {diff}")
                return 1
            print(f"exported {name}: step {meta['step']:,}, round trip exact")
        (out / "README.md").write_text(card(name, spec, a.org, model, meta), encoding="utf-8", newline="\n")
    print("\nupload (after `huggingface-cli login`), one repository per model:")
    for name in names:
        print(f"  huggingface-cli upload {a.org}/{name} {Path(a.out) / name} . --repo-type model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
