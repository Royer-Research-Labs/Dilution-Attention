"""Render the needle-in-a-haystack retrieval heatmaps used in the README.

    PYTHONPATH=src python scripts/make_retrieval_figure.py [--outdir docs/figures]

One panel per training regime, because model size and token budget are confounds that must not be
mixed inside a single chart:

    retrieval-209m-1x.png   209M params, 16K context, 4.19B tokens (1.0x Chinchilla)
    retrieval-209m-2x.png   the same NoPE models trained on to 8.40B tokens (2.0x)
    retrieval-209m-1k.png   209M params, 1K context, 4.19B tokens (1.0x)
    retrieval-64m-1x.png     64M params, 16K context, 1.27B tokens (1.0x)

One row per model and seed: extrapolation range varies between seeds of the same recipe, so the
rows show every seed instead of a mean. Cell colour = NIAH accuracy (8 needles, one disqualified
by the control prior, so 7 scored; chance 1/7). Cells up to the trained context come from the
20-placement in-context run; cells beyond it come from the 10-placement extrapolation ladders.
RoPE-bearing rows use dynamic NTK base scaling beyond the trained context (their unscaled
ladders are at or near chance, docs/signals.md S12-S14, S21); NoPE rows have nothing to scale.

Results are read from results/niah_v4/ (niah-v4-spec re-scores) when present, else results/niah/.
A row that still contains niah-v3 cells is marked with a dagger and footnoted: its checkpoint was
trained on past the evaluated step, so it cannot be re-scored.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
NIAH = ROOT / "results" / "niah"
NIAH_V4 = ROOT / "results" / "niah_v4"      # niah-v4-spec re-scores of earlier niah-v3 results
CHANCE = 1.0 / 7.0          # one of the eight needles is disqualified by the control prior
CHINCHILLA_TOKENS_PER_PARAM = 20

COLS_16K = [1024, 1920, 3840, 7936, 12288, 16000, 24000, 32000, 48000, 64000, 96000, 128000, 192000, 256000]
COLS_1K = [512, 1024, 1920, 3072, 3840, 6144, 7936, 12288, 16000, 24000, 32000]


def label(c):
    """Haystack lengths in the NIAH ladders are just under powers of two (1920, 3840, 7936) or
    round thousands (16000, 24000, ...); label them by the nearest K."""
    if c < 1000:
        return str(c)
    return f"{round(c / 1024)}K" if c < 12500 else f"{round(c / 1000)}K"


def seeds(name, trained, stems_for, seed_list):
    """Rows for one configuration: (label, trained context, in-context stem, ladder stems)."""
    rows = []
    for s in seed_list:
        inctx, ladders = stems_for(s)
        rows.append((f"{name}   seed {s}", trained, inctx, ladders))
    return rows


def s16(arm, s, ntk=False):
    tag = "_ntk" if ntk else ""
    return (f"scale1x__{arm}_seed{s}",
            [f"extrap{k}{tag}__scale_16384_{arm}_seed{s}" for k in ("64k", "128k", "256k")])


def s2x(arm, s):
    return (f"scale2x__{arm}_seed{s}", [f"extrap{k}__scale2x__{arm}_seed{s}" for k in ("64k", "128k", "256k")])


def s1k(arm, s, ntk=False):
    return (f"scale1k__{arm}_seed{s}", [f"extrap32k{'_ntk' if ntk else ''}__{arm}_seed{s}"])


def m64(cfg, s, ntk=False):
    stem = f"nope_16384__{cfg}_seed{s}"
    tag = "_ntk" if ntk else ""
    return (stem, [f"extrap128k{tag}__{stem}", f"extrap128k{tag}__{stem.replace('nope_16384__', 'nope_16384_')}"])


SEEDS2 = (1337, 2357)
SEEDS3 = (1337, 2357, 7331)
PANELS = [
    dict(name="retrieval-209m-1x", params=208.5e6, tokens=4.19e9, cols=COLS_16K, rows=
         seeds("dilution, no position encoding", 16000, lambda s: s16("all_dilution_nope", s), SEEDS2)
         + seeds("alt6: dilution + RoPE softmax", 16000,
                 lambda s: s16("alt6_softmax_rope_dilution", s, ntk=True), SEEDS2)
         + seeds("softmax + RoPE (control)", 16000, lambda s: s16("all_softmax_rope", s, ntk=True), SEEDS2)
         + seeds("softmax, no position encoding", 16000, lambda s: s16("all_softmax_nope_lr6e-4", s), SEEDS2),
         note="24-layer d768 models trained at 16K on the same corpus, schedule and learning rate; rows differ "
              "only in the attention operator, its position encoding and the seed. Softmax without position "
              "encoding seed 2357 hit gradient spikes early in training (docs/signals.md S24)."),
    dict(name="retrieval-209m-2x", params=208.5e6, tokens=8.40e9, cols=COLS_16K, rows=
         seeds("dilution, no position encoding", 16000, lambda s: s2x("all_dilution_nope", s), SEEDS2)
         + seeds("softmax, no position encoding", 16000, lambda s: s2x("all_softmax_nope", s), SEEDS2),
         note="The no-position-encoding pair from the 1x panel trained on to the end of the same cosine schedule. "
              "Loss replicates to 0.003 between the dilution seeds; extrapolation range does not (S19, S24)."),
    dict(name="retrieval-209m-1k", params=208.5e6, tokens=4.19e9, cols=COLS_1K, rows=
         seeds("dilution, no position encoding", 1024, lambda s: s1k("all_dilution_nope_1k", s), SEEDS2)
         + seeds("softmax, no position encoding", 1024, lambda s: s1k("all_softmax_nope_1k", s), SEEDS2)
         + seeds("softmax + RoPE 5e5", 1024, lambda s: s1k("all_softmax_rope_1k", s, ntk=True), SEEDS2),
         note="The 209M models trained at a 1,024-token context (S21, S24). At this length softmax + RoPE has "
              "the best loss; only dilution retrieves beyond the trained context."),
    dict(name="retrieval-64m-1x", params=63.5e6, tokens=1.27e9, cols=COLS_16K[:-2], rows=
         seeds("dilution, no position encoding", 16000,
               lambda s: m64("all_dilution_nope_chinchilla_lr3e4", s), SEEDS3)
         + seeds("dilution, RoPE alternate layers", 16000,
                 lambda s: m64("all_dilution_rope_alt_chinchilla_lr3e4", s, ntk=True), SEEDS3)
         + seeds("softmax + RoPE 5e5 (control)", 16000,
                 lambda s: m64("all_softmax_rope_theta5e5_chinchilla_lr3e4", s, ntk=True), SEEDS3)
         + seeds("softmax, no position encoding", 16000,
                 lambda s: m64("all_softmax_nope_chinchilla_lr1.5e-4", s), (1337,)),
         note="12-layer d512 models at 16K. With no position encoding, dilution forms in-context retrieval in "
              "two of three seeds at identical loss; RoPE on alternate layers makes it reliable (S22, S23). "
              "Softmax without position encoding needs half the learning rate (1.5e-4) to train (S16)."),
]


def load(stem):
    """{context_length: accuracy} for one result file, and whether it is niah-v3. The v4 re-score
    wins when it exists; v4 rows carry length_basis "input" (the full scored prompt)."""
    for d in (NIAH_V4, NIAH):
        p = d / f"{stem}.json"
        if p.exists():
            break
    else:
        return {}, False
    j = json.load(p.open(encoding="utf-8"))
    rows = j["results"] if isinstance(j, dict) and "results" in j else j
    v3 = any(x.get("length_basis") != "input" for x in rows if "accuracy" in x)
    return {int(x["context_length"]): float(x["accuracy"]) for x in rows if "accuracy" in x}, v3


V3_NOTE = ("  \u2020 Scored with the earlier niah-v3 prompts (question appended past the stated length; "
           "control prior measured on a prompt capped at 5,400 tokens). These checkpoints were trained on "
           "past this step and cannot be re-scored; every other v3 result in these figures was re-scored under v4 "
           "(docs/research-results.md, NIAH protocol versions).")


def column_for(cols, ctx):
    best = min(cols, key=lambda c: abs(c - ctx))
    return cols.index(best) if abs(best - ctx) <= 0.06 * max(best, ctx) else None


def build_cmap():
    c = LinearSegmentedColormap.from_list("niah", [
        (0.00, "#8c1c13"), (0.25, "#c0392b"), (0.45, "#e08b3c"),
        (0.65, "#e8c547"), (0.85, "#7fb069"), (1.00, "#2d6a4f")])
    c.set_bad("#f2f2f2")
    return c


def render(panel, outdir):
    rows, cols = panel["rows"], panel["cols"]
    grid = np.full((len(rows), len(cols)), np.nan)
    missing, v3_rows = [], set()
    for i, (_, trained, inctx, ladders) in enumerate(rows):
        got, v3 = load(inctx)
        if not got:
            missing.append(inctx)
        for ctx, acc in got.items():                        # in context: 20-placement run
            j = column_for(cols, ctx)
            if j is not None and ctx <= trained * 1.06:
                grid[i, j] = acc
                if v3:
                    v3_rows.add(i)
        for stem in ladders:                                # beyond the trained context: ladders
            got, v3 = load(stem)
            for ctx, acc in got.items():
                j = column_for(cols, ctx)
                if j is not None and ctx > trained * 1.06:
                    grid[i, j] = acc
                    if v3:
                        v3_rows.add(i)

    cm = build_cmap()
    h = 1.55 + 0.36 * len(rows)
    fig, ax = plt.subplots(figsize=(13.4, h))
    ax.imshow(grid, cmap=cm, vmin=CHANCE, vmax=1.0, aspect="auto", interpolation="nearest")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isnan(grid[i, j]):
                continue
            v = grid[i, j]
            ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=8.3,
                    color="white" if (v < 0.45 or v > 0.80) else "#222222",
                    fontweight="bold" if v >= 0.85 else "normal")
    for i, (_, trained, _, _) in enumerate(rows):
        edge = max(j for j, c in enumerate(cols) if c <= trained * 1.06) + 0.5
        ax.plot([edge, edge], [i - 0.5, i + 0.5], color="#111111", lw=2.6, solid_capstyle="butt", zorder=5)
    # white separators between configurations
    names = [r[0].split("   seed")[0] for r in rows]
    for i in range(1, len(rows)):
        if names[i] != names[i - 1]:
            ax.axhline(i - 0.5, color="white", lw=3.2)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([label(c) for c in cols], fontsize=10)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] + (" \u2020" if i in v3_rows else "") for i, r in enumerate(rows)], fontsize=9.2)
    ax.set_xlabel("haystack length (tokens)", fontsize=11)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=CHANCE, vmax=1.0))
    cb = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.03)
    cb.set_label("retrieval accuracy", fontsize=10)
    cb.set_ticks([CHANCE, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["chance", ".50", ".75", "1.0"])

    ratio = panel["tokens"] / (CHINCHILLA_TOKENS_PER_PARAM * panel["params"])
    head = (f"{int(panel['params'] / 1e6 + 0.5)}M params   ·   {panel['tokens']/1e9:.2f}B training tokens"
            f"   ·   {ratio:.1f}x Chinchilla")
    ax.set_title(head + "\nblack bar = trained context; everything to its right is extrapolation "
                        "with no fine-tuning", fontsize=12.5, pad=12, loc="left")
    note = textwrap.fill(
        panel["note"] + "  Likelihood-scored keyword retrieval, 7 scored needles, chance 1/7: 20 depth placements "
        "in context, 10 beyond. RoPE rows use dynamic NTK base scaling beyond the trained context."
        + (V3_NOTE if v3_rows else ""), width=150)
    n_lines = len(note.splitlines())
    fig.subplots_adjust(left=0.265, right=0.935, top=1 - 0.72 / h, bottom=(0.62 + 0.16 * n_lines) / h + 0.02)
    fig.text(0.012, 0.012, note, fontsize=7.4, color="#666666", linespacing=1.6, va="bottom")
    out = Path(outdir) / f"{panel['name']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190)
    plt.close(fig)
    print(f"wrote {out}  ({int(np.isfinite(grid).sum())} cells)" + (f"  MISSING {missing}" if missing else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(ROOT / "docs" / "figures"))
    a = ap.parse_args()
    for panel in PANELS:
        render(panel, a.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
