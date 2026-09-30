"""Compare each results/niah_v4/<name>.json (niah-v4-spec) with its niah-v3 original in
results/niah/<name>.json, length by length, then summarise where the protocol change moves scores.

    python scripts/compare_niah_versions.py [--summary-only]

Per length: accuracy under both specs, the change, the binomial standard error of the v4 score,
and the mean log-likelihood margin of the correct keyword over the best other scored keyword
(avg_retrieval_gap_ll, in nats). The result files record how many keywords the control prior
disqualified but not which one, so a flip cannot be traced to a changed disqualification.

The summary uses only cells scored with the same number of placements under both specs, and
reports: the direction of the changes (is v4 systematically easier?), the changes split at 5,400
tokens (v3 capped its control prompt there), and the size of the changes by v3 margin and by v3
accuracy (where does the protocol add variance?).
"""
import argparse
import json
import math
import statistics as st
from pathlib import Path

R = Path(__file__).resolve().parent.parent
V3_CONTROL_CAP = 5400   # niah-v3 built its needle-free control from at most 5,400 filler tokens


def rows(path):
    return {int(r["context_length"]): r for r in json.loads(path.read_text(encoding="utf-8"))}


def sign_test(xs):
    """Two-sided sign test over the nonzero changes: (up, down, p)."""
    xs = [x for x in xs if abs(x) > 1e-9]
    n, k = len(xs), sum(x > 0 for x in xs)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n) if n else 1.0
    return k, n - k, p


def binned(cells, key, edges, label):
    print(f"\n|accuracy change| by v3 {label}:")
    for lo, hi in zip(edges, edges[1:]):
        s = [c for c in cells if lo <= c[key] < hi]
        if s:
            d = [abs(c["d"]) for c in s]
            print(f"  [{lo:>5}, {hi:>5}): {len(s):3d} cells, mean v3 accuracy {st.mean(c['a3'] for c in s):.2f}, "
                  f"mean |change| {st.mean(d):.3f}, largest {max(d):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    cells, worst = [], []
    for f in sorted((R / "results/niah_v4").glob("*.json")):
        old = R / "results/niah" / f.name
        if not old.exists():
            print(f"{f.stem}: no v3 original")
            continue
        v3, v4 = rows(old), rows(f)
        if not a.summary_only:
            print(f"== {f.stem}")
        for L in sorted(v4):
            if L not in v3:
                continue
            r3, r4 = v3[L], v4[L]
            a3, a4, n = r3["accuracy"], r4["accuracy"], r4.get("total", 70)
            se = math.sqrt(max(a4 * (1 - a4), 1e-9) / n)
            g3, g4 = r3.get("avg_retrieval_gap_ll", float("nan")), r4.get("avg_retrieval_gap_ll", float("nan"))
            same = r3.get("total") == r4.get("total")
            if not a.summary_only:
                print(f"  {L:>7}  v3 {a3:.3f}  v4 {a4:.3f}  change {a4 - a3:+.3f}  se {se:.3f}  "
                      f"margin {g3:+.2f} -> {g4:+.2f}" + ("" if same else "  (placements changed)"))
            worst.append((abs(a4 - a3), f.stem, L, a3, a4))
            if same:
                cells.append(dict(L=L, a3=a3, d=a4 - a3, g3=g3, dg=g4 - g3))
    worst.sort(reverse=True)
    if not a.summary_only:
        print("\nlargest changes:")
        for d, s, L, a3, a4 in worst[:12]:
            print(f"  {d:.3f}  {s} @ {L}: {a3:.3f} -> {a4:.3f}")

    print(f"\nsummary over {len(cells)} cells scored with the same placements under both specs")
    up, down, p = sign_test([c["d"] for c in cells])
    print(f"accuracy change: up {up}, down {down}, unchanged {len(cells) - up - down}; "
          f"mean {st.mean(c['d'] for c in cells):+.4f}; sign test p = {p:.2f}")
    for name, sel in ((f"<= {V3_CONTROL_CAP:,} tokens", lambda c: c["L"] <= V3_CONTROL_CAP),
                      (f"> {V3_CONTROL_CAP:,} tokens", lambda c: c["L"] > V3_CONTROL_CAP)):
        s = [c for c in cells if sel(c)]
        if not s:
            continue
        au, ad, ap_ = sign_test([c["d"] for c in s])
        mu, md, mp = sign_test([c["dg"] for c in s])
        print(f"  {name:>15}: {len(s):3d} cells; accuracy up {au} / down {ad} (p = {ap_:.2f}); "
              f"margin median {st.median(c['dg'] for c in s):+.3f} nats, up {mu} / down {md} (p = {mp:.2f})")
    binned(cells, "g3", [-99, 0.5, 1.5, 3, 99], "mean margin (nats)")
    binned(cells, "a3", [0, 0.3, 0.7, 0.95, 1.01], "accuracy")


if __name__ == "__main__":
    main()
