"""Recompute the sampling evaluation's published summaries from its per-sample metric records.

    PYTHONPATH=src python scripts/verify_samples.py [results/samples/scale209m_1x.json ...]

For each summary <set>.json it reads <set>.metrics.jsonl.gz (one JSON line per sample: generator,
prompt length, decoding, prompt index and end position, the text statistics, the judge score and
the cross-scores, without token ids or text), reruns dilution.sampling.summarize with the settings
the summary records, and compares every number in the summary's "table" and "paired" sections.
The bootstrap uses a fixed seed, so agreement is exact up to float formatting. Exits non-zero on
any mismatch.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from dilution.sampling import read_metrics, summarize

ROOT = Path(__file__).resolve().parent.parent


def numbers(obj, path=""):
    """Every numeric leaf of a JSON value, keyed by its path."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from numbers(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from numbers(v, f"{path}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, float(obj)


def same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def verify(summary_path: Path) -> bool:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = read_metrics(summary_path.with_suffix(".metrics.jsonl.gz"))
    table, paired = summarize(records, models=list(summary["models"]), lengths=summary["lengths"],
                              decodings=list(summary["decodings"]), baselines=summary["baselines"])
    want = dict(numbers({"table": summary["table"], "paired": summary["paired"]}))
    got = dict(numbers({"table": table, "paired": paired}))
    missing = sorted(set(want) ^ set(got))
    wrong = [k for k in want if k in got and not same(want[k], got[k])]
    ok = not missing and not wrong
    print(f"{summary_path.name}: {len(records)} samples, {len(want)} published numbers; "
          + ("all reproduced" if ok else f"{len(wrong)} differ, {len(missing)} unmatched"))
    for k in (wrong + missing)[:10]:
        print(f"  {k}: published {want.get(k)} recomputed {got.get(k)}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summaries", nargs="*", type=Path)
    a = ap.parse_args()
    paths = a.summaries or sorted(p for p in (ROOT / "results/samples").glob("*.json"))
    if not paths:
        print("no summaries found under results/samples")
        return 2
    return 0 if all([verify(p) for p in paths]) else 1


if __name__ == "__main__":
    sys.exit(main())
