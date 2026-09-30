"""Blend two GPT-2 uint16 corpora at the document level into a new harness-compatible corpus.

    python scripts/blend_corpus.py --out data/wiki50_fineweb50_2b \\
        --source wiki=path/to/wikipedia_gpt2/train.bin:1000000000 \\
        --source fineweb=data/fineweb_edu_100bt_40b/train.bin:1000000000

Each source contributes whole documents (split on the GPT-2 end-of-text token 50256) up to
its token quota, taken from the start of the file. The combined document list is shuffled
with a fixed seed and written as headless little-endian uint16, so every training window
mixes both sources at document granularity with an exact token balance. A sibling
manifest.json in the format runtime.verify_token_manifest enforces records the sources,
their quotas and the output's sha256. Only a "train" split is written: point the config's
val_path at an existing validation file so losses stay comparable across data blends.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

EOT = 50256


def load_docs(path, quota):
    """First `quota` tokens' worth of whole documents from `path`, as (array, starts, ends)."""
    mm = np.memmap(path, dtype=np.uint16, mode="r")
    # read a little past the quota so the last document can be closed cleanly
    take = min(mm.size, int(quota * 1.02) + 1_000_000)
    a = np.array(mm[:take])
    eot = np.flatnonzero(a == EOT)
    starts = np.concatenate([[0], eot[:-1] + 1])
    ends = eot + 1                                   # keep the EOT with its document
    lens = ends - starts
    keep = np.searchsorted(np.cumsum(lens), quota, side="right")
    return a, starts[:keep], ends[:keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", action="append", required=True, help="name=path:quota_tokens")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    arrays, docs, meta = [], [], []
    for spec in a.source:
        name, rest = spec.split("=", 1)
        path, quota = rest.rsplit(":", 1)
        arr, st, en = load_docs(path, int(quota))
        k = len(arrays)
        arrays.append(arr)
        docs.append(np.stack([np.full(len(st), k), st, en], axis=1))
        n_tok = int((en - st).sum())
        meta.append(dict(name=name, path=str(Path(path).resolve()), quota_tokens=int(quota),
                         documents=int(len(st)), tokens=n_tok,
                         median_doc_tokens=int(np.median(en - st))))
        print(f"{name}: {len(st):,} documents, {n_tok:,} tokens (median doc {meta[-1]['median_doc_tokens']})", flush=True)

    alldocs = np.concatenate(docs)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(alldocs)

    total = int((alldocs[:, 2] - alldocs[:, 1]).sum())
    train = out / "train.bin"
    written = 0
    h = hashlib.sha256()
    with open(train, "wb") as f:
        buf, buf_n = [], 0
        for src, s, e in alldocs:
            buf.append(arrays[src][s:e]); buf_n += e - s
            if buf_n >= 16_000_000:
                chunk = np.concatenate(buf).astype("<u2").tobytes()
                f.write(chunk); h.update(chunk); written += buf_n; buf, buf_n = [], 0
        if buf:
            chunk = np.concatenate(buf).astype("<u2").tobytes()
            f.write(chunk); h.update(chunk); written += buf_n
    assert written == total, (written, total)

    manifest = {
        "manifest_version": 1,
        "token_format": {"dtype": "uint16", "byte_order": "little", "header_bytes": 0, "bytes_per_token": 2},
        "source": {"kind": "document-level blend", "seed": a.seed, "components": meta},
        "split_policy": {"document_level_shuffle": True, "eot_token": EOT,
                         "note": "train only; use an existing validation file for comparable losses"},
        "splits": {"train": {"path": "train.bin", "num_tokens": total, "num_bytes": total * 2,
                             "sha256": h.hexdigest()}},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    shares = ", ".join(f"{m['name']} {m['tokens'] / total:.1%}" for m in meta)
    print(f"wrote {train} ({total:,} tokens: {shares}) and manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
