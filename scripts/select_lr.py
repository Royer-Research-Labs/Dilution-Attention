"""Pick the learning rate for an arm from an LR sweep by the stability judge.

    python scripts/select_lr.py runs/lr_8192 hybrid_softmax4 --baseline runs/12l_8192_600m/hybrid_softmax4_seed1337:3e-4

Prints one LR string (e.g. "6e-4"). Among runs that the judge calls stable,
the one with the lowest best val loss wins; if none is stable, the lowest
best val loss among all runs (a marginally lower grad median is not worth a
worse loss). Exit 1 if no runs are found.
"""
import argparse, glob, json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_stability import judge  # noqa: E402


def score(run_dir: Path):
    ok, why = judge(run_dir)
    ev = [json.loads(l) for l in (run_dir / "metrics.jsonl").open(encoding="utf-8") if l.strip()]
    g = [e["grad_norm"] for e in ev if e.get("event") == "train" and e["step"] >= 1000]
    done = [e for e in ev if e.get("event") == "complete"]
    if not g or not done:
        return None
    return ok, done[-1]["best_val_loss"], statistics.median(g)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep_root")
    ap.add_argument("arm")
    ap.add_argument("--baseline", help="run_dir:lr to include, e.g. runs/x/arm_seed1337:3e-4")
    ap.add_argument("--seed", default="1337")
    args = ap.parse_args()
    cands = []
    for d in glob.glob(f"{args.sweep_root}/{args.arm}_lr*_seed{args.seed}"):
        lr = Path(d).name.split("_lr")[1].split("_seed")[0]
        s = score(Path(d))
        if s: cands.append((lr, *s))
    if args.baseline:
        d, lr = args.baseline.rsplit(":", 1)
        s = score(Path(d))
        if s: cands.append((lr, *s))
    if not cands:
        print("no completed runs", file=sys.stderr); return 1
    for lr, ok, best, med in sorted(cands, key=lambda c: c[2]):
        print(f"  {'STABLE  ' if ok else 'UNSTABLE'} lr {lr:8s} best {best:.4f} grad med {med:.2f}", file=sys.stderr)
    stable = [c for c in cands if c[1]]
    pick = min(stable or cands, key=lambda c: c[2])
    print(pick[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
