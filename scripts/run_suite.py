"""Sequential suite runner: every config in a directory x every seed.

    python scripts/run_suite.py configs/4l_128_20m --seeds 1337
    python scripts/run_suite.py configs/4l_128_20m --include competition --stop-after-steps 500

Conventions:
  - output dir runs/<suite>/<arm>_seed<seed> (suite = config dir name), anchored at the
    repo root regardless of the invoking cwd
  - a run whose metrics.jsonl already holds a terminal event (complete, screen_complete, or a
    deliberate --stop-at-step 'stopped') is skipped
  - a run with a latest.pt but no terminal event is resumed automatically
  - a run with records but no checkpoint is NEVER overwritten implicitly: it is skipped with a
    message. Archived result records (published without checkpoints) look exactly like this.
    Pass --restart-incomplete to discard such records and start those runs fresh.
  - arms run in --include order when given, else alphabetically
  - to reproduce published runs without touching the archived records under runs/, use a
    separate --output-root (e.g. runs_repro)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TERMINAL_EVENTS = {"complete", "screen_complete", "stopped"}


def has_terminal_event(metrics_path: Path) -> bool:
    """True when the run's most recent event ends it (complete, screen_complete or a deliberate
    'stopped'). Status comes from the latest lifecycle event, not from any historical one: a run
    stopped at a milestone and later resumed ('resume', train, checkpoint events) is resumable
    again if it is interrupted before completing."""
    if not metrics_path.is_file():
        return False
    last = None
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line).get("event")
            except json.JSONDecodeError:
                continue
    return last in TERMINAL_EVENTS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_dir", help="directory of experiment YAMLs (one per arm)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337])
    parser.add_argument("--include", nargs="+", help="arm names (config stems) to run, in order")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "bf16", "float32", "fp32"))
    parser.add_argument(
        "--stop-after-steps", type=int, default=0,
        help="screen every arm for N steps (separate _screenN run dirs, no checkpoints)",
    )
    parser.add_argument(
        "--stop-at-step", type=int, default=0,
        help="checkpointed stop at step N on the full schedule (kept checkpoint_step_N.pt; resumable)",
    )
    parser.add_argument(
        "--restart-incomplete", action="store_true",
        help="discard run records that have no checkpoint and restart those runs (off by default: "
             "archived records have no checkpoints)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    if not config_dir.is_dir():
        print(f"config directory does not exist: {config_dir}", file=sys.stderr)
        return 2
    suite = config_dir.name
    configs = {path.stem: path for path in sorted(config_dir.glob("*.yaml"))}
    if not configs:
        print(f"no *.yaml configs in {config_dir}", file=sys.stderr)
        return 2

    if args.include:
        missing = [name for name in args.include if name not in configs]
        if missing:
            print(f"unknown arm(s): {', '.join(missing)}", file=sys.stderr)
            return 2
        queue = [(name, configs[name]) for name in args.include]
    else:
        queue = sorted(configs.items())

    # The child runs with cwd=ROOT; anchor every path the runner inspects there
    # too, so skip/resume decisions and the child agree on the same directories.
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root

    failures = 0
    for seed in args.seeds:
        for arm, config_path in queue:
            run_name = f"{arm}_seed{seed}"
            if args.stop_after_steps > 0:
                run_name += f"_screen{args.stop_after_steps}"
            output_dir = output_root / suite / run_name
            metrics_path = output_dir / "metrics.jsonl"
            if has_terminal_event(metrics_path):
                print(f"skip     {suite}/{run_name} (terminal event present)")
                continue

            command = [
                sys.executable, "-m", "dilution.train",
                "--config", str(config_path.resolve()),
                "--output-dir", str(output_dir),
                "--seed", str(seed),
            ]
            latest = output_dir / "latest.pt"
            if latest.is_file():
                command += ["--resume", str(latest)]
            elif output_dir.is_dir() and any(output_dir.iterdir()) and not any(output_dir.glob("*.pt")):
                # Records without a checkpoint: a run that died before its first checkpoint,
                # or an archived result record. Never discard them implicitly.
                if not args.restart_incomplete:
                    print(f"skip     {suite}/{run_name} (run records but no checkpoint; "
                          "use --restart-incomplete to discard them, or a separate --output-root)")
                    continue
                print(f"restart  {suite}/{run_name} (run records but no checkpoint; --restart-incomplete)")
                command += ["--overwrite-run"]
            if args.device:
                command += ["--device", args.device]
            if args.dtype:
                command += ["--dtype", args.dtype]
            if args.stop_after_steps > 0:
                command += ["--stop-after-steps", str(args.stop_after_steps)]
            if args.stop_at_step > 0:
                command += ["--stop-at-step", str(args.stop_at_step)]

            print(("dry-run  " if args.dry_run else "run      ") + " ".join(command), flush=True)
            if args.dry_run:
                continue
            env = dict(os.environ)
            env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            result = subprocess.run(command, cwd=ROOT, env=env)
            if result.returncode != 0:
                failures += 1
                print(f"FAILED   {suite}/{run_name} (exit {result.returncode})", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
