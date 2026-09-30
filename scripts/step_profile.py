"""In-step per-kernel CUDA times for the production dilution kernels (torch profiler).

    PYTHONPATH=src python scripts/step_profile.py [--shape 8x8x4096] [--steps 200]
    PYTHONPATH=src python scripts/step_profile.py --variants   # launch-config variants

Times the full autograd step (forward + backward) the way training runs it and
attributes CUDA time per kernel, so cache state and launch overlap are the
real ones. Isolated per-kernel timing (scripts/profile_kernels.py) was found to
misreport in-sequence costs by up to 1.5 ms (S26). Refuses to run while
dilution.train is on the GPU.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import torch
from torch.profiler import ProfilerActivity, profile

from dilution import kernels as K


def gpu_busy():
    out = subprocess.run(["pgrep", "-f", "dilution[.]train"], capture_output=True, text=True)
    return out.stdout.strip() != ""


def step_profile(q, k, v, do, steps, **kw):
    def step():
        out = K.dilution_attention(q, k, v, dot_precision="bf16", **kw)
        out.backward(do)
        q.grad = k.grad = v.grad = None
    for _ in range(20):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
    rows = {}
    for e in prof.key_averages():
        if e.device_time_total <= 0:
            continue
        key = e.key if e.key.startswith("_dilution") else "torch (elementwise, scans, fills, casts)"
        rows[key] = rows.get(key, 0.0) + e.device_time_total / steps / 1000.0
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", default="8x8x4096")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--variants", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if gpu_busy() and not args.force:
        print("refusing: dilution.train is on the GPU"); return 2
    B, H, T = (int(x) for x in args.shape.split("x"))
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, T, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    do = torch.randn(B, H, T, 64, device="cuda", dtype=torch.bfloat16)
    print(torch.cuda.get_device_name(0), f"| B{B} H{H} T{T} D64 | {args.steps} steps after 20 warmup\n")
    configs = [("production", {})]
    if args.variants:
        configs += [
            ("BC 8 warps", dict(b3_num_warps=8)),
            ("BA 8 warps", dict(b4_num_warps=8)),
            ("bwd num_stages 2", dict(bwd_num_stages=2)),
            ("fwd num_warps 8", dict(num_warps=8)),
            ("production (repeat)", {}),
        ]
    for label, kw in configs:
        rows = step_profile(q, k, v, do, args.steps, **kw)
        order = ["_dilution_k1", "_dilution_scan", "_dilution_k2", "_dilution_key_forward", "_dilution_normalize", "_dilution_ba", "_dilution_bc", "_dilution_dq_epilogue", "torch (elementwise, scans, fills, casts)"]
        parts = "  ".join(f"{name.replace('_dilution_', ''):>5s} {rows.get(name, 0.0):5.2f}" for name in order)
        extra = {k_: v_ for k_, v_ in rows.items() if k_ not in order}
        print(f"{label:22s} total {sum(rows.values()):5.2f} ms | {parts}" + (f"  other {sum(extra.values()):.2f}" if extra else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
