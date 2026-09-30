"""Full training step (forward + backward) of the fused dilution kernel with the key-parallel
forward on versus off, at a given shape. Answers whether ctx-1024 training should cross over.

    PYTHONPATH=src python scripts/bench_crossover.py --shape 32x12x1024
"""
from __future__ import annotations

import argparse
import sys

import torch

from dilution import kernels as K


def time_step(q, k, v, do, steps):
    def step():
        K.dilution_attention(q, k, v, dot_precision="bf16").backward(do)
        q.grad = k.grad = v.grad = None
    for _ in range(20):
        step()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        step()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / steps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", default="32x12x1024")
    ap.add_argument("--steps", type=int, default=100)
    a = ap.parse_args()
    B, H, T = (int(x) for x in a.shape.split("x"))
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, T, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    do = torch.randn(B, H, T, 64, device="cuda", dtype=torch.bfloat16)
    saved = K.KEY_FORWARD_MIN_SEQ
    outs = {}
    for label, cross in (("row-parallel", 1 << 30), ("key-parallel", 0), ("row-parallel (repeat)", 1 << 30), ("key-parallel (repeat)", 0)):
        K.KEY_FORWARD_MIN_SEQ = cross
        ms = time_step(q, k, v, do, a.steps)
        outs[label.split()[0]] = outs.get(label.split()[0], []) + [ms]
        print(f"B{B} H{H} T{T} | {label:22s} {ms:7.3f} ms/step", flush=True)
    K.KEY_FORWARD_MIN_SEQ = saved
    # both paths must agree numerically
    K.KEY_FORWARD_MIN_SEQ = 1 << 30
    a1 = K.dilution_attention(q, k, v, dot_precision="bf16"); a1.backward(do); g1 = [t.grad.clone() for t in (q, k, v)]
    q.grad = k.grad = v.grad = None
    K.KEY_FORWARD_MIN_SEQ = 0
    a2 = K.dilution_attention(q, k, v, dot_precision="bf16"); a2.backward(do); g2 = [t.grad.clone() for t in (q, k, v)]
    K.KEY_FORWARD_MIN_SEQ = saved
    rel = lambda x, y: ((x.float() - y.float()).norm() / y.float().norm()).item()
    print(f"parity: out {rel(a2, a1):.2e}  dq {rel(g2[0], g1[0]):.2e}  dk {rel(g2[1], g1[1]):.2e}  dv {rel(g2[2], g1[2]):.2e}")
    r, kf = min(outs["row-parallel"]), min(outs["key-parallel"])
    print(f"key-parallel / row-parallel = {kf / r:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
