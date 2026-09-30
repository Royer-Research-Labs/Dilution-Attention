"""Isolated kernel / historical-decomposition diagnostics.

Use step_profile.py for production in-step costs: long-context training now
uses the key-parallel forward, which this historical forward comparison omits.

    PYTHONPATH=src python scripts/profile_kernels.py [--shape 8x8x4096] [--iters 30]

1. Runs the forward + backward once to build the O(T) statistics, then times
   each kernel launch (K1, scan, K2, dbar, B1, scan, B2, B3, B4) in isolation
   with CUDA events, and prints Triton's register / spill / shared-memory
   metadata per kernel.
2. Times B4 diagnostic variants (timing only, not parity-correct) that remove
   one suspected cost at a time: the in-tile cumsum, the dq atomics, the do.v
   product, or all three -- so the gap between B4 and its siblings gets a
   named cause. Refuses to run while dilution.train is on the GPU.
"""
from __future__ import annotations

import argparse
import math
import statistics
import subprocess
import sys

import torch

from dilution import kernels as K

if not K.HAS_TRITON:
    sys.exit("triton not available")
import triton
import triton.language as tl
from dilution.kernels import _dot, _ds_from_v, _share_from_k


@triton.jit
def _b4_diag(
    Q, K_, V, DO, DK, DQ, DBAR, SSUM, RMAX, RDEN, CARRY, SUFK, GHAT,
    stride_qn, stride_qt, stride_qd,
    stride_sn, stride_sb, stride_st,
    T, NB, HEAD_DIM: tl.constexpr, SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr,
    NO_SCAN: tl.constexpr, NO_ATOMIC: tl.constexpr, NO_DOV: tl.constexpr, RELAXED: tl.constexpr,
):
    """Production B4 with optional pieces removed (timing only)."""
    pid_n = tl.program_id(0)
    pid_kb = tl.program_id(1)
    j_offs = pid_kb * BLOCK_N + tl.arange(0, BLOCK_N)
    j_mask = j_offs < T
    d_offs = tl.arange(0, HEAD_DIM)
    q_base = Q + pid_n * stride_qn
    k_base = K_ + pid_n * stride_qn
    v_base = V + pid_n * stride_qn
    do_base = DO + pid_n * stride_qn
    k_blk = tl.load(k_base + j_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                    mask=j_mask[:, None], other=0.0)
    v_blk = tl.load(v_base + j_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                    mask=j_mask[:, None], other=0.0)
    dk_acc = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    dq_acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    first_rb = (pid_kb * BLOCK_N) // BLOCK_M
    for rb in range(first_rb, NB):
        m_offs = rb * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_offs < T
        q_tile = tl.load(q_base + m_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                         mask=m_mask[:, None], other=0.0)
        do_tile = tl.load(do_base + m_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                          mask=m_mask[:, None], other=0.0)
        rmax = tl.load(RMAX + pid_n * T + m_offs, mask=m_mask, other=0.0)
        rden = tl.load(RDEN + pid_n * T + m_offs, mask=m_mask, other=1.0)
        dbar = tl.load(DBAR + pid_n * T + m_offs, mask=m_mask, other=0.0)
        ssum = tl.load(SSUM + pid_n * T + m_offs, mask=m_mask, other=1.0)
        ghat = tl.load(GHAT + pid_n * T + m_offs, mask=m_mask, other=0.0)
        c_base = CARRY + pid_n * stride_sn + rb * stride_sb
        sufk_base = SUFK + pid_n * stride_sn + rb * stride_sb

        causal, p, committed, inv = _share_from_k(
            q_tile, k_blk, j_offs, m_offs, T, rmax, 1.0 / rden, c_base, stride_st,
            SCALE, DOT_PREC, LOWP)
        if NO_DOV:
            ds = tl.where(causal, p * 0.5 - dbar[:, None], 0.0)
        else:
            ds = _ds_from_v(do_tile, v_blk, dbar, 1.0 / (ssum + 1e-9), causal, DOT_PREC, LOWP)
        inv2 = inv * inv
        dk_share = -ds * p * inv2
        sufk = tl.load(sufk_base + j_offs * stride_st, mask=j_mask, other=0.0)
        if NO_SCAN:
            suffix = sufk[None, :] + tl.sum(dk_share, axis=0)[None, :] - dk_share
        else:
            suffix = sufk[None, :] + tl.sum(dk_share, axis=0)[None, :] - tl.cumsum(dk_share, axis=0)
        da = tl.where(causal, p * (ds * (committed + 1e-9) * inv2 + suffix - ghat[:, None]), 0.0)
        dk_acc += _dot(tl.trans(da), q_tile, LOWP, DOT_PREC)
        dq_part = _dot(da, k_blk, LOWP, DOT_PREC) * SCALE
        if NO_ATOMIC:
            dq_acc += dq_part
        elif RELAXED:
            tl.atomic_add(DQ + pid_n * stride_qn + m_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                          dq_part, mask=m_mask[:, None], sem="relaxed")
        else:
            tl.atomic_add(DQ + pid_n * stride_qn + m_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                          dq_part, mask=m_mask[:, None])
    if NO_ATOMIC:  # timing only: dump the (BLOCK_M, D) accumulator at a row-block-shaped offset
        mo = pid_kb * BLOCK_M + tl.arange(0, BLOCK_M)
        tl.store(DQ + pid_n * stride_qn + mo[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                 dq_acc, mask=(mo < T)[:, None])
    tl.store(DK + pid_n * stride_qn + j_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
             dk_acc * SCALE, mask=j_mask[:, None])


def gpu_busy():
    out = subprocess.run(["pgrep", "-f", "dilution[.]train"], capture_output=True, text=True)
    return out.returncode == 0 and out.stdout.strip() != ""


def timed(fn, iters):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort()
    return statistics.median(ts)


def spread(fn, iters):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 10], statistics.median(ts), ts[-max(1, len(ts) // 10)]


def meta(compiled):
    m = getattr(compiled, "metadata", None)
    regs = getattr(compiled, "n_regs", None) or getattr(m, "num_regs", None)
    spills = getattr(compiled, "n_spills", None) or getattr(m, "num_spills", None)
    shared = getattr(m, "shared", None) if m is not None else None
    return f"regs {regs} spills {spills} smem {shared}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", default="8x8x4096")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--diag", action="store_true", help="also run the (old) B4 diagnostic variants")
    ap.add_argument("--ba-warps", type=int, default=4)
    args = ap.parse_args()
    if gpu_busy() and not args.force:
        print("refusing: dilution.train is on the GPU"); return 2
    B, H, T = (int(x) for x in args.shape.split("x"))
    D = 64
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    do = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    n = B * H
    bm = bn = 64
    nb = (T + bm - 1) // bm
    qf, kf, vf, dof = (t.reshape(n, T, D).contiguous() for t in (q, k, v, do))
    dev = q.device
    lowp, prec = K._resolve_precision("bf16")
    scale = 1.0 / math.sqrt(D)
    common_f = dict(BLOCK_M=bm, BLOCK_N=bn, DOT_PREC=prec, LOWP=lowp, num_warps=4, num_stages=2)
    common_b = dict(BLOCK_M=bm, BLOCK_N=bn, DOT_PREC=prec, LOWP=lowp, num_warps=4, num_stages=1)
    grid = (n, nb)

    # forward buffers
    out = torch.empty(n, T, D, device=dev, dtype=torch.float32)
    rmax = torch.empty(n, T, device=dev); rden = torch.empty(n, T, device=dev); ssum = torch.empty(n, T, device=dev)
    s = torch.empty(n, nb, T, device=dev)
    pk = torch.empty(n, T, D, device=dev)
    carry = torch.empty_like(s)
    committed_total = torch.empty(n, T, device=dev)
    qs = (qf.stride(0), qf.stride(1), qf.stride(2)); ss = (s.stride(0), s.stride(1), s.stride(2))
    os_ = (out.stride(0), out.stride(1), out.stride(2))
    k1 = lambda: K._dilution_k1[grid](qf, kf, rmax, rden, s, *qs, *ss, T, D, scale, PK=pk, **common_f)
    c1 = k1()
    scan = lambda: K._dilution_scan[(triton.cdiv(T, 32), n)](
        s, carry, committed_total, T, nb, BLOCK_M=bm,
        SCAN_B=triton.next_power_of_2(nb), KEYS=32, num_warps=4)
    scan()
    k2 = lambda: K._dilution_k2[grid](qf, kf, vf, out, ssum, rmax, rden, carry, *qs, *os_, *ss, T, D, scale, **common_f)
    c2 = k2()
    # backward buffers
    of = out
    dbar = (dof.float() * of).sum(dim=-1).contiguous()
    sk = torch.zeros(n, nb, T, device=dev); gpart = torch.empty(n, T, device=dev); ghat = torch.empty(n, T, device=dev)
    dq = torch.zeros(n, T, D, device=dev); dk = torch.empty(n, T, D, device=dev); dv = torch.empty(n, T, D, device=dev)
    b1 = lambda: K._dilution_b1[grid](qf, kf, vf, dof, dbar, ssum, rmax, rden, carry, sk, gpart, *qs, *ss, T, D, scale, **common_b)
    cb1 = b1()
    sufk = sk.flip(1).cumsum(dim=1).flip(1) - sk
    b2 = lambda: K._dilution_b2[grid](qf, kf, rmax, rden, sufk, gpart, ghat, *qs, *ss, T, D, scale, **common_b)
    cb2 = b2()
    nkb = (T + bn - 1) // bn
    kgrid = (n, nkb)
    b3 = lambda: K._dilution_b3[kgrid](qf, kf, dof, dv, ssum, rmax, rden, carry, *qs, *ss, T, nb, D, scale, **common_b)
    cb3 = b3()
    b4 = lambda: K._dilution_b4[kgrid](qf, kf, vf, dof, dk, dq, dbar, ssum, rmax, rden, carry, sufk, ghat, *qs, *ss, T, nb, D, scale, **common_b)
    cb4 = b4()
    # Backward: BA (64-row geometry here; production uses 32) + BC (+ elementwise)
    ghat2 = torch.zeros(n, T, device=dev)
    dk_out = torch.empty_like(dk)
    common_ba = dict(common_b, num_warps=args.ba_warps)
    ba = lambda: K._dilution_ba[kgrid](qf, kf, vf, dof, dk, dq, dv, ghat2, dbar, ssum, rmax, rden, carry, *qs, *ss, T, nb, D, scale, **common_ba)
    cba = ba()
    bc = lambda: K._dilution_bc[kgrid](qf, kf, dk, dk_out, ghat2, rmax, rden, *qs, T, nb, D, scale, **common_b)
    cbc = bc()
    elem = lambda: dq.sub_(ghat2.unsqueeze(-1) * pk)

    print(torch.cuda.get_device_name(0), f"| B{B} H{H} T{T} D64, tiles {bm}x{bn}, median of {args.iters}\n")
    print("kernel   ms      metadata")
    rows = [("K1", k1, c1), ("scan_f", scan, None), ("K2", k2, c2),
            ("dbar", lambda: (dof.float() * of).sum(dim=-1), None),
            ("B1", b1, cb1), ("scan_b", lambda: (sk.flip(1).cumsum(dim=1).flip(1) - sk), None),
            ("B2", b2, cb2), ("B3", b3, cb3), ("B4", b4, cb4)]
    total = 0.0
    for name, fn, comp in rows:
        ms = timed(fn, args.iters); total += ms
        print(f"{name:7s} {ms:6.2f}   {meta(comp) if comp is not None else ''}")
    print(f"{'sum':7s} {total:6.2f}\n")

    print("Two-kernel backward, isolated (BA at 64-row geometry, BC):")
    total2 = 0.0
    for name, fn, comp in [("BA", ba, cba), ("BC", bc, cbc), ("elem", elem, None)]:
        p10, ms, p90 = spread(fn, args.iters); total2 += ms
        print(f"{name:7s} {ms:6.2f}  (p10 {p10:5.2f} p90 {p90:5.2f})   {meta(comp) if comp is not None else ''}")
    fwd = sum(timed(fn, args.iters) for fn in (k1, scan, k2, lambda: (dof.float() * of).sum(dim=-1)))
    print(f"{'bwd':7s} {total2:6.2f}   fwd {fwd:5.2f}   step {fwd + total2:5.2f}")
    print()
    if not args.diag:
        return 0
    print("B4 diagnostics (timing only; each removes one suspected cost)")
    base = dict(NO_SCAN=0, NO_ATOMIC=0, NO_DOV=0, RELAXED=0)
    for label, flags, warps, bn_ in [
        ("B4 as-is (diag kernel)", {}, 4, 64),
        ("relaxed atomics", dict(RELAXED=1), 4, 64),
        ("no dq atomics (local acc)", dict(NO_ATOMIC=1), 4, 64),
        ("8 warps", {}, 8, 64),
        ("8 warps, relaxed atomics", dict(RELAXED=1), 8, 64),
        ("8 warps, no atomics", dict(NO_ATOMIC=1), 8, 64),
        ("BLOCK_N 128, 8 warps", {}, 8, 128),
        ("BLOCK_N 128, 8 warps, relaxed", dict(RELAXED=1), 8, 128),
        ("BLOCK_N 128, 8 warps, no atomics", dict(NO_ATOMIC=1), 8, 128),
        ("no cumsum, no do.v, no atomics", dict(NO_SCAN=1, NO_ATOMIC=1, NO_DOV=1), 4, 64),
        ("relaxed + no cumsum", dict(RELAXED=1, NO_SCAN=1), 4, 64),
        ("relaxed + no do.v", dict(RELAXED=1, NO_DOV=1), 4, 64),
        ("relaxed + no cumsum + no do.v", dict(RELAXED=1, NO_SCAN=1, NO_DOV=1), 4, 64),
        ("relaxed, 4 warps, 2 stages", dict(RELAXED=1), 4, 64),
    ]:
        stages = 2 if "2 stages" in label else 1
        cfg = dict(BLOCK_M=bm, BLOCK_N=bn_, DOT_PREC=prec, LOWP=lowp, num_warps=warps, num_stages=stages, **{**base, **flags})
        kg = (n, (T + bn_ - 1) // bn_)
        fn = lambda cfg=cfg, kg=kg: _b4_diag[kg](qf, kf, vf, dof, dk, dq, dbar, ssum, rmax, rden, carry, sufk, ghat,
                                                *qs, *ss, T, nb, D, scale, **cfg)
        comp = fn()
        print(f"  {label:28s} {timed(fn, args.iters):6.2f} ms   {meta(comp)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
