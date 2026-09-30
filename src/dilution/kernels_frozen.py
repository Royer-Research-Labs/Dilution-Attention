"""Experimental frozen-demand backward; never selected by the production model.

Forward is the original dilution operator. Backward treats the inclusive
cumulative demand plus epsilon as a constant. This is a surrogate gradient,
not an optimization of the original derivative. It ran 23.5% faster in a
1,000-step screen and diverged over a full schedule (S26); retain it only for
reproduction.
"""
from __future__ import annotations

import math
import torch
from . import kernels as K

if K.HAS_TRITON:
    import triton
    import triton.language as tl
    from .kernels import _dot, _share_from_k

    @triton.jit
    def _dilution_frozen_backward(
        Q, K, V, DO, DK, DQ, DV, DBAR, SSUM, RMAX, RDEN, CARRY,
        stride_qn, stride_qt, stride_qd,
        stride_sn, stride_sb, stride_st,
        T, NB, HEAD_DIM: tl.constexpr, SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr,
    ):
        """Surrogate dlogit = A * (do.v - do.out), freezing cumulative demand."""
        pid_n = tl.program_id(0)
        pid_kb = tl.program_id(1)
        j_offs = pid_kb * BLOCK_N + tl.arange(0, BLOCK_N)
        j_mask = j_offs < T
        d_offs = tl.arange(0, HEAD_DIM)
        q_base = Q + pid_n * stride_qn
        k_base = K + pid_n * stride_qn
        v_base = V + pid_n * stride_qn
        do_base = DO + pid_n * stride_qn
        k_blk = tl.load(k_base + j_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                        mask=j_mask[:, None], other=0.0)
        v_blk = tl.load(v_base + j_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                        mask=j_mask[:, None], other=0.0)
        dk_acc = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
        dv_acc = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
        first_rb = (pid_kb * BLOCK_N) // BLOCK_M
        for step in range(0, NB - first_rb):
            rb = NB - 1 - step
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
            c_base = CARRY + pid_n * stride_sn + rb * stride_sb
            causal, p, _, inv = _share_from_k(
                q_tile, k_blk, j_offs, m_offs, T, rmax, 1.0 / rden, c_base, stride_st,
                SCALE, DOT_PREC, LOWP)
            dov = _dot(do_tile, tl.trans(v_blk), LOWP, DOT_PREC)
            attn = tl.where(causal, p * inv / (ssum[:, None] + 1e-9), 0.)
            pdp = attn * (dov - dbar[:, None])
            dv_acc += _dot(tl.trans(attn), do_tile, LOWP, DOT_PREC)
            dk_acc += _dot(tl.trans(pdp), q_tile, LOWP, DOT_PREC)
            dq_part = _dot(pdp, k_blk, LOWP, DOT_PREC) * SCALE
            tl.atomic_add(DQ + pid_n * stride_qn + m_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                          dq_part, mask=m_mask[:, None], sem="relaxed")

        tl.store(DV + pid_n * stride_qn + j_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                 dv_acc, mask=j_mask[:, None])
        tl.store(DK + pid_n * stride_qn + j_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                 dk_acc * SCALE, mask=j_mask[:, None])


def dilution_attention_reference(q, k, v):
    t, d = q.shape[-2:]
    aff = q.float() @ k.float().transpose(-2, -1) / math.sqrt(d)
    causal = torch.ones(t, t, device=q.device, dtype=torch.bool).tril()
    p = aff.masked_fill(~causal, float("-inf")).softmax(-1)
    share = p / (p.cumsum(-2) + 1e-9).detach()
    return (share / (share.sum(-1, keepdim=True) + 1e-9)) @ v.float()


class _FrozenDemandFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, dot_precision):
        out, aux = K.dilution_attention_forward(q, k, v, dot_precision=dot_precision, return_aux=True)
        ctx.save_for_backward(q, k, v, out, aux["rmax"], aux["rden"], aux["carry"], aux["ssum"])
        ctx.dot_precision = dot_precision
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, rmax, rden, carry, ssum = ctx.saved_tensors
        b, h, t, d = q.shape
        n = b * h
        qf, kf, vf, dof = (x.reshape(n, t, d).contiguous() for x in (q, k, v, dout))
        dbar = (dof.float() * out.reshape(n, t, d)).sum(-1)
        dq = torch.zeros(n, t, d, device=q.device, dtype=torch.float32)
        dk = torch.empty(n, t, d, device=q.device, dtype=k.dtype)
        dv = torch.empty(n, t, d, device=q.device, dtype=v.dtype)
        lowp, prec = K._resolve_precision(ctx.dot_precision)
        nb = triton.cdiv(t, 64)
        _dilution_frozen_backward[(n, nb)](
            qf, kf, vf, dof, dk, dq, dv, dbar, ssum, rmax, rden, carry,
            *qf.stride(), *carry.stride(), t, nb, d, 1.0 / math.sqrt(d),
            BLOCK_M=64, BLOCK_N=64, DOT_PREC=prec, LOWP=lowp, num_warps=4, num_stages=1)
        return dq.to(q.dtype).reshape_as(q), dk.reshape_as(k), dv.reshape_as(v), None


def dilution_attention(q, k, v, dot_precision="bf16"):
    """Original forward, frozen-demand surrogate backward; research use only."""
    return _FrozenDemandFn.apply(q, k, v, dot_precision)
