"""Parity tests for the fused Triton dilution kernels (docs/kernels.md)."""

from __future__ import annotations

import math

import pytest
import torch

from dilution import kernels
from dilution.kernels import dilution_attention_reference as eager_dilution_core

needs_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and kernels.HAS_TRITON),
    reason="needs CUDA + triton",
)


@needs_gpu
@pytest.mark.parametrize("seq", [1, 65, 257, 1025])
def test_block_scan_ignores_unwritten_future_keys(seq):
    """The scan must mask K1's unwritten triangle, including partial blocks."""
    import triton

    block_m = 64
    nb = triton.cdiv(seq, block_m)
    torch.manual_seed(seq)
    sums = torch.rand(2, nb, seq, device="cuda")
    rows = torch.arange(nb, device="cuda")[:, None]
    keys = torch.arange(seq, device="cuda")[None, :]
    future = keys >= (rows + 1) * block_m
    sums.masked_fill_(future, 0)
    expected = sums.cumsum(1) - sums
    total = sums.sum(1)
    sums.masked_fill_(future, float("nan"))
    carry = torch.empty_like(sums)
    got_total = torch.empty_like(total)
    kernels._dilution_scan[(triton.cdiv(seq, 32), 2)](
        sums, carry, got_total, seq, nb, BLOCK_M=block_m,
        SCAN_B=triton.next_power_of_2(nb), KEYS=32, num_warps=4,
    )
    torch.testing.assert_close(carry, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(got_total, total, atol=2e-6, rtol=2e-6)


@needs_gpu
@pytest.mark.parametrize("head_dim", [16, 32, 64, 128])
def test_saved_query_correction_matches_reference(head_dim):
    torch.manual_seed(head_dim)
    q, k, v = (torch.randn(1, 2, 257, head_dim, device="cuda") for _ in range(3))
    # IEEE D128 needs one stage to fit K2 in this GPU's shared memory.
    _, aux = kernels.dilution_attention_forward(
        q, k, v, return_aux=True, _save_pk=True, num_stages=1,
    )
    aff = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
    causal = torch.ones(257, 257, device="cuda", dtype=torch.bool).tril()
    p = aff.masked_fill(~causal, float("-inf")).softmax(-1)
    expected = (p @ k) / math.sqrt(head_dim)
    torch.testing.assert_close(aux["pk"].reshape_as(q), expected, atol=2e-5, rtol=2e-5)


@needs_gpu
@pytest.mark.parametrize("seq", [8, 64, 128, 257])
@pytest.mark.parametrize("shape", [(1, 1), (2, 4)])
def test_fused_forward_matches_eager_fp32(seq, shape):
    batch, heads = shape
    torch.manual_seed(seq)
    head_dim = 64
    q, k, v = (torch.randn(batch, heads, seq, head_dim, device="cuda") for _ in range(3))

    ref = eager_dilution_core(q, k, v)
    got = kernels.dilution_attention_forward(q, k, v)
    assert torch.isfinite(got).all()
    err = (got - ref).abs().max().item()
    assert err < 1e-3, f"max abs err {err}"


@needs_gpu
def test_model_kernel_path_matches_eager():
    """enable_dilution_kernel() must not change eval logits beyond tf32 noise."""

    from dilution.config import ModelConfig
    from dilution.model import DilutionLM

    cfg = ModelConfig(context_length=128, vocab_size=256, n_layers=2, n_heads=4,
                      d_model=128, attention="dilution")
    torch.manual_seed(0)
    model = DilutionLM(cfg).cuda().eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 128), device="cuda")
    with torch.no_grad():
        eager_logits, _ = model(idx, idx)
        model.enable_dilution_kernel()
        kernel_logits, _ = model(idx, idx)
    assert torch.isfinite(kernel_logits).all()
    assert (kernel_logits - eager_logits).abs().max().item() < 2e-2


@needs_gpu
def test_fused_forward_scalar_weights_and_bf16_inputs():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 128, 64, device="cuda") for _ in range(3))
    ref = eager_dilution_core(q, k, v)
    got = kernels.dilution_attention_forward(q.bfloat16(), k.bfloat16(), v.bfloat16())
    # bf16 inputs vs fp32 reference: loose tolerance, must remain finite/normalized.
    assert torch.isfinite(got).all()
    assert (got - ref).abs().max().item() < 5e-2


@needs_gpu
@pytest.mark.parametrize("seq", [8, 64, 257, 1024, 2049])
def test_v3_backward_matches_reference_autograd(seq):
    """Full gradient parity: dq, dk, dv vs autograd through the reference."""

    torch.manual_seed(seq)
    B, H, D = 2, 4, 64
    mk = lambda: torch.randn(B, H, seq, D, device="cuda", requires_grad=True)
    q1, k1, v1 = mk(), mk(), mk()
    do = torch.randn(B, H, seq, D, device="cuda")

    kernels.dilution_attention_reference(q1, k1, v1).backward(do)
    clones = [t.detach().clone().requires_grad_(True) for t in (q1, k1, v1)]
    kernels.dilution_attention(*clones, dot_precision="ieee").backward(do)

    for name, ref, ker in zip(("dq", "dk", "dv"), (q1, k1, v1), clones):
        scale = ref.grad.abs().max().item() + 1e-6
        rel = (ker.grad - ref.grad).abs().max().item() / scale
        assert rel < 2e-3, f"{name} relative error {rel}"


@needs_gpu
@pytest.mark.parametrize("seq", [1, 65, 257, 2049])
def test_key_forward_checkpoints_and_totals(seq):
    """Training's running demand agrees with complete reference prefixes."""
    torch.manual_seed(seq)
    q, k, v = (torch.randn(1, 2, seq, 32, device="cuda") for _ in range(3))
    out, aux = kernels.dilution_attention_forward(q, k, v, return_aux=True, _save_pk=True)
    torch.testing.assert_close(out, eager_dilution_core(q, k, v), atol=1e-5, rtol=1e-4)
    aff = (q @ k.transpose(-2, -1)) / math.sqrt(32)
    causal = torch.ones(seq, seq, device="cuda", dtype=torch.bool).tril()
    p = aff.masked_fill(~causal, float("-inf")).softmax(-1)
    inclusive = p.cumsum(-2).reshape(2, seq, seq)
    torch.testing.assert_close(aux["committed_total"], p.sum(-2), atol=2e-5, rtol=2e-5)
    bm = aux["bwd_block_m"]  # 32 on the key-parallel path, else the forward tile
    assert aux["carry"].shape[1] == (seq + bm - 1) // bm
    for rb in range(aux["carry"].shape[1]):
        # Only keys in this row block's causal range are initialized.
        end = min((rb + 1) * bm, seq)
        expected = inclusive[:, rb * bm - 1, :end] if rb else torch.zeros(2, end, device="cuda")
        torch.testing.assert_close(aux["carry"][:, rb, :end], expected, atol=2e-5, rtol=2e-5)


@needs_gpu
@pytest.mark.parametrize("seq", [64, 257])
def test_frozen_demand_experiment_matches_its_surrogate_reference(seq):
    from dilution import kernels_frozen as frozen

    torch.manual_seed(seq)
    xs = [torch.randn(1, 2, seq, 64, device="cuda", requires_grad=True) for _ in range(3)]
    refs = [x.detach().clone().requires_grad_(True) for x in xs]
    do = torch.randn_like(xs[0])
    ref = frozen.dilution_attention_reference(*refs)
    ref.backward(do)
    got = frozen.dilution_attention(*xs, dot_precision="ieee")
    got.backward(do)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-4)
    for x, reference in zip(xs, refs):
        torch.testing.assert_close(x.grad, reference.grad, atol=2e-5, rtol=2e-3)


@needs_gpu
def test_v3_backward_finite_grads():
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, 2, 96, 64, device="cuda", requires_grad=True) for _ in range(3))
    out = kernels.dilution_attention(q, k, v, dot_precision="ieee")
    out.sum().backward()
    for t in (q, k, v):
        assert t.grad is not None and torch.isfinite(t.grad).all()


@needs_gpu
def test_bf16_operand_mode_forward_and_backward_close_to_fp32():
    """dot_precision="bf16" (bf16 tensor-core q.k / p.v, fp32 cumsum math)."""

    torch.manual_seed(5)
    B, H, T, D = 2, 4, 257, 64
    q32, k32, v32 = (torch.randn(B, H, T, D, device="cuda", requires_grad=True) for _ in range(3))
    do = torch.randn(B, H, T, D, device="cuda")

    ref = kernels.dilution_attention_reference(q32, k32, v32)
    ref.backward(do)
    ref_grads = [t.grad.clone() for t in (q32, k32, v32)]

    qb, kb, vb = (t.detach().bfloat16().requires_grad_(True) for t in (q32, k32, v32))
    out = kernels.dilution_attention(qb, kb, vb, dot_precision="bf16")
    out.backward(do.bfloat16())
    assert torch.isfinite(out).all()
    assert (out.float() - ref).abs().max().item() < 5e-2
    for name, gr, t in zip(("dq", "dk", "dv"), ref_grads, (qb, kb, vb)):
        g = t.grad.float()
        assert torch.isfinite(g).all(), name
        rel = (g - gr).abs().max().item() / (gr.abs().max().item() + 1e-6)
        assert rel < 1e-1, f"{name} bf16-mode relative error {rel}"


@needs_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decode_kernel_matches_reference_rows(dtype):
    """Fused decode step == reference row t-1 over the full prefix, token after token
    (each step consumes the accumulators the previous step advanced in place)."""

    torch.manual_seed(1)
    batch, heads, head_dim, prefix, steps = 2, 4, 32, 37, 6
    total = prefix + steps
    q, k, v = (torch.randn(batch, heads, total, head_dim, device="cuda") for _ in range(3))
    ref = eager_dilution_core(q, k, v)

    prec = "bf16" if dtype == torch.bfloat16 else "ieee"
    _, aux = kernels.dilution_attention_forward(
        q[:, :, :prefix].to(dtype), k[:, :, :prefix].to(dtype), v[:, :, :prefix].to(dtype),
        dot_precision=prec, return_aux=True,
    )
    capacity = total + 3  # spare rows: the kernel must never touch them
    kc = torch.zeros(batch, heads, capacity, head_dim, device="cuda", dtype=dtype)
    vc = torch.zeros_like(kc)
    kc[:, :, :prefix] = k[:, :, :prefix].to(dtype)
    vc[:, :, :prefix] = v[:, :, :prefix].to(dtype)
    committed = torch.zeros(batch, heads, capacity, device="cuda")
    committed[:, :, :prefix] = aux["committed_total"]

    tol = 1e-3 if dtype == torch.float32 else 5e-2
    for t in range(prefix, total):
        kc[:, :, t] = k[:, :, t].to(dtype)
        vc[:, :, t] = v[:, :, t].to(dtype)
        out = kernels.dilution_decode_step(
            q[:, :, t : t + 1].to(dtype), kc, vc, committed, t + 1
        )
        assert torch.isfinite(out).all()
        err = (out[:, :, 0] - ref[:, :, t]).abs().max().item()
        assert err < tol, f"t={t} max abs err {err}"
    assert committed[:, :, total - 1].abs().max().item() > 0
    assert committed[:, :, total:].abs().max().item() == 0


@needs_gpu
def test_model_kernel_decode_matches_eager_decode():
    """The fused decode step must reproduce the eager cached step and the full forward."""

    from dilution.config import ModelConfig
    from dilution.model import DilutionLM

    cfg = ModelConfig(context_length=128, vocab_size=256, n_layers=2, n_heads=4,
                      d_model=128, attention="dilution")
    torch.manual_seed(0)
    model = DilutionLM(cfg).cuda().eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 40), device="cuda")
    prefix = 24

    def run():
        logits, caches = model.prefill(tokens[:, :prefix])
        outs = [logits[:, -1]]
        for pos in range(prefix, tokens.shape[1]):
            logits = model.decode_step(tokens[:, pos : pos + 1], pos, caches)
            outs.append(logits[:, -1])
        return torch.stack(outs, dim=1)

    with torch.no_grad():
        eager = run()
        model.enable_dilution_kernel()
        fused = run()
        full, _ = model(tokens, tokens)
    assert torch.isfinite(fused).all()
    assert (fused - eager).abs().max().item() < 2e-2
    assert (fused - full[:, prefix - 1 :]).abs().max().item() < 2e-2


# --- bid ablations (kernels_bid: softmax / relu / relu2) -----------------------

BIDS = ["softmax", "relu", "relu2", "softplus", "minshift", "sigmoid", "expshift", "sigshift"]


@needs_gpu
@pytest.mark.parametrize("bid", BIDS)
@pytest.mark.parametrize("share", [True, False])
@pytest.mark.parametrize("seq", [64, 257])
def test_bid_forward_matches_reference(bid, share, seq):
    from dilution import kernels_bid as kb
    torch.manual_seed(seq)
    q, k, v = (torch.randn(2, 4, seq, 64, device="cuda") for _ in range(3))
    ref = kb.dilution_attention_reference(q, k, v, bid=bid, share_cumsum=share)
    got = kb.dilution_attention_forward(q, k, v, bid=bid, share_cumsum=share)
    assert torch.isfinite(got).all()
    assert (got - ref).abs().max().item() < 1e-3


@needs_gpu
@pytest.mark.parametrize("bid", BIDS)
@pytest.mark.parametrize("share", [True, False])
@pytest.mark.parametrize("seq", [64, 257])
def test_bid_backward_matches_reference_autograd(bid, share, seq):
    from dilution import kernels_bid as kb
    torch.manual_seed(seq)
    B, H, D = 2, 4, 64
    mk = lambda: torch.randn(B, H, seq, D, device="cuda", requires_grad=True)
    q1, k1, v1 = mk(), mk(), mk()
    do = torch.randn(B, H, seq, D, device="cuda")
    kb.dilution_attention_reference(q1, k1, v1, bid=bid, share_cumsum=share).backward(do)
    clones = [t.detach().clone().requires_grad_(True) for t in (q1, k1, v1)]
    kb.dilution_attention(*clones, dot_precision="ieee", bid=bid, share_cumsum=share).backward(do)
    # relu2 squares the affinities, doubling their dynamic range; its fp32
    # summation-order noise sits at ~3e-3 relative at T=257 (measured), so it gets
    # a looser bound than the 2e-3 the other bids meet.
    bound = 5e-3 if bid == "relu2" else 2e-3
    for name, ref, ker in zip(("dq", "dk", "dv"), (q1, k1, v1), clones):
        scale = ref.grad.abs().max().item() + 1e-6
        rel = (ker.grad - ref.grad).abs().max().item() / scale
        assert rel < bound, f"{bid} {name} relative error {rel}"


@needs_gpu
def test_bid_softmax_path_is_kernels_py():
    """kernels_bid at bid=softmax must equal production kernels.py bit-for-bit."""
    from dilution import kernels_bid as kb
    torch.manual_seed(3)
    q, k, v = (torch.randn(2, 4, 200, 64, device="cuda") for _ in range(3))
    a = kernels.dilution_attention_forward(q, k, v)
    b = kb.dilution_attention_forward(q, k, v, bid="softmax")
    assert torch.equal(a, b)


@needs_gpu
@pytest.mark.parametrize("bid", ["relu", "relu2", "softplus", "minshift", "sigmoid", "expshift", "sigshift"])
def test_bid_decode_matches_reference_rows(bid):
    from dilution import kernels_bid as kb
    torch.manual_seed(5)
    B, H, T, D = 2, 3, 70, 64
    q, k, v = (torch.randn(B, H, T, D, device="cuda") for _ in range(3))
    full = kb.dilution_attention_reference(q, k, v, bid=bid)
    prefix = T - 8
    _, aux = kb.dilution_attention_forward(q[:, :, :prefix], k[:, :, :prefix], v[:, :, :prefix],
                                           return_aux=True, bid=bid)
    committed = torch.zeros(B, H, T, device="cuda"); committed[:, :, :prefix] = aux["committed_total"]
    kc, vc = k.contiguous(), v.contiguous()
    for t in range(prefix, T):
        y = kb.dilution_decode_step(q[:, :, t:t + 1].contiguous(), kc, vc, committed, t + 1, bid=bid)
        err = (y[:, :, 0] - full[:, :, t]).abs().max().item()
        assert err < 1e-3, f"{bid} step {t}: {err}"


@needs_gpu
def test_bid_no_cumsum_softmax_is_plain_attention():
    """bid=softmax, share_cumsum=False must equal SDPA causal attention."""
    from dilution import kernels_bid as kb
    torch.manual_seed(11)
    q, k, v = (torch.randn(2, 4, 200, 64, device="cuda") for _ in range(3))
    sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    got = kb.dilution_attention_forward(q, k, v, bid="softmax", share_cumsum=False)
    assert (got - sdpa).abs().max().item() < 1e-3


@needs_gpu
@pytest.mark.parametrize("bid", ["relu", "relu2", "softplus", "minshift", "sigmoid", "expshift", "sigshift"])
def test_bid_model_kernel_matches_eager(bid):
    from dilution.config import ModelConfig
    from dilution.model import DilutionLM
    torch.manual_seed(7)
    cfg = ModelConfig(vocab_size=128, n_layers=2, d_model=64, n_heads=2, context_length=96,
                      layer_attention=["dilution", "softmax"], bid=bid)
    model = DilutionLM(cfg).cuda().eval()
    idx = torch.randint(0, 128, (2, 96), device="cuda")
    with torch.no_grad():
        eager, _ = model(idx)
        model.enable_dilution_kernel()
        fused, _ = model(idx)
    assert (eager - fused).abs().max().item() < 5e-2

@needs_gpu
@pytest.mark.parametrize("seq", [64, 257])
def test_exclusive_history_collapses_to_the_identity(seq):
    """share = p / (committed + eps) -- temporal attention's exclusive denominator on
    row-normalised bids -- is degenerate in causal self-attention (S18).

    Key i is first visible to query i, and no earlier query can have claimed it, so
    committed[i, i] is exactly zero and that entry wins the row by 1/eps. The forward
    is exact against the reference and dv is exact; dq/dk are fp32 noise on cancelling
    1e9-scale intermediates, which is a property of the operator, not of the kernel.
    """
    from dilution import kernels_bid as kb

    torch.manual_seed(seq)
    q, k, v = (torch.randn(1, 2, seq, 64, device="cuda") for _ in range(3))
    out = kb.dilution_attention_forward(q, k, v, dot_precision="ieee", share_cumsum="exclusive")
    ref = kb.dilution_attention_reference(q, k, v, share_cumsum="exclusive")
    assert (out - ref).abs().max().item() < 1e-3

    aff = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(64)
    causal = torch.ones(seq, seq, device="cuda", dtype=torch.bool).tril()
    p = aff.masked_fill(~causal, float("-inf")).softmax(-1)
    committed = p.cumsum(-2) - p
    assert committed.diagonal(dim1=-2, dim2=-1).abs().max().item() == 0.0
    share = p / (committed + 1e-9)
    attn = share / (share.sum(-1, keepdim=True) + 1e-9)
    diag = attn.diagonal(dim1=-2, dim2=-1)
    assert diag.min().item() > 0.99, f"expected identity collapse, got {diag.min().item()}"
    off = attn - torch.diag_embed(diag)
    assert off.abs().max().item() < 1e-3, "off-diagonal mass should be ~1/eps smaller"
    assert off.abs().sum(-1).max().item() < 1e-2

    # dv still matches exactly: it only needs the (collapsed) attention matrix.
    xs = [t.clone().requires_grad_(True) for t in (q, k, v)]
    refs = [t.clone().requires_grad_(True) for t in (q, k, v)]
    do = torch.randn_like(q)
    kb.dilution_attention_reference(*refs, share_cumsum="exclusive").backward(do)
    kb.dilution_attention(*xs, dot_precision="ieee", share_cumsum="exclusive").backward(do)
    rel = (xs[2].grad - refs[2].grad).abs().max().item() / (refs[2].grad.abs().max().item() + 1e-9)
    assert rel < 2e-3, f"dv relative error {rel}"


@needs_gpu
def test_exclusive_first_row_is_the_softmax_row():
    """With committed == 0 on row 0, the row normalise cancels eps: temporal attention's
    published first-step definition falls out rather than being special-cased."""
    from dilution import kernels_bid as kb

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, 32, 64, device="cuda") for _ in range(3))
    out = kb.dilution_attention_reference(q, k, v, share_cumsum="exclusive")
    assert torch.allclose(out[:, :, 0], v[:, :, 0], atol=1e-5)


@needs_gpu
@pytest.mark.parametrize("share_cumsum", [True, "exclusive", False])
def test_kernel_cached_decode_matches_full_forward_for_share_modes(share_cumsum):
    # The Triton prefill + decode path must honour the share mode (the exclusive-history
    # variant divides by earlier demand only, committed + eps).
    from dilution.config import ModelConfig
    from dilution.model import DilutionLM
    torch.manual_seed(3)
    cfg = ModelConfig(vocab_size=128, n_layers=2, d_model=64, n_heads=2, context_length=96,
                      attention="dilution", share_cumsum=share_cumsum)
    model = DilutionLM(cfg).cuda().eval()
    model.enable_dilution_kernel()
    idx = torch.randint(0, 128, (2, 80), device="cuda")
    with torch.no_grad():
        full, _ = model(idx, idx)
        logits, caches = model.prefill(idx[:, :64])
        assert (logits[:, -1] - full[:, 63]).abs().max().item() < 2e-2
        for pos in range(64, 80):
            step = model.decode_step(idx[:, pos:pos + 1], pos, caches)
            err = (step[:, -1] - full[:, pos]).abs().max().item()
            assert err < 2e-2, f"decode differs at {pos}: {err}"
