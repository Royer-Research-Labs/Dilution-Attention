# Fused kernels

`src/dilution/kernels.py`. The exclusive column-cumsum is the operator's only
sequential coupling, so it decomposes into per-row-block partial key-sums
(fully parallel) joined by one fused Triton block scan. This document describes
the current design and its measured cost; how it got here, what was tried and
the full-schedule validation are summarised in S26.

## Training forward (two score sweeps)

    K1 (grid N x row-blocks): online causal-softmax row statistics, with
        pk = SCALE * p @ k accumulated during that same statistics sweep
    KF (grid N x key-blocks; query blocks in causal order): reconstruct p,
        retain per-key cumulative demand in registers, save incoming carry
        checkpoints, and accumulate share @ V and row share sums with atomics
    normalize: out = accumulated numerator / row share sum

KF replaces a second score sweep, the block scan, and a separate output pass.
It stores the carry checkpoints that the full-gradient backward needs, but does
not allocate the block-sum buffer S. It runs on 32-row query tiles (K1 stays at
64), so the checkpoints come every 32 rows, which is the tile height the
backward's BA pass needs; KF itself is neutral at 32 rows. Its output/row-sum
atomics use relaxed ordering; only the next kernel reads their completed sums.
Moving pk into K1's online sweep is necessary for this to pay: calculating it
with more atomics in KF lost the saving. The online accumulator is rescaled
whenever the running row maximum changes, as is the softmax denominator.

A training step makes four score products. Full cumulative demand and its
backward suffix remain; no gradient is detached.

## Forward-only and prefill (row-parallel)

    K1 (grid N x nB): causal-softmax row stats (rmax, rden) + per-block
        key-sums S of p
        carry = exclusive block-cumsum(S)           # Triton, (N, nB, T)
    K2 (grid N x nB): rebuild p from the stats, committed = carry + in-block
        exclusive cumsum(p), share = p / (committed + p + eps)
        -> row-normalize -> @ V

Nothing (T x T)-shaped reaches DRAM; scratch is O(nB * T). Tiles are loaded
in their native dtype (bf16 under autocast) and the q.k / p.v products run on
bf16 tensor cores with fp32 accumulation (`dot_precision="bf16"`, the
default); the cumsum/share math stays fp32. `return_aux=True` also hands back
the inclusive column totals of p, which are the decode accumulator after the
prefix, for free from the scan buffer.

The scan also reduces the inclusive column totals in the same launch. It
masks K1's unwritten future-key triangle, so S needs no zero fill; there is no
separate full-size cumsum temporary or totals reduction. Forward-only calls and
cached prefill omit pk and use this path, avoiding the training forward's
output atomics and final normalization launch.

Training sequences of at most `KEY_FORWARD_MIN_SEQ` = 256 tokens also use this
decomposition, with online pk enabled. This path saves its carry every 64 rows,
so its backward runs on 64-row tiles, which spill heavily; the threshold used to
be 1024, and moving the 209M 1K run onto the key-parallel path took it from 88K to
110K tok/s (S26). With the current backward, key-parallel over row-parallel time
is 1.01 / 1.00 / 0.95 / 0.86 at 128 / 256 / 512 / 1024 tokens
(`scripts/bench_crossover.py`), so the threshold is 256. Outputs and gradients
of the two paths agree to bf16 rounding (at most 1.8e-4 relative).

## Backward

Let ds be the gradient at the share (from the row-normalize and the p.v
product), inv = 1/(committed + p + eps), and dk_share = -ds * p * inv^2.
The gradient reaching p_ij is dp_ij = ds (committed + eps) inv^2 + suffix_ij,
where suffix_ij is the sum of dk_share over every *later* query on the same
key. The softmax Jacobian then gives da = p (dp - ghat) with ghat_i =
sum_j p_ij dp_ij. Two facts make this two kernels: the suffix is a running sum
if row blocks are visited last to first, and the -ghat term is linear, so it
can be applied after the fact:

    BA  (grid N x key-blocks; 32-row blocks in REVERSE order): recompute p
        from the saved row stats, ds from do.v, dk_share; keep a running
        per-key suffix sum so dp is exact; accumulate
            dv[j]      += sum_i attn_ij do_i               (in registers)
            dk_main[j] += SCALE sum_i (p dp)_ij q_i        (in registers)
            dq_main[i] += SCALE sum_j (p dp)_ij k_j        (fp32 atomics, relaxed)
            ghat[i]    += sum_j (p dp)_ij                  (64-wide atomics, relaxed)
    BC  (grid N x key-blocks, 64-row blocks): the Jacobian correction
        dk[j] = dk_main[j] - SCALE sum_i p_ij ghat_i q_i. Needs only p (row
        stats): no carry, no cumsum, no do tile
    epilogue kernel: dq = dq_main - ghat * pk, stored in the gradient dtype;
        BA stores dv and BC the corrected dk in the gradient dtype directly

dv is accumulated in BA from the same attn tile. At 64-row tiles that extra
product doubles BA (255 registers, spills 46 -> 200), whether dv is a register
accumulator or per-tile atomics; at 32-row tiles BA fits (26 spills) and the
product costs 0.45 ms. Forming attn before ds keeps the two tiles from being
live together. 16-row tiles lose to iteration count.

BA loads the next row block's q and do tiles one iteration ahead and runs with
num_stages=2 on the bf16 path, so the remaining per-row loads (row stats,
carry) are pipelined as well. Neither change helps alone; together BA drops
from 3.14 to 2.82 ms at B8 H8 T4096 and from 18.1 to 16.2 ms at the 209M 16K
shape (B2 H12). Prefetching the row stats and carry by hand as well is slower
(42 spills), as are register caps (`maxnreg` 128-200), 8 warps, and 128-key
tiles.

The epilogue reads pk saved by K1. Moving that product out of BA removes one
tensor-core product and one atomic accumulation per tile from the most
register-constrained pass. It increases K1's time, but saves more in BA.

BA reads `dout` in fp32: a bf16 copy made its compiled schedule 0.6 ms slower
before the prefetch and is neutral with it (2.84 vs 2.82 ms), so it stays
fp32. BC does not read `dout`.

Both kernels recompute tiles from the forward's saved stats (rmax, rden,
carry, ssum) plus dbar_i = do_i . out_i. Exposed as
`dilution.kernels.dilution_attention` (torch.autograd.Function). Gradient
parity vs autograd through the reference on dq, dk, dv at <= 2e-3 relative
in fp32 mode and <= 1e-1 relative in bf16-operand mode (tests/test_kernels.py).

The atomics use relaxed memory ordering: nothing in-kernel reads the
accumulated buffers and the launch boundary is a full sync. The default
acquire-release ordering cost 2.4 ms per step at B8 H8 T4096.

An earlier four-kernel backward (row-parallel key-sums and partial ghat, a
reverse block scan, then separate ghat, dv and dk/dq kernels) is still defined
in kernels.py but is not launched.

## Cached decode

Decode state per dilution layer: the KV cache plus **one** fp32 accumulator
vector (committed column totals of p). Prefill obtains it free from the
forward kernel's scan buffer. Each token step is one fused kernel
(`dilution_decode_step`, grid = batch x heads):

    sweep 1: affinities q.k over the cached keys -> O(T) fp32 scratch;
             online max/sum
    sweep 2: p from the stats, share = p/(committed+p+eps), sum(share . v)
             and sum(share); committed += p stored in place

K is streamed once, V once, the scratch once; output = acc / sum(share) is
stored in the activation dtype. Caches are preallocated (at the context length,
or at prompt + new tokens when generating past it) and filled by slice
assignment, so a step never reallocates.

## Long context (S14, S15)

Three limits surface between 64K and 256K tokens, all in addressing rather
than arithmetic:

- **64-bit scratch offsets.** The carry/S buffers are `(n, T/BLOCK_M, T)` fp32,
  so `n * nB * T` passes 2^31 at 12 heads and T = 131,072: the per-head stride
  is 2.7e8 and the last head's base is 2.95e9, which wraps int32 and raises an
  illegal memory access. Only the per-program base multiply is promoted
  (`pid_n.to(tl.int64) * stride_sn`), hoisted out of the row-block loops so the
  tile offsets stay 32-bit. In-step cost at B8 H8 T4096: 5.78 ms against
  5.76-5.80 unpatched, i.e. free.
- **Row-chunked forward scratch.** That scratch is quadratic in T -- 24 GiB at
  H=12 T=128K and 96 GiB at 256K, which no longer fits. The (batch, head) rows
  are independent, so `dilution_attention_forward` processes them in chunks
  sized to `scratch_budget_gib` (default 16) whenever the caller does not need
  the carry back (`return_aux=False`). Bitwise identical to the unchunked path.
  256K at B1 H12 D64 now runs in **17.9 GiB and 1.8 s**.
- **Grid dimension y.** CUDA caps grid dim 1 at 65,535, and the block scan
  launches `T / KEYS` programs on it -- 65,536 at T = 256K with the KEYS = 4
  tile the shared-memory budget forces at that block count. The scan's grid
  axes are swapped so key tiles ride axis 0.

Inference also skips the autograd Function entirely (`torch.is_grad_enabled()`
is false), so no backward carry is allocated at all, and the NIAH loader turns
the fused kernels on: the eager tensor path materialises a (T x T) matrix per
head and needs ~12 GiB per layer at 16K, which OOMs a 24-layer eval.

## Ablation kernels (`kernels_bid.py`)

`SHARE` is a three-way constexpr: 0 = no share step (`attn = p`, the softmax
control), 1 = inclusive `p / (committed + p + eps)` (dilution), 2 = exclusive
`p / (committed + eps)` (temporal attention's denominator, S18). Only `inv` and
the backward's direct term differ between 1 and 2; the suffix term
`-p * inv^2` is common. Both the forward and the backward must be launched with
the same code -- the backward previously used `int(bool(share_cumsum))`, which
silently computed the inclusive gradient for mode 2 (fixed; the parity test
caught it before any run).

## Precision and limits

- `dot_precision`: "ieee" for parity tests, "tf32" for fp32 activations,
  "bf16" (default) for bf16 activations. fp32 accumulators and fp32
  cumsum/share math in every mode.
- Tiles 64x64, 4 warps everywhere, except the training path's KF and BA at
  32 rows x 64 keys; forward num_stages=2, backward 1 except BA, which runs
  2 stages with bf16 operands. fp32 operand tiles keep one stage, since two
  can exceed shared memory at larger head dims. An 88-combination sweep of
  {64,128}^2 x {4,8} warps x {1,2,3} stages (forward) and {64,128}^2 x
  {4,8}^2 x {1,2} (backward) found nothing better: every 128-wide tile is out
  of shared memory or slower, and every 8-warp forward is slower.
- head_dim >= 16 required (Triton dot minimum); the model falls back to the
  tensor path below that.
- IEEE fp32 D128 can exceed shared memory with the default two-stage
  forward on this GPU; `dilution_attention_forward(..., num_stages=1)`
  works for the saved-correction parity test.
- Training retains an extra fp32 `(B, H, T, D)` tensor per dilution layer
  for pk: 64 MiB at B8 H8 T4096 D64, or 32 MiB at B2 H8 T8192 D64.
- The saved carry checkpoints come every 32 rows on the training path, so
  that buffer is `(B*H, T/32, T)` fp32: 128 MiB per layer at B2 H8 T8192,
  256 MiB at B1 H8 T16384; +0.75 GiB peak at the 12-layer 8K training config
  (10.54 -> 11.29 GiB) and about 1.9 GiB at 16K.
- Training uses atomics, so it matches the reference to tolerance but is not
  bitwise deterministic.

## Measured

Forward plus backward of the attention op inside the model step (torch
profiler, bf16, D64, RTX PRO 6000 Blackwell): **5.46 ms at B8 H8 T4096, 5.05 ms
at B2 H8 T8192, 30.18 ms at the 209M 16K shape B2 H12 T16384, and 3.67 ms at the
209M 1K shape B32 H12 T1024.** At B8 H8 T4096, BA is 2.82 ms; the other kernels,
unchanged by the last pass and measured just before it, are K1 0.51, KF 1.28,
normalize 0.04, BC 0.55, dq epilogue 0.09 and torch glue 0.22 ms.

Model throughput (median tok/s over steps 200-1000 of a screen): 159.9K for the
64M model at 8K against about 280K for SDPA softmax at the same shape, and 33.4K
for the 209M dilution model at 16K against 72-75K for softmax. From the first
working kernels, model throughput rose 4% / 19% / 30% / 41% / 52% at 1K / 2K /
4K / 8K / 16K, and the last pass added 2.6-4.1%, with loss unchanged; a
full-schedule replay reproduces the first kernels' loss to 0.0007 nats (S26).

At long context (forward only, bf16, B1 D64): 1.5 s / ~17 GiB at T=128K H12,
1.8 s / 17.9 GiB at T=256K H12, 1.1 s / 17.3 GiB at T=256K H8.

Profiling tools: `scripts/step_profile.py` times each kernel in the order the
step runs it (isolated timing can misreport in-sequence cost by up to 1.5 ms),
`scripts/profile_kernels.py` times kernels in isolation, and
`scripts/bench_crossover.py` compares the two forward paths at short context.

## Where the time goes

BA is 52% of the step (2.82 of 5.46 ms at B8 H8 T4096; 16.2 of 30.2 ms at the
209M 16K shape). Its SASS runs about 1,650 instructions per warp per 32x64 tile,
including 80 tensor-core ops, 29 barriers, about 220 shared-memory accesses and
23 spill accesses. With two blocks of 4 warps per SM, it issues at about 27% of
the SM's rate: latency, not arithmetic, bounds it. Removing one piece at a time
(timing only): atomics 0.33 ms, the two row scans 0.33, the dv product 0.29,
reciprocals 0.08. No single piece dominates.

Triton's scan cannot run in the tensor-core accumulator layout, so each of BA's
two scans converts the tile through shared memory to a row-split layout and
back (4 of the loop's 6 layout conversions). Gluon in Triton 3.7.1 cannot lower
a scan in that layout or its linear equivalent either. An inline PTX
warp-shuffle scan on the accumulator layout is exact (1.5e-7 relative) and
removes those conversions (BA barriers 45 -> 33, spills 32 -> 16), but saves
nothing on top of the prefetch and 4% of KF. It is not adopted: its correctness
depends on the compiler's choice of accumulator layout, which the language
cannot assert.

Structurally, p is rebuilt four times per tile per step, about 4.9
matmul-equivalents against a theoretical 4, and the cumsum chain (the in-tile
scans, the share division and its backward) is about a quarter of the step
(S4). Launch configuration, tile shapes, pass ordering, atomics placement and
the other levers reachable from Triton have been measured (S26); further gains
need register-level scheduling of BA that Triton does not expose, or a
different decomposition.
