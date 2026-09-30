# Evidence ledger

The numbered findings behind [the README](../README.md) and
[docs/research-results.md](research-results.md), in the order they were measured
(2026-09-06 to 2026-09-30). Each signal records the question, the configuration, the
numbers, and the run records and result files that back them.

How to read it:

- **Scope.** Exploratory studies outside the scope of this release are not included. These
  were hybrids with token-local mixers and learned-position comparisons that were not re-run
  with modern controls. No claim in the public docs rests on them. The kernel development log
  (profiling, variants and null results) is condensed into one signal, S26, with the
  full-schedule validation and the frozen-demand result that the rest of the ledger relies on.
- **Later signals correct earlier ones.** Corrections are kept in place and marked in the
  signal they correct. The consolidated list is the "Superseded and withdrawn claims" table
  in [docs/research-results.md](research-results.md). S19, S21 and S22 in particular were
  revised by second and third seeds (S22's three-seed update, S24).
- **Conventions.** "64M" is the 12-layer d512 model and "209M" the 24-layer d768 model.
  "1x Chinchilla" is 20 tokens per parameter. NIAH is likelihood-scored keyword retrieval
  with 7 scored needles, so chance is 1/7 = 0.14. Loss comparisons use a common validation
  slice (`scripts/eval_checkpoint.py`) unless a signal says otherwise. Seed counts are
  stated per result.
- **Evidence paths.** `runs/<group>/<run>/` holds each run's `metrics.jsonl`, resolved config
  and data provenance. Checkpoints are not included. `results/{niah,niah_v4,eval,structure,needle,samples}/`
  holds the evaluation outputs. Corpus paths are repo-relative (`data/...`). Build the
  corpora with `dilution-prepare-data` (see the README).
  Local absolute paths in the run records were rewritten to this repo-relative form on
  export. Corpus file hashes are untouched, but a recorded manifest-copy hash refers to the
  record before its paths were rewritten.

## S1 — Context ladder complete at LR 3e-4: dilution's margin over softmax is 0.011 / 0.072 / 0.122 / 0.191 / 0.425 at 1024 / 2048 / 4096 / 8192 / 16384 (measured, 1-3 seeds)

Runs at ctx 2048 and 4096 complete the ladder (best val, seed 1337 unless
noted; U = unstable):

    ctx      all_dilution   all_softmax   margin   tok/s dil / sm
    1024     3.6828         3.6942        0.011    262K / 342K
    2048     3.6607         3.7328        0.072    234K / 368K
    4096     3.6620         3.7843        0.122    169K / 333K
    8192     3.6524 (3s)    3.8434 (3s)   0.191    108K / 278K
    16384    3.6564 (3s)    4.0861 U      0.425     62K / 213K

1. Dilution's loss is flat across a 16x context range (3.6524-3.6828, and
   the 1024 value is the worst); softmax degrades monotonically (3.6942 ->
   4.0861). The margin doubles roughly per context doubling from 2048.

2. Retrieval along the ladder (acc / margin at the longest cell that fits):
   all_dilution 1.000/7.30 (1920) -> 0.943/3.02 (3840) -> 0.957/6.2 (7936,
   2 of 3 seeds) -> 0.957/4.92 (16000); all_softmax 1.000/3.40 -> 0.900/2.54
   -> 0.33-0.50 -> 0.343. Dilution's margin is 2x softmax's at every length
   and its accuracy holds where softmax's falls to chance.

## S2 — NoPE at ctx 8192: dilution improves without position embeddings (-0.010), softmax loses 0.099 and becomes unstable; 16k seeds: dilution 3.6564 +/- 0.004 with seed-dependent 16000-token retrieval (measured, 1-3 seeds)

NoPE at ctx 8192 (configs/nope_8192, model.positional: none, LR 3e-4):

    ctx 8192              learned pos        NoPE       delta     grad NoPE   NIAH 1024 / 3840 / 7936 (NoPE)
    all_dilution          3.6533             3.6435     -0.010    1.00/1.6    0.943/7.44 0.929/6.35 0.929/4.63
    all_softmax           3.8494             3.9481     +0.099    1.08/25.8 U 0.186/-0.70 0.143/-1.69 0.143/-1.35

1. Dilution does not need a position signal: removing the 4.2M-parameter
   embedding table improves dilution, and NoPE dilution retrieves at 0.929
   over 7936 tokens. The operator's exclusive cumulative sum is an order
   signal in itself (query i is divided by what queries 0..i-1 took), and
   the model uses it. Softmax without positions loses 0.1 nats, develops
   gradient spikes (p99 25.8), and retrieves at chance at every length.

2. This is also the cleanest evidence yet that dilution's mechanism is
   positional: it substitutes for learned positions where softmax cannot do
   without them.

16k seeds (LR 3e-4):

    ctx 16384             1337     2357     7331     mean     sd       NIAH 7936 / 16000 by seed
    all_dilution          3.6609   3.6551   3.6533   3.6564   0.0040   0.986/6.05 0.957/4.92 | 0.943/5.45 0.871/3.46 | 0.886/3.68 0.529/1.06

3. Loss replicates (sd 0.004), and it stays context-flat against its 8192
   mean (+0.004). Retrieval at the 16000-token cell is seed-dependent:
   0.957 / 0.871 / 0.529, with margins 4.9 / 3.5 / 1.1, while 7936 holds at
   0.886-0.986 on every seed. So "retrieves at 0.957 over 16000 tokens"
   (seed 1337, S1) is the best of three seeds; the median seed reads 0.871.

## S3 — Bid ablation: the softmax is load-bearing for a structural reason. The share step's gradient scales as 1/bid, so only bid functions whose Jacobian carries a factor of the bid (softmax, sigmoid) train; relu, relu2, softplus and minshift diverge. With softmax bids the cumsum is worth 0.178 nats and takes long-haystack retrieval from 0.59 to 1.00; with sigmoid bids it costs 0.106 and supplies all the retrieval (measured, 1 seed per cell)

*Correction (release review, 2026-09-30). The structural argument explains relu, relu2 and
minshift but not softplus. Softplus's Jacobian does carry the bid factor: f'(a)/f(a) =
sigmoid(a)/softplus(a) is at most 1, as sigmoid's (1 - sigmoid(a)) is. Its divergence is
measured but not explained by point 1. It trained normally to step ~500 (loss 6.33, gradient
norm 0.8) before blowing up, which points at a numerical cause, like expshift's overflow. One
candidate, untested: the kernel computes softplus in fp32 as max(a, 0) + log(1 + exp(-|a|)),
which is exactly 0 below a ~ -16.6, while its Jacobian sigmoid(a)/R is not, so such keys get
a zero bid with a nonzero gradient through the share step's 1/(committed + eps) term. Point 1
is corrected below; the heading's list of diverging bids stands as measured.*

The operator's first step, p = causal_softmax(a), was replaced by "percent of
share" alternatives (src/dilution/kernels_bid.py, model.bid), all with the
dilution step unchanged, plus a share_cumsum=false control (attn = p /
rowsum(p), no committed division). ctx 8192, LR 6e-4, seed 1337
(configs/bid_8192). Baseline: softmax bids + cumsum = the operator (taken
from the ctx-8192 LR sweep, `runs/lr_8192/all_dilution_lr6e-4_seed1337`).

    ctx 8192, 6e-4                     best val   stable   NIAH 1024 / 3840 / 7936        tok/s
    softmax + cumsum                   3.6046     yes      1.000/7.27 1.000/5.34 1.000/4.62  108K
    softmax, no cumsum                 3.7823     yes      0.971/3.55 0.900/3.04 0.586/0.76  132K
    sigmoid, no cumsum                 3.8319     yes      chance at every length           133K
    relu, no cumsum                    3.8569     yes      0.871/1.77 0.729/1.28 0.586/0.54  135K
    sigmoid + cumsum                   3.9375     yes      0.929/3.84 0.829/2.96 0.771/2.26  106K
    relu2 + cumsum                     4.5195     no       chance                            108K
    relu + cumsum                      4.9895     no       chance                            108K
    softplus + cumsum                  NaN at step ~1100 (grad norms 1e6-1e11 from step 950)
    minshift + cumsum                  NaN at step 25
    expshift + cumsum                  NaN at step 400 (fp32 overflow of the unnormalised exp)

1. Why the relu family fails. share = p / (committed + p + eps), so
   d share / dp = (committed + eps) / (committed + p + eps)^2, which is ~1/p at
   a key nobody has bid on yet. The gradient reaching the affinity is
   jac * (dp - ghat) with jac = dp/da. Softmax has jac = p, which cancels
   the 1/p exactly; sigmoid has jac = p(1-p), which does the same. relu has
   jac = 1[a>0]/R (no factor of p: unbounded where p is tiny but positive),
   relu2 has 2 relu(a)/R ~ sqrt(p), and minshift has jac = 1/R while its
   bid (a - rowmin)/R goes to zero at the row minimum -- so every early-row,
   low-bid key injects a 1/p-sized gradient. Softplus is *not* in this
   group: its jac = sigmoid(a)/R = p * sigmoid(a)/softplus(a), a factor of
   p times at most 1 (see the correction note above). A CPU probe on random
   inputs at T=257 puts the largest |d out / d a| at 0.15 (sigmoid), 0.30
   (softmax) and 9.4 (relu). Minshift exploded from the first steps and
   softplus after ~500 normal ones; relu and relu2 limp with grad medians of
   2-4 and p99 of 40-100 and finish 0.9-1.4 nats behind. Of the bids tried,
   only softmax and sigmoid train with the cumulative share.

2. A per-element floor is not a fix. The first attempt gave every rejected
   key a 1e-6 bid so no row was empty; the share step is scale-free per key,
   so every universally rejected key then collected an O(1/i) share and 16%
   of attention mass landed on keys with zero affinity. The rule that
   works is: dead keys bid exactly 0, and a row with no mass falls back to
   uniform over its window with zero Jacobian.

3. What the cumsum is worth, with the right bids. Softmax attention through
   the same kernels with the share step off is 3.7823; with it on, 3.6046.
   That 0.178 is the operator's whole loss advantage isolated from kernel,
   data and schedule, and retrieval at 7936 goes from 0.586 to 1.000.

4. Sigmoid bids (each in (0,1), no row sum) are the one alternative that
   trains cleanly, and they separate two effects. Sigmoid attention on its
   own is 0.050 behind softmax attention -- the bounded bid costs little on
   loss -- but it retrieves at chance at every length: bounded, independent
   bids cannot single out one key among thousands. Adding the cumsum costs
   sigmoid 0.106 on loss (over-dilution: a key liked by n earlier queries
   carries committed mass ~n, against a fraction per query under softmax)
   and supplies its entire retrieval (0.771 at 7936). So the share step is
   what makes retrieval possible, and softmax's row normalisation is what
   sets the dilution scale correctly.

5. Unnormalised exp (expshift, p = exp(a)/T) has the right Jacobian and no
   saturation but overflows fp32 as affinities grow (NaN at step 400 after a
   normal trajectory to step 325); the max-subtraction is what the row
   normalisation was buying. sigshift, p = exp(a)/(exp(a)+T) = sigmoid(a -
   log T), is the safe form and is measured in S4.

6. Kernel structure, not the operator, is the throughput deficit. Plain
   softmax attention through our two-pass forward / four-kernel backward
   runs at 132K tok/s against SDPA's 280K at this shape; the full dilution
   operator runs at 108K. The cumsum, share division and extra backward
   chain cost 18% on top of a structure that is already 2.1x slower than
   flash attention doing the same math. The first kernel variants tried were
   operator-specific and null; the deficit is generic (seven q.k recomputations and
   exponentials per tile per step against flash's two, one extra forward
   pass, B4 at a third of its siblings' per-product efficiency).

## S4 — sigshift (exp(a)/(exp(a)+T)) is unstable with the dilution step, and the elementwise-bid kernel paths are no faster than the softmax path; the cumsum chain is 25% of the step and the row normalisation is not a cost centre (measured)

sigshift, the overflow-safe form of expshift (S3.5), at ctx 8192 / 6e-4:
val 5.0188 at 10k with grad median 2.17 and p99 26 -- worse than plain
sigmoid at the same step (4.7313) and in relu territory. With the shift the
bids start at ~1/T, so committed mass on early rows is tiny and the share
step's inverse-squared term dominates the gradients despite the p(1-p)
Jacobian; the un-shifted sigmoid, whose bids start near 0.5, over-dilutes
but trains. Cut at 10k. Between S3 and this, every bid we tried at 64M
is measured: softmax bids with the cumsum are the operator; sigmoid is the
only alternative that trains and it loses 0.33; nothing else survives.
*(Release-review qualification: this does not close the bid axis. Raw
exponential bids failed by fp32 overflow in our implementation, and the
numerically stable cumulative-logsumexp form has not been built; softplus's
failure is not explained by S3's argument. See the S3 correction note and
docs/related-work.md.)*

Kernel timing of the bid paths (fwd+bwd, bf16, median of 30; the benchmark
script and its log are not part of the release):

                               B8 H8 T4096      B2 H8 T8192
    production kernels          9.64 ms          9.22 ms
    kernels_bid, softmax        9.68  1.00x      9.30  0.99x
    kernels_bid, sigmoid        9.73  0.99x      9.42  0.98x
    kernels_bid, expshift       9.51  1.01x      9.22  1.00x
    kernels_bid, sigshift       9.99  0.97x      9.65  0.96x
    softmax, no cumsum          7.28  1.32x      6.91  1.33x

Two conclusions. The elementwise bids, which skip K1's statistics sweep,
save nothing measurable: the sweep is ~7% of the step in isolation, the
elementwise forms trade exp2 for an exp plus a divide, and B2 (ghat) still
launches. The row normalisation is not where the time goes. And the cumsum
chain -- the in-tile scans, the share division, the sufk/dk_share backward
-- is 25% of the step (2.4 ms of 9.6), which is larger than the 18%
model-level estimate in S3.6 and is the operator's true inherent cost.
The remaining 2.1x between our no-cumsum path (7.28 ms) and flash
attention is the generic structure: recomputation count and B4's per-
product efficiency. That is where kernel work has to go (S26); the bid
function is not a lever for speed.

## S5 — RoPE at ctx 8192: softmax with RoPE closes the ladder's loss gap (3.6381, from 3.8494), RoPE in every layer destroys dilution's retrieval, and RoPE on alternate layers gives dilution the best cell in the project on both axes: 3.6323 with NIAH 1.00 at every haystack to 7936 (measured, 1 seed per cell)

`configs/rope_8192`: the S2 recipe (12L
64M, B2 T8192, LR 3e-4, 36,622 steps, seed 1337) with two new positional
modes. `rope` rotates q and k in every attention layer; `rope_alt` rotates
layers 1, 3, 5, ... counted from 1, so the first layer always has position
data and the others see only causal order. Both drop the learned position
table (63.5M params, as NoPE). Learned and NoPE cells are S1/S2.

| positional | softmax loss | softmax NIAH 3840 / 6144 / 7936 | dilution loss | dilution NIAH 3840 / 6144 / 7936 | dilution - softmax |
|---|---:|---|---:|---|---:|
| learned (S1) | 3.8494 | 0.50 / 0.41 / 0.46 | 3.6542 | 0.96 / 0.87 / 0.80 | -0.195 |
| none (S2) | 3.9774 unstable | 0.14 / 0.14 / 0.14 | **3.6438** | 0.93 / 0.91 / 0.93 | -0.334 |
| rope | **3.6381** | 0.57 / 0.40 / 0.40 | 3.6527 | 0.20 / 0.16 / 0.20 | +0.015 |
| rope_alt | 3.6694 | 0.53 / 0.40 / 0.34 | **3.6323** | **1.00 / 1.00 / 1.00** | -0.037 |

Throughput: softmax 264-271K tok/s under RoPE, dilution 149K (the kernels
of S26). All RoPE arms stable (grad median 0.95-0.98, p99 < 1.7).

1. **The ladder's 8K margin was mostly softmax's position table.** RoPE
   takes all-softmax from 3.8494 to 3.6381, a 0.211 gain. Against
   dilution's learned-position cell (3.6542) that is a 0.016 softmax
   lead; against dilution's best cell (rope_alt, 3.6323) it is a 0.006
   dilution lead, at the edge of seed noise (S8: 8K cells replicate within 0.006). S1's
   0.191 figure measured learned positions against learned positions;
   on loss, with each operator given its best positional mode, the two
   are close to tied at this context and size.
2. **Dilution does not care how position arrives.** Learned 3.6542, none
   3.6438, RoPE 3.6527: a 0.01 band, with no signal at all the best. The
   cumsum supplies the order the operator needs; extra position
   information is redundant for loss.
3. **RoPE and the cumsum fight over retrieval.** Dilution with RoPE loses
   long-haystack NIAH almost entirely (0.16-0.20 past 6K against 0.87-0.93
   without), while its loss is unchanged. RoPE's decay of attention with
   distance is exactly the bias the share step must overcome to reach a
   far needle: the cumsum favours *earlier* keys, RoPE favours *nearer*
   ones, and the model lands on the loss-optimal compromise, which is not
   the retrieval-optimal one. Softmax with RoPE keeps the modest
   retrieval it had with learned positions (0.40 at 7936).
4. **Alternating RoPE is a worse softmax and the best dilution.** For
   softmax, rope_alt costs 0.031 against every-layer RoPE with the same
   retrieval profile. For dilution it is the opposite: 3.6323, the lowest
   loss of any 8K cell in the project, with NIAH 1.00 at every haystack
   from 256 to 7936, where no other configuration of either operator
   exceeds 0.93 at 7936. The reading that fits both facts: the first
   layer (rotated) gets position to build local structure with, and the
   un-rotated layers keep a distance-neutral affinity, which is what the
   cumsum needs to reach a far needle. Every-layer RoPE removes the
   second; NoPE removes the first.
5. **Where this leaves the comparison at 8K.** Softmax with RoPE matches
   dilution on loss at 1.8x the throughput but retrieves at 0.40 past 6K.
   Dilution with rope_alt is 0.006 better on loss and retrieves perfectly
   across the whole context. The operator's contribution has moved from
   "0.19 nats" to "a small loss edge plus the retrieval", and the
   retrieval part is the one no softmax cell reproduces. One seed per
   cell; the rope_alt dilution cell wants seeds before it carries weight.

Precedent: alternating RoPE and NoPE layers is not a project novelty;
"Rope to Nope and Back Again" (Yang et al., 2025) studies hybrid RoPE/NoPE
stacks for long context (docs/related-work.md). What is new here is the
interaction of that layout with the dilution operator: rope_alt is neutral
for softmax and the best mode for dilution on both axes.

What this changes upstream: S1's per-context margins were measured with
learned positions on both sides and should be read as "against a
learned-position softmax". The 16K rung (0.425) needs the RoPE control
before it means anything; S6 runs it, with dilution NoPE and
rope_alt at 16K. Configs: `configs/rope_8192`; runs:
`runs/rope_8192/*_seed1337`; the rope_alt dilution arm was rerun
after a machine restart interrupted it at step 2,000.

## S6 — The 16K rung with position controls: softmax with RoPE reaches 3.7273 (learned: 4.1195), dilution with alternating RoPE reaches 3.6359 with NIAH 0.93 at 16000 tokens; the operator's 16K margin is 0.09 nats plus retrieval, not 0.425 (measured, 1 seed per cell)

The S5 positional modes at ctx
16384 (12L 64M, B1 T16384, LR 3e-4, 36,622 steps, seed 1337), against
the S1/S2 learned-position cells. NIAH ladder to 16000 tokens.

| cell | val loss | tok/s | NIAH 7936 / 12288 / 16000 | stable |
|---|---:|---:|---|---|
| softmax, learned (S1) | 4.1195 | 215K | 0.33 / 0.20 / 0.34 | no (grad p99 26) |
| softmax, rope | 3.7273 | 202K | 0.30 / 0.26 / 0.23 | yes |
| dilution, learned (S1, seed 1337) | 3.6611 | 62K (now 94K) | 0.99 / 0.97 / 0.96 | yes |
| dilution, learned, seeds 2357 / 7331 (S2) | 3.6564 mean | | 0.94 / 0.94 / 0.87 and 0.89 / 0.67 / 0.53 | yes |
| dilution, none | 3.6408 | 95K | 0.76 / 0.53 / 0.50 | yes |
| dilution, rope_alt | **3.6359** | 94K | **0.96 / 0.94 / 0.93** | yes |

1. **The RoPE control shrinks the 16K margin from 0.425 to 0.09.** RoPE
   fixes both of softmax's 16K problems at once: the loss (-0.392) and
   the instability (grad-norm p99 26 -> 1.9). Against dilution's best
   cell the operator is worth 0.091 nats at 16K, against 0.006 at 8K
   (S5): the edge grows with context, as S1 claimed, but from a much
   lower base.
2. **Alternating RoPE is again dilution's best mode, and it fixes the
   retrieval lottery.** S2 found dilution's 16000-token retrieval
   seed-dependent with learned positions (0.96 / 0.87 / 0.53 over three
   seeds) and NoPE lands at 0.50 here. rope_alt gives 0.93-0.99 at every
   haystack from 256 to 16000 in one seed, at the lowest 16K loss in the
   project. Same pattern as 8K: a rotated first layer plus un-rotated
   cumsum layers is better than either alone, on both axes.
3. **Softmax with RoPE does not retrieve at this context.** 0.40 by 3840
   tokens, 0.23 at 16000, the same profile as its learned-position cell.
   Nothing on the softmax side of this table reaches half the haystack.
4. **Position-mode sensitivity is the operator difference.** Softmax
   moves 0.39 nats between its worst and best positional mode at 16K;
   dilution moves 0.025. The ladder's large margins were mostly
   softmax's sensitivity to a bad position signal at long context.

Runs: `runs/rope_16384/all_softmax_rope_seed1337`,
`runs/nope_16384/all_dilution_nope_seed1337`,
`runs/rope_16384/all_dilution_rope_alt_seed1337`. What is still open:
seeds on the rope_alt cells (one seed each at 8K and 16K) and the alt6
hybrid (softmax on the RoPE layers, dilution on the rest; S7).

## S7 — alt6: softmax on the RoPE layers, dilution on the un-rotated layers. At 8K it is the best cell on every axis (3.6303, NIAH 1.00 everywhere, 193K tok/s); at 16K it ties all-dilution rope_alt on loss (3.6402 vs 3.6359) with better retrieval (0.97 at 16000) at 1.37x the throughput (measured, 1 seed per cell)

`configs/rope_{8192,16384}/alt6_softmax_rope_dilution.yaml`:
`positional: rope_alt` with `layer_attention` alternating softmax
(layers 1, 3, 5, ... counted from 1, the rotated ones) and dilution (the
un-rotated ones), six of each. Each operator gets the positional mode S5
found best for it. Same recipe as S5/S6 (LR 3e-4, seed 1337); the 16K
arm ran only because the 8K arm passed a gate of stable and val < 3.66.

| ctx | cell | val loss | tok/s | NIAH 3840 / 7936 / 12288 / 16000 |
|---|---|---:|---:|---|
| 8K | all-dilution rope_alt (S5) | 3.6323 | 152K | 1.00 / 1.00 / - / - |
| 8K | all-softmax rope (S5) | 3.6381 | 264K | 0.57 / 0.40 / - / - |
| 8K | **alt6** | **3.6303** | 193K | **1.00 / 1.00** / - / - |
| 16K | all-dilution rope_alt (S6) | **3.6359** | 94K | 0.97 / 0.96 / 0.94 / 0.93 |
| 16K | all-softmax rope (S6) | 3.7273 | 202K | 0.40 / 0.30 / 0.26 / 0.23 |
| 16K | **alt6** | 3.6402 | 129K | 0.99 / 0.96 / 0.96 / **0.97** |

Stable at both contexts (grad median 0.92-0.93, p99 1.8-2.0).

1. **The two operators are complementary, not competing, once each has
   its position mode.** Six RoPE softmax layers and six un-rotated
   dilution layers match or beat twelve of either on loss and keep
   dilution's full-context retrieval. With RoPE the softmax layers are
   no longer the weak half, and the dilution layers still supply what
   softmax cannot, retrieval past 4K.
2. **Throughput.** alt6 runs at 1.27x all-dilution at 8K and 1.37x at 16K
   because half the layers are SDPA. Any equal-wall-clock comparison
   would now favour it further over both pure stacks.
3. **What decides between alt6 and all-dilution rope_alt is seeds.** The
   loss differences (0.002 at 8K, 0.004 at 16K) are inside the 0.004 seed
   spread; the retrieval difference at 16000 (0.97 vs 0.93) is inside the
   NIAH sampling noise at 10 samples. Both cells rest on one seed. The
   throughput difference is not noise.

Runs: `runs/rope_8192/alt6_softmax_rope_dilution_seed1337`,
`runs/rope_16384/alt6_softmax_rope_dilution_seed1337`. Open: seeds on
alt6 and rope_alt at both contexts; whether the 6/6 split is the right
ratio (a 4-dilution / 8-softmax RoPE stack would run faster still).

## S8 — Second seeds for the RoPE-era cells: every 8K result replicates within 0.006; at 16K the losses replicate within 0.008 but far-haystack retrieval has a seed spread for both dilution stacks (0.81-0.97 all-dilution rope_alt, 0.59-0.97 alt6 at 16000 tokens) (measured, 2 seeds per cell)

Seed 2357 for the five S5-S7 headline cells, same recipe.

| cell | seed 1337 loss | seed 2357 loss | mean | NIAH at the longest haystack, 1337 / 2357 |
|---|---:|---:|---:|---|
| 8K alt6 | 3.6303 | 3.6323 | **3.6313** | 1.00 / 1.00 |
| 8K all-dilution rope_alt | 3.6323 | 3.6383 | 3.6353 | 1.00 / 1.00 |
| 8K all-softmax rope | 3.6381 | 3.6374 | 3.6378 | 0.40 / 0.31 |
| 16K alt6 | 3.6402 | 3.6484 | 3.6443 | 0.97 / 0.59 |
| 16K all-dilution rope_alt | 3.6359 | 3.6359 | **3.6359** | 0.93 / 0.81 |

All ten runs stable. Throughput per cell reproduces to 1%.

1. **8K is settled to two seeds.** The three cells sit within 0.007 on
   loss (alt6 best by 0.004 over all-dilution rope_alt, 0.006 over
   softmax RoPE) and the two with un-rotated dilution layers retrieve at
   1.00 across the whole context in both seeds; softmax RoPE does not.
2. **16K loss is settled; 16K far retrieval is not.** all-dilution
   rope_alt repeats its loss to four decimals and stays ahead of alt6 by
   0.008 on the mean. Retrieval at 12288-16000 varies by seed for both
   dilution stacks: 0.83 / 0.81 against 0.94 / 0.93 for the pure stack,
   0.70 / 0.59 against 0.96 / 0.97 for alt6. Alternating RoPE narrows
   S2's learned-position lottery (0.96 / 0.87 / 0.53 at 16000) for the
   pure stack but does not remove it, and alt6, with half the dilution
   layers, is more exposed to it. Retrieval through 7936 is 0.90+ in
   every dilution run at both contexts.
3. **What the operator is worth, two seeds, best position mode each
   side.** 8K: 0.003-0.007 nats plus retrieval past 2K. 16K: 0.091 nats
   (3.6359 vs 3.7273, one softmax seed) plus retrieval past 4K. The
   retrieval part is the robust half of the claim; the loss part grows
   with context from a base near zero.

Runs: `runs/rope_8192/*_seed2357`, `runs/rope_16384/*_seed2357`.

Third seed (7331) of 16K all-dilution rope_alt: loss 3.6366
(3.6359 / 3.6359 / 3.6366 over three seeds, spread 0.001), NIAH 1.00 at
every haystack to 12288 and 0.99 at 16000. Far retrieval over three seeds
is 0.99 / 0.89 / 0.99 at 16000 (50-sample values for the first two), so
the second seed was the low draw and the cell's expected far retrieval is
about 0.95.

## S9 — RoPE base at 16K: raising theta from 1e4 to 1e5 or 5e5 gains softmax 0.03 nats; only 5e5 extends its retrieval (0.90 at 3840, 0.56 at 7936 vs ~0.5 / ~0.35); alt6 is indifferent to the base on both axes (measured, 1 seed per cell)

`model.rope_theta` (new field, default 1e4) at ctx 16384, seed 1337.

| cell | val loss | NIAH 3840 / 7936 / 12288 / 16000 |
|---|---:|---|
| softmax rope, 1e4 (S6) | 3.7273 | 0.40 / 0.30 / 0.26 / 0.23 |
| softmax rope, 1e5 | 3.6947 | 0.51 / 0.37 / 0.31 / 0.27 |
| softmax rope, 5e5 | 3.6978 | **0.90 / 0.56** / 0.41 / 0.27 |
| alt6, 1e4 (S7) | 3.6402 | 0.99 / 0.96 / 0.96 / 0.97 |
| alt6, 5e5 | 3.6439 | 0.97 / 0.94 / 0.94 / 0.94 |

1. The base-1e4 control understated softmax by 0.03 nats at 16K; the
   loss gain saturates by 1e5. With the stronger control the 16K
   operator margin is 3.6359 vs 3.6947-3.6978: **0.06 nats**, down from
   0.09 (S6) and 0.425 (S1).
2. Retrieval is a separate axis: 5e5 roughly doubles softmax's
   mid-range retrieval where 1e5 does nothing, consistent with the base
   flattening RoPE's distance decay only once it is large relative to
   the context. Past 12K every softmax cell is at 0.27-0.41.
3. alt6 moves 0.004 on loss and 0.02-0.03 on NIAH with the base: inside
   noise. Its retrieval comes from the un-rotated dilution layers, which
   the base does not touch.

The 209M scale pair (S11) uses 5e5 for both arms: the strongest
softmax control on both axes, and the same rotation on alt6's softmax
layers so the two differ only in the operator of the un-rotated layers.

## S10 — Length extrapolation: an 8K-trained all-dilution rope_alt model retrieves at 1.00 / 1.00 / 0.99 at 9216 / 12288 / 16000 tokens, twice its trained context, in both seeds; softmax RoPE is at chance past 8K; the alt6 hybrid decays through its RoPE softmax layers (measured, eval-only, 1-2 seeds per cell)

The 8K checkpoints of S5-S8 scored on haystacks past the
trained context, with the model's context cap raised to 17408
(`niah_eval --context-override`; only models without a learned position
table can be evaluated this way). Same NIAH protocol (8 needles, 10
placements). No training involved.

| 8K-trained model | 7936 (in-context) | 9216 | 12288 | 16000 |
|---|---:|---:|---:|---:|
| all-dilution rope_alt, seed 1337 | 1.00 | 1.00 | 1.00 | 1.00 |
| all-dilution rope_alt, seed 2357 | 1.00 | 1.00 | 1.00 | 0.99 |
| all-dilution none | 0.93 | 0.90 | 0.90 | 0.77 |
| alt6, seed 1337 | 1.00 | 0.94 | 0.51 | 0.33 |
| alt6, seed 2357 | 1.00 | 1.00 | 0.64 | 0.14 |
| all-softmax rope, seed 1337 | 0.40 | 0.33 | 0.17 | 0.14 |
| all-softmax rope, seed 2357 | 0.31 | 0.14 | 0.14 | 0.16 |

1. **Un-rotated dilution layers extrapolate.** The share step has no
   notion of absolute or relative distance: a key's weight is its bid
   divided by the demand already committed on it, and neither quantity
   changes meaning when the sequence is longer than any seen in
   training. The pure dilution stacks retrieve at 2x their trained
   context with no fine-tuning; rope_alt perfectly, NoPE at 0.77-0.90.
2. **The rotated first layer helps extrapolation too.** rope_alt beats
   NoPE by 0.10-0.23 at every extrapolated length, so the first layer's
   RoPE is supplying local structure that survives the length change
   (its own rotation has been seen at every relative offset up to 8K,
   and that is all a first layer needs).
3. **RoPE softmax layers are the part that does not extrapolate.**
   Softmax RoPE is at chance (0.14-0.17; 8 choices) by 12288 tokens, the
   standard result for RoPE past its trained length. alt6 inherits this:
   its six softmax layers decay from 1.00 to 0.14-0.33 by 16000 even
   though its six dilution layers would not, so the hybrid's output is
   gated by its weakest half at extrapolated lengths. The 16K in-context
   seed spread of S8 (alt6 0.59-0.97 at 16000) looks like the same
   mechanism at the training length.
4. **Consequence for the trade.** At the trained length alt6 and pure
   dilution rope_alt are equivalent (S7/S8) and alt6 is 1.3-1.4x
   faster. Past the trained length only the pure dilution stack works.
   That is a real difference between the two "best" configurations,
   and the first capability in the project that softmax has no
   configuration for at all.

Results: `results/niah/extrap16k__*.json`.

**50-sample NIAH on the 16K checkpoints** (7936 / 12288 /
16000; 10-sample values from S6-S9 in parentheses). The 16K seed
spread is the model, not the eval:

| 16K checkpoint | 50 samples | 10 samples |
|---|---|---|
| all-dilution rope_alt, seed 1337 | 0.99 / 0.99 / 0.99 | 0.96 / 0.94 / 0.93 |
| all-dilution rope_alt, seed 2357 | 0.97 / 0.90 / 0.89 | 0.93 / 0.83 / 0.81 |
| alt6, seed 1337 | 0.99 / 0.99 / 0.99 | 0.96 / 0.96 / 0.97 |
| alt6, seed 2357 | 0.92 / 0.69 / 0.59 | 0.90 / 0.70 / 0.59 |
| alt6, base 5e5 | 0.99 / 0.99 / 0.99 | 0.94 / 0.94 / 0.94 |
| softmax rope, base 5e5 | 0.58 / 0.42 / 0.29 | 0.56 / 0.41 / 0.27 |

Ten placements per needle reproduce the fifty-placement values within
0.05 in every cell. The pure dilution stack's far-haystack floor across
two seeds is 0.89, the hybrid's 0.59, softmax's best 0.29.

## S11 — 209M at 16K, 1x Chinchilla (4.19B tokens): alt6 beats the softmax RoPE control by 0.050 nats on a common validation slice (3.0385 vs 3.0883) and retrieves 1.00 at every haystack to 16000 where the control drops to 0.80; per unit wall clock the control is ahead because alt6 runs at 0.6x its throughput (measured, 1 seed)

24L d768 12H (208.5M params), ctx 16384, 65,536
tokens per step (control batch 4; alt6 batch 2 x grad-accum 2, see
below), LR 6e-4 with 1,000 warmup on the 128,174-step (8.4B-token)
cosine schedule, RoPE base 5e5 on both arms, fresh 40B-token FineWeb-Edu
corpus, seed 1337. Both arms
were stopped right after their step-64,000 checkpoint (4.19B tokens, 1x
Chinchilla by total parameters; the learning rate is still at 55.7% of
peak there, so these are mid-schedule losses, comparable between the
arms but not to a finished run).

| step 64,000 | softmax RoPE 5e5 | alt6 (softmax on RoPE layers, dilution on the rest) |
|---|---:|---:|
| val loss, common 2.1M-token slice (batch 2, `scripts/eval_checkpoint.py`) | 3.0883 | **3.0385** |
| val loss, in-run window (different windows, see 3) | 3.1192 | 3.0056 |
| NIAH 7936 / 12288 / 16000, 20 placements | 0.99 / 0.93 / 0.80 | **1.00 / 1.00 / 1.00** |
| NIAH at 16000, early / mid / late needle | 0.49 / 0.93 / 1.00 | 1.00 / 1.00 / 1.00 |
| throughput | 72.0K tok/s | 44.1K tok/s |
| wall clock to step 64,000 | 16.4 h | 26.7 h |
| peak memory | 66.5 GiB | 45.6 GiB (batch 2 x 2) |

1. **Loss.** 0.050 nats at matched tokens on identical data, one seed.
   The 12L cells at 600M tokens gave 0.006 at 8K (S5) and 0.06-0.09 at
   16K (S6/S9); at 3.3x the parameters and 7x the tokens the 16K gap
   is at the low end of that range, not gone. Seed 2357 of both arms
   (below) says whether 0.05 is real.
2. **Retrieval.** The control learns most of its retrieval with scale
   (0.99 at 7936 against 0.56 for the 12L cell) but keeps the RoPE
   distance-decay failure on early needles at 16000 (0.49). The
   un-rotated dilution layers remove it: 1.00 at every haystack and
   every depth. This is the part of the result that did not compress
   with scale at all.
3. **Correction to the in-run comparison.** The trainer evaluates
   `eval_batches x batch_size` sequences, so the control's in-run window
   was 524K tokens and alt6's the first 262K of them; the common-slice
   eval shows alt6's window is 0.064 nats easier. The matched-step
   "0.11 lead" reported during the run overstated the gap by that
   amount, and the in-run claim that alt6 reached the control's best
   loss at step 34,000 / 14.2 h is wrong: corrected for the window
   offset, alt6 reaches the control's 1x loss at about step 51,000 /
   21.3 h, against 16.4 h for the control. At equal wall clock the
   control is ahead; at equal tokens alt6 is ahead. Both evals now run
   under no_grad with a chunked loss, and the eval window will be made
   batch-independent before the next scale run.
4. **Memory.** The batch probe chose 4 for alt6 (86.8 GiB peak), but at
   that point the allocator's reserved pool crossed the card's 95.6 GB
   and the WSL driver paged into shared system memory: throughput fell
   from 43K to 19-24K tok/s with no OOM error. alt6 was restarted from
   scratch at batch 2 x accumulation 2 (same tokens per step, 45.6 GiB,
   45K tok/s). Size batches by reserved memory with real margin.
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` is unsupported under
   WSL2 here (CUDA driver error). The eager dilution path (no kernels)
   needs 12 GiB per layer at 16K and OOMed the common eval of the 24L
   model; the eval loader now enables the fused kernels on CUDA.

Runs: `runs/scale_16384/{all_softmax_rope,alt6_softmax_rope_dilution}_seed1337`
(paused at step 64,000, resumable); NIAH `results/niah/scale1x__*`;
common eval `results/eval/scale1x__*`. Follow-ups: extrapolation
with and without NTK scaling (S12-S14), seed 2357 to 1x (below), and the 2x
continuation (S19, S24).


**Second seed (seed 2357, same schedule, stopped at step 64,000).**

| 1x, common 2.1M-token slice | seed 1337 | seed 2357 | mean |
|---|---:|---:|---:|
| softmax RoPE control | 3.0883 | 3.0787 | 3.0835 |
| alt6 | 3.0385 | 3.0399 | **3.0392** |
| gap | 0.050 | 0.039 | **0.044** |

NIAH at 12288 / 16000: control 0.93 / 0.80 and 0.78 / 0.66; alt6 1.00 /
1.00 in both seeds (1.00 at every haystack and depth). Seed spread on the
common slice is 0.010 for the control and 0.001 for alt6, so the 0.044
mean gap is about four times the larger spread. The retrieval margin is
wider than S11 stated: the control's 16000-token retrieval is 0.66-0.80
across seeds while alt6's is 1.00 in both. Runs `runs/scale_16384/*_seed2357`,
NIAH `results/niah/scale1x__*_seed2357.json`, eval `results/eval/`.


**Wall-clock comparison, both seeds** (in-run curves shifted by each run's
common-slice offset, so both arms are on the common-slice scale; wall clock
from each run's final start event):

| equal wall clock | control step / val (s1337) | alt6 step / val (s1337) | alt6 - control | s2357 gap |
|---|---|---|---:|---:|
| 4 h | 15,000 / 3.352 | 9,000 / 3.432 | +0.080 | +0.083 |
| 8 h | 31,000 / 3.229 | 19,000 / 3.260 | +0.031 | +0.052 |
| 12 h | 46,000 / 3.162 | 28,000 / 3.194 | +0.032 | +0.046 |
| 16.4 h (control's 1x finish) | 64,000 / 3.088 | 39,000 / 3.135 | +0.043 | +0.054 |
| 21.3 h / 22.5 h | control stopped | 51,000 / 54,000: alt6 reaches the control's 1x loss | 0 | 0 |
| 26.7 h (alt6's 1x finish) | | 64,000 / 3.039 | -0.050 | -0.039 |

Throughput 71.8-71.9K tok/s (control, batch 4) against 44.0K (alt6,
batch 2 x 2), a 1.63x ratio. At equal wall clock the control leads by
0.03-0.05 nats throughout its run, the hybrid catches its 1x loss about
five to six hours after the control finished, and ends its own 1x leg
0.04-0.05 ahead. Equal-tokens edge 0.044 (mean of seeds); equal-time
deficit at the control's finish 0.043-0.054; the two are the same size
with opposite sign, which is what a 1.63x throughput ratio on a curve
losing ~0.03 per doubling of tokens predicts. Retrieval does not enter
this trade: the control's 16000-token NIAH (0.66-0.80) does not reach
alt6's (1.00) at any wall clock.

Note the alt6 arm ran at batch 2 x accumulation 2 after the batch-4
paging incident (item 4); a non-paging batch-4 alt6 would run about the
same 43-45K tok/s, so the throughput ratio is a property of the kernels
(S26), not of the batch choice.

## S12 — Extrapolation until it breaks, with an NTK ablation: pure dilution with alternating RoPE holds 1.00 to 3x its 8K training context plain and, with the first layer's RoPE base NTK-scaled, 0.83-0.87 at 4x; from 16K the best seed keeps 0.90 at 4x (64K); softmax RoPE is at chance past its context with or without NTK; the alt6 hybrid needs NTK and then tracks the typical pure-dilution seed (measured, eval-only, 1-3 seeds per cell)

*Correction: `rope_alt` rotates six of twelve layers and `--ntk` rescales all of them, so "the first layer's RoPE base" in this heading overstates what these runs isolate.*

Every RoPE-bearing checkpoint scored plain and with dynamic
NTK-aware scaling (`niah_eval --ntk`: for a haystack of L tokens past the
trained context T, every RoPE base becomes theta * (L/T)^(D/(D-2)); the
NoPE stacks have nothing to scale). Context cap raised to 66,000. The block
scan kernel needed a smaller tile past 32K (`_dilution_scan` KEYS now
scales with the block count; exact vs the eager path at T=40,000).

**8K-trained models** (ratio = haystack / 8K):

| model | mode | 12K (1.5x) | 16K (2x) | 24K (3x) | 32K (4x) | 48K (6x) | 64K (8x) |
|---|---|---:|---:|---:|---:|---:|---:|
| all-dilution rope_alt, s1337 | plain | 1.00 | 1.00 | 0.93 | 0.43 | 0.21 | 0.14 |
| all-dilution rope_alt, s1337 | NTK | 1.00 | 1.00 | 1.00 | **0.83** | 0.50 | 0.46 |
| all-dilution rope_alt, s2357 | plain | 1.00 | 0.99 | 0.24 | 0.14 | 0.14 | 0.14 |
| all-dilution rope_alt, s2357 | NTK | 1.00 | 1.00 | 0.99 | **0.87** | 0.44 | 0.34 |
| all-dilution NoPE | (none) | 0.81 | 0.77 | 0.71 | 0.69 | 0.27 | 0.23 |
| alt6 | plain | 0.57 | 0.33 | 0.14 | 0.14 | 0.14 | 0.14 |
| alt6 | NTK | 0.99 | 0.87 | 0.66 | 0.54 | 0.31 | 0.19 |
| all-softmax rope, s1337 | plain | 0.20 | 0.14 | 0.14 | 0.14 | 0.14 | 0.14 |
| all-softmax rope, s1337 | NTK | 0.33 | 0.34 | 0.31 | 0.31 | 0.29 | 0.26 |
| all-softmax rope, s2357 | plain / NTK | 0.14 / 0.23 | 0.16 / 0.16 | 0.14 | 0.14 | 0.14 | 0.14 |

**16K-trained models** (ratio = haystack / 16K; chance is 0.14):

| model | mode | 16K (1x) | 24K (1.5x) | 32K (2x) | 48K (3x) | 64K (4x) |
|---|---|---:|---:|---:|---:|---:|
| all-dilution rope_alt, s1337 | plain | 0.93 | 0.91 | 0.90 | 0.27 | 0.26 |
| all-dilution rope_alt, s1337 | NTK | 0.93 | 0.93 | 0.91 | **0.90** | **0.84** |
| all-dilution rope_alt, s2357 | plain / NTK | 0.81 | 0.66 / 0.84 | 0.47 / 0.69 | 0.16 / 0.49 | 0.14 / 0.47 |
| all-dilution rope_alt, s7331 | plain / NTK | 0.99 | 0.67 / 0.86 | 0.17 / 0.71 | 0.14 / 0.37 | 0.14 / 0.27 |
| all-dilution NoPE | (none) | 0.50 | 0.40 | 0.26 | 0.17 | 0.17 |
| alt6, base 5e5 | plain / NTK | 0.94 | 0.70 / 0.89 | 0.40 / 0.73 | 0.17 / 0.50 | 0.16 / 0.47 |
| all-softmax rope, base 5e5 | plain / NTK | 0.27 | 0.14 / 0.19 | 0.14 | 0.14 | 0.14 |

1. **The cliff is the rotation, not the share mechanism.** Plain, both
   pure-dilution stacks break at an absolute length of about 32-48K
   whatever the training context (3-4x from 8K, 2-3x from 16K): the
   rotated layers run out of wavelengths they were trained on. (`rope_alt`
   rotates every other attention layer starting at layer 0 -- six of
   twelve -- and `--ntk` rescales all of them, so these runs do not
   isolate a single layer as the bottleneck.) NTK moves the break out to
   the end of the eval:
   0.83-0.87 at 4x from 8K in both seeds, 0.90 at 4x (64K) from 16K in
   the best seed. The NoPE stack, with nothing to break, degrades
   slowly instead (0.69 at 4x from 8K) from a lower base. The dilution
   layers themselves impose no length limit that this eval can find.
2. **Softmax RoPE does not extrapolate, and NTK does not help it at this
   size.** At chance past the trained context in every seed and both
   bases; the NTK arms reach 0.3 at best. NTK preserves retrieval a model
   has; the 12L softmax models had little to preserve (0.27-0.40
   in-context at the top of their range).
3. **alt6 extrapolates only through its dilution half.** Plain it is
   capped by its six softmax layers (0.33 at 2x from 8K). With NTK on
   those layers it tracks the typical pure-dilution seed from 16K (0.73
   at 2x, 0.47 at 4x, against 0.69-0.91 and 0.27-0.84), still short of
   the best one. The softmax layers stop destroying retrieval once
   scaled; they do not supply it.
4. **Seeds.** From 16K the three pure-dilution seeds with NTK give 0.91 /
   0.69 / 0.71 at 2x and 0.90 / 0.49 / 0.37 at 3x: reliable to 2x, seed-
   dependent beyond. From 8K both seeds agree to 4x. In-context
   retrieval strength predicts extrapolation strength (s1337 > s7331 >
   s2357 on both).
5. **Scope.** NIAH here is likelihood-scored keyword retrieval, not
   long-context reasoning or generation, and chance is 1/7 = 0.143: one
   of the eight needles is disqualified by the control prior, leaving 7
   needles x 10-20 placements per length.

Results: `results/niah/extrap64k__*` (plain) and `extrap64k_ntk__*`.
What is not covered: the 209M checkpoints past 16K (eval-only, about an
hour), and training with NTK-scaled or larger-base RoPE on the first
layer only, which the mechanism in (1) suggests would let a pure
dilution stack train at 8K and serve at 32K+ without any scaling.

## S13 — 209M NoPE: at scale a pure dilution stack with no position signal at all retrieves 1.00 across its 16K context AND 0.96-0.99 out to 64K (4x) with no scaling, no fine-tuning and no lucky seed; the 12L NoPE weakness (0.50 at 16000) was undertraining, not the operator (measured, 1 seed)

Same recipe as the S11 pair: 24L d768 12H (208.5M), ctx 16384,
`positional: none`, all 24 layers dilution, 65,536 tokens/step (batch 2 x
accum 2), LR 6e-4, 40B-token corpus, seed 1337, stopped at the step-64,000
checkpoint (4.19B tokens, 1x Chinchilla). 32.0K tok/s, 55.8 GiB peak.

**1x, common 2.1M-token slice** (all four 209M arms, same data):

| arm | val loss | NIAH 7936 / 12288 / 16000 | tok/s |
|---|---:|---|---:|
| alt6 (mean of 2 seeds) | **3.0392** | 1.00 / 1.00 / 1.00 | 44.0K |
| all-dilution NoPE | 3.0569 | 1.00 / 1.00 / 1.00 | 32.0K |
| softmax RoPE 5e5 (mean of 2 seeds) | 3.0835 | 0.96-0.99 / 0.78-0.93 / 0.66-0.80 | 71.9K |

**Extrapolation from 16K** (10 placements, context cap 66,000; chance 0.14):

| arm | mode | 16K (1x) | 24K | 32K (2x) | 48K (3x) | 64K (4x) |
|---|---|---:|---:|---:|---:|---:|
| all-dilution NoPE | none to scale | 1.00 | 0.99 | 0.96 | 0.97 | **0.99** |
| alt6, s1337 | plain | 1.00 | 0.96 | 0.33 | 0.14 | 0.16 |
| alt6, s1337 | NTK | 1.00 | 1.00 | 1.00 | 0.89 | 0.36 |
| alt6, s2357 | plain | 1.00 | 1.00 | 0.91 | 0.46 | 0.23 |
| alt6, s2357 | NTK | 1.00 | 1.00 | 1.00 | 1.00 | **0.97** |
| softmax rope, s1337 | plain / NTK | 0.80 | 0.17 / 0.17 | 0.14 | 0.14 | 0.14 |
| softmax rope, s2357 | plain / NTK | 0.66 | 0.16 / 0.63 | 0.14 / 0.20 | 0.14 | 0.14 |

1. **NoPE's 12L weakness was undertraining.** At 12L/600M tokens the 16K
   NoPE cell retrieved 0.50 at 16000 and was the worst dilution variant
   (S6); at 24L/4.19B tokens it is 1.00 at every haystack and depth.
   Its extrapolation ladder is flat -- 1.00 / 0.99 / 0.96 / 0.97 / 0.99
   from 16K to 64K -- where the 12L version fell to 0.17. Scale and
   tokens turn "gradual decay from a weak base" into "no decay at all".
2. **With no rotation there is nothing to break and nothing to tune.**
   The share step is scale-free in sequence length: a longer context is
   just more keys competing under the same rule, and no wavelength can
   run out. The NoPE arm needs no NTK arm, no base choice, and no
   position-scaling rule at serving time. This is the cleanest
   demonstration in the project that the length behaviour belongs to
   the operator and not to a positional trick.
3. **RoPE is the binding constraint at this scale, not capacity.** The
   same 209M budget with six rotated softmax layers (alt6) caps at
   1.5-2x plain and needs NTK to reach 3x; its 4x result swings 0.36 to
   0.97 by seed. The pure NoPE stack beats both alt6 seeds' plain
   ladders everywhere past 24K and matches the better seed's NTK ladder
   without any scaling.
4. **Softmax does not extrapolate at any scale tested.** Both control
   seeds are at chance by 24K plain; NTK adds nothing for s1337 and one
   point (0.63 at 24K) for s2357. 3.3x the parameters and 7x the tokens
   bought the control in-context retrieval (0.66-0.80 at 16000, up from
   0.27 at 12L) and no length generalisation whatsoever.
5. **The cost is loss and speed.** NoPE is 0.018 behind alt6 on the
   common slice (and 0.027 ahead of the control), and runs at 32.0K
   tok/s against alt6's 44.0K and the control's 71.9K. So at 209M the
   ordering is alt6 for loss, NoPE for length generalisation, softmax
   for throughput; all three dilution arms retrieve perfectly in
   context where the control does not.

Runs: `runs/scale_16384/all_dilution_nope_seed1337`; NIAH
`results/niah/scale1x__all_dilution_nope_seed1337.json` and
`results/niah/extrap64k*__scale_16384_*.json`; eval `results/eval/`.
One seed for the NoPE arm. The untested follow-up is whether a NoPE
dilution stack trained at 8K also reaches 4x at this scale, which would
separate "trained long" from "trained large".

## S14 — NIAH to 128K: the 209M NoPE dilution stack is flat at 0.96-0.99 from 16K to 128K (8x its training context) with no position signal and no scaling; every RoPE-bearing arm decays past 4x even with NTK; a 64-bit scratch-indexing fix was needed to run the kernels past ~110K tokens (measured, 1-2 seeds per cell)

Every checkpoint still above chance at 64K (S12/S13) taken to
96K and 128K. RoPE-bearing arms use `--ntk` (their plain ladders were at
chance by 64K); the NoPE arm has nothing to scale. 10 placements, 8
needles, context cap 132,000; chance is 0.14.

| model (training context) | mode | 64K | 96K | 128K |
|---|---|---:|---:|---:|
| **209M all-dilution NoPE (16K)** | none to scale | 0.99 | **0.99** | **0.96** |
| 209M alt6, s2357 (16K) | NTK | 0.97 | 0.64 | 0.43 |
| 209M alt6, s1337 (16K) | NTK | 0.36 | 0.14 | 0.14 |
| 64M all-dilution rope_alt, s1337 (16K) | NTK | 0.84 | 0.61 | 0.50 |
| 64M all-dilution rope_alt, s2357 (16K) | NTK | 0.47 | 0.34 | 0.29 |
| 64M alt6 base 5e5 (16K) | NTK | 0.47 | 0.19 | 0.17 |
| 64M all-dilution rope_alt, s1337 (8K) | NTK | 0.46 | 0.36 | 0.21 |
| 64M all-dilution rope_alt, s2357 (8K) | NTK | 0.34 | 0.23 | 0.17 |
| softmax RoPE controls (8K and 16K, 64M and 209M) | plain / NTK | chance from 24K (S12/S13) | | |

1. **No position signal, no decay.** The 209M NoPE stack reads
   1.00 / 0.96 / 0.99 / 0.99 / 0.96 at 16K / 32K / 64K / 96K / 128K.
   Across an eightfold range there is no decay curve to fit, which is
   what a length-free operator predicts: the share step divides a bid by
   the demand already on that key, and neither quantity acquires a
   distance scale. Nothing was tuned for the longer contexts -- the same
   checkpoint, no scaling rule, no fine-tuning.
2. **Rotation sets the ceiling, scaling only moves it.** Every arm that
   contains RoPE decays past about 4x even with NTK, and the decay is
   faster the more rotated layers there are: the pure dilution stacks
   (one rotated layer) fall to 0.21-0.50 at 128K, the alt6 hybrids (six)
   to 0.14-0.43. Scale does not rescue it -- the 209M hybrid's weaker
   seed is at chance by 96K while a 64M pure stack holds 0.50 at 128K.
3. **Seed spread grows with the ratio.** At 4x the two 209M hybrid seeds
   differ by 0.61 (0.36 vs 0.97) and the two 64M 16K pure seeds by 0.37.
   The NoPE arm is one seed; its flatness should be replicated before
   the claim is leaned on.
4. **Kernel fix (S14).** The `(n, nB, T)` carry/S scratch overflows
   int32 pointer arithmetic once `n * nB * T > 2^31`: at 12 heads and
   T=131072 the per-head stride is 2.7e8 and the last head's base is
   2.95e9, which raised an illegal memory access. Only the per-program
   base multiplication is promoted to int64, hoisted out of the
   row-block loops so tile offsets stay 32-bit. Full suite passes (154
   tests); in-step cost at B8 H8 T4096 is 5.78 ms against 5.76-5.80 for
   the unpatched kernels, i.e. free. A 128K forward at B1 H12 D64 runs
   in 1.5 s at 25.0 GiB.

**Extension to 256K.** Every checkpoint still above chance at
128K, carried to 192K and 256K:

| model (trained context) | mode | 128K | 192K | 256K |
|---|---|---:|---:|---:|
| 209M all-dilution NoPE (16K) | none to scale | 0.96 | **0.80** | **0.63** |
| 64M all-dilution rope_alt, s1337 (16K) | NTK | 0.50 | 0.34 | 0.26 |
| 64M all-dilution rope_alt, s2357 (16K) | NTK | 0.29 | 0.26 | 0.21 |
| 64M all-dilution rope_alt, s1337 (8K) | NTK | 0.21 | 0.23 | 0.20 |
| 209M alt6, s2357 (16K) | NTK | 0.43 | 0.14 | 0.14 |

The 209M NoPE arm's flat region ends: 0.96-0.99 holds from 1x to 8x
(16K to 128K) and then falls, to 0.80 at 12x and 0.63 at 16x. So the
behaviour is not unbounded, it is flat to roughly 8x and degrades
gracefully after -- still four times chance at 256K, where the best
RoPE-bearing arm (alt6 with NTK) is at chance and the pure dilution
stacks with one rotated layer are at 0.20-0.26. 256,000 tokens is the
longest haystack run; the kernels handle it in 17.9 GiB and 1.8 s after
the row-chunked scratch of S15.

**Second seed.** Seed 2357 of the 209M NoPE arm, same protocol,
stopped at the same 1x Chinchilla checkpoint:

| | seed 1337 | seed 2357 |
|---|---:|---:|
| val loss, common 2.1M slice | 3.0569 | 3.0548 |
| NIAH, 256 to 16000 (in context) | 1.00 everywhere | 1.00 everywhere |
| 24K / 32K / 48K / 64K (2-4x) | 0.99 / 0.96 / 0.97 / 0.99 | 1.00 / 1.00 / 1.00 / 1.00 |
| 96K / 128K (6-8x) | 0.99 / 0.96 | 1.00 / 0.96 |
| 192K / 256K (12-16x) | 0.80 / 0.63 | **0.84 / 0.79** |

Loss reproduces to 0.002 and the flat region to the digit: both seeds are
at or above 0.96 through 8x, and the second seed decays more slowly past
it (0.79 vs 0.63 at 16x). The headline no longer rests on one run. The
remaining control is the operator itself -- every softmax arm in the
figures carries RoPE, and softmax without positional encoding is known to
length-generalise (Kazemnejad et al. 2023, docs/related-work.md), so
"dilution extrapolates" is only an operator claim once softmax+NoPE is
measured at the same protocol (S16).

Results: `results/niah/extrap128k*__*.json`. 128,000 tokens is the
longest haystack the eval harness has been run at; the NoPE arm shows no
sign of a limit there, so the ceiling of this behaviour is unmeasured.

## S15 — Retrieval and loss dissociate, and the learning rate decides which one you get: at 64M the same NoPE dilution model reaches 0.97 retrieval everywhere at LR 3e-4 and 0.14 at LR 6e-4, while 6e-4 is 0.08 nats BETTER on loss. The S6 weakness was undertraining after all, once the learning rate is held (measured, 1 seed per cell)

*Later context (S23): at 3e-4, retrieval formed in two of three seeds of this recipe at the same loss, so the single-seed contrast here is one draw from a seed-dependent outcome; the 6e-4 arm was not repeated.*

The 64M all-dilution NoPE model at ctx 16384 retrieved
only 0.50 at 16000 (S6) where the 209M version retrieves 1.00 and
extrapolates to 8x (S13/S14). Those cells differ in parameters AND token
budget, so this isolates the causes. All rows are the same model and
corpus family, scored on one common 1.05M-token validation slice.

| run | tokens | LR | schedule | val loss | NIAH 1024 / 3840 / 7936 / 16000 |
|---|---:|---:|---|---:|---|
| S6 baseline | 600M | 3e-4 | full cosine | 3.6757 | 0.86 / 0.83 / 0.76 / 0.50 |
| Mid-cosine | 1.27B | 6e-4 | 2x cosine, stopped at 50% | 3.6138 | 0.42 / 0.16 / 0.16 / 0.24 |
| Control A | 2.54B | 6e-4 | 2x cosine, fully annealed | **3.4911** | 0.52 / 0.20 / 0.16 / 0.14 |
| Control B | 1.27B | 3e-4 | full cosine | 3.5717 | **0.97 / 0.97 / 0.97 / 0.97** |

1. **The learning rate, not the token budget, destroyed retrieval.** At
   6e-4 the model is near chance past 2K at both 1.27B and 2.54B tokens;
   at 3e-4 with the same 1.27B tokens it is 0.97 at every haystack from
   256 to 16000. Annealing is not the mechanism either -- Control A is
   fully annealed and still at chance (it improved only the shortest
   haystacks, 0.60-0.96 below 1K).
2. **The original question is answered: undertraining, once LR is held.**
   At a matched 3e-4 full cosine, going from 600M to 1.27B tokens takes
   16000-token retrieval from 0.50 to 0.97 and removes the length
   gradient entirely (0.97 flat from 1K to 16K, against 0.86 falling to
   0.50). The S6 cell was undertrained; it was not a property of NoPE
   or of the 64M size.
3. **Loss and retrieval move in opposite directions.** The best model on
   loss is the worst on retrieval by a wide margin: Control A beats
   Control B by 0.081 nats and scores 0.14 against 0.97 at 16000. Loss
   alone would have selected exactly the wrong configuration. Any
   comparison in this project that reports loss without NIAH is
   under-determined. **Attribution caveat:** that 0.081-nat pair differs
   in token budget (2.54B vs 1.27B) and schedule shape as well as
   learning rate. Mid-cosine vs Control B is matched on tokens and differs
   in rate and schedule position (0.42 vs 0.97 at 1024). The evidence
   strongly implicates the recipe and points at the rate; a matched-budget,
   matched-schedule rate sweep is what would make it causal.
4. **Where the hazard lives.** 6e-4 was chosen to match the 209M
   protocol (S11), and at 209M it produces 1.00 retrieval. The same rate
   at 64M destroys it. So the safe learning rate for retrieval scales
   with model size, and 12L-at-16K results should use 3e-4 (as S5-S9
   did) rather than the 209M's 6e-4.
5. **Length invariance is separate from retrieval strength.** The mid-cosine
   run's extrapolation ladder is flat at 0.23-0.24 from 24K to 128K, exactly
   its own in-context value: with no rotation there is no distance scale
   to decay against, whatever the absolute level. The S13/S14 claim is
   therefore two claims -- dilution+NoPE is length-invariant (robust),
   and a well-trained one retrieves at 0.96-0.99 (training-dependent).

Runs: `runs/nope_16384/all_dilution_nope_chinchilla_seed1337` (mid-cosine run,
annealed to 2x by Control A) and `..._chinchilla_lr3e4_seed1337`
(Control B). Open: whether Control B also extrapolates like the 209M
model (its ladder past 16K has not been run), and a second seed.

## S16 — The operator control for NoPE: with no position encoding on either side and each operator at its own best stable learning rate, dilution retrieves 0.97 flat across 16K and softmax reaches 0.15; softmax+NoPE also needs half the learning rate and still degrades late. Extrapolation range grows with scale, so the flat-to-8x behaviour is a 209M property, not a 64M one (measured, 1 seed per cell)

*Item 3's reading, that extrapolation range grows with scale, is superseded by S22-S24: range is seed-dependent at both 64M and 209M.*

Every softmax arm in S5-S14 carries RoPE, so the
S13/S14 claim ("a dilution model with no position encoding retrieves 8x
past its context") is an operator claim only if softmax with no position
encoding does not do the same. The literature makes that a live
possibility: Kazemnejad et al. (2023) find decoder-only transformers
without positional encoding length-generalise (docs/related-work.md).
This measures it. 64M, ctx 16384, 1.27B tokens (1.0x Chinchilla), full
cosine, 40B corpus, seed 1337 -- identical except the operator and the
learning rate, which softmax cannot share (see 2).

| 64M, positional: none | LR | val loss (common 1.05M slice) | NIAH 1024 / 3840 / 7936 / 16000 | 24K / 64K / 128K |
|---|---:|---:|---|---|
| all-dilution | 3e-4 | **3.5717** | **0.97 / 0.97 / 0.97 / 0.97** | 0.91 / 0.40 / 0.23 |
| all-softmax | 1.5e-4 | 4.0846 | 0.51 / 0.55 / 0.27 / 0.15 | 0.17 / 0.19 / 0.10 |
| all-softmax | 3e-4 | 4.6760 (diverged) | 0.14 everywhere | 0.14 everywhere |

1. **The result is the operator, not the absence of rotation.** At equal
   tokens, equal data and no position encoding on either side, dilution
   is 0.51 nats better on the common slice (3.5717 vs 4.0846) and
   retrieves 0.97 flat where softmax reaches 0.15
   at the same context length. Removing RoPE does not by itself buy
   retrieval in this harness.
2. **Softmax without position encoding is the fragile side.** At 3e-4 it
   diverges (grad median 19, max 61,071; loss rising from 4.35 at step
   20,000 to 4.96 at 50,000; NIAH at chance even in context), reproducing
   the S2 instability at 8K. 10,000-step screens put 1.5e-4 and 7.5e-5
   in the stable band (grad median 1.0, p99 4.6-5.4) with 1.5e-4 clearly
   better (4.94 vs 5.47 at step 10,000). Even the 1.5e-4 full run decays:
   grad median 1.26 over steps 0-20k, 1.49 over 20-40k, 7.13 over 40-60k
   and 15.74 over 60-77.5k with spikes to 3,370, while its loss still
   improves monotonically to 3.9957. Dilution took 3e-4 with grad median
   0.8. So the comparison is each operator at its own best stable rate;
   a matched-rate comparison is not available because softmax has no
   stable run at 3e-4.
3. **Correction to S13/S14: extrapolation range grows with scale.** The
   well-trained 64M dilution NoPE model (0.97 in context) holds only to
   ~2x -- 0.91 at 24K, 0.80 at 32K, then 0.43 / 0.40 / 0.26 / 0.23 at
   48K / 64K / 96K / 128K. The 209M model is flat at 0.96-1.00 to 8x in
   both seeds. So "dilution + NoPE is length-invariant" is not a size-free
   property: 64M reaches about 2x, 209M about 8x. The S15 note that a
   poorly-trained model is "equally flat" was flat at chance (0.23), which
   is not the same phenomenon and should not be read as invariance.

Runs: `runs/nope_16384/all_softmax_nope_chinchilla_lr{3e4,1.5e-4}_seed1337`
and `..._all_dilution_nope_chinchilla_lr3e4_seed1337`; screens in
`runs/smoke/softmax_nope_screen_*`. Open: the same control at 209M, where
the headline lives and where softmax may be stable at a higher rate.

## S17 — The operator control at 209M, learning-rate matched: with no position encoding on either side, dilution retrieves 1.00 across 16K and extrapolates to 8x while softmax falls to 0.19 in context and is at chance everywhere past it; dilution without any position encoding also beats softmax WITH RoPE on loss (measured, 1-2 seeds)

*Item 3 is superseded in part by S24: at 209M and 6e-4, softmax without position encoding was unstable in one of two seeds.*

The S13/S14 headline is an operator claim only if softmax with
no position encoding does not behave the same way; Kazemnejad et al.
(2023) make that a live possibility (docs/related-work.md). S16 answered
it at 64M, but there softmax could not train at the dilution arm's
learning rate, so the comparison was each operator at its own best rate.
At 209M that objection disappears: 6,000-step screens put softmax+NoPE
at **grad median 0.39, p99 2.0** at 6e-4 -- the dilution arm's own rate --
and 6e-4 beat 3e-4 on screen loss (3.8342 vs 3.9082), so the control ran
fully matched. 24L d768, ctx 16384, 4.19B tokens (1x Chinchilla), 40B
corpus, seed 1337, LR 6e-4, identical in every respect except the
attention operator. The full run was clean: grad median 0.36, p99 0.81.

| 209M, common 2.1M-token slice | val loss | NIAH 1024 / 1920 / 3840 / 7936 / 16000 | 24K-256K |
|---|---:|---|---|
| alt6 (dilution + RoPE softmax) | **3.0392** | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | to 3x with NTK |
| all-dilution, **no position encoding** | 3.0559 | 1.00 / 1.00 / 1.00 / 1.00 / **1.00** | **0.96-1.00 to 8x**, 0.63-0.79 at 16x |
| softmax + RoPE | 3.0835 | 1.00 / 1.00 / 1.00 / 0.96-0.99 / 0.66-0.80 | chance past 1.5x |
| softmax, **no position encoding** | 3.2118 | 0.99 / 0.78 / 0.46 / 0.29 / **0.19** | chance everywhere |

1. **The extrapolation result is the operator, not the absence of
   rotation.** Matched on architecture size, corpus, schedule, seed and
   learning rate, softmax with no position encoding retrieves perfectly
   only to ~1K tokens and reaches 0.19 at its own trained context, then
   chance at every extrapolated length from 24K to 256K. Dilution under
   identical conditions is 1.00 across 16K and 0.96-1.00 out to 128K in
   both seeds. Removing RoPE does not buy softmax retrieval at either
   scale tested.
2. **Dilution needs no position encoding at all to beat a RoPE softmax.**
   The NoPE dilution arm is 0.027 nats ahead of softmax+RoPE on loss and
   retrieves 1.00 against 0.66-0.80. Softmax pays 0.128 nats for losing
   RoPE (3.0835 -> 3.2118). The alt6 -> dilution-NoPE difference is only
   0.017 (3.0392 -> 3.0559), but that pair changes half the attention
   layers as well as the position encoding, so it bounds rather than
   measures what dilution pays for dropping position. The clean statement
   is the one above: position encoding is doing work for softmax that the
   share step supplies structurally.
3. **Softmax+NoPE's fragility is a small-model property.** It diverges at
   3e-4 at 64M (S16: grad median 19, max 61,071) and trains cleanly at
   twice that rate at 209M (grad median 0.36, p99 0.81). The instability
   of S2 should therefore be read as a scale artefact, while the
   retrieval failure reported here is not: the 209M run is healthy by
   every stability measure and still cannot retrieve.

Runs: `runs/scale_16384/all_softmax_nope_lr6e-4_seed1337`; screens in
`runs/smoke/softmax_nope_209m_*`. One seed for this control; the dilution
arm it is compared against has two. Open: whether the ordering holds
beyond 1x Chinchilla and beyond 209M.

## S18 — The mechanism controls the literature asked for: temporal attention's exclusive denominator is degenerate at init in causal self-attention (the diagonal wins every row by 1/eps), training escapes it everywhere except layer 0, and at a matched learning rate it costs 0.072 nats and 0.38 retrieval at 7.9K (corrected from 0.023 / 0.11); attention-structure measurements show the share step makes queries ~20x more selective AND key usage 4x more balanced at the same time (measured, 1 seed per cell)

*Correction (release review, 2026-09-30). The first version of this signal compared the
exclusive cell (LR 6e-4) with inclusive dilution and the softmax control trained at 3e-4, and
reported that exclusive history costs 0.023 nats and ~0.11 retrieval and that the share step is
worth 0.128 nats. Runs with the same recipe exist for every cell (8K, 64M, 600M tokens, learned
positions, LR 6e-4, seed 1337; the resolved configs differ only in `bid`, `share_cumsum` and an
unused `rope_theta`). Matched, exclusive history costs 0.072 nats and 0.38 retrieval at 7,936
tokens, and the share step is worth 0.177 nats (the same comparison S3 made with best-so-far
validation, 0.178). The no-cumsum row's 3840 value was also mis-transcribed (0.97 is its 1024
value; 3840 is 0.90). All four cells were re-scored under niah-v4 with 20 placements each
(results/niah_v4/); the table below uses those. The structure table further down uses the 3e-4
dilution and softmax checkpoints; the matched 6e-4 dilution checkpoint is close to it (layer 0
diagonal 0.200, entropy 4.08; token-0 mass 0.002 / 0.007 / 0.009 at layers 0 / 6 / 11, against
the no-cumsum control's 0.006 / 0.052 / 0.024; results/structure/lr_8192_all_dilution_lr6e-4_seed1337.json).*

*The "~20x more selective, ~4x more balanced" figures are layer-0 measurements against a near-uniform control layer; S20 corrects them.*

docs/related-work.md asks for the factorial that separates dilution from
its closest equation-level precedent, temporal attention (Sankaran et al.
2016; Paulus et al. 2018): raw vs row-normalised bids, crossed with
exclusive vs inclusive history. The row-normalised half is measured here;
the raw-score half is not.

| current bid | history | status |
|---|---|---|
| raw exp(z) | exclusive | degenerate at initialisation by the same argument as below; whether training escapes it is untested |
| raw exp(z) | inclusive | **untested.** The `expshift` bid is exp(a)/SHIFT, so share = p/(committed+p) has the SHIFT cancel and is algebraically raw inclusive temporal attention, but our fp32 implementation overflowed at step 400 (S4). That is an implementation failure; the stable per-key cumulative-logsumexp form has not been built |
| row-normalised p | exclusive | this signal |
| row-normalised p | inclusive | dilution |

**Exclusive history is degenerate at initialisation.** In causal
self-attention key j first becomes visible to query j, and no earlier
query can have claimed it, so `committed[j, j]` is *exactly* zero.
`share = p / (committed + eps)` therefore gives the diagonal a 1/eps
advantage over every other entry in its row, and the attention matrix
collapses to the identity: measured diagonal 1.0000, off-diagonal ~3e-10
(tests/test_kernels.py::test_exclusive_history_collapses_to_the_identity).
The first row needs no special case -- committed is zero across it, so the
row normalise cancels eps and reproduces temporal attention's published
first-step definition exactly. Dilution's inclusive denominator is what
removes the singularity: on the diagonal it gives p/(0+p+eps) = 1.

**Training escapes it, except in layer 0.** Run to the full 36,622-step
8K schedule the exclusive cell is stable (grad median 0.77, p99 1.78) and
lands between dilution and the controls:

| 8K, 64M, 600M tokens, learned positions, LR 6e-4 | final val | NIAH 3840 / 6144 / 7936 |
|---|---:|---|
| dilution (inclusive) | **3.6055** | 1.00 / 1.00 / 1.00 |
| exclusive history | 3.6771 | 0.83 / 0.72 / 0.62 |
| no-cumsum control (attn = p) | 3.7829 | 0.91 / 0.71 / 0.61 |
| softmax control (unstable at this LR; best 3.7654 at 1.3e-3) | 4.0124 | 0.61 / 0.37 / 0.27 |

NIAH: niah-v4, 20 placements per length for every cell (140 scored prompts per length).

So at a matched learning rate the inclusive-vs-exclusive choice is worth 0.072 nats and 0.38
of retrieval at 7,936 tokens, and the share step itself 0.177 nats against the no-cumsum
control. Exclusive history keeps about 60% of the share step's loss gain but none of its
long-haystack retrieval: at 6,144 and 7,936 tokens it retrieves no better than the no-cumsum
control.

**Where the cost comes from (scripts/attention_structure.py, seq 1024,
mean over heads):**

| model | layer | diag | first | top1 | entropy | col_gini |
|---|---|---:|---:|---:|---:|---:|
| exclusive history | 0 | **0.911** | 0.000 | 0.937 | **0.28** | 0.111 |
| exclusive history | 6 | 0.107 | 0.003 | 0.498 | 2.10 | 0.237 |
| exclusive history | 11 | 0.463 | 0.037 | 0.633 | 1.57 | 0.230 |
| dilution (inclusive) | 0 | 0.212 | 0.002 | 0.212 | 4.04 | 0.131 |
| dilution (inclusive) | 6 | 0.175 | 0.002 | 0.253 | 3.72 | 0.155 |
| dilution (inclusive) | 11 | 0.255 | 0.004 | 0.301 | 3.14 | 0.197 |
| no-cumsum (attn = p) | 0 | 0.007 | 0.006 | 0.011 | 5.90 | 0.500 |
| no-cumsum (attn = p) | 11 | 0.007 | 0.024 | 0.037 | 5.60 | 0.556 |
| softmax control | 0 | 0.007 | 0.007 | 0.014 | 5.86 | 0.507 |
| softmax control | 11 | 0.007 | 0.033 | 0.054 | 5.42 | 0.580 |

1. **The exclusive model loses layer 0 to the degeneracy.** Its first
   layer never escapes: diagonal 0.911, entropy 0.28 nats, top-1 mass
   0.937 -- an identity map that mixes nothing. Layers 6 and 11 escape
   (diagonal 0.107 / 0.463). Dilution's layer 0 is a working attention
   layer (diagonal 0.212, entropy 4.04). One sacrificed layer is a
   plausible account of part of the 0.072-nat matched deficit; whether it
   is all of it is untested.
2. **The share step is selective and balanced at once -- with a caveat at
   layer 0.** Against the no-cumsum control at layer 0, dilution raises
   top-1 mass from 0.011 to 0.212 while cutting the column-load Gini from
   0.500 to 0.131 (~4x more balanced across keys). Those normally trade
   off. But at layer 0 the top-1 weight *is* the diagonal (top1 0.21161,
   diag 0.21160), so that 20x is self-attention mass, and part of the
   Gini drop follows mechanically from mass moving onto each query's own
   key. At layers 6 and 11 the two separate (top1 0.253 / 0.301 against
   diag 0.175 / 0.255), so genuine off-diagonal sharpening exists but is
   smaller than the layer-0 figure suggests. Off-diagonal top-k and
   off-diagonal column load would be the discriminating measurement and
   have not been run.
3. **Dilution barely uses the attention sink.** Mass on token 0 is
   0.002-0.004 at every depth, against 0.007-0.033 for softmax and
   0.006-0.024 for the no-cumsum control, and softmax's sink mass grows
   with depth while dilution's does not. Consistent with the operator
   taxing exactly the keys that accumulate demand (cf. StreamingLLM /
   StableMask, docs/related-work.md).
4. **Softmax and the no-cumsum control are nearly identical in
   structure** (entropy 5.86 vs 5.90, Gini 0.507 vs 0.500 at layer 0), as
   they should be -- with the share step off, the operator *is* softmax
   attention. That the two agree is a check on the measurement.

Kernel fix found while building this: `kernels_bid`'s backward launched
with `SHARE=int(bool(ctx.share_cumsum))`, which silently ignored any mode
other than inclusive, so an exclusive or no-cumsum gradient was computed
against the wrong operator. Now `_share_code(...)` in both directions;
157 tests pass. Runs:
`runs/bid_8192/all_dilution_exclusive_lr6e-4_seed1337`; structure in
`results/structure/`.

## S19 — 2x Chinchilla: the loss gap barely compresses (0.156 -> 0.147) and in-context retrieval holds, but dilution's 8x extrapolation does NOT survive further training -- the fully annealed 2x model holds only to 2x (0.51 at 64K, 0.24 at 128K) where both 1x checkpoints were flat to 8x (measured, 1 seed at 2x)

*(Seed 2357 does not replicate the shortened extrapolation: 0.97 at 64K and 0.83 at 128K after 2x. See S24.)*

Both 209M NoPE arms, stopped mid-cosine at the 1x checkpoint in
S13/S17, were resumed to the end of the same 128,174-step schedule: 8.40B
tokens, 2x Chinchilla, fully annealed. Same corpus, seed, rate (6e-4),
architecture; only the operator differs. Both runs were clean (dilution grad
median 0.39 p99 0.83; softmax 0.45 / 0.83).

**Loss, common 2.1M-token slice:**

| | 1x (4.19B) | 2x (8.40B) | gained |
|---|---:|---:|---:|
| dilution, NoPE | 3.0559 (2-seed mean) | **2.8574** | 0.199 |
| softmax, NoPE | 3.2118 | 3.0039 | 0.208 |
| gap | 0.156 | **0.147** | |

The gap narrows by 0.009 for twice the data. In-run evaluations on a
shared 262K window tell the same story in finer steps: 0.191 at 1.3B
tokens, 0.155 at 4.2B, 0.148 at 6.5B, with the per-eval noise at 0.003.
The early narrowing (0.08 over the first 2.6B tokens) slowed by roughly
ten times; a gap near 0.14-0.15 at 2x is about fifteen times the larger
1x seed spread (0.010). Dilution passed softmax's final 2x loss (2.9719
in-run) at step 78,000, 61% of softmax's tokens. On wall clock the
reverse holds: softmax took ~31 h of training for 8.4B tokens, dilution
~73 h.

**In-context retrieval (NIAH, 20 placements, chance 1/7):**

| | 1920 | 3840 | 7936 | 16000 |
|---|---:|---:|---:|---:|
| dilution 2x | 1.00 | 1.00 | 0.99 | 0.97 |
| softmax 1x | 0.78 | 0.46 | 0.29 | 0.19 |
| softmax 2x | 0.97 | 0.78 | 0.45 | 0.29 |

Softmax learns in-context retrieval from data (0.19 -> 0.29 at 16K,
0.46 -> 0.78 at 3840); dilution was already saturated and stays there.

**Extrapolation -- the surprise:**

| dilution NoPE | 16K | 24K | 32K | 48K | 64K | 96K | 128K | 192K | 256K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x, seed 1337 | 1.00 | 0.99 | 0.96 | 0.97 | 0.99 | 0.99 | 0.96 | 0.80 | 0.63 |
| 1x, seed 2357 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.84 | 0.79 |
| **2x, seed 1337** | 0.97 | 0.94 | 0.94 | 0.73 | **0.51** | 0.29 | **0.24** | 0.23 | 0.23 |
| softmax 2x | 0.29 | 0.21 | 0.24 | 0.19 | 0.16 | 0.14 | 0.14 | 0.14 | 0.14 |

Further training bought loss and cost extrapolation. The same weights that
held 0.96-0.99 to 8x at the 1x checkpoint hold only to ~2x after the
second half of the schedule, then decay steadily. It is still far above
softmax at every extrapolated length (0.51 vs 0.16 at 64K), but the S13/S14
statement "flat to 8x" describes the 1x checkpoint, not the trained-out
model.

**What is confounded.** The 1x checkpoints were mid-cosine, learning rate
still 55.7% of peak; the 2x checkpoint is fully annealed. So this does not
separate "more tokens" from "annealing". S15 found annealing did not
*restore* retrieval at 64M, but whether it *removes* extrapolation is a
different question. The cheap disentangling test is at 64M: two runs to
the same 1.27B tokens at 3e-4, one ending its cosine there (S15 Control B,
already run: holds ~2x) and one stopped mid-cosine, compared on the same
extrapolation ladder.

**Structure at 2x** (scripts/attention_structure.py, with the new
off-diagonal columns):

| dilution 2x | diag | first | col Gini | off_top1 | off Gini |
|---|---:|---:|---:|---:|---:|
| layer 0 | 0.252 | 0.001 | 0.139 | 0.163 | 0.147 |
| layer 12 | 0.134 | **0.028** | **0.367** | 0.137 | 0.408 |
| layer 23 | 0.203 | 0.001 | 0.199 | 0.203 | 0.211 |

The middle layer has moved toward softmax's pattern: a small sink (0.028
on token 0, against 0.144 for softmax 2x at the same depth) and a more
concentrated key load (Gini 0.367 against 0.540). Whether that
concentration is what costs extrapolation is a hypothesis the matched 1x
structure numbers can check. *(Checked in S20: the middle-layer Gini did
not rise from 1x to 2x, so key concentration does not explain it.)*

Runs: `runs/scale_16384/all_{dilution,softmax}_nope*_seed1337` (both now at
step 128,174); NIAH `results/niah/{scale2x,extrap*__scale2x}__*`; eval
`results/eval/scale2x__*`.

## S20 — Attention structure across every 209M checkpoint: the missing sink is set by the operator, not the position encoding (it differs between layers of the same model), but S18's "4x more balanced" was largely dilution's local layer 0 against softmax's uniform one; and key concentration does not explain S19's lost extrapolation (measured, one 1,024-token window per model)

*Updated at release review (2026-09-30): see "Update: second seeds" at the end of this signal.
The softmax sink is larger than point 3 shows (0.75 in one seed without position encoding), the
hybrid comparison does not isolate the operator on its own, and point 5 now has a same-seed
comparison.*

This study ran `scripts/attention_structure.py` on all seven 209M checkpoints
and re-measured five 64M ones, this time with the off-diagonal columns
(self-attention removed, rows renormalised). Files: `results/structure/scale*.json`.
Three layers per model (first / middle / last), averaged over heads.

**1. Reference values that were missing.** Uniform causal attention
(every query spreading evenly over its prefix) gives a column-load Gini of
**0.4995** and mean row entropy of **5.94** on a 1,024-token window, purely
from the causal mask: early keys are visible to more queries. A 32-token
sliding window gives Gini **0.03**. So Gini 0.5 means "no preference", not
"concentrated", and a very low Gini can come from locality alone.

**2. Correction to S18 and to the README's mechanism summary.** Softmax's layer 0 matches
uniform averaging in every model measured (Gini 0.493-0.507, entropy
5.86-5.92, mean age 0.49-0.50 of the row), and so does the no-cumsum
control's (0.500 / 5.90). Dilution's layer 0 is recency-weighted instead
(age 0.11-0.13, diagonal 0.21-0.25). The "0.500 -> 0.131, ~4x more balanced"
comparison therefore contrasts a local pattern with a flat one. It is a real
difference between the trained layers, but not evidence that dilution spreads
*retrieval* evenly over the keys. Off the diagonal, dilution's layer 0 still
puts 0.13-0.16 of each row on its single best other key, against 0.010-0.017
for softmax, but that sharpness is mostly short-range.

**3. The sink.** Mass on token 0, first / middle / last layer:

| model | first | middle | last |
|---|---|---|---|
| dilution NoPE 209M, 1x (seed 2357) | 0.001 | 0.014 | 0.002 |
| dilution NoPE 209M, 2x | 0.001 | 0.028 | 0.001 |
| alt6 209M, seed 1337: softmax+RoPE / softmax+RoPE / **dilution** | 0.006 | 0.100 | **0.002** |
| alt6 209M, seed 2357 | 0.006 | 0.051 | **0.006** |
| softmax RoPE 209M, seed 1337 | 0.006 | 0.056 | 0.248 |
| softmax RoPE 209M, seed 2357 | 0.006 | 0.315 | 0.289 |
| softmax NoPE 209M, 2x | 0.006 | 0.144 | 0.076 |
| dilution 64M 8K (learned pos) | 0.002 | 0.002 | 0.004 |
| dilution NoPE 64M 16K | 0.001 | 0.011 | 0.000 |
| no-cumsum control 64M 8K | 0.006 | 0.052 | 0.024 |
| softmax 64M 8K | 0.007 | 0.030 | 0.033 |

The alt6 rows are the cleanest control yet: in the same network, trained on the
same data, the softmax layers park 5-10% of their mass on token 0 and the
dilution layer parks 0.2-0.6%. Dilution never exceeds 0.028 at any depth or
scale, while 209M softmax reaches 0.25-0.32. Removing the share step while
keeping everything else (no-cumsum) brings the sink back (0.052). The softmax
sink also varies a lot by seed (middle layer 0.056 vs 0.315), which matters for
point 5.

**4. Deeper layers, where the comparison is fair.** Off-diagonal, middle layer:
dilution 209M off_top1 0.14-0.15 with off_gini 0.41; softmax RoPE 209M
0.08-0.32 with 0.54-0.74; softmax NoPE 2x 0.18 with 0.54. Last layer: dilution
off_gini 0.21-0.27 against softmax's 0.63-0.64. At depth dilution is about as
selective per query as softmax, while its key load is clearly less
concentrated. Much of softmax's extra concentration comes from the sink: the
row with the largest sink (softmax RoPE seed 2357, 0.315) also has the highest
Gini (0.744).

**5. S19's hypothesis is not supported.** S19 asked whether key
concentration explains why the 2x model lost extrapolation. From the 1x
checkpoint (seed 2357, flat to 8x) to the 2x one (seed 1337, holds ~2x),
the middle layer's Gini did not rise (0.379 -> 0.367). The sink doubled
(0.014 -> 0.028), the diagonal rose (0.104 -> 0.134) and off-diagonal top-1
fell (0.154 -> 0.137). These are different seeds, and the seed-1337 1x
checkpoint no longer exists because it was resumed to 2x, so the changes cannot
be separated from seed variation. Across models Gini does not follow
extrapolation range either: 64M NoPE (holds ~2x) has middle Gini 0.383, while
209M 1x (holds 8x) has 0.379. Whatever shortened extrapolation, this probe does
not see it.

**6. Other rows.** The exclusive (temporal-attention) model's layer 0 has
diagonal 0.911, confirming S18's identity-map reading. Dilution with NoPE at
16K has a more concentrated middle layer than dilution with learned positions
at 8K (0.383 vs 0.155), but context length and position encoding both change
between those two runs.

Caveats: one validation window of 1,024 tokens per model, three layers,
averages over heads. These are descriptions of trained models, not causal
accounts of 16K-128K retrieval.

**Update: second seeds (release review, 2026-09-30).** Six more checkpoints were measured
after this signal was written, with the same probe (`results/structure/`): dilution NoPE seed
2357 at 2x, softmax NoPE seed 2357 at 1x and 2x, and the second seeds of the three 1K models
(S21, S24). Middle-layer values:

| model | sink | column Gini | off-diagonal Gini | top-1 | entropy |
|---|---:|---:|---:|---:|---:|
| dilution NoPE 209M, seed 2357, 1x | 0.014 | 0.379 | 0.408 | 0.19 | 3.90 |
| dilution NoPE 209M, seed 2357, 2x | 0.026 | 0.380 | 0.410 | 0.19 | 4.06 |
| dilution NoPE 209M, seed 1337, 2x | 0.028 | 0.367 | 0.408 | 0.20 | 4.02 |
| softmax NoPE 209M, seed 2357, 1x | **0.750** | 0.994 | 0.994 | 1.00 | 0.00 |
| softmax NoPE 209M, seed 2357, 2x | **0.747** | 0.993 | 0.993 | 0.98 | 0.12 |
| softmax NoPE 209M, seed 1337, 2x | 0.144 | 0.540 | 0.543 | 0.18 | 4.51 |
| dilution NoPE 1K 209M, seeds 1337 / 2357 | 0.028 / 0.020 | | 0.37 / 0.32 | | |
| softmax NoPE 1K 209M, seeds 1337 / 2357 | 0.363 / 0.381 | | 0.57 / 0.59 | | |
| softmax RoPE 1K 209M, seeds 1337 / 2357 | 0.190 / 0.277 | | 0.73 / 0.71 | | |

1. *Point 5 now has a same-seed and a same-budget comparison, and the answer holds.* Seed 2357
   went from 1x to 2x keeping its reach (0.97 at 64K), and its middle layer changed as little as
   the cross-seed comparison above suggested: sink 0.014 -> 0.026, Gini 0.379 -> 0.380. At 2x the
   two seeds differ in reach (0.51 against 0.97 at 64K) but have the same sink (0.028 / 0.026)
   and off-diagonal Gini (0.408 / 0.410). The sink's growth from 1x to 2x happens in the seed
   that kept its reach too. Neither statistic tracks extrapolation.
2. *The softmax sink is larger than point 3 shows.* Softmax NoPE seed 2357, the seed whose
   training hit a burst of gradient spikes (S24), has a collapsed middle layer at both 1x and
   2x: three quarters of its attention on token 0, top-1 mass ~1 and entropy ~0, so every head
   sends each row to a single key, usually token 0. The 209M softmax middle-layer sink is
   therefore 0.06-0.32 with RoPE and 0.14-0.75 without position encoding, and "209M softmax
   reaches 0.25-0.32" in point 3 understates it. Whether the collapsed layer caused the seed's
   loss penalty or followed from the spikes is not tested.
3. *Scope of the sink claim.* Everything here is three sampled layers of one 1,024-token window,
   averaged over heads, so a sink confined to one head or to an unsampled layer would not show.
   "Dilution never exceeds 0.028" holds for every sampled dilution layer in every dilution
   checkpoint measured (64M and 209M, 1K-16K), not for every layer or head.
4. *What isolates the operator.* The alt6 rows in point 3 are consistent with the operator
   setting the sink but are not a clean control on their own: the hybrid's softmax layers also
   carry RoPE, and its sampled dilution layer is the last layer while its sinks sit in the
   middle. The position-matched evidence is the NoPE pairs: softmax 0.14-0.75 against dilution
   0.014-0.028 at 209M/16K, and 0.36-0.38 against 0.02-0.03 at 209M/1K (two seeds each). The
   no-cumsum row (0.052) was compared with a dilution checkpoint trained at a different learning
   rate; against the matched 6e-4 dilution checkpoint the comparison holds (token-0 mass 0.007 /
   0.009 at the middle / last layer against 0.052 / 0.024; S18 correction).

## Methods note (2026-09-22) — NIAH lengths now mean the full scored input

The NIAH harness (https://github.com/Royer-Research-Labs/Niah, spec `niah-v3-spec`) treated `context_length` as the
haystack alone and appended the question afterwards, so every scored prompt ran 12 tokens
(GPT-2) past the requested length. A request equal to the trained context could never run,
which is why the 1K arm's in-context ladder had no 1024 row. The package's own length guard
also never fired for DilutionLM: its adapter read `block_size`/`max_seq_len`, and our
config names it `context_length`. Our adapter worked around both with a 64-token headroom
skip.

Fixed in the harness (`niah-v4-spec`): `context_length` is the scored model input
(haystack + separator + question, plus all but the last token of a multi-token choice).
Every row and the control are built to exactly that length. The needle's insert point is
fixed and only the tail filler is fitted, because moving the needle changes BPE merges at
its joins. This was verified on CPU at 256-131,072 tokens: every row re-encodes to exactly
the requested length, and an old v3 spec rebuilds byte-identical rows. The adapter now reads
`context_length` too, and our headroom workaround has been removed.

Effect on earlier results: every NIAH number through S20 is v3, with the haystack at the
stated length and the prompt 12 tokens longer. From the 1K study onward the results files
carry `length_basis: "input"`. The 1K dilution arm's NIAH is re-run under v4, and its v3 files
are kept as `*.v3haystack.json`.

*Correction (release review, 2026-09-30). This note first said the difference was "at most
1.3% in length at the lengths used (896 and up), and no result depends on it". The ladders
start at 256 tokens, where 12 tokens is 4.7% (under 1.4% from 896 up). v3 also capped the
needle-free control prompt, which picks the disqualified keyword, at 5,400 tokens (the 54-token
filler repeated 100 times) whatever the requested length, so beyond that length the prior was
measured on a shorter prompt than the rows. "No result depends on it" was asserted, not
measured. It has since been measured:*

**Update: v4 re-score (release review, 2026-09-30).** Every v3 result behind a headline table or
figure (31 files) was re-scored under niah-v4 from the same checkpoint, with the same lengths,
placements, context override and NTK setting, into `results/niah_v4/<same name>.json`
(`scripts/compare_niah_versions.py` prints the comparison). The four LR-matched S18 cells were
re-scored with 20 placements each.

| group | cells | mean change | largest change |
|---|---:|---:|---|
| 209M, in context | 88 | 0.003 | 0.03 (softmax NoPE 2x, 16K: 0.29 -> 0.31) |
| 209M, beyond 16K | 67 | 0.013 | 0.09 (dilution NoPE 2x, 96K: 0.29 -> 0.37; alt6 seed 2357 without NTK, 64K: 0.23 -> 0.14) |
| 64M, in context | 22 | 0.020 | 0.16 (softmax NoPE, 1K: 0.51 -> 0.67) |
| 64M, beyond 16K | 12 | 0.042 | 0.17 (dilution NoPE seed 1337, 48K: 0.43 -> 0.60) |
| S18 exclusive cell (same placements) | 9 | 0.031 | 0.10 (6144: 0.62 -> 0.72); 7936: 0.69 -> 0.62 |

The large changes sit at mid-range accuracies, where a few of the 70-140 scored prompts
flipping moves the score; accuracies near 1.0 or near chance barely move. No conclusion in
this ledger changes direction. Headline numbers that moved: 209M dilution NoPE seed 2357 at
128K 0.96 -> 0.91 and at 192K 0.84 -> 0.91; the 2x seed-1337 model at 64K 0.51 -> 0.54 and at
128K 0.24 -> 0.29; 64M softmax NoPE at 16K 0.15 -> 0.21. The README, docs/research-results.md
and the figures use the v4 values. The signals above keep the numbers as first reported.

**Is v4 easier, and where does the protocol add variance?** (`scripts/compare_niah_versions.py
--summary-only`, the 198 cells scored with the same placements under both specs.) Not easier:
accuracy rose in 35 cells and fell in 35 (mean +0.001, sign test p = 1.0). Beyond 5,400 tokens,
where v3's control prompt was shorter than the scored prompt, the mean margin of the correct
keyword over the best other scored keyword rose slightly (median +0.04 nats, 72 of 128 cells up,
p = 0.18). That is the direction expected if v4 now disqualifies the keyword the model actually
prefers at that length, removing its strongest distractor, but accuracy did not follow (26 up,
28 down). The variance tracks the margin:

| v3 mean margin | cells | mean v3 accuracy | mean change | largest |
|---|---:|---:|---:|---:|
| 3 nats or more | 115 | 0.98 | 0.003 | 0.07 |
| 1.5-3 | 13 | 0.75 | 0.019 | 0.05 |
| 0.5-1.5 | 9 | 0.56 | 0.052 | 0.17 |
| under 0.5 | 61 | 0.24 | 0.021 | 0.16 |

The protocol change shifts the filler and question by 12 tokens and can change which keyword is
disqualified; confident comparisons survive that and near-ties flip either way. Near chance (1/7)
scores moved little (0.02 on average), though accuracy can fall below chance (one 64M cell is 0.086). So a v3 score near 1.0 or near chance carries almost
no protocol uncertainty, while a mid-range v3 score can differ from v4 by about 0.05,
occasionally 0.15, on top of its binomial error. The result files record how many keywords were
disqualified but not which, so individual flips cannot be traced to a changed disqualification.

Not re-scorable: the 209M 1x seed-1337 models without position encoding (dilution and softmax).
Their checkpoints were trained on to 2x before the trainer could keep a copy of the 1x
checkpoint, so their 1x ladders exist only under v3; the figure marks them. Other v3 results in
this ledger (the 8K and 16K studies through S12, the S10-S12 extrapolation ladders) were not
re-scored; read their mid-range cells with the uncertainty above.

## S21 — 209M at 1K context: softmax + RoPE wins on loss (by 0.038 nats), all three retrieve in context, and only dilution extrapolates (flat to ~4x); the dilution-vs-softmax loss gap is a long-context effect (measured, 1 seed each)

*(Second seed (S24): dilution loss 3.1611 and perfect retrieval to ~8x, not ~4x.)*

Three 209M arms (24L d768 12H) at context 1024, 65,536 tokens/step, the
same 128,174-step 2x cosine stopped at the 1x checkpoint (step 64,000 = 4.19B
tokens), LR 6e-4, 40B fineweb-edu corpus, seed 1337. The dilution arm switched to
the key-parallel forward at step 55,000 (S26); loss was continuous across it.

| 1K arm | common-slice loss | tok/s |
|---|---:|---:|
| softmax + RoPE 5e5 | **3.1276** | 131.0K |
| dilution, no position encoding | 3.1653 | 88K -> 110K (crossover fix, S26) |
| softmax, no position encoding | 3.1852 | 142.6K |

Against the same recipe at 16K (S11/S17), the ordering flips. At 16K, dilution
NoPE (3.0559) beat softmax RoPE (3.0835) by 0.028. At 1K, softmax RoPE beats
dilution by 0.038. The NoPE-vs-NoPE gap shrinks from 0.156 at 16K to 0.020 at 1K.
The loss advantage is a long-context effect: at short context a positioned softmax
is better.

**NIAH, chance 0.14** (niah-v4-spec lengths; the dilution arm re-run under v4):

| | 256-1024 | 1920 | 3072 | 3840 | 6144 | 7936 | 12288 | 16000 | 24000 | 32000 |
|---|---|---|---|---|---|---|---|---|---|---|
| dilution NoPE | 1.00 | 1.00 | 1.00 | 1.00 | 0.66 | 0.54 | 0.31 | 0.27 | 0.17 | 0.17 |
| softmax NoPE | 0.98-1.00 | 0.56 | 0.27 | 0.23 | 0.16 | 0.16 | 0.14 | 0.14 | 0.14 | 0.14 |
| softmax RoPE | 0.97-1.00 | 0.16 | 0.16 | 0.17 | 0.14 | 0.14 | 0.14 | 0.14 | 0.14 | 0.14 |
| softmax RoPE + NTK | | 0.14 | 0.16 | 0.17 | 0.17 | 0.16 | 0.14 | 0.14 | 0.14 | 0.14 |

In context, all three retrieve. Softmax without position encoding retrieves at 1K
where it managed only 0.19 at 16K (S17), so its retrieval failure is a length
failure. Beyond training length, dilution holds perfect to ~4x (the 16K-trained
1x checkpoints held to 8x). Softmax NoPE reaches ~2x. Softmax RoPE is at chance at
1.9x, with or without NTK scaling.

**Structure** (1,024-token window, first / middle / last layer): sink mass
0.001 / 0.028 / 0.001 for dilution, 0.007 / 0.363 / 0.006 for softmax NoPE,
0.004 / 0.190 / 0.133 for softmax RoPE. Middle-layer key Gini: 0.37 / 0.57 /
0.73. The pattern matches S20 at 1K.

Runs: `runs/scale_1024/*_seed1337`; eval `results/eval/scale1k__*`; NIAH
`results/niah/{scale1k,extrap32k,extrap32k_ntk}__*`; structure
`results/structure/scale1k_*`.

## S22 — [Superseded by the three-seed update below] A 50% Wikipedia blend restores 8x extrapolation to the 64M dilution model (0.23 -> 0.94 at 128K) and lifts softmax's in-context retrieval (0.15 -> 0.52 at 16K) without letting it extrapolate; the document-length account is ruled out (measured, 1 seed; see the replication and three-seed update below)

This tests the hypothesis that the data blend drives extrapolation.
Corpus: the 50/50 Wikipedia / FineWeb-Edu blend built by `scripts/blend_corpus.py`: 2.0B
tokens, exactly 50% English Wikipedia and 50% fineweb-edu, shuffled at the
document level. The configs differ from the fineweb-only references (S15 Control B
dilution at 3e-4; S16 softmax at 1.5e-4) only in the training file. Both are 64M,
no position encoding, 16K context, 1.27B tokens with a full cosine. Validation is
the fineweb split in every row. Wikipedia documents are *shorter* than fineweb's
(median 339 vs 625 tokens), so any gain cannot come from longer documents.

| fineweb val loss | fineweb only | wiki50 | change |
|---|---:|---:|---:|
| dilution | 3.5717 | 3.6600 | +0.088 |
| softmax | 4.0846 | 4.1680 | +0.083 |

Fineweb loss worsens by the same amount for both operators (half the training data
is out of domain), and the operator gap is unchanged (0.51).

**NIAH, chance 0.14:**

| | 3840 | 7936 | 12288 | 16000 | 24000 | 32000 | 48000 | 64000 | 96000 | 128000 |
|---|---|---|---|---|---|---|---|---|---|---|
| dilution, fineweb | 0.97 | 0.97 | 0.97 | 0.97 | 0.91 | 0.80 | 0.43 | 0.40 | 0.26 | 0.23 |
| **dilution, wiki50** | 0.99 | 0.99 | 0.97 | 0.98 | **0.96** | **0.94** | **0.94** | **0.94** | **0.94** | **0.94** |
| softmax, fineweb | 0.55 | 0.27 | 0.23 | 0.15 | 0.17 | 0.19 | 0.14 | 0.19 | 0.14 | 0.10 |
| softmax, wiki50 | 0.60 | 0.49 | 0.62 | **0.52** | 0.11 | 0.11 | 0.14 | 0.14 | 0.14 | 0.14 |

1. **Dilution extrapolation is data-dependent.** The same 64M recipe that held
   ~2x on fineweb holds flat to 8x on the blend. This is the profile S13/S14 found
   only at 209M (1x checkpoints). S16.3 read that as "extrapolation range grows
   with scale", but at least part of it is data. It also reframes S19: the 2x
   model's lost extrapolation may be a property of fineweb-only training rather
   than of more tokens or annealing.
2. **Softmax in-context retrieval is partly data-substitutable.** Retrieval at 16K
   rises 0.15 -> 0.52.
   Softmax still cannot extrapolate: chance from 24K either way.
3. **Why:** document length is ruled out (Wikipedia documents are shorter). The
   remaining candidates are text type (entity-dense, facts restated and referred
   back to) and topic diversity. Not yet separated.

Caveats: one seed per arm. Both softmax runs are unstable (gradient norm > 20 on
318 / 379 logged steps; blend median 4.0), as S16 found for softmax NoPE at 64M,
so its within-16K numbers are noisy (0.62 at 12K beside 0.49 at 8K). The blend
softmax's middle layer is extreme: top-1 0.82, key Gini 0.97. The fineweb
dilution reference used the kernels before the last kernel pass (S26) and the v3 NIAH harness (prompts 12
tokens longer); neither can move 0.23 to 0.94. Dilution structure is ordinary
(sink 0.001 / 0.013 / 0.007).

Replication: seed 2357 of both dilution arms, a paired two-seed
comparison. Runs: `runs/blend_16384/*`; NIAH `results/niah/{blend,extrap128k__blend}__*`;
eval `results/eval/blend__*`; structure `results/structure/blend_*`.

### S22 replication: the blend's effect is larger and more robust than one seed suggested, because fineweb-only training is seed-fragile

Seed 2357 of both dilution arms, same recipes (64M NoPE, 16K, 1.27B tokens, 3e-4).
Both runs were stable (gradient median 1.10, p99 2.06-2.56).

| dilution 64M | fineweb loss | 1024 | 3840 | 7936 | 16000 | 32000 | 64000 | 128000 |
|---|---:|---|---|---|---|---|---|---|
| fineweb, seed 1337 | 3.5717 | 0.97 | 0.97 | 0.97 | 0.97 | 0.80 | 0.40 | 0.23 |
| **fineweb, seed 2357** | 3.5721 | 0.86 | **0.46** | **0.29** | **0.19** | 0.20 | 0.29 | 0.36 |
| wiki50, seed 1337 | 3.6600 | 0.99 | 0.99 | 0.99 | 0.98 | 0.94 | 0.94 | 0.94 |
| wiki50, seed 2357 | 3.6621 | 0.96 | 0.96 | 0.96 | 0.96 | 0.89 | 0.63 | 0.49 |

1. **Loss replicates to 0.002 in every arm, and retrieval does not.** The second
   fineweb-only seed has the same loss as the first (3.5721 vs 3.5717) but lost
   in-context retrieval. It is 0.86 at 1K, 0.46 at 3.8K and 0.19 at its own trained
   16K: S15's loss/retrieval dissociation, now at the *default* rate and seed level.
   Its structure is unremarkable (middle-layer key Gini 0.31, sink 0.012), so the
   1,024-token probe does not see the failure.
2. **Both blend seeds retrieve 0.96-0.99 at every in-context length.** On the blend
   the in-context result is robust; on fineweb it is a seed lottery (1 of 2 seeds).
3. **Extrapolation:** both blend seeds beat both fineweb seeds at every length
   beyond 16K. Two-seed means at 64K / 128K: wiki50 0.79 / 0.72, fineweb 0.35 / 0.30.
   Flat-to-8x (seed 1337) was the favourable end of the blend's seed spread.

Revised reading of S22: on this recipe, a Wikipedia-heavy blend makes dilution's
retrieval *reliable* in context and roughly doubles its 4-8x extrapolation, at a
0.09-nat cost on fineweb loss. Earlier single-seed fineweb-only retrieval results
at 64M (S15, S16) should be read with this seed fragility in mind. Runs:
`runs/{blend,nope}_16384/*_seed2357`.

### S22 three-seed update: the blend effect is within seed noise; the replicated finding is that 64M NoPE retrieval forms in only about two seeds in three, on either corpus, at unchanged loss. This supersedes the two readings above.

Third seed (7331) of both dilution arms, plus a matched softmax + RoPE 5e5 control (LR 3e-4, otherwise
identical) on both corpora for all three seeds.

| dilution 64M NoPE | loss | 1024 | 3840 | 7936 | 16000 | 32000 | 64000 | 128000 |
|---|---:|---|---|---|---|---|---|---|
| fineweb 1337 | 3.5717 | 0.97 | 0.97 | 0.97 | 0.97 | 0.80 | 0.40 | 0.23 |
| fineweb 2357 | 3.5721 | 0.86 | 0.46 | 0.29 | **0.19** | 0.20 | 0.29 | 0.36 |
| fineweb 7331 | 3.5749 | 0.99 | 0.99 | 0.99 | 0.96 | 0.90 | 0.79 | 0.59 |
| wiki50 1337 | 3.6600 | 0.99 | 0.99 | 0.99 | 0.98 | 0.94 | 0.94 | 0.94 |
| wiki50 2357 | 3.6621 | 0.96 | 0.96 | 0.96 | 0.96 | 0.89 | 0.63 | 0.49 |
| wiki50 7331 | 3.6548 | 0.79 | 0.69 | 0.55 | **0.28** | 0.23 | 0.29 | 0.24 |

Three-seed means at 16K / 64K / 128K: fineweb 0.71 / 0.49 / 0.39, wiki50 0.74 / 0.62 / 0.56. With one
failed seed on each side, the blend's advantage is within seed noise. The single-seed claim above
("restores 8x extrapolation") paired a favourable blend seed with an ordinary fineweb seed. What does
replicate: fineweb loss is tight (3.5717-3.5749); the blend costs 0.09 nats of it; retrieval forms in
4 of 6 seeds and loss cannot tell which.

| softmax + RoPE 5e5, 64M | loss | 3840 | 7936 | 12288 | 16000 | 24000+ |
|---|---:|---|---|---|---|---|
| fineweb 1337 / 2357 / 7331 | 3.6753 / 3.6120 / 3.6612 | 0.99 / 0.98 / 0.99 | 0.72 / 0.79 / 0.92 | 0.55 / 0.41 / 0.73 | 0.44 / 0.47 / 0.56 | chance |
| wiki50 1337 / 2357 / 7331 | 3.6990 / 3.6652 / 3.6552 | 0.96 / 1.00 / 0.96 | 0.76 / 0.96 / 0.59 | 0.54 / 0.79 / 0.39 | 0.44 / 0.62 / 0.30 | chance |

Softmax RoPE never collapses, fades to 0.30-0.62 at 16K, and is at chance beyond it on either corpus.
Its loss varies 0.06 between seeds where dilution's varies 0.003. Dilution NoPE beats it by 0.077 nats
on the fineweb three-seed means (3.5729 vs 3.6495). The mechanism of the failed dilution seeds and a
test of fixes are S23. Runs: `runs/{nope,blend}_16384/*_seed7331`, `*softmax_rope_theta5e5*`.

## S23 — Why 64M NoPE retrieval fails in some seeds, and what makes it reliable: failed seeds never grow a strong retrieval head; a lower learning rate does not change the odds; RoPE on alternate layers makes retrieval reliable in 3 of 3 seeds, with the best loss (measured, 3 seeds per cell)

**1. The mechanism (`scripts/needle_attention.py`, built on the NIAH package's
`needle_attention_profile`).** On niah-v4 prompts at 7,936 tokens (4 needles x 5 depths), the
diagnostic measures the attention from the last prompt position onto the needle's answer token, for
every layer and head. Reported is the best single head:

| 64M dilution NoPE, 3e-4 | best head on answer token | layer | NIAH at 7.9K |
|---|---:|---|---|
| fineweb 1337 | 0.53 | 9 (0.46 at 6) | 0.97 |
| fineweb 7331 | 0.76 | 7 | 0.99 |
| wiki50 1337 | 0.72 | 6 (0.53 at 7) | 0.99 |
| wiki50 2357 | 0.63 | 6 | 0.96 |
| **fineweb 2357 (failed)** | **0.19** | 6 | 0.29 |
| **wiki50 7331 (failed)** | **0.18** | 7 | 0.55 |

Healthy seeds grow at least one head in layers 6-9 that puts 53-76% of its attention on the answer
token. Failed seeds have the same layers but top out at 18-19%. Layers 0-4 put essentially nothing on
the answer in any model. This separates the seeds cleanly where loss (identical to 0.003) and the
1,024-token structure probe (S20) could not. The 16K pass needs ~90 GB in the eager path and ran out
of memory; 7,936 was sufficient.

**2. A lower learning rate does not fix it.** The same recipe at 1.5e-4, same three seeds:

| NoPE, fineweb | loss | 3840 | 7936 | 16000 | 32000 | 64000 | 128000 |
|---|---:|---|---|---|---|---|---|
| 3e-4: 1337 / 2357 / 7331 | 3.5717 / 3.5721 / 3.5749 | 0.97 / 0.46 / 0.99 | 0.97 / 0.29 / 0.99 | 0.97 / **0.19** / 0.96 | 0.80 / 0.20 / 0.90 | 0.40 / 0.29 / 0.79 | 0.23 / 0.36 / 0.59 |
| 1.5e-4: 1337 / 2357 / 7331 | 3.6469 / 3.6419 / 3.6404 | 0.90 / 0.95 / 0.92 | 0.61 / 0.96 / 0.94 | **0.41** / 0.96 / 0.86 | 0.30 / 0.91 / 0.47 | 0.23 / 0.91 / 0.31 | 0.21 / 0.79 / 0.29 |

The lower rate costs 0.07 nats on every seed and still leaves one seed weak, a *different* one.
Seed 2357 fails at 3e-4 and is the best extrapolator at 1.5e-4; seed 1337 is the reverse. Retrieval
formation is therefore not tied to the initialisation or data order. It is a fragile event in
training at this size.

**3. Partial position encoding does.** `rope_alt` (RoPE on layers 1, 3, 5, ..., 1-indexed), 3e-4,
same seeds:

| rope_alt, fineweb | loss | 3840 | 7936 | 16000 | 32000 | 64000 | 128000 |
|---|---:|---|---|---|---|---|---|
| seed 1337 | 3.5586 | 1.00 | 1.00 | 1.00 | 0.69 (NTK 0.91) | 0.14 (0.41) | 0.14 (0.21) |
| seed 2357 | 3.5553 | 0.99 | 1.00 | 0.99 | 0.97 (0.99) | 0.64 (0.94) | 0.20 (0.56) |
| seed 7331 | 3.5582 | 0.97 | 0.97 | 0.94 | 0.66 (0.83) | 0.19 (0.61) | 0.19 (0.33) |

All three seeds retrieve at 0.94-1.00 through 16K. The mean loss, 3.5574, is the best 64M dilution
result on this recipe, 0.016 below NoPE. The trade is S10-S12's: without scaling the rotated layers
cap extrapolation near 2x, and dynamic NTK scaling extends it to 2-4x (0.41-0.94 at 64K) depending on
the seed.

**Reading.** At 64M, dilution with no position encoding at all learns in-context retrieval in about
two seeds in three, and loss cannot tell which. Position information in half the layers makes it
reliable and improves loss, at the cost of long-range extrapolation. At 209M, NoPE retrieval formed in
every seed tested (two at 16K, one at 1K); S24 adds more. Earlier single-seed 64M NoPE retrieval
results (S15, S16) should be read as one draw from this distribution. Runs:
`runs/nope_16384/all_dilution_{nope_chinchilla_lr1.5e-4,rope_alt_chinchilla_lr3e4}_seed*`; needle
profiles `results/needle/`.

## S24 — Second seeds for every 209M and 1K claim: dilution's loss replicates to 0.002-0.004 and softmax NoPE's failure to retrieve replicates, but S19's shortened extrapolation at 2x Chinchilla and S21's ~4x reach at 1K do not; extrapolation range is seed-dependent (measured, 2 seeds per cell)

Seed 2357 throughout, same configs and evaluation recipes as the originals (S13/S17 for 1x, S19 for
2x, S21 for 1K). Retrieval at or below the trained length is from the 20-placement in-context run.
The 10-placement extrapolation ladder also measures 16K (it read 0.16 / 0.14 for softmax at 1x), so it
is used only beyond the trained length. Each 1x checkpoint is kept as `latest_1x.pt` before it is resumed to 2x.

**1. 209M softmax NoPE (LR 6e-4), 1x Chinchilla.** Retrieval replicates; stability does not.

| 209M NoPE, 1x | common-slice loss | 1920 | 3840 | 7936 | 16000 | 32000-256000 |
|---|---:|---|---|---|---|---|
| softmax seed 1337 (S17) | 3.2118 | 0.78 | 0.46 | 0.29 | 0.19 | chance |
| **softmax seed 2357** | **3.2907** | 0.66 | 0.42 | 0.28 | 0.17 | chance |
| dilution seeds 1337 / 2357 | 3.0569 / 3.0548 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96-1.00 to 128K |

The retrieval failure of softmax without position encoding is identical across seeds. The second
seed hit a burst of gradient spikes starting near step 13,400 (94 logged steps with norm > 5, p99
62.5, against 3 for seed 1337) and trailed from then on (in-run 3.654 vs 3.488 at step 16,000). It
ends 0.079 behind seed 1337, where the two dilution seeds differ by 0.002. The operator loss gap is
0.156 against the stable softmax seed and 0.195 on two-seed means, the latter inflated by the
instability. At 209M and 6e-4, softmax NoPE sits near its stability edge and dilution does not,
consistent with S16 (at 64M softmax NoPE needed half the rate to train). Runs:
`runs/scale_16384/all_softmax_nope_lr6e-4_seed2357`; NIAH `results/niah/{scale1x,extrap*k}__*seed2357*`.

**2. 209M dilution NoPE seed 2357, 1x -> 2x: S19's shortened extrapolation does not replicate.**

| 209M dilution NoPE | loss | 16000 | 32000 | 48000 | 64000 | 96000 | 128000 | 192000 | 256000 |
|---|---:|---|---|---|---|---|---|---|---|
| seed 1337, 1x | 3.0569 | 1.00 | 0.96 | 0.97 | 0.99 | 0.99 | 0.96 | 0.80 | 0.63 |
| seed 1337, 2x (S19) | 2.8574 | 0.97 | 0.94 | 0.73 | 0.51 | 0.29 | 0.24 | 0.23 | 0.23 |
| seed 2357, 1x | 3.0548 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.84 | 0.79 |
| **seed 2357, 2x** | **2.8541** | 1.00 | 1.00 | 1.00 | **0.97** | **0.93** | **0.83** | 0.77 | 0.67 |

Loss replicates to 0.003 at 2x, as at 1x, and training was clean (no gradient norm above 2 in the
second half). But the second seed keeps near-flat retrieval to 4x and 0.83 at 8x after the full
schedule, where seed 1337 fell to ~2x. S19's reading ("further training bought loss and cost
extrapolation") is one seed's outcome, not a property of the recipe. Extrapolation range at 2x is
seed-dependent, just as retrieval formation is at 64M (S23), and loss does not reveal it. The
tokens-versus-annealing question S19 raised is moot for the recipe as a whole. Runs:
`runs/scale_16384/all_dilution_nope_seed2357` (2x `latest.pt`, 1x `latest_1x.pt`); NIAH
`results/niah/{scale2x,extrap*k__scale2x}__all_dilution_nope_seed2357.json`.

**3. 209M softmax NoPE seed 2357, 1x -> 2x: replicates S19's softmax side.**

| 209M softmax NoPE, 2x | loss | 1920 | 3840 | 7936 | 16000 | 32000 | 64000+ |
|---|---:|---|---|---|---|---|---|
| seed 1337 (S19) | 3.0039 | 0.97 | 0.78 | 0.45 | 0.29 | 0.24 | chance |
| seed 2357 | 3.0716 | 0.96 | 0.71 | 0.38 | 0.29 | 0.23 | chance |

Softmax learns some in-context retrieval from the extra data in both seeds (0.17-0.19 -> 0.29
at 16K) and none beyond it. Seed 2357 carries its early-instability deficit to the end: 0.068 behind
seed 1337, and still spikier after 1x (41 logged steps with gradient norm > 5 against 4). The 2x
operator gap is 0.147 against the stable softmax seed and 0.182 on two-seed means (dilution 2.8574 /
2.8541, mean 2.8558; softmax mean 3.0378). As at 1x, the dilution seeds agree to 0.003 and the softmax
seeds do not. Runs: `runs/scale_16384/all_softmax_nope_lr6e-4_seed2357` (2x `latest.pt`, 1x
`latest_1x.pt`).

**4. 1K study seed 2357 (three arms).**

| 209M 1K, 1x | loss | 1024 | 3840 | 6144 | 7936 | 12288 | 16000 | 32000 |
|---|---:|---|---|---|---|---|---|---|
| dilution NoPE seed 1337 (S21) | 3.1653 | 1.00 | 1.00 | 0.66 | 0.54 | 0.31 | 0.27 | 0.17 |
| **dilution NoPE seed 2357** | **3.1611** | 1.00 | 1.00 | **1.00** | **1.00** | 0.81 | 0.63 | 0.39 |

Loss replicates (0.004 apart). Reach again differs by seed: perfect to ~4x in seed 1337, to ~8x in
seed 2357 (0.81 at 12x, 0.63 at 16x). S21's "flat to ~4x" is the lower of the two.

| 209M 1K softmax NoPE | loss | 512 | 1024 | 1920 | 3072 | 3840 | 6144+ |
|---|---:|---|---|---|---|---|---|
| seed 1337 (S21) | 3.1852 | 1.00 | 0.98 | 0.56 | 0.27 | 0.23 | chance |
| seed 2357 | 3.1746 | 1.00 | **0.71** | 0.39 | 0.17 | 0.14 | chance |

Softmax NoPE's second 1K seed was stable (gradient p99 0.60) and retrieves less even at its own
trained length (0.71 at 1024). The 1K NoPE loss gap is 0.0135 on this seed and 0.017 on two-seed
means (dilution 3.1632, softmax 3.1799), which confirms S21's small short-context margin.

| 209M 1K softmax RoPE 5e5 | loss | 1024 | 1920 | 3840 | 7936+ | NTK 1920 / 3840 |
|---|---:|---|---|---|---|---|
| seed 1337 (S21) | 3.1276 | 0.97 | 0.16 | 0.17 | chance | 0.14 / 0.17 |
| seed 2357 | 3.1259 | 1.00 | 0.27 | 0.16 | chance | 0.29 / 0.30 |

Softmax RoPE replicates to 0.002 and is at or near chance beyond its trained length in both seeds,
with or without NTK scaling. At 1K it leads dilution NoPE by 0.036 on two-seed means (3.1268 vs
3.1632), confirming S21's ordering reversal at short context.

**S24 summary (all runs landed 2026-09-29).** Every 209M and 1K claim now has two seeds.
- *Replicates:* dilution loss (to 0.002-0.004 in every cell); the operator loss gap (16K 1x 0.156
  stable-seed / 0.195 mean; 2x 0.147 / 0.182; 1K NoPE 0.017; 1K RoPE -0.036); softmax NoPE's failure
  to retrieve beyond ~1.5x in every seed; dilution's in-context retrieval at 209M.
- *Does not replicate:* S19's shortened extrapolation at 2x (seed 2357 keeps 0.83 at 128K) and
  S21's ~4x reach at 1K (seed 2357 reaches ~8x). Extrapolation range is seed-dependent. It should be
  reported as a range over seeds.
- *New:* softmax NoPE at 209M, 6e-4 is unstable in one of two seeds (a 0.07-0.08 loss penalty that
  persists to 2x); dilution showed no instability in any 209M run.

## S25 — Sampling evaluation: inside the trained context sample quality tracks loss (level with softmax + RoPE at 16K, 0.03-0.12 bits/byte better than softmax without position encoding, about 0.01 behind softmax + RoPE at 1K where softmax has the better loss); past it every softmax model writes incoherent text (1.9-3.3 bits/byte at 2x-4x) while dilution stays within 0.06 of its in-context quality (measured, 128-256 prompts, two seeds per arm except one dilution seed at 16K 1x)

**Why.** Loss is teacher-forced and NIAH is likelihood-scored, so no result so far made a model
write. Generation conditions on the model's own tokens, and in dilution those tokens keep adding
to every key's demand total, so there was a plausible failure mode (drift, loss of the prompt)
that teacher-forced loss cannot show. `scripts/sample_eval.py` (library `src/dilution/sampling.py`)
makes every model continue the same prompts with the same random streams.

**Setup.** 256 validation prompts, each ending at least 64 tokens into a FineWeb-Edu document
with the next 256 human tokens still inside it; prompt lengths 256 / 1024 / 4096 / 16128 share
end positions (only the amount of earlier context changes), plus 32768 (2x the trained context;
RoPE layers use dynamic NTK scaling, as in the NIAH ladders). 256 new tokens each, under
ancestral sampling, nucleus p = 0.9 and greedy decoding. Judge: HuggingFaceTB/SmolLM2-1.7B
(revision effd688), bits per UTF-8 byte of the continuation given the last 4,096 judge tokens of
the prompt; lower is more plausible, and the human continuation scores 0.66-0.68. Models: the
1x checkpoints of dilution NoPE (seed 2357 only; seed 1337's 1x checkpoint was trained on to
2x), softmax + RoPE 5e5 and alt6 (two seeds each). Loss for reference (common slice): dilution
3.0548, softmax + RoPE 3.0883 / 3.0787, alt6 3.0385 / 3.0399.

Judge bits per byte, nucleus (ancestral in brackets), by prompt length:

| model | 256 | 1024 | 4096 | 16128 | 32768 (2x) |
|---|---:|---:|---:|---:|---:|
| human continuation | 0.679 | 0.664 | 0.660 | 0.660 | 0.660 |
| dilution NoPE, s2357 | 1.116 (1.496) | 1.091 (1.488) | 1.128 (1.467) | 1.118 (1.476) | **1.142 (1.471)** |
| softmax + RoPE, s1337 | 1.102 (1.450) | 1.098 (1.436) | 1.086 (1.466) | 1.108 (1.436) | 1.868 (2.068) |
| softmax + RoPE, s2357 | 1.145 (1.494) | 1.114 (1.489) | 1.124 (1.491) | 1.115 (1.489) | 2.145 (2.303) |
| alt6, s1337 | 1.077 (1.437) | 1.059 (1.426) | 1.086 (1.374) | 1.102 (1.413) | 1.197 (1.528) |
| alt6, s2357 | 1.094 (1.493) | 1.088 (1.493) | 1.070 (1.481) | 1.058 (1.443) | 1.196 (1.546) |

Paired differences, dilution minus softmax + RoPE (95% bootstrap interval over the 256 prompts),
nucleus:

| length | vs seed 1337 | vs seed 2357 |
|---|---|---|
| 256 | +0.014 [-0.024, +0.053] | -0.029 [-0.074, +0.008] |
| 1024 | -0.007 [-0.041, +0.025] | -0.023 [-0.055, +0.007] |
| 4096 | +0.041 [+0.003, +0.081] | +0.003 [-0.032, +0.041] |
| 16128 | +0.010 [-0.023, +0.042] | +0.003 [-0.026, +0.034] |
| 32768 | **-0.726** [-0.768, -0.677] | **-1.002** [-1.050, -0.950] |

1. **Inside the trained context the two operators sample alike.** Averaged over the four
   in-context lengths, dilution minus softmax is +0.015 (seed 1337) and -0.012 (seed 2357) bits
   per byte under nucleus sampling, and +0.035 / -0.009 under ancestral sampling. The two softmax
   seeds differ from each other by as much (seed 2357 is 0.01-0.05 worse than seed 1337), so
   dilution sits inside the softmax seed spread. Its 0.03-nat loss advantage (about 0.01 bits per
   byte) is below what 256 prompts resolve, and it does not show up as better samples. Similar,
   not better.
2. **alt6 is the best in-context sampler**: under nucleus both alt6 seeds score 0.00-0.05 below
   softmax seed 1337 at every in-context length, matching alt6's best loss.
3. **Past the trained context softmax + RoPE breaks down; dilution does not move.** At 32K its
   judge score is 1.87 / 2.15 (nucleus) against 1.09-1.15 in context. The text keeps topic words
   from the prompt but loses syntax ("special scenery for American colleges, where Shakespeare
   Agha.30 Jauer, college games the student. $1844 Click a nonrated."), and nothing is copied from
   the prompt (copy-8 0.000). Dilution NoPE at 32K scores 1.142, against 1.09-1.13 in
   context, and continues the same documents coherently (a bibliography with plausible entries; the
   bat-lighting study with further percentages). alt6 degrades by about 0.1 (1.20) although its
   NIAH retrieval is still 1.00 at 32K with NTK (presumably its softmax half; inference). This is
   S13-S17's extrapolation result in generative form, with a different kind of evidence: judged text rather than keyword likelihoods.
4. **Greedy decoding loops in every model** (96-100% of samples have rep-4 >= 0.5 over 256
   tokens; judge 0.13-0.16 in context, below the human text because repetition is predictable).
   It does not separate the operators at this scale; at 32K softmax loops slightly harder (rep-4 0.89-0.90
   against 0.82-0.86).
5. **Nothing degenerate in sampled text.** Nucleus and ancestral rep-4 is 0.00-0.04 for every
   model (human 0.02), looping at most 2%. Models end the document early (EOT) in 11-34% of samples.
6. **Cross-scoring is symmetric.** Each model rates its own nucleus samples at about 2.7 nats per
   token and the others' at 2.9-3.2; softmax rates dilution's samples no worse than dilution rates
   softmax's. Samples from models that share a seed (the same data order) are rated more alike
   across operators than same-operator models across seeds. On the human continuation the models
   rank as their losses do (alt6 3.107-3.109, dilution 3.124, softmax 3.131-3.149 at 256 tokens).



**The 2x set (no position encoding on either side, two seeds each).** The position-matched
pair after 2x Chinchilla (loss: dilution 2.8574 / 2.8541, softmax NoPE 3.0039 / 3.0716; softmax
seed 2357 is the one that hit gradient spikes). 128 prompts, a subset of the 256 above (the first
128 positions drawn; the sets share end positions, not row order).

| model | 256 | 1024 | 4096 | 16128 | 32768 (2x) |
|---|---:|---:|---:|---:|---:|
| human continuation | 0.670 | 0.653 | 0.649 | 0.649 | 0.649 |
| dilution NoPE, s1337 | 1.027 (1.465) | 1.020 (1.442) | 1.025 (1.370) | 1.003 (1.377) | **0.997 (1.430)** |
| dilution NoPE, s2357 | 1.044 (1.427) | 0.995 (1.414) | 1.016 (1.404) | 0.999 (1.354) | **1.016 (1.392)** |
| softmax NoPE, s1337 | 1.086 (1.450) | 1.054 (1.471) | 1.077 (1.416) | 1.071 (1.472) | 3.325 (3.052) |
| softmax NoPE, s2357 | 1.111 (1.493) | 1.117 (1.479) | 1.135 (1.477) | 1.105 (1.455) | 2.314 (2.447) |

(judge bits per byte, nucleus, ancestral in brackets)

7. **With position encoding matched, dilution samples better, by about what its loss lead
   predicts.** Under nucleus sampling every one of the 16 in-context dilution-minus-softmax pairs is
   negative: -0.03 to -0.07 against softmax seed 1337 (6 of 8 intervals exclude zero) and -0.07 to
   -0.12 against seed 2357 (all 8 exclude zero). Under ancestral sampling 15 of 16 are negative and
   10 exclude zero. Each softmax model rates both dilution models' samples as more plausible than
   the other softmax seed's samples, and dilution scores the human text better (2.94 against 3.07 / 3.14 nats
   per token at 256), which is the loss gap again.
8. **Softmax without position encoding collapses harder past its context than softmax + RoPE**:
   3.33 / 2.31 bits per byte at 32K under nucleus, nothing copied from the prompt, and seed 1337
   ends the document in 61% of samples (16-18% in context). Both dilution seeds are flat (1.00 / 1.02
   at 32K against 1.00-1.04 in context), including seed 1337, whose NIAH reach after 2x shrank to
   about 2x (0.93 at 32K, 0.54 at 64K).
9. **Calibration of the test.** The 2x prompts are a subset of the 1x set, so the same dilution
   seed (2357) can be compared before and after the extra training on identical prompts, matched
   by recorded end position: 2x minus 1x is -0.047 to -0.112 bits per byte under nucleus at every
   length (all five intervals exclude zero), for a 0.20-nat loss improvement. A loss difference of about 0.2 nats therefore shows as
   roughly 0.05-0.1 bits per byte in samples, and the 0.03-nat in-context gap at 1x (point 1) is
   below what this test resolves: "similar" there means "within about 0.03 bits per byte".

**The 1K set (the sensitivity check).** The 209M models trained at a 1,024-token context
(S21, S24), two seeds per arm. Here softmax + RoPE has the better loss (3.1276 / 3.1259 against
dilution 3.1653 / 3.1611 and softmax NoPE 3.1852 / 3.1746), so a fair test should favour it in
context. 256 prompts; lengths 128 / 256 / 768 in context, 1792 (2x) and 3840 (4x) beyond it.

| model | 128 | 256 | 768 | 1792 (2x) | 3840 (4x) |
|---|---:|---:|---:|---:|---:|
| human continuation | 0.700 | 0.688 | 0.674 | 0.668 | 0.666 |
| dilution NoPE, s1337 | 1.087 | 1.104 | 1.095 | **1.124** | **1.157** |
| dilution NoPE, s2357 | 1.068 | 1.071 | 1.091 | **1.106** | **1.134** |
| softmax + RoPE, s1337 | 1.065 | 1.085 | 1.093 | 2.445 | 2.974 |
| softmax + RoPE, s2357 | 1.063 | 1.053 | 1.081 | 2.577 | 2.616 |
| softmax NoPE, s1337 | 1.094 | 1.069 | 1.099 | 2.259 | 2.247 |
| softmax NoPE, s2357 | 1.075 | 1.089 | 1.092 | 2.372 | 2.774 |

(judge bits per byte, nucleus; RoPE rows use NTK scaling beyond 1K)

10. **The test does not simply favour dilution.** In context, softmax + RoPE samples slightly
    better: over the 12 paired nucleus comparisons (2 dilution seeds x 2 softmax seeds x 3
    lengths) dilution is +0.013 bits per byte on average, the direction and roughly the size its
    0.036-nat loss deficit predicts (about 0.012), with 11 of 12 intervals including zero. All six
    models are within 0.05 of each other in context.
11. **Past the trained context every softmax model collapses and dilution does not.** At 2x and
    4x all four softmax models write incoherent text (2.2-3.0 bits per byte, nothing copied from
    the prompt; at 4x even RoPE greedy decoding stops looping and writes noise, rep-4 0.35-0.61 with
    judge 1.0-1.4). Dilution rises by 0.03-0.06 over its in-context mean (1.11-1.16 against
    1.07-1.10 in context).
12. **The teacher-forced version of the same effect.** Cross-scoring at 2x and 4x scores the human
    continuation too: every softmax model assigns it 6.2-8.8 nats per token (against 3.0-3.1 in
    context), while both dilution models stay at 3.03-3.08, within 0.04 of their in-context
    values. A 1K-trained dilution model predicts real text at 4x its context as well as inside it.

**Summary of S25.** Sample quality tracks loss inside the trained context: level with softmax +
RoPE at 16K (a 0.03-nat loss gap, below the test's resolution), better than softmax without
position encoding (a 0.15-0.21-nat gap), slightly behind softmax + RoPE at 1K (where softmax has
the better loss). Nothing in dilution's samples is degenerate. Past the trained context the
operators separate completely: every softmax model, with or without RoPE and NTK scaling, writes
incoherent text at 2x-4x, and dilution's samples stay within 0.06 bits per byte of their
in-context quality.

Caveats: one dilution seed at 1x; one prompt set; the judge sees at most the last 4,096 of its
tokens of each prompt, so it measures local fluency and coherence, not long-range consistency; a
low judge score can reflect blandness, so it is read together with rep-4.

**Files.** `results/samples/scale209m_{1x,2x,1k}.json` hold the aggregates, the paired
differences with their bootstrap intervals, and the provenance: prompt end positions in the
validation stream, seeds, decoding settings, checkpoint SHA-256s and the judge revision.
`results/samples/scale209m_{1x,2x,1k}.metrics.jsonl.gz` hold every sample's metrics (text
statistics, judge score, cross-scores; no token ids or text), and `scripts/verify_samples.py`
recomputes every number in the summaries from them. `results/samples/scale209m_{1x,2x,1k}.md`
show two prompts at every length with the human continuation and every model's samples. The
full samples (token ids and text, 150 MB) are not published; the summaries identify everything
needed to regenerate them from the checkpoints with `scripts/sample_eval.py`, whose judge is
pinned to the recorded revision.

## S26 — Kernel engineering summary (2026-09-10 to 2026-09-29): exact-gradient fused Triton kernels made dilution training 4-52% faster in model throughput from 1K to 16K context, and a later pass added 2.6-4.1%; a full-schedule replay reproduces the first kernels' loss to 0.0007 nats; freezing the demand gradient is faster but trains an unstable objective; the levers reachable from Triton are exhausted (measured)

This signal condenses the kernel development log (profiling, variants tried and null results)
into what the rest of the ledger relies on. The design itself is described in docs/kernels.md.
Timings are forward plus backward of the attention op inside the model step, bf16, head
dimension 64, on the RTX PRO 6000; model throughput is the median over steps 200-1000 of a
1,000-step screen of the 64M model.

**1. What changed.** From the first working kernels (a two-pass forward and a four-kernel backward
with fp32 atomics), in order: relaxed memory ordering for the dq atomics (-10% step time); a
restructured backward whose reverse key-parallel pass computes the demand suffix sums itself,
removing three kernels (-19%); saving the query correction in the first forward kernel and fusing
the block scan (-5 to -7%); a key-parallel forward with online softmax statistics and a bf16
input to the correction kernel (-13 to -16%); the dv product moved into the heavy backward pass,
which now runs on 32-row tiles (-5 to -6%); and prefetching that pass's query and gradient tiles
with two pipeline stages (-3 to -6%). Every step passed the parity tests against the fp32
reference before it was measured.

| context (batch) | attention fwd+bwd, first kernels -> after the 32-row backward | model tok/s | gain | loss at step 1000 |
|---|---:|---:|---:|---|
| 1024 (8) | 0.69 -> 0.60 ms | 263.8K -> 274.4K | +4.0% | 5.7179 / 5.7176 |
| 2048 (8) | 2.35 -> 1.43 | 238.5K -> 283.4K | +18.8% | 5.7476 / 5.7484 |
| 4096 (4) | 4.48 -> 2.70 | 172.0K -> 223.2K | +29.7% | 5.8072 / 5.8071 |
| 8192 (2) | 8.97 -> 5.29 | 108.4K -> 153.2K | +41.3% | 5.8596 / 5.8635 |
| 16384 (1) | 18.66 -> 10.48 | 62.1K -> 94.3K | +51.9% | 5.8595 / 5.8628 |

The last pass (tile prefetch plus two pipeline stages) then took the step from 5.76-5.84 to
5.46 ms at B8 H8 T4096, 5.30-5.36 to 5.05 at B2 H8 T8192, 31.98-32.22 to 30.18 at the 209M 16K
shape (B2 H12 T16384) and 3.78-3.82 to 3.67 at the 209M 1K shape, and model throughput by 2.6%
(64M at 8K: 155.7K -> 159.9K tok/s) and 4.1% (209M at 16K: 32.1K -> 33.4K), with loss unchanged.
The kernels cost extra activation memory: the 32-row carry checkpoints and saved softmax
statistics add about 0.75 GiB at the 12-layer 8K configuration and 1.9 GiB at 16K.

**2. The rewritten kernels are training-equivalent, not only parity-equivalent.** The 8K
all-dilution run of the context ladder (`runs/12l_8192_600m/all_dilution_seed1337`, S1) was
replayed over its full schedule with the kernels after the 32-row backward, recording the kernel
source hash with the run (`runs/validate_s27/all_dilution_seed1337`): stable (gradient median 0.92,
p99 1.52) at 152.3K tok/s against 108K for the original.

| step | replay val | original val | difference |
|---|---:|---:|---:|
| 5000 | 4.4326 | 4.4297 | +0.0029 |
| 10000 | 4.0929 | 4.0887 | +0.0041 |
| 20000 | 3.8344 | 3.8341 | +0.0003 |
| 30000 | 3.6956 | 3.6941 | +0.0014 |
| 36622 | 3.6548 | 3.6542 | +0.0007 |

Retrieval (10 placements, 8 needles, niah-v3) is 0.99 / 0.97 / 0.96 / 0.96 / 0.96 / 0.97 / 0.97 /
0.97 / 0.94 from 256 to 7936 tokens, against 1.00 / 1.00 / 0.97 / 1.00 / 0.99 / 0.97 / 0.96 /
0.87 / 0.80 for the original: the same within sampling noise at short haystacks. Results
measured with different kernel versions in this ledger are therefore comparable.

**3. Exact demand gradients are required.** A surrogate backward that treats the cumulative demand
as a constant (`dilution.kernels_frozen`) was 23.5% faster in model throughput and only 0.165
nats behind at step 1,000. Over the full schedule
(`runs/validate_s27/frozen_demand_seed1337`) it reached validation 5.1674 at step 5,000 (+0.74
over the replay above) and 5.2668 at 8,500 (+1.11) and rising, with gradient-norm p99 168, and
was stopped as unstable (`scripts/judge_stability.py`). It trains a different, unstable
objective rather than approximating the derivative, so the exact suffix pass stays and
`kernels_frozen` is research-only.

**4. The short-context crossover.** Training used the key-parallel forward only above 1,024 tokens,
so the 209M 1K run began on the row-parallel path, whose backward runs 64-row tiles that spill
heavily. Lowering the threshold mid-run (at its step-55,000 checkpoint, after the kernel tests
passed) took it from 88.4K to 109.9K tok/s (+24%) with the loss continuous (3.13 at step 55,100).
With the final backward, key-parallel over row-parallel time is 1.01 / 1.00 / 0.95 / 0.86 at
128 / 256 / 512 / 1024 tokens (`scripts/bench_crossover.py`), so the threshold is now 256.

**5. Method, and what is exhausted.** Kernel timings are taken in the full ordered attention step,
not in isolation (a bf16 register change that looked neutral in isolation cost 1.5 ms in the real
step), then confirmed by a 1,000-step model screen, and only survivors get a full schedule.
Measured null or negative: transposed tiles and atomic fusion; 8 warps; extra pipeline stages
without the prefetch; 128-wide, 32-key and 16-row tiles; swapped grid axes; a bf16 upstream
gradient; per-tile atomics for dv and dk; dense materialisation with batched GEMMs; tensor-core
triangular products in place of scans; a persistent tile queue; register caps; explicit Gluon
layouts (the compiler cannot lower a scan in the tensor-core layout); and an exact inline-PTX
warp-shuffle scan, which saves a little but was not adopted because its correctness depends on a
compiler layout the language cannot assert. The heaviest backward pass issues at about 27% of the
SM's rate at the 255-register ceiling, so it is latency-bound, and further gains need a
lower-level implementation (docs/kernels.md).

**6. Inferred, not measured: equal wall clock at 8K after the speedup.** Dilution at 8K trains at
152K tok/s after the rewrite against SDPA softmax's 279K. Interpolating dilution's tuned
compressed-schedule loss log-linearly in tokens between its two measured points
(`runs/eqtime_tuned/all_{dilution,softmax}_*`, `runs/lr_8192/all_dilution_lr6e-4_seed1337`;
240M -> 3.8230, 600M -> 3.6046), in softmax's 37-minute budget dilution would reach about 3.742
against softmax's 3.7654, and in the 95-minute budget roughly 3.52-3.55 against 3.6477. Not run.

**Records.** `runs/validate_s27/` holds the replay and the frozen-demand run, each with its kernel
source hash. The 1,000-step screens behind the timing tables compared intermediate kernel versions
that are not part of this release, so their records are not included; the timings are as measured.

## S27 — Long-document perplexity (PG-19): past the trained context dilution's loss stays flat to 8x for the 16K models and rises gradually for the 1K models, while every softmax model's loss climbs steeply (1.7-1.9 nats worse than dilution at 8x with RoPE and NTK scaling, 2.6-5.1 worse without position encoding); inside the trained context, on these out-of-domain books, dilution trails softmax + RoPE by 0.05-0.08 nats at 16K and 0.18-0.24 at 1K (measured, 29 / 120 books; two seeds per arm, one dilution seed at 16K 1x)

**Why.** NIAH is synthetic. Loss by position on long real documents is the standard test of
length extrapolation, and it also shows how the operators compare inside the trained context on
text unlike the training data.

**Setup.** The PG-19 test and validation splits (150 Project Gutenberg books published before 1919;
parquet mirror `emozilla/pg19` at revision c021754), GPT-2-tokenized into 15.2M tokens
(`scripts/long_perplexity.py prepare`; the manifest records the token file's SHA-256). The 16K
models read the first 128K tokens (8x) of the 29 books that long; the 1K models read the first 32K
tokens (32x) of the 120 books that long. Each model reads each book teacher-forced, and the loss at
every position is averaged within doubling position ranges. Models with RoPE get one pass per
length with NTK base scaling to that length, as in the NIAH ladders (range (L/2, L] comes from the
pass at L), and one pass without scaling. Results: `results/longppl/pg19_209m_{16k_1x,16k_2x,1k}.json`
(every book's per-range loss is stored); figure: `docs/figures/pg19-perplexity-209m.png`.

Loss (nats per token, mean over books) by position range; "a / b" is seeds 1337 / 2357:

| 16K, 1x Chinchilla | 8K-16K (in context) | 16K-32K (2x) | 32K-64K (4x) | 64K-128K (8x) |
|---|---:|---:|---:|---:|
| dilution, no position encoding (seed 2357) | 3.554 | 3.577 | 3.607 | 3.605 |
| softmax + RoPE, NTK scaling | 3.537 / 3.513 | 3.710 / 3.723 | 4.517 / 4.680 | 5.338 / 5.539 |
| softmax + RoPE, no scaling | | 5.228 / 5.458 | 6.083 / 6.439 | 6.174 / 6.923 |
| alt6, NTK scaling | 3.500 / 3.553 | 3.463 / 3.526 | 3.661 / 3.818 | 3.980 / 4.213 |
| softmax, no position encoding (seed 2357) | 3.820 | 3.978 | 5.544 | 6.203 |

| 16K, 2x Chinchilla | 8K-16K (in context) | 16K-32K (2x) | 32K-64K (4x) | 64K-128K (8x) |
|---|---:|---:|---:|---:|
| dilution, no position encoding | 3.372 / 3.353 | 3.334 / 3.349 | 3.395 / 3.382 | 3.413 / 3.389 |
| softmax, no position encoding | 3.498 / 3.542 | 5.462 / 5.153 | 7.827 / 6.758 | 8.471 / 7.317 |

| 1K context | 512-1K (in context) | 1K-2K (2x) | 2K-4K (4x) | 4K-8K (8x) | 16K-32K (32x) |
|---|---:|---:|---:|---:|---:|
| dilution, no position encoding | 3.374 / 3.343 | 3.502 / 3.445 | 3.648 / 3.581 | 3.737 / 3.676 | 4.179 / 4.226 |
| softmax + RoPE, NTK scaling | 3.134 / 3.158 | 4.159 / 4.413 | 6.687 / 6.845 | 7.579 / 7.382 | 8.239 / 7.525 |
| softmax, no position encoding | 3.234 / 3.308 | 5.018 / 5.384 | 7.753 / 8.410 | 8.437 / 9.955 | 9.033 / 10.457 |

1. **Past the trained context, the retrieval result holds for loss on real books.** Paired over
   books (dilution minus the other model, with its standard error): at 16K and 1x, against softmax
   + RoPE with NTK scaling, -0.13 / -0.15 at 2x (+-0.11, within noise), -0.91 / -1.07 at 4x and
   -1.73 / -1.94 at 8x; against alt6, +0.05 / +0.11 at 2x (within noise), -0.05 / -0.21 at 4x and
   -0.38 / -0.61 at 8x. At 2x Chinchilla, against softmax without position encoding: -1.8 to -2.1
   at 2x, -3.4 to -4.4 at 4x and -3.9 to -5.1 at 8x. At 1K, against every softmax model: -0.66 to
   -1.94 at 2x and -3.0 to -6.6 from 4x on.
2. **Dilution's own curve.** The 16K models stay flat: the 1x model is 0.05 above its in-context
   loss at 4x-8x, and the 2x models stay within 0.05 of theirs at every range to 8x. The 1K models
   degrade gradually: +0.10-0.13 at 2x, +0.24-0.27 at 4x, +0.33-0.36 at 8x and +0.81-0.88 at 32x
   over their 512-1K loss, where softmax rises by 1.0-2.1 nats at 2x and 3.5-5.1 by 4x. alt6 with NTK
   scaling matches dilution at 2x and degrades from 4x, in line with its NIAH reach (3x-4x).
3. **Inside the trained context, on these out-of-domain books, dilution trails softmax + RoPE.**
   Averaged over the positions inside the trained context and paired over books: at 16K and 1x,
   dilution is behind softmax + RoPE by 0.047 +- 0.017 (seed 1337) and 0.084 +- 0.018 (seed 2357)
   and behind alt6 by 0.04-0.08, and ahead of softmax without position encoding by 0.25. At 2x
   Chinchilla it is ahead of softmax without position encoding by 0.12-0.19. At 1K it is behind
   softmax + RoPE by 0.18-0.24 and behind softmax without position encoding by 0.09-0.15. On the
   in-domain FineWeb-Edu validation slice the same checkpoints put dilution 0.02-0.03 ahead of
   softmax + RoPE at 16K, 0.036 behind it at 1K and 0.01-0.02 ahead of softmax without position
   encoding at 1K, so on these books the in-context comparison moves against dilution, most at
   short context. The largest gap is at the start of each book (positions 0-512: 4.06 for dilution
   against 3.68-3.87 for softmax + RoPE at 16K). Why is not tested; the books are pre-1919
   fiction and non-fiction, far from the educational web text the models were trained on.

Caveats: 29 books at 16K (only books of at least 128K tokens qualify), one dilution seed at 16K 1x,
one domain; RoPE models use dynamic NTK scaling, and their unscaled loss is reported too.
