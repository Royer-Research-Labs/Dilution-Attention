# Dilution Attention research results

The concise, current record, reviewed through `docs/signals.md` S25 (2026-09-30). The numbered
signals, per-run `metrics.jsonl` records and `results/{niah,niah_v4,eval,structure,needle,samples}` are the
evidence and take precedence if this summary disagrees with them. Every number below states its
seed count. Claims that did not survive replication are collected under
[Superseded](#superseded-and-withdrawn-claims) rather than left in place.

## The operator

For causal logits `z_ij`:

```text
p_ij     = exp(z_ij) / sum(k <= i) exp(z_ik)        # softmax bids
d_ij     = sum(r <= i) p_rj                         # inclusive cumulative demand on key j
share_ij = p_ij / (d_ij + eps)
A_ij     = share_ij / sum(k <= i) share_ik          # renormalise the row
out_i    = sum(j <= i) A_ij v_j
```

It is causal, has no parameters, is invariant to shifting a row's logits, and is O(T^2) like
softmax attention. Cached decoding needs the KV cache plus one fp32 demand accumulator per key and
head. The load-bearing choices are row-normalised softmax bids, *inclusive* cumulative demand, the
final row normalisation, and exact demand gradients.

## Setup

- **64M family:** 12 layers, width 512, 8 heads, SwiGLU, no biases, GPT-2 vocabulary (50,304).
  About 63.5M parameters without position tables. 16,384 tokens per optimiser step, bf16,
  `torch.compile`, fused AdamW, 1,000 warmup steps, cosine schedule. The early ladders (through S14)
  used a 1B-token FineWeb-Edu corpus and 600M tokens. The reliability studies (S15-S23) used a
  40B-token FineWeb-Edu corpus and 1.27B tokens (1x Chinchilla) at 16K context.
- **209M family:** 24 layers, width 768, 12 heads, 208.5M parameters, 65,536 tokens per step,
  LR 6e-4, the 40B corpus, and a 128,174-step cosine schedule (8.4B tokens, 2x Chinchilla). "1x"
  means the step-64,000 checkpoint (4.19B tokens, learning rate 55.7% of peak). "2x"
  means the end of the schedule. Trained at 16K (S11-S19) and at 1K (S21).
- **Retrieval (NIAH):** a synthetic, likelihood-scored keyword test
  (<https://github.com/Royer-Research-Labs/Niah>, pinned at `7cbf29f`). The haystack is one fixed
  54-token paragraph of procedural office text repeated to length, so it is highly repetitive and
  unlike natural documents. The sentence "The secret keyword is *w*." is inserted at a depth, and
  the prompt ends "### IMPORTANT DATA: The secret keyword is". The model's likelihood is compared
  across 8 fixed single-token candidates (gold, life, time, love, work, play, game, rain), each of
  which serves as the needle in turn at evenly spaced depths: 20 per length in context and 10
  beyond the trained length. Before scoring, the candidates are scored on a needle-free control
  prompt and the model's most-preferred one is disqualified, so each model is scored on its own
  7 of the 8 and chance is 1/7 = 0.14. The test measures copying one planted token back out of
  repetitive filler, not reasoning or retrieval from natural text.
- **NIAH protocol versions:** `niah-v4` (from 2026-09-22) builds every prompt, and the control,
  to exactly the requested length, question included. `niah-v3` built the haystack to the
  requested length and appended the question, so prompts ran 12 tokens over (4.7% at the
  256-token rung, under 1.4% from 896 tokens up). It also capped the control prompt at 5,400
  tokens whatever the requested length, so at longer lengths the disqualified keyword was chosen
  on a shorter prompt than the one scored. Every v3 result behind the tables and figures here was
  re-scored under v4 from the same checkpoint and settings (`results/niah_v4/`, compared by
  `scripts/compare_niah_versions.py`). At 209M, 88 in-context cells moved by at most 0.03 and 67
  extrapolation cells by at most 0.09 (mean 0.01). At 64M, two of 34 cells moved by 0.16-0.17 (one
  dilution cell at 48K, 0.43 -> 0.60; one softmax cell at 1K) and the rest by at most 0.06. The large
  changes sit at mid-range accuracies, where a few of the 70-140 scored prompts flipping moves the
  score; no finding changed direction. v4 is not systematically easier: over the 198 cells scored
  with the same placements, accuracy rose in 35 and fell in 35 (mean +0.001). Beyond 5,400 tokens
  the mean margin of the correct keyword rose slightly (median +0.04 nats, 72 of 128 cells up,
  sign test p = 0.18), the direction expected when the control removes the keyword the model
  actually prefers at that length, but accuracy did not follow (26 up, 28 down). The protocol adds
  variance where the correct keyword only narrowly wins. Cells with a mean margin of 3 nats or more
  (115 cells, accuracy ~0.98) moved 0.003 on average; cells at 0.5-1.5 nats (accuracy ~0.5) moved
  0.05 on average and up to 0.17; near chance (1/7) scores moved 0.02 on average, though accuracy can
  fall below chance (one 64M cell is 0.086). A v3 score near
  1.0 or near chance therefore carries almost no protocol uncertainty, while a mid-range v3 score
  can differ from v4 by about 0.05, occasionally 0.15, on top of its binomial error
  (`scripts/compare_niah_versions.py --summary-only`). The numbers here and in the figures are
  the v4 values, with two exceptions: the 209M 1x seed-1337 models without position encoding
  (dilution and softmax) were trained on to 2x before a copy of the 1x checkpoint was kept, so
  their 1x ladders exist only under v3 and are marked in the figure. Older ledger results (the 8K
  and 16K studies through S12) are v3 as recorded; read their mid-range cells with the extra
  uncertainty above.
- **Sampling:** `scripts/sample_eval.py` makes the models write. Every model continues the same
  validation prompts (each ending inside a FineWeb-Edu document, with the real next 256 tokens as
  the human reference) with the same random streams, so comparisons are paired prompt by prompt.
  Prompts of different lengths share their end point; only the amount of earlier context changes.
  256 new tokens under ancestral, nucleus (p 0.9) and greedy decoding. The main score is bits per
  byte of the continuation under a judge model, HuggingFaceTB/SmolLM2-1.7B, given the last 4,096
  of its tokens of the prompt (lower is more natural; the human text scores 0.65-0.70), read with
  repetition (seq-rep-4) because repetitive text also scores low. The judge sees local context
  only, so this measures fluency and local coherence, not long-range consistency (S25).
- **Loss:** 209M and 64M comparisons use one common validation slice per scale
  (`scripts/eval_checkpoint.py`). In-run validation windows differ with batch size and are not
  comparable across runs (S11).

## Results

### 209M, trained at 16K

| arm | loss 1x (seeds) | NIAH at 16K | beyond 16K |
|---|---|---|---|
| alt6: RoPE softmax + NoPE dilution, alternating | 3.0385 / 3.0399 | 1.00 / 1.00 | 3x-4x with NTK scaling (0.66 at 6x in one seed) |
| dilution, no position encoding | 3.0569 / 3.0548 | 1.00 / 1.00 | 0.91-1.00 to 128K (8x) in both seeds; 0.80-0.91 at 12x |
| softmax + RoPE 5e5 | 3.0883 / 3.0787 | 0.79 / 0.64 | 1.5x in one seed with NTK scaling (0.64); chance by 2x |
| softmax, no position encoding | 3.2118 / 3.2907* | 0.19 / 0.17 | chance to 256K |

\*This seed hit gradient spikes near step 13,400 (94 logged steps with norm above 5) and never
recovered the difference (S24). At 2x Chinchilla (S19, S24), dilution reaches 2.8574 / 2.8541 and
softmax 3.0039 / 3.0716. Dilution keeps 0.97-1.00 retrieval at 16K. Softmax improves to 0.29-0.31
at 16K and stays at chance beyond. Dilution's reach after 2x differs by seed: 0.54 at 64K in one,
and 0.97 at 64K and 0.83 at 128K in the other.

Throughput (logged training rate, RTX PRO 6000): softmax 72-75K tok/s, alt6 44K, dilution 33.4K with
the final kernels (S26).

### 209M, trained at 1K (S21, S24)

| arm | loss (seeds) | NIAH at 1K | beyond 1K |
|---|---|---|---|
| softmax + RoPE 5e5 | **3.1276 / 3.1259** | 0.97 / 1.00 | one seed at chance by 2x; the other 0.27 near 2x, or 0.29-0.33 to about 3.75x with NTK scaling |
| dilution, no position encoding | 3.1653 / 3.1611 | 1.00 / 1.00 | perfect to 4x / 8x, then declining |
| softmax, no position encoding | 3.1852 / 3.1746 | 0.98 / 0.71 | partial near 2x (0.56 / 0.39), near chance by about 4x |

### 64M, trained at 16K, 1x Chinchilla (S15-S23)

| arm | loss (3 seeds unless noted) | NIAH at 16K | beyond 16K |
|---|---|---|---|
| dilution, no position encoding, 3e-4 | 3.5717 / 3.5721 / 3.5749 | 0.97 / **0.19** / 0.96 | healthy seeds 0.34-0.79 at 64K |
| dilution, no position encoding, 1.5e-4 | 3.6469 / 3.6419 / 3.6404 | **0.41** / 0.96 / 0.86 | 0.23-0.91 at 64K |
| dilution, RoPE on alternate layers, 3e-4 | 3.5586 / 3.5553 / 3.5582 | 1.00 / 0.99 / 0.94 | 0.41-0.94 at 64K with NTK scaling |
| softmax + RoPE 5e5, 3e-4 | 3.6753 / 3.6120 / 3.6612 | 0.44 / 0.47 / 0.56 | near chance beyond 1.5x, with NTK scaling |
| softmax, no position encoding, 1.5e-4 (1 seed) | 4.0846 | 0.21 | chance |

Softmax without position encoding diverges at 3e-4 at this size (S16). The same models trained on a
50/50 Wikipedia / FineWeb-Edu blend (S22): dilution 3.6600 / 3.6621 / 3.6548 on the FineWeb
validation slice, with 16K retrieval 0.98 / 0.96 / **0.28**; softmax + RoPE 3.6990 / 3.6652 /
3.6552 with 0.44 / 0.62 / 0.30.

### Attention structure

From S18 and S20: `scripts/attention_structure.py` on one 1,024-token validation window per model, three layers
(first / middle / last), averaged over heads, so a sink confined to one head or to an unsampled
layer would not show. *Sink* is the fraction of attention mass on token 0; *key Gini* is how
concentrated the received load is across keys, excluding self-attention, at the middle / last
layer (0 means every key is used equally; uniform causal attention scores 0.50).

| model | sink (first / mid / last) | key Gini (mid / last) |
|---|---|---|
| dilution, no position encoding, 209M, 1x Chinchilla | 0.001 / 0.014 / 0.002 | 0.41 / 0.27 |
| dilution, no position encoding, 209M, 2x, two seeds | 0.001 / 0.026-0.028 / 0.001-0.002 | 0.41 / 0.21-0.22 |
| softmax, no position encoding, 209M, 1x | 0.006 / 0.750 / 0.033 | 0.99 / 0.66 |
| softmax, no position encoding, 209M, 2x, two seeds | 0.006 / 0.14-0.75 / 0.07-0.08 | 0.54-0.99 / 0.64-0.66 |
| softmax + RoPE, 209M, 1x, two seeds | 0.006 / 0.06-0.32 / 0.25-0.29 | 0.54-0.74 / 0.63-0.64 |
| alt6 hybrid, 209M: softmax / softmax / **dilution** layers, two seeds | 0.006 / 0.05-0.10 / **0.002-0.006** | 0.59-0.62 / **0.17-0.18** |
| dilution, 64M, 8K, learned positions | 0.002 / 0.002 / 0.004 | 0.17 / 0.23 |
| softmax, 64M, 8K, learned positions | 0.007 / 0.030 / 0.033 | 0.37 / 0.58 |

The position-matched pairs are the cleanest comparison: with no position encoding on either side,
209M softmax puts 14-75% of its middle layer's attention on token 0 and dilution 1.4-2.8%; at 1K
context, 36-38% against 2-3%. The 75% is the softmax seed whose training hit gradient spikes; its
middle layer sends every row to a single key, usually token 0. The alt6 row fits the same picture
but does not isolate the operator, because the hybrid's softmax layers also carry RoPE. The
layer-0 difference is not evidence of balanced retrieval: softmax's first layer attends almost
uniformly and dilution's is recency-weighted, and a local pattern looks balanced automatically; the
fair comparison is at depth. Neither statistic tracks extrapolation: the two 2x dilution seeds
differ in reach (0.54 against 0.97 at 64K) but not in sink or key Gini. These are snapshots of
trained models, not a causal account of long-context retrieval.

## Findings

1. **Dilution, not the absence of rotation, is what extrapolates.** With no position encoding on
   either side, at the same scale, data, schedule and learning rate, dilution retrieves at its
   trained context and far beyond it. Softmax fades to 0.17-0.19 at 16K and is at chance beyond it
   (209M, two seeds each). No tested softmax configuration sustains strong retrieval far beyond its
   training context: the 16K-trained controls are at chance by 2x (one softmax + RoPE seed holds
   0.64 at 1.5x with NTK scaling), and the 1K-trained controls keep partial retrieval near 2x (0.56
   / 0.39 without position encoding; one RoPE seed 0.29-0.33 to about 3.75x with NTK scaling)
   before degrading. The RoPE-bearing dilution and hybrid configurations tested also degrade
   eventually; NTK scaling extends some of them but does not remove the degradation. Pure dilution
   with no position encoding produced the longest extrapolation observed (S12-S17, S21, S24).
2. **Dilution's loss is reproducible; its extrapolation range is not.** Dilution seeds agree to
   0.002-0.004 nats in every cell measured, while softmax seeds differ by up to 0.08. How far
   retrieval reaches varies by seed at every scale: 4x-8x at 1K context, 2x-8x after 2x Chinchilla
   at 209M, 2x-8x among 64M seeds that retrieve at all. The 209M 1x checkpoints are the most
   consistent (flat to 8x in both seeds). Report extrapolation as a range over seeds (S22-S24).
3. **At 64M without position encoding, in-context retrieval forms in only about two seeds in three,
   and loss cannot tell which.** Two seeds of the same recipe match loss to 0.0004, while one
   retrieves 0.97 at 16K and the other 0.19. The failed seeds never grow a strong retrieval head.
   In healthy seeds, at least one head in layers 6-9 puts 53-76% of its attention on the needle's
   answer token; in failed seeds the best head reaches 0.18-0.19 (`scripts/needle_attention.py`).
   Halving the learning rate costs 0.07 nats and leaves one seed in three weak, a different seed. A
   Wikipedia-heavy blend does not change the odds. RoPE on alternate layers makes retrieval form in
   all three seeds and gives the best 64M loss, with a reach that varies by seed (with NTK scaling
   0.41-0.94 at 4x, one seed 0.80 at 6x) (S22, S23). At 209M,
   retrieval formed in every dilution seed tested (four runs).
4. **The loss margin depends on context length.** Once each operator has its best position mode,
   softmax + RoPE is *ahead* by 0.036 at 1K (209M). The two are level within 0.007 at 8K (64M), and
   dilution leads at 16K: 0.028 without position encoding and 0.044 for alt6 at 209M, and 0.077 /
   0.092 without position encoding / with RoPE on alternate layers at 64M (three seeds). The early
   0.19-0.43-nat margins used learned position tables for softmax and mostly measured their failure
   (S1, S5-S9, S11, S21-S24).
5. **Dilution strongly suppresses the token-0 attention sink, in the layers measured.** The probe
   samples three
   layers (first, middle, last) of one 1,024-token window per checkpoint, averaged over heads. No
   sampled dilution layer puts more than 2.8% of its attention on token 0. With no position
   encoding on either side, the position-matched comparison, 209M softmax puts 14-75% there in its
   middle layer (two seeds; the 75% seed is the one whose training hit gradient spikes, and its
   middle layer sends every row to a single key) and 1K softmax 36-38%. With RoPE, 209M softmax
   puts 25-32% there. In the alt6 hybrid the softmax layers carry a 5-10% sink and the sampled
   dilution layer under 1%, consistent with the operator setting it; the hybrid alone does not
   isolate the operator, because its softmax layers also carry RoPE. At depth, dilution's key
   load is less concentrated than softmax's for similar per-query selectivity. Sink size does not
   track extrapolation: the two 2x dilution seeds differ in reach but not in sink (S18, S20).
6. **Softmax bids are load-bearing.** Of the bids tried, only softmax and sigmoid train with the
   share step, and sigmoid is 0.33 nats worse. The share step's gradient scales as 1/bid, and
   bids whose Jacobian lacks a factor of the bid (ReLU, ReLU^2, min-shift) diverge, as that
   predicts. Softplus also diverged, after ~500 normal steps, although its Jacobian does carry
   the factor; its failure is measured, not explained, and may be numerical (the fp32 kernel
   rounds softplus to exactly 0 below about -16.6 while its derivative stays nonzero). Raw
   exponential bids overflowed fp32 in our implementation; a numerically stable form is untested.
   With softmax bids, the cumulative-demand step is worth 0.178 nats and lifts 8K retrieval from
   0.59 to 1.00 (one seed per cell, LR-matched; S3, S4).
7. **Inclusive demand matters.** The closest published precedent, temporal attention (Sankaran
   2016, Paulus 2018), divides by *exclusive* history. In causal self-attention that denominator is
   zero at each key's first query (itself), so the diagonal wins by 1/eps. Training escapes this
   everywhere except layer 0. Against inclusive dilution trained with the same recipe (64M, 8K,
   LR 6e-4, one seed each), exclusive history costs 0.072 nats (3.6771 against 3.6055) and
   retrieval at 7,936 tokens falls from 1.00 to 0.62. The share step as a whole is worth 0.177
   nats against the matched no-cumsum control, so exclusive history keeps about 60% of it (S18).
   The raw-exponential half of the temporal-attention factorial is untested: our fp32 version
   overflowed, and the stable cumulative-logsumexp form has not been built.
8. **Stability.** No dilution run at either scale showed a loss-affecting instability. Softmax
   without position encoding needs half the learning rate at 64M (S16). At 209M and 6e-4 it was
   unstable in one of two seeds (S24).
9. **Generated text tracks loss in context; past it only dilution still writes coherently.** At
   209M, judged bits per byte under nucleus sampling (two seeds per arm, 128-256 prompts): inside
   the trained context dilution is level with softmax + RoPE at 16K (within the 0.01-0.04 that
   separates the two softmax seeds; one dilution seed), 0.03-0.12 better than softmax without
   position encoding at 2x Chinchilla, and about 0.01 behind softmax + RoPE at 1K, where softmax
   has the better loss. The test resolves loss gaps of about 0.2 nats (the same dilution seed
   improves 0.05-0.11 from 1x to 2x Chinchilla), not 0.03. Past the trained context all the softmax
   models tested, with or without RoPE and NTK scaling, write incoherent text at 2x-4x (1.9-3.3 bits
   per byte against 1.05-1.15 in context). Dilution's samples stay within 0.06 of their in-context
   quality. Scored teacher-forced on the human continuation (1K-trained models), each softmax model
   tested gives 6-9 nats per token at 2x-4x its context, and dilution stays within 0.04 of its
   in-context value. The judge sees only the last 4,096 tokens of each prompt, so this shows that
   dilution remains a working language model past its trained window, not that it uses information
   from far back. Greedy decoding loops in every model at this scale (S25).

## Kernels and cost

Triton kernels (`src/dilution/kernels.py`, `docs/kernels.md`). The training forward computes
online softmax statistics, then runs a key-parallel pass that carries demand in registers and saves
checkpoints every 32 rows. The backward walks row blocks in reverse, so the per-key suffix is a
running sum, and applies the softmax-Jacobian correction in a separate lean pass. Decoding keeps
one accumulator per key. Forward plus backward, bf16 D64 (S26): 5.46 ms at B8 H8 T4096, 5.05 ms at
B2 H8 T8192, 30.2 ms at B2 H12 T16384. Softmax trains 1.3x faster at 1K context and 2.2x faster at
16K. The heaviest backward pass is latency-bound at the register ceiling. The levers reachable from
Triton have been measured, including an exact in-register warp-shuffle scan that was not adopted.
Further gains need a lower-level implementation. A 256K-token forward runs in 17.9 GiB (S14).

## Superseded and withdrawn claims

Kept here so the ledger's earlier wording is not mistaken for current results.

| earlier claim | status | where |
|---|---|---|
| Dilution beats softmax by 0.19 / 0.43 nats at 8K / 16K | learned-position artefact; see Finding 4 | S1 -> S5-S9 |
| Extrapolation range grows with scale (64M ~2x, 209M ~8x) | seed-dependent at both scales | S16.3 -> S22-S24 |
| 2x Chinchilla shortens dilution's reach to ~2x | one seed; the second holds 0.83 at 128K | S19 -> S24 |
| 1K-trained dilution reaches ~4x | one seed; the second reaches ~8x | S21 -> S24 |
| A Wikipedia blend restores 8x extrapolation at 64M | within seed noise over three seeds | S22 |
| Softmax without position encoding is stable at 209M | unstable in one of two seeds | S17 -> S24 |
| The 64M NoPE loss is 3.4840 (gap 0.60) | in-run window; common slice 3.5717 (gap 0.51) | S16 correction |
| Dilution is ~20x more selective, ~4x more balanced | measured at layer 0 against a uniform layer | S18 -> S20 |
| Exclusive history costs 0.023 nats; the share step is worth 0.128 | compared 6e-4 cells with dilution at 3e-4; matched: 0.072 and 0.177 | S18 correction |
| Only exponential-family bids can train (softplus fails structurally) | softplus's Jacobian carries the bid factor; its failure is unexplained | S3 correction |
| The raw-exponential temporal-attention cells are settled | our fp32 version overflowed; a stable version is untested | S18 correction |
| Dilution's sink never exceeds 0.028 at any depth; 209M softmax reaches 0.32 | three sampled layers per model; softmax without position encoding reaches 0.75 | S20 update |
| The 1x checkpoint has the learning rate at ~70% of peak | 55.7% (3.34e-4 of 6e-4) | S11, S19 correction |
| No softmax configuration extrapolates beyond ~1.5x; all are near chance by 2x | the 1K-trained controls keep partial retrieval near 2x (0.56 / 0.39) | S21, S24 |
| RoPE-bearing configurations cap out sooner the more layers they rotate | the number of rotated layers was not isolated as a variable | S10-S12 |
| `rope_alt`'s rotated first layer is the bottleneck | it rotates 6 of 12 layers; not isolated | S10-S12 correction |
| Frozen (stop-gradient) demand is a faster equivalent | diverged over a full schedule | S26 |

## Caveats

- Scope: two model sizes (64M, 209M), one tokenizer and corpus family (GPT-2, FineWeb-Edu), and
  keyword retrieval. NIAH here is not long-context reasoning or generation.
- Seeds: two per arm at 209M, three at 64M for the reliability studies. The early 64M ladders
  (through S14) are mostly single-seed, and Finding 3 applies to any single-seed 64M retrieval number.
- Compare matched position modes, token budgets and wall clock separately. Dilution is slower per
  token, so equal-token and equal-time comparisons differ.
- Fused training uses atomics, so it matches the reference to tolerance but is not bitwise
  deterministic.

## Open questions

- What makes a retrieval head form, or fail to form, in a given seed. Tracking the needle-attention
  profile during training could catch a failing run early.
- What sets extrapolation range once retrieval has formed.
- Scale beyond 209M and other data. Whether the 64M retrieval lottery disappears with size (all four
  209M dilution runs retrieved) needs more seeds to establish.
- Hybrids with token-local mixers such as [TriGLU](https://github.com/Royer-Research-Labs/TriGLU),
  which early learned-position runs suggested but did not establish.
- Generation beyond local fluency: whether dilution's samples stay consistent with information
  far back in a long prompt (the sampling judge sees only the last 4K tokens), and retrieval by
  generation (greedy decoding of the NIAH answer rather than likelihood scoring).
- A lower-level backward kernel (CUDA or a full Gluon pass) and a key-parallel prefill without
  checkpoint writes (`docs/kernels.md`).
