# Related work and contribution boundary

Literature reviewed through 2026-09-13. This review positions both the
attention operator and its fused implementation. It is deliberately
conservative: a literature and public-code search cannot prove that an exact
precedent does not exist, and this document is not a patentability or
freedom-to-operate opinion.

## Mechanism being positioned

For causal score matrix `z`, dilution attention computes

```text
p_ij     = exp(z_ij) / sum(k <= i) exp(z_ik)
d_ij     = sum(r <= i) p_rj
share_ij = p_ij / (d_ij + eps)
A_ij     = share_ij / sum(k <= i) share_ik
out_i    = sum(j <= i) A_ij v_j
```

Thus each query first distributes one unit of bid mass over its visible keys.
Its claim on key `j` is then divided by the **inclusive cumulative normalized
bid** that the key has received from queries through `i`, and the resulting
shares are normalized across the row. The implementation in
[`operator.md`](operator.md) writes `d_ij` as an exclusive cumulative sum plus
the current bid; the two forms are identical.

The details in bold matter. The operator is invariant to adding an arbitrary
constant to one query's logits, each pre-renormalization share lies in
`[0, 1]`, and the first row follows the same equation as every other row. The
operator adds no learned parameters. It remains quadratic in sequence length,
although cached decoding needs only the ordinary KV cache plus one cumulative
demand scalar per cached key and head.

## The closest equation-level precedent: temporal attention

[Temporal Attention Model for Neural Machine Translation](https://arxiv.org/abs/1608.02927)
(Sankaran et al., 2016) is the closest located prior work. For decoder step
`i > 1` and source position `j`, it computes

```text
history_ij = sum(r < i) exp(z_rj)
b_ij       = exp(z_ij) / history_ij
A_ij       = b_ij / sum(k) b_ik
```

and defines the first decoder step separately as `b_1j = exp(z_1j)`.
[A Deep Reinforced Model for Abstractive Summarization](https://openreview.net/forum?id=HkAClQgA-)
(Paulus et al., ICLR 2018) uses the same intra-temporal ratio. This is not merely a
shared motivation: both temporal attention and dilution divide the current
claim on a location by claims made at earlier output steps and then normalize
the resulting row.

Ignoring `eps`, there is also an exact algebraic reduction. Define the
row-calibrated logits `z'_ij = log(p_ij) = z_ij - logsumexp_k(z_ik)`. Then
dilution's share is

```text
share_ij = exp(z'_ij) / sum(r <= i) exp(z'_rj).
```

Dilution is therefore inclusive temporal attention applied to row-calibrated
logits. The operator distinction is row calibration, inclusion of the current
bid, and use inside causal Transformer self-attention; division by temporal
attention history has clear precedent.

The distinction is narrower than “attention with memory,” but it is material:

| Property | Temporal / intra-temporal attention | Dilution attention |
| --- | --- | --- |
| Setting | RNN decoder cross-attention over a fixed encoded source | Causal Transformer self-attention whose visible key set grows with the query position |
| Quantity accumulated per key | Unnormalized `exp(z_rj)` alignments | Row-normalized causal probabilities `p_rj` |
| Denominator | Earlier steps only; first step is special-cased | Earlier plus current bid at every step |
| Per-query logit-shift invariance | No: a shift of row `r` rescales its contribution to every future denominator | Yes: row softmax removes that degree of freedom before accumulation |
| Bound before final row normalization | The ratio can exceed one or become arbitrarily large | `share_ij <= 1` up to epsilon |
| Intended role | Prevent repeated source coverage during task-specific sequence generation | Replace attention normalization inside every selected causal language-model layer |
| Implementation studied | Sequential decoder recurrence | Parallel training, prefill, exact backward, and cached autoregressive decode |

[Get To The Point](https://aclanthology.org/P17-1099/) (See et al., 2017)
is especially informative negative evidence. It describes temporal attention
as dividing the current distribution by accumulated previous attention, but
reports that this post-hoc intervention distorted the attention signal in its
summarization system and preferred learned coverage instead. That result makes
the two normalization choices above necessary controls rather than cosmetic
implementation details.

A source-attribution wrinkle is worth preserving. Paulus et al. and See et al.
also attribute a summarization use of temporal attention to
[Nallapati et al. (2016)](https://aclanthology.org/K16-1028/), but the method
section of the published CoNLL paper available from ACL Anthology presents
feature-rich, pointer, and hierarchical-attention models and does not state
the temporal ratio above. This review therefore uses Sankaran et al. and
Paulus et al. as the auditable equation-level precedents.

## Closely related research families

| Family and primary sources | Material overlap | Distinction from this project |
| --- | --- | --- |
| Attention coverage and fertility: [Tu et al. (2016)](https://aclanthology.org/P16-1008/), [Mi et al. (2016)](https://aclanthology.org/D16-1096/), [See et al. (2017)](https://aclanthology.org/P17-1099/) | Maintains a per-source-position state derived from previous attention to prevent over-translation, under-translation, or repetition. See et al. use the simple sum of previous normalized attention as coverage. | Coverage is fed back into a learned scoring function, updated by a learned recurrent state, added as an auxiliary loss, or some combination of these. Dilution deterministically divides the already-normalized bid by its inclusive cumulative demand and has no coverage parameters or loss. |
| Coverage regularization: [Show, Attend and Tell](https://proceedings.mlr.press/v37/xuc15.html) (Xu et al., 2015) | Encourages each image location's attention, summed across decoder steps, to approach a target total. This is an early column-usage constraint on an attention matrix. | It is a soft training penalty for cross-attention. It does not alter each current attention row with a causal prefix demand. |
| Globally balanced attention: [Sinkformers](https://proceedings.mlr.press/v151/sander22a.html) (Sander et al., 2022) | Alternates row and column normalization so tokens compete for attention and the final matrix is approximately doubly stochastic. | Sinkhorn balances the complete matrix through multiple global iterations. Dilution performs one directional prefix division between two row normalizations. Exact double stochasticity is also incompatible with nontrivial square causal attention: a lower-triangular matrix whose rows and columns all sum to one is the identity. |
| Flow-based source competition: [Flowformer](https://proceedings.mlr.press/v162/wu22m.html) (Wu et al., 2022) | Treats values and outputs as sources and sinks, then uses incoming- and outgoing-flow conservation to create competition and allocation across the two attention axes. | Flow-Attention factorizes a nonnegative similarity to obtain a linear-time operator and gates conserved capacities. It neither preserves exact causal softmax nor treats earlier query mass as consumption that discounts a key's later use. |
| Competitive assignment: [Slot Attention](https://proceedings.neurips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html) (Locatello et al., 2020) | Normalizes assignments over slots so slots compete for each input, then divides by each slot's total assignment to form a weighted mean. | It is iterative set-to-slot cross-attention with learned recurrent slot updates over a static input. It has no causal prefix, and its two-axis normalization expresses clustering rather than prior use by earlier queries. |
| Cumulative attention for cache policy: [H2O](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6ceefa7b15572587b78ecfcebb2827f8-Abstract-Conference.html) (Zhang et al., 2023) | Sums attention received by cached tokens, exposing the same per-key “how much attention has this position received?” statistic. | H2O uses the statistic outside the attention operator to retain heavy-hitter KV entries at inference. Dilution uses it during training and inference to reduce a repeatedly claimed key's share; the direction of the intervention is therefore different. |
| Attention sinks and forced allocation: [StreamingLLM](https://openreview.net/forum?id=NG7sS51zVF) (Xiao et al., 2024), [StableMask](https://proceedings.mlr.press/v235/yin24a.html) (Yin et al., 2024), and [Gated Attention](https://proceedings.neurips.cc/paper_files/paper/2025/hash/904e89bb4e632e75fb47f093b620b257-Abstract-Conference.html) (Qiu et al., 2025) | Diagnose or mitigate excessive allocation to initial or semantically weak tokens. This is a natural behavioral comparison because cumulative reuse pressure should interact with sinks. | StreamingLLM preserves sink tokens for cache stability; StableMask adds pseudo-attention mass through the causal mask; Gated Attention gates the SDPA output per head and query. None divides a key's current attention by its cumulative prior use. |
| Query-dependent sparsity and long-context flattening: [Selective Attention](https://proceedings.neurips.cc/paper_files/paper/2024/hash/14fc4a68da97a3d31eb11c642b0b10fc-Abstract-Conference.html) (Zhang et al., 2024) and [TransNormer](https://aclanthology.org/2022.emnlp-main.473/) (Qin et al., 2022) | Both use the phrase “attention dilution” for attention becoming too diffuse as context grows, and modify sharpness or local structure. | Their dilution means low per-query selectivity. This repository's name refers to cumulative per-key demand: a popular key dilutes later claims. The terms should not be treated as the same mechanism. |

Coverage, temporal attention, doubly stochastic attention, and competitive slot
assignment establish that attention history and competition over the opposite
axis are mature ideas. The narrow question is the exact causal normalization
used here and what it does as a repeated Transformer language-model primitive.

## Position encoding and length extrapolation precedents

Position encoding is a separate causal variable in this repository's results,
not part of the dilution equation. The literature makes two contribution
boundaries especially important:

| Work | Relevance to this project | Boundary |
| --- | --- | --- |
| [The Impact of Positional Encoding on Length Generalization](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4e85362c02172c0c6567ce593122d31c-Abstract-Conference.html) (Kazemnejad et al., 2023) | Finds strong length generalization from decoder-only Transformers without explicit positional encoding. | The project's NoPE behavior and the finding that RoPE can limit extrapolation have clear precedent. |
| [Rope to Nope and Back Again](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5c9ab393551b7a39b4c02d88fe5e7e69-Abstract-Conference.html) (Yang et al., 2025) | Studies hybrid RoPE/NoPE layer stacks for long context and combines local RoPE attention with global NoPE attention. | Alternating RoPE and NoPE layers is not a project novelty. The result here is the interaction between that positional layout, dilution versus softmax operators, and full-attention/hybrid placement. |
| [Position Interpolation](https://arxiv.org/abs/2306.15595) (Chen et al., 2023) and [YaRN](https://proceedings.iclr.cc/paper_files/paper/2024/hash/874a4d89f2d04b4bcf9a2c19545cf040-Abstract-Conference.html) (Peng et al., 2024) | Establish RoPE rescaling and interpolation as context-extension methods, including passkey-retrieval evaluation. | Any 2x–4x extrapolation result must report the position-scaling rule separately from the attention operator and include the corresponding softmax control. |

Accordingly, `rope_alt`, the alt6 RoPE/NoPE layout, and NTK-style scaling are
controls or compositions with prior positional methods. They strengthen the
operator study by separating attention behavior from position behavior; they
do not broaden the operator's novelty claim.

## Kernel and systems precedents

The exact operator has an additional prefix dependence down each key column,
so a standard fused SDPA kernel cannot implement it as a pointwise score
modifier. The current kernels use online row-softmax statistics, tiled
recomputation, query-block carry checkpoints, a reverse suffix recurrence in
backward, and an O(T) decode accumulator. These choices sit next to several
well-developed systems lines:

| Work | Lesson that transfers | Limit of the analogy |
| --- | --- | --- |
| [FlashAttention](https://proceedings.neurips.cc/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html), [FlashAttention-2](https://proceedings.iclr.cc/paper_files/paper/2024/file/98ed250b203d1ac6b24bbcf263e3d4a7-Paper-Conference.pdf), and [FlashAttention-3](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7ede97c3e082c6df10a8d6103a2eebd2-Abstract-Conference.html) | IO-aware tiling, online softmax, score recomputation, sequence-parallel work partitioning, warp specialization, and overlap of matrix and scalar work are the baseline design principles for an exact attention kernel. | Dilution's column-prefix state couples query blocks, and its backward needs a reverse per-key suffix. Those dependencies prevent a direct call to an SDPA implementation. |
| [FlashAttention-4](https://arxiv.org/abs/2603.05451) (Zadouri et al., 2026, preprint) | On Blackwell, non-MMA work, shared-memory traffic, exponentials, register pressure, and atomics become primary limits. It uses asynchronous MMA, tensor memory, software exponentials, conditional rescaling, and two-CTA backward work. | It targets B200/GB200 and standard attention. Hardware-specific primitives must be checked on the project's RTX PRO 6000 before adopting the schedule. |
| [Gated Linear Attention](https://proceedings.mlr.press/v235/yang24ab.html) (Yang et al., 2024) and [Tiled Flash Linear Attention](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6cb81234ab47027e991728ed7dd76735-Abstract-Conference.html) (Beck et al., 2025) | Chunkwise recurrent/parallel duality, state passing, reverse recurrences, and an added level of sequence tiling reduce the number of states materialized to HBM. This is the closest systems precedent for the demand and suffix scans. | Their recurrent state comes from associative linear attention. Dilution must first compute a causal row softmax, so the complete attention operator does not factor into a fixed-size linear recurrence. |
| [FlexAttention](https://proceedings.mlsys.org/paper_files/paper/2025/file/61a9278dfef5f871b5e472389f8d6fa1-Paper-Conference.pdf), [AttentionEngine](https://arxiv.org/abs/2502.15349), and [Flashlight](https://arxiv.org/abs/2511.02043) | Compiler and template systems can generate optimized kernels for increasingly broad attention programs and are worth using as implementation baselines. AttentionEngine explicitly includes parallel and recurrent templates; Flashlight claims data-dependent programs beyond score modifiers. | FlexAttention's scalar `score_mod` cannot express a post-softmax reduction across earlier queries. The other two systems require a feasibility prototype before assuming they can schedule dilution's row-softmax-plus-column-scan dependency. |

This systems literature supports the conclusion already reached by the local
profiles in [`kernels.md`](kernels.md): another Triton launch-parameter sweep
is unlikely to remove the remaining gap. The highest-value kernel directions
are a lower-level Blackwell schedule that can control live-tile placement and
producer/consumer overlap, or a hierarchical sequence decomposition that
stores fewer carry checkpoints.

## Claims ruled out by the literature

The project should not claim:

- the first attention mechanism to remember previous attention;
- the first deterministic division by accumulated per-key alignments;
- the first coverage, fertility, or anti-repetition attention mechanism;
- the first column-balanced or competitive attention normalization;
- the first use of cumulative attention mass in a causal Transformer system;
- the first parameter-free modification intended to reduce attention sinks;
- the first alternating RoPE/NoPE Transformer or position-scaling method;
- the first IO-aware, tiled, recomputing, scan-based attention kernel; or
- a linear-time or subquadratic training operator.

A narrower description is supported by the search:

> Dilution attention adapts the temporal-attention idea of reducing claims on
> previously used keys to causal Transformer self-attention, but accumulates
> row-normalized bids and divides by inclusive cumulative demand before a
> final row normalization. The contribution is the exact normalization, its
> controlled causal-language-model study across context, scale, placement,
> position encoding, and retrieval, and exact fused training, prefill, and
> decode implementations.

No located paper applied this exact normalized-inclusive equation throughout
a causal Transformer language model. That is a search finding, not proof of
priority. “Temporal-attention-style causal self-attention with normalized
inclusive demand” is safer and more informative than an unqualified claim of
a new cumulative-attention concept.

## Direct differentiation controls suggested by prior work

**Status: half run (S18).** The row-normalized half of the factorial below is
measured; the raw-score half is not. The `expshift` bid is `exp(a) / SHIFT`, so
`p / (committed + p)` has the SHIFT cancel and is algebraically raw inclusive
temporal attention. Our fp32 implementation of it overflowed at step 400 (S4),
but that is a failure of the implementation, not a measurement of the
operator: the stable per-key cumulative `logsumexp` form recommended below has
not been built, so both raw-score cells remain open. Both exclusive cells share
a structural degeneracy this review did not anticipate: key j first becomes
visible to query j, so `sum(r < j)` over its prior claims is *exactly* zero and
that entry wins its row by 1/eps, collapsing attention to the identity at
initialization (measured diagonal 1.0000, off-diagonal 3e-10). The published
first-row special case turns out to be unnecessary rather than merely awkward
-- the row normalise cancels eps and reproduces it. Whether training escapes
the degeneracy is measured only for the normalized cell. Trained to completion
(64M, 8K context, 600M tokens, learned positions, LR 6e-4, one seed per cell),
it escapes the collapse in every layer but the first, which stays an identity
map (diagonal 0.911, entropy 0.28 nats). Against dilution trained with the same
recipe it loses 0.072 nats (3.6771 against 3.6055), and retrieval at 7,936
tokens falls from 1.00 to 0.62. The share division as a whole is worth 0.177
nats against a no-cumsum control with the same recipe (3.7829), so exclusive
history keeps about 60% of the loss gain and the inclusive denominator
supplies the rest. An earlier version of this paragraph reported 0.023 and
0.128 nats; both compared these 6e-4 cells against dilution trained at 3e-4.

The strongest missing mechanism control is the exact temporal-attention
predecessor in the same Transformer harness. A compact factorial isolates the
two differences:

| Current quantity | Historical denominator | Purpose |
| --- | --- | --- |
| `exp(z_ij)` | `sum(r < i) exp(z_rj)` | Sankaran/Paulus temporal attention; requires the published first-row special case |
| `exp(z_ij)` | `sum(r <= i) exp(z_rj)` | Isolates inclusive demand while retaining raw-score history |
| `p_ij` | `sum(r < i) p_rj` | Isolates normalized bids while retaining exclusive history |
| `p_ij` | `sum(r <= i) p_rj` | Dilution attention |

The raw-score arms should be implemented with a stable per-key cumulative
`logsumexp`; subtracting a different row maximum before accumulation changes
their operator. The exclusive normalized arm also needs an explicit first-row
definition and can produce unbounded ratios. These are properties to measure,
not silently repair.

**Status: implemented (`scripts/attention_structure.py`), reported in S18 and
extended to every 209M checkpoint with off-diagonal statistics in S20.** The
measurement samples three layers (first, middle, last) of one 1,024-token
validation window per checkpoint and averages over heads, so a sink confined to
one head or an unsampled layer would not show. Within that scope the sink
result holds and is sharper at scale. No sampled dilution layer puts more than
0.028 of its attention on token 0. The position-matched comparison is the
no-position-encoding pair: 209M softmax without position encoding puts 0.14-0.75
there in its middle layer (0.75 in the seed whose training hit a gradient-spike
burst, S24), against 0.014-0.028 for dilution. With RoPE, 209M softmax reaches
0.25-0.32. In the alt6 hybrid the softmax layers carry a 5-10% sink and the
sampled dilution layer does not. That is consistent with the operator setting
the sink, but it does not isolate the operator: the hybrid's softmax layers
also carry RoPE, and the sampled dilution layer is the last layer while the
sinks sit in the middle. The S18 "~20x more selective, ~4x more balanced"
figures were measured at layer 0 and do not hold as stated. The top-1 weight
there is the diagonal. The no-cumsum and softmax controls attend uniformly at
layer 0 (Gini 0.50 and entropy 5.9 are the uniform-causal values), while
dilution's layer 0 is recency-weighted. At depth, dilution is comparably
selective per query and less concentrated across keys.

The most useful behavioral measurements are per-layer and per-head cumulative
column mass, first-token/sink mass, maximum column load, column-load Gini,
attention entropy, attended-token age, and gradient tails. H2O supplies the
heavy-hitter comparison; StreamingLLM and StableMask supply sink diagnostics;
coverage work supplies overuse/underuse interpretations. Reporting these with
the existing loss and NIAH results would show whether dilution improves by
flattening key utilization, by preserving selected old keys, or by some other
effect.

For position, the decisive controls remain matched softmax and dilution models
under the same NoPE, RoPE, alternating RoPE/NoPE, and extrapolation scaling.
The RNoPE precedent should be cited wherever `rope_alt` or alt6 is presented.

For the kernel, the literature suggests this order:

1. Prototype whether a CuTe-DSL/CUTLASS path on the actual workstation GPU can
   keep the backward's live tiles outside the register file and overlap the
   reverse suffix work with MMA, using FlashAttention-4 as the schedule model.
2. Test a two-level carry layout inspired by Tiled Flash Linear Attention to
   reduce `(T / 32) x T` checkpoint traffic without increasing score
   recomputation enough to lose the saving.
3. Check AttentionEngine and Flashlight with a minimal forward-only expression
   before investing in another handwritten implementation; reject them early
   if their reduction algebra cannot cross the row-softmax boundary.
4. Keep exact temporal-attention and inclusive/exclusive controls in the
   reference path first. Only build custom kernels for variants that survive a
   short quality and stability screen.

## Search coverage and limits

The search followed equation terms and citation chains across ACL Anthology,
PMLR, NeurIPS, ICLR/OpenReview, MLSys, and arXiv. Queries covered temporal and
intra-temporal attention, coverage and fertility, cumulative attention,
column normalization, doubly stochastic attention, competitive attention,
attention sinks, RoPE/NoPE hybrids, scan and recurrent attention kernels, and
custom-attention compilers. Primary papers are linked above. Generic uses of
“temporal attention” for time-series axes or document timestamps, including
[Temporal Attention for Language Models](https://arxiv.org/abs/2202.02093),
are terminological collisions and do not use cumulative prior attention.
