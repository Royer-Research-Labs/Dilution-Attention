# The dilution attention operator

A query's claim on a key is diluted by the demand that key has already
received from earlier queries. Everything is cumulative and causal by
construction:

    p         = causal_softmax(q k^T / sqrt(d))     # bids
    committed = exclusive_cumsum_cols(p)            # taken by earlier queries
    share     = p / (committed + p + eps)           # my bid vs cumulative demand
    A         = share / rowsum(share);  out = A @ v

One causal softmax, one exclusive cumulative sum down the query axis, one
renormalise. Ops: softmax, cumsum, elementwise -- O(T^2), the same complexity
class as standard attention.

**The operator has no parameters.** There are no learnable scalars and only one
softmax. The order asymmetry -- earlier positions face less
competition for a key than later ones -- falls out of the running sum alone:
query i divides by everything queries 0..i took from that key.

The `cumsum(dim=-2)` runs down columns, not along rows. For each key j it
accumulates how much attention mass earlier queries already placed on j, so a
key that earlier queries wanted presents a larger denominator to later ones.

The authoritative implementation is
`dilution.kernels.dilution_attention_reference` (four lines of math); the
fused Triton kernels are parity-pinned against it in tests/test_kernels.py.

## Variants the config can select

`model.share_cumsum` selects the denominator, and `model.bid` the bid function;
both exist to make the ablations in docs/signals.md expressible rather than to
offer tuning knobs. The operator above is the default.

| `share_cumsum` | denominator | what it is |
|---|---|---|
| `true` (default) | `committed + p + eps` | dilution: inclusive demand |
| `"exclusive"` | `committed + eps` | temporal attention's history (Sankaran 2016, Paulus 2018) on row-normalised bids |
| `false` | none (`attn = p`) | the softmax control: identical to standard attention |

The inclusive denominator is not cosmetic. In causal self-attention key j first
becomes visible to query j, and no earlier query can have claimed it, so the
exclusive `committed[j, j]` is *exactly* zero: that entry wins its row by 1/eps
and the attention matrix collapses to the identity. Including the current bid
bounds the diagonal at `p / (0 + p + eps) = 1`. A model trained with exclusive
history escapes the collapse in every layer but the first, which stays an
identity map. Against inclusive dilution trained with the same recipe (64M, 8K
context, LR 6e-4, one seed each) it loses 0.072 nats, and retrieval at 7,936
tokens falls from 1.00 to 0.62 (S18).

`eps` is 1e-9 throughout and is load-bearing only in the exclusive variant;
under the inclusive denominator every row already has `committed + p >= p > 0`.

