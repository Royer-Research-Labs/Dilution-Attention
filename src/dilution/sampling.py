"""Open-ended sampling evaluation: do dilution models generate text as well as softmax models?

The loss and NIAH evaluations are teacher-forced or likelihood-scored; this one makes each model
write. Every model continues the *same* prompts with the *same* random streams, so the
comparison between models is paired prompt by prompt.

Prompts come from the validation stream. One set of end positions is drawn so that the prompt
ends at least `min_doc_tail` tokens into a document and the next `new_tokens` tokens (the human
reference) stay inside it; each prompt length L then takes the L tokens before the same end
position. Across lengths the local context is identical and only the amount of earlier context
(the rest of the document, then earlier documents) changes.

Per sample (continuation truncated after its first end-of-text token):

    judge_bpb   bits per UTF-8 byte of the continuation under an external judge LM, conditioned on
                the last `judge_context` judge tokens of the prompt (lower = more plausible; the
                same number for the human continuation is the reference point)
    rep_n       1 - unique n-grams / n-grams within the continuation (Welleck et al. 2020 seq-rep-n)
    looping     rep_4 >= 0.5: half the 4-grams are repeats, i.e. the sample is stuck in a loop
    copy_8      fraction of continuation 8-grams that occur verbatim in the prompt
    eot         the model ended the document inside the continuation
    xnll[m]     per-token NLL of the continuation under our model m, given the full prompt
                (the cross-scoring matrix; the diagonal is each model's surprise at its own text)

A low judge score alone can mean bland or repetitive text, so read it together with rep_n.

`summarize` turns per-sample metric records into the published aggregates and paired bootstrap
intervals; `scripts/verify_samples.py` reruns it on the published `<set>.metrics.jsonl.gz` files.
"""
from __future__ import annotations

import gzip
import io
import json
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch

EOT = 50256                 # GPT-2 <|endoftext|>: the document separator in the token streams
GPT2_VOCAB = 50257          # real GPT-2 ids; embedding rows above this are padding


# ---- prompt selection ---------------------------------------------------------------------

def select_end_positions(read: Callable[[int, int], Sequence[int]], num_tokens: int, *, count: int,
                         max_prompt: int, new_tokens: int, seed: int, min_doc_tail: int = 64,
                         max_tries: int = 100_000) -> list[int]:
    """End positions e with tokens[e - min_doc_tail : e + new_tokens] free of end-of-text, so the
    prompt ends inside a document and the reference continuation stays in it. `read(start, n)`
    returns n tokens from the stream; e ranges over [max_prompt, num_tokens - new_tokens)."""

    lo, hi = max_prompt, num_tokens - new_tokens
    if hi <= lo:
        raise ValueError(f"stream of {num_tokens} tokens is too short for {max_prompt}+{new_tokens}")
    rng = random.Random(seed)
    ends: list[int] = []
    seen: set[int] = set()
    for _ in range(max_tries):
        if len(ends) == count:
            break
        e = rng.randrange(lo, hi)
        if e in seen:
            continue
        seen.add(e)
        window = read(e - min_doc_tail, min_doc_tail + new_tokens)
        if EOT not in list(window):
            ends.append(e)
    if len(ends) < count:
        raise RuntimeError(f"found only {len(ends)} of {count} prompt positions")
    return sorted(ends)


# ---- per-sample text statistics -----------------------------------------------------------

def truncate_at_eot(ids: Sequence[int]) -> tuple[list[int], bool]:
    """Continuation up to and including its first end-of-text token, and whether it had one."""
    ids = list(ids)
    if EOT in ids:
        return ids[: ids.index(EOT) + 1], True
    return ids, False


def ngrams(ids: Sequence[int], n: int) -> list[tuple[int, ...]]:
    return [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]


def rep_n(ids: Sequence[int], n: int) -> float:
    """1 - unique n-grams / n-grams (0 = no repeated n-gram; nan if shorter than n)."""
    grams = ngrams(ids, n)
    return 1.0 - len(set(grams)) / len(grams) if grams else float("nan")


def copy_rate(ids: Sequence[int], prompt: Sequence[int], n: int = 8) -> float:
    """Fraction of the continuation's n-grams that appear verbatim in the prompt."""
    grams = ngrams(ids, n)
    if not grams:
        return float("nan")
    source = set(ngrams(prompt, n))
    return sum(g in source for g in grams) / len(grams)


def sample_stats(cont: Sequence[int], prompt: Sequence[int]) -> dict:
    ids, eot = truncate_at_eot(cont)
    body = ids[:-1] if eot else ids                       # statistics over text, not the marker
    r4 = rep_n(body, 4)
    return {"tokens": len(ids), "eot": eot, "rep_2": rep_n(body, 2), "rep_3": rep_n(body, 3),
            "rep_4": r4, "looping": bool(r4 == r4 and r4 >= 0.5), "copy_8": copy_rate(body, prompt, 8)}


# ---- aggregation --------------------------------------------------------------------------

def mean_se(xs: Iterable[float]) -> tuple[float, float, int]:
    xs = [float(x) for x in xs if x == x]                 # drop nan
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(xs) / n
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n) if n > 1 else float("nan")
    return m, se, n


def paired_bootstrap(a: Sequence[float], b: Sequence[float], *, resamples: int = 2000,
                     seed: int = 0) -> dict:
    """Mean of a - b over pairs where both are finite, with a percentile 95% bootstrap interval."""
    d = [x - y for x, y in zip(a, b) if x == x and y == y]
    if not d:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = random.Random(seed)
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return {"mean": sum(d) / n, "lo": means[int(0.025 * resamples)],
            "hi": means[int(0.975 * resamples) - 1], "n": n}


# ---- the judge ----------------------------------------------------------------------------

@dataclass
class JudgeScore:
    nll_nats: float
    n_bytes: int
    n_tokens: int

    @property
    def bpb(self) -> float:
        return self.nll_nats / math.log(2) / self.n_bytes if self.n_bytes else float("nan")


class Judge:
    """An external causal LM that scores continuation text given prompt text. The prompt and
    continuation are tokenized separately (as lm-eval-harness does) and the prompt is cut to its
    last `context - len(continuation)` tokens, so the judge sees local context only. A
    continuation that does not fit the context with at least one prompt token is rejected rather
    than truncated: its byte count would no longer match the tokens scored.

    `revision` pins the model and tokenizer to one Hugging Face commit; `self.revision` records
    the commit actually loaded."""

    def __init__(self, name: str, device: str = "cuda", context: int = 4096, token_budget: int = 32768,
                 revision: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision)
        self.tok.model_max_length = 1 << 30   # prompts are cut to `context` below, not by the tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, dtype=torch.bfloat16).to(device).eval()
        self.device = device
        self.context = min(context, int(getattr(self.model.config, "max_position_embeddings", context)))
        self.token_budget = token_budget
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision

    @torch.no_grad()
    def score(self, prompts: Sequence[str], conts: Sequence[str]) -> list[JudgeScore]:
        items = []
        for p, c in zip(prompts, conts):
            c_ids = self.tok(c, add_special_tokens=False)["input_ids"]
            if len(c_ids) >= self.context:
                raise ValueError(
                    f"a continuation of {len(c_ids)} judge tokens does not fit the judge context of "
                    f"{self.context} with any prompt; use fewer new tokens or a larger judge context")
            p_ids = self.tok(p, add_special_tokens=False)["input_ids"]
            keep = max(1, self.context - len(c_ids))
            items.append((p_ids[-keep:], c_ids, len(c.encode("utf-8"))))
        out: list[JudgeScore | None] = [None] * len(items)
        order = sorted(range(len(items)), key=lambda i: len(items[i][0]) + len(items[i][1]))
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        i = 0
        while i < len(order):
            longest = len(items[order[i]][0]) + len(items[order[i]][1])
            batch = [order[i]]
            while i + len(batch) < len(order):
                nxt = order[i + len(batch)]
                longest = max(longest, len(items[nxt][0]) + len(items[nxt][1]))
                if longest * (len(batch) + 1) > self.token_budget:
                    break
                batch.append(nxt)
            seqs = [items[j][0] + items[j][1] for j in batch]
            width = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), width), pad, dtype=torch.long)
            mask = torch.zeros((len(seqs), width), dtype=torch.long)
            for r, s in enumerate(seqs):
                ids[r, : len(s)] = torch.tensor(s)
                mask[r, : len(s)] = 1
            # hidden states for the whole batch, the LM head only at continuation positions
            hidden = self.model.get_decoder()(input_ids=ids.to(self.device),
                                               attention_mask=mask.to(self.device)).last_hidden_state
            head = self.model.get_output_embeddings()
            for r, j in enumerate(batch):
                p_len, c_ids, n_bytes = len(items[j][0]), items[j][1], items[j][2]
                if not c_ids:
                    out[j] = JudgeScore(0.0, n_bytes, 0)
                    continue
                logp = torch.log_softmax(head(hidden[r, p_len - 1:p_len - 1 + len(c_ids)]).float(), dim=-1)
                tgt = torch.tensor(c_ids, device=logp.device)
                nll = -logp.gather(-1, tgt[:, None]).sum().item()
                out[j] = JudgeScore(nll, n_bytes, len(c_ids))
            del hidden
            i += len(batch)
        return out  # type: ignore[return-value]


# ---- published per-sample metrics and the summary built from them ------------------------

METRIC_FIELDS = ("gen", "L", "dec", "prompt", "end", "tokens", "eot", "rep_2", "rep_3", "rep_4",
                 "looping", "copy_8", "judge_bpb", "judge_tokens", "bytes", "xnll")
SUMMARY_METRICS = ("judge_bpb", "rep_2", "rep_4", "looping", "copy_8", "eot", "tokens")
PAIRED_METRICS = ("judge_bpb", "rep_4")


def metrics_record(rec: dict) -> dict:
    """A sample's record without its token ids and text: everything the summary is computed from."""
    return {k: rec[k] for k in METRIC_FIELDS if k in rec}


def write_metrics(path: str | Path, records: Iterable[dict]) -> None:
    """Per-sample metric records as gzipped JSON lines, byte-reproducible (no gzip timestamp)."""
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz, \
            io.TextIOWrapper(gz, encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(metrics_record(rec)) + "\n")


def read_metrics(path: str | Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summarize(records: Iterable[dict], *, models: Sequence[str], lengths: Sequence[int],
              decodings: Sequence[str], baselines: Sequence[str], human: str = "human"):
    """(table, paired) from per-sample records: per (generator, length, decoding) means and
    standard errors, cross-scores, and paired bootstrap differences against each baseline and
    against the human continuation. Deterministic (fixed bootstrap seed), so the published
    summaries can be recomputed exactly from the published metric records."""
    by_key = {(r["gen"], r["L"], r["dec"], r["prompt"]): r for r in records}
    n_prompts = 1 + max(k[3] for k in by_key)
    groups: dict[tuple[str, int, str], list[dict]] = {}
    for (gen, L, dec, i) in sorted(by_key, key=lambda k: (k[1], k[2], k[0], k[3])):
        groups.setdefault((gen, L, dec), []).append(by_key[(gen, L, dec, i)])
    table = []
    for (gen, L, dec), recs in groups.items():
        row = {"gen": gen, "L": L, "dec": dec, "n": len(recs)}
        for m in SUMMARY_METRICS:
            mu, se, _ = mean_se(float(r[m]) for r in recs if m in r)
            row[m], row[m + "_se"] = mu, se
        for scorer in models:
            row[f"xnll[{scorer}]"] = mean_se(r.get("xnll", {}).get(scorer, float("nan")) for r in recs)[0]
        table.append(row)

    def per_prompt(gen, L, dec, m):
        return [float(by_key[(gen, L, dec, i)].get(m, float("nan"))) if (gen, L, dec, i) in by_key
                else float("nan") for i in range(n_prompts)]

    paired = []
    for L in lengths:
        for dec in decodings:
            for name in models:
                if (name, L, dec) not in groups:
                    continue
                entry = {"L": L, "dec": dec, "model": name, "vs": {}}
                for b in baselines:
                    if b != name and (b, L, dec) in groups:
                        entry["vs"][b] = {f"{m}_minus_baseline": paired_bootstrap(
                            per_prompt(name, L, dec, m), per_prompt(b, L, dec, m)) for m in PAIRED_METRICS}
                entry["judge_bpb_minus_human"] = paired_bootstrap(
                    per_prompt(name, L, dec, "judge_bpb"), per_prompt(human, L, "reference", "judge_bpb"))
                paired.append(entry)
    return table, paired


# ---- scoring with our own models ----------------------------------------------------------

@torch.no_grad()
def continuation_nll(model, prompts: torch.Tensor, conts: Sequence[Sequence[int]]) -> list[float]:
    """Per-token NLL (nats) of each continuation under `model`, given its prompt. prompts is
    (B, L); conts are the (EOT-truncated) continuations, scored up to and including the EOT.
    One forward over prompt + continuation; the LM head runs on the continuation positions only."""
    import torch.nn.functional as F

    batch, length = prompts.shape
    width = max(len(c) for c in conts)
    # ragged rows are padded after their continuation; causal attention never lets the pad (id 0)
    # reach a scored position
    idx = torch.zeros((batch, length + width), dtype=torch.long, device=prompts.device)
    idx[:, :length] = prompts
    tgt = torch.full((batch, width), -100, dtype=torch.long, device=prompts.device)
    for r, c in enumerate(conts):
        c = torch.tensor(list(c), dtype=torch.long, device=prompts.device)
        idx[r, length:length + len(c)] = c
        tgt[r, : len(c)] = c
    x = model.transformer.drop(model._embed(idx, 0))
    for block in model.transformer.h:
        x = block(x)
    x = model.transformer.ln_f(x[:, length - 1:length - 1 + width])   # predicts continuation tokens
    logits = model.lm_head(x).float()
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100,
                          reduction="none").view(batch, width)
    valid = (tgt != -100).float()
    per_row = (nll * valid).sum(-1) / valid.sum(-1).clamp(min=1)
    return [float(v) if n > 0 else float("nan") for v, n in zip(per_row.tolist(), valid.sum(-1).tolist())]
