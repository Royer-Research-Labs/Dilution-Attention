"""Sampling-eval helpers: prompt selection, text statistics, paired statistics, cross-scoring,
the judge, and recomputing published summaries from per-sample metrics."""

from __future__ import annotations

import math
import sys
import types

import pytest
import torch

from dilution.config import ModelConfig
from dilution.model import DilutionLM
from dilution.sampling import (EOT, Judge, continuation_nll, copy_rate, mean_se, paired_bootstrap,
                               read_metrics, rep_n, sample_stats, select_end_positions, summarize,
                               truncate_at_eot, write_metrics)


class CharTokenizer:
    """One token per character, ids 1..vocab-1 (0 is padding)."""
    pad_token_id = 0

    def __init__(self, vocab: int = 64):
        self.vocab = vocab

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1 + ord(ch) % (self.vocab - 1) for ch in text]}


def bare_judge(tok, model=None, context=64):
    judge = Judge.__new__(Judge)
    judge.tok, judge.model, judge.device, judge.context, judge.token_budget = tok, model, "cpu", context, 256
    return judge


def test_judge_rejects_continuations_longer_than_its_context():
    judge = bare_judge(CharTokenizer(), context=16)
    with pytest.raises(ValueError, match="does not fit the judge context"):
        judge.score(["prompt"], ["x" * 16])        # 16 tokens leave no room for a prompt token


def test_judge_pins_the_revision_for_model_and_tokenizer(monkeypatch):
    calls = []

    class FakeTokenizer:
        pad_token_id, model_max_length = 0, 8

        @classmethod
        def from_pretrained(cls, name, revision=None):
            calls.append(("tokenizer", name, revision))
            return cls()

    class FakeModel:
        config = types.SimpleNamespace(max_position_embeddings=64, _commit_hash="abc123")

        @classmethod
        def from_pretrained(cls, name, revision=None, dtype=None):
            calls.append(("model", name, revision))
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    fake = types.ModuleType("transformers")
    fake.AutoTokenizer, fake.AutoModelForCausalLM = FakeTokenizer, FakeModel
    monkeypatch.setitem(sys.modules, "transformers", fake)
    judge = Judge("org/judge", device="cpu", revision="abc123")
    assert calls == [("tokenizer", "org/judge", "abc123"), ("model", "org/judge", "abc123")]
    assert judge.revision == "abc123"


def test_judge_score_matches_teacher_forced_nll_per_byte():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                      max_position_embeddings=64)
    model = transformers.LlamaForCausalLM(config).eval()
    tok = CharTokenizer()
    judge = bare_judge(tok, model, context=24)
    prompts = ["the long prompt that gets cut to the window", "short"]
    conts = [" and a continuation", " é!"]                  # multi-byte character: bytes != chars
    scores = judge.score(prompts, conts)
    for p, c, sc in zip(prompts, conts, scores):
        c_ids = tok(c)["input_ids"]
        p_ids = tok(p)["input_ids"][-(24 - len(c_ids)):]
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([p_ids + c_ids])).logits[0]
        logp = torch.log_softmax(logits.float(), -1)[len(p_ids) - 1:len(p_ids) - 1 + len(c_ids)]
        nll = -logp[torch.arange(len(c_ids)), torch.tensor(c_ids)].sum().item()
        assert sc.n_tokens == len(c_ids) and sc.n_bytes == len(c.encode("utf-8"))
        assert sc.bpb == pytest.approx(nll / math.log(2) / len(c.encode("utf-8")), rel=1e-4)


def test_published_summary_is_recomputed_from_the_metrics_file(tmp_path):
    """summarize() over the metric records written to disk gives exactly the in-memory summary."""
    import random
    rng = random.Random(1)
    records = []
    for gen, decs in (("human", ["reference"]), ("a", ["nucleus"]), ("b", ["nucleus"])):
        for L in (8, 16):
            for dec in decs:
                for i in range(12):
                    records.append({"gen": gen, "L": L, "dec": dec, "prompt": i, "end": 100 + i,
                                    "ids": [1, 2, 3], "text": "abc", "tokens": 3, "eot": False,
                                    "rep_2": rng.random(), "rep_3": 0.0, "rep_4": rng.random(),
                                    "looping": False, "copy_8": 0.0, "judge_bpb": 1 + rng.random(),
                                    "judge_tokens": 3, "bytes": 3, "xnll": {"a": rng.random(), "b": rng.random()}})
    kw = dict(models=["a", "b"], lengths=[8, 16], decodings=["nucleus"], baselines=["b"])
    table, paired = summarize(records, **kw)
    path = tmp_path / "set.metrics.jsonl.gz"
    write_metrics(path, records)
    back = read_metrics(path)
    assert "ids" not in back[0] and "text" not in back[0]
    assert summarize(back, **kw) == (table, paired)
    first = path.read_bytes()
    write_metrics(path, records)
    assert path.read_bytes() == first                      # byte-reproducible
    a_vs_b = [p for p in paired if p["model"] == "a" and p["L"] == 8][0]["vs"]["b"]
    assert a_vs_b["judge_bpb_minus_baseline"]["n"] == 12


def test_rep_n_and_copy_rate():
    assert rep_n([1, 2, 3, 4, 5], 2) == 0.0
    assert rep_n([7, 7, 7, 7, 7], 2) == pytest.approx(1 - 1 / 4)
    assert rep_n([1, 2, 1, 2, 1, 2], 2) == pytest.approx(1 - 2 / 5)
    assert math.isnan(rep_n([1, 2], 4))
    prompt = list(range(20))
    assert copy_rate(list(range(5, 15)), prompt, 8) == 1.0
    assert copy_rate([99] * 10, prompt, 8) == 0.0


def test_sample_stats_truncate_at_eot():
    ids, eot = truncate_at_eot([5, 6, EOT, 9, 9])
    assert ids == [5, 6, EOT] and eot
    s = sample_stats([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, EOT, 8, 8, 8], prompt=[0])
    assert s["eot"] and s["tokens"] == 13
    assert s["rep_4"] == pytest.approx(1 - 4 / 9)       # statistics exclude the EOT marker
    assert s["looping"] is True
    assert sample_stats(list(range(40)), prompt=[0])["looping"] is False


def test_select_end_positions_stays_inside_one_document():
    torch.manual_seed(0)
    tokens = torch.randint(0, 100, (20_000,))
    tokens[torch.randint(0, 20_000, (400,))] = EOT       # ~ one document boundary per 50 tokens
    read = lambda s, n: tokens[s:s + n].tolist()
    ends = select_end_positions(read, len(tokens), count=20, max_prompt=512, new_tokens=16,
                                seed=3, min_doc_tail=8)
    assert len(ends) == 20 == len(set(ends))
    for e in ends:
        assert e >= 512 and e + 16 <= len(tokens)
        assert EOT not in tokens[e - 8:e + 16].tolist()
    assert ends == select_end_positions(read, len(tokens), count=20, max_prompt=512, new_tokens=16,
                                        seed=3, min_doc_tail=8)


def test_mean_se_and_paired_bootstrap():
    m, se, n = mean_se([1.0, 2.0, 3.0, float("nan")])
    assert (m, n) == (2.0, 3) and se == pytest.approx(math.sqrt(1 / 3))
    a = [1.0 + 0.1 * i for i in range(50)]
    b = [x - 0.5 for x in a]
    r = paired_bootstrap(a, b)
    assert r["mean"] == pytest.approx(0.5) and r["lo"] == pytest.approx(0.5) and r["hi"] == pytest.approx(0.5)
    r = paired_bootstrap([0.0, 1.0] * 20, [0.5] * 40)
    assert r["lo"] < 0.0 < r["hi"]


@pytest.mark.parametrize("attention", ["dilution", "softmax"])
def test_continuation_nll_matches_teacher_forced_loss(attention):
    torch.manual_seed(4)
    model = DilutionLM(ModelConfig(vocab_size=64, n_layers=2, d_model=16, n_heads=2, context_length=32,
                                   mlp="swiglu", attention=attention)).eval()
    prompts = torch.randint(0, 60, (3, 10))
    conts = [[3, 4, 5, 6], [7, 8], [9, 10, 11]]           # ragged: padded internally
    got = continuation_nll(model, prompts, conts)
    for r, c in enumerate(conts):
        seq = torch.cat([prompts[r], torch.tensor(c)])[None]
        with torch.no_grad():
            logits, _ = model(seq, seq)
        logp = torch.log_softmax(logits[0, 9:9 + len(c)].float(), -1)
        want = -logp[torch.arange(len(c)), torch.tensor(c)].mean().item()
        assert got[r] == pytest.approx(want, abs=1e-5)
