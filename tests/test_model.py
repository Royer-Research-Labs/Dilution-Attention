"""Model tests: arms, causality, parameter matching, cached decode parity."""

from __future__ import annotations

import pytest
import torch

from dilution.config import ModelConfig
from dilution.model import (
    CausalSelfAttention,
    DilutionAttention,
    DilutionLM,
)


def tiny(**overrides) -> ModelConfig:
    values = dict(vocab_size=32, n_layers=2, d_model=16, n_heads=2,
                  context_length=16, mlp="swiglu")
    values.update(overrides)
    return ModelConfig(**values)


@pytest.mark.parametrize("attention", ["dilution", "softmax"])
def test_arms_train_and_are_finite(attention):
    torch.manual_seed(0)
    model = DilutionLM(tiny(attention=attention))
    idx = torch.randint(0, 32, (2, 12))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 12, 32)
    assert torch.isfinite(loss)
    loss.backward()


def test_layer_attention_dispatch_and_param_match():
    mixed = DilutionLM(tiny(layer_attention=["dilution", "softmax"]))
    assert type(mixed.transformer.h[0].attn) is DilutionAttention
    assert type(mixed.transformer.h[1].attn) is CausalSelfAttention

    torch.manual_seed(0)
    n_soft = DilutionLM(tiny(attention="softmax")).num_parameters()
    n_dil = DilutionLM(tiny(attention="dilution")).num_parameters()
    assert n_dil == n_soft  # the operator adds no parameters at all


@pytest.mark.parametrize("attention", ["dilution", "softmax"])
def test_causality(attention):
    torch.manual_seed(0)
    model = DilutionLM(tiny(attention=attention)).eval()
    idx = torch.randint(0, 32, (1, 12))
    changed = idx.clone()
    changed[:, 5:] = torch.randint(0, 32, (1, 7))
    with torch.no_grad():
        a, _ = model(idx, idx)
        b, _ = model(changed, changed)
    assert torch.allclose(a[:, :5], b[:, :5], atol=1e-5)


def _greedy_full_recompute(model, idx, n_new):
    tokens = idx.clone()
    with torch.no_grad():
        for _ in range(n_new):
            logits, _ = model(tokens)
            tokens = torch.cat([tokens, logits[:, -1, :].argmax(-1, keepdim=True)], dim=1)
    return tokens


@pytest.mark.parametrize("positional", ["learned", "none", "rope", "rope_alt"])
@pytest.mark.parametrize("layers", [
    ["dilution", "dilution"],
    ["softmax", "dilution"],
])
def test_cached_generate_matches_full_recompute(layers, positional):
    torch.manual_seed(0)
    model = DilutionLM(tiny(layer_attention=layers, positional=positional)).eval()
    idx = torch.randint(0, 32, (2, 6))
    assert torch.equal(model.generate(idx, 8), _greedy_full_recompute(model, idx, 8))


@pytest.mark.parametrize("positional", ["rope", "rope_alt"])
def test_rope_layers_and_decode_offsets(positional):
    """rope: every attention layer rotated; rope_alt: layers 1, 3, ... (1-indexed), so
    the first layer always carries position. Cached decode must rotate the
    appended token by its absolute position."""
    torch.manual_seed(5)
    model = DilutionLM(tiny(layer_attention=["softmax", "dilution", "dilution", "softmax"],
                            n_layers=4, positional=positional)).eval()
    has_rope = [block.attn.rope is not None for block in model.transformer.h]
    assert has_rope == ([True] * 4 if positional == "rope" else [True, False, True, False])
    tokens = torch.randint(0, 32, (2, 12))
    with torch.no_grad():
        logits, caches = model.prefill(tokens[:, :6])
        full, _ = model(tokens, tokens)
        assert torch.allclose(logits[:, -1], full[:, 5], atol=1e-5)
        for pos in range(6, 12):
            logits = model.decode_step(tokens[:, pos : pos + 1], pos, caches)
            assert torch.allclose(logits[:, -1], full[:, pos], atol=1e-5), f"pos {pos}"


def test_cached_decode_logits_match_teacher_forced():
    torch.manual_seed(3)
    model = DilutionLM(tiny(attention="dilution")).eval()
    tokens = torch.randint(0, 32, (2, 12))
    with torch.no_grad():
        logits, caches = model.prefill(tokens[:, :6])
        full, _ = model(tokens, tokens)
        assert torch.allclose(logits[:, -1], full[:, 5], atol=1e-5)
        for pos in range(6, 12):
            logits = model.decode_step(tokens[:, pos : pos + 1], pos, caches)
            assert torch.allclose(logits[:, -1], full[:, pos], atol=1e-5), f"pos {pos}"


def test_generate_rejects_overflow():
    model = DilutionLM(tiny(context_length=8)).eval()
    with pytest.raises(ValueError, match="context_length"):
        model.generate(torch.zeros(1, 6, dtype=torch.long), 4)


def test_generate_sampling_shapes():
    torch.manual_seed(0)
    model = DilutionLM(tiny()).eval()
    idx = torch.randint(0, 32, (2, 4))
    out = model.generate(idx, 5, temperature=1.0, top_k=8)
    assert out.shape == (2, 9)
    assert torch.equal(out[:, :4], idx)


def test_filter_logits_top_k_and_nucleus():
    from dilution.model import filter_logits

    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    kept = lambda x: torch.isfinite(x)[0].tolist()
    assert kept(filter_logits(logits, top_k=2)) == [True, True, False, False]
    # nucleus keeps the smallest prefix whose mass reaches p: 0.5 < 0.6 -> add 0.3 (0.8 >= 0.6)
    assert kept(filter_logits(logits, top_p=0.6)) == [True, True, False, False]
    assert kept(filter_logits(logits, top_p=0.5)) == [True, False, False, False]
    assert kept(filter_logits(logits, top_p=0.95)) == [True, True, True, False]
    assert kept(filter_logits(logits, top_p=0.01)) == [True, False, False, False]   # top token always kept
    assert kept(filter_logits(logits, top_p=1.0)) == [True] * 4
    shuffled = logits[:, [2, 0, 3, 1]]                                              # order-independent
    assert kept(filter_logits(shuffled, top_p=0.6)) == [False, True, False, True]


@pytest.mark.parametrize("attention", ["dilution", "softmax"])
def test_cached_sampling_matches_uncached_and_is_reproducible(attention):
    """Seeded sampling through the decode cache draws exactly the tokens a full re-forward at
    every step would draw from the same generator."""
    torch.manual_seed(1)
    model = DilutionLM(tiny(attention=attention, context_length=32)).eval()
    idx = torch.randint(0, 32, (3, 5))
    kw = dict(temperature=1.0, top_p=0.9, vocab_limit=30)
    a = model.generate(idx, 10, generator=torch.Generator().manual_seed(7), **kw)
    b = model.generate(idx, 10, generator=torch.Generator().manual_seed(7), **kw)
    assert torch.equal(a, b)
    assert int(a[:, 5:].max()) < 30

    from dilution.model import filter_logits
    gen, tokens = torch.Generator().manual_seed(7), idx
    with torch.no_grad():
        for _ in range(10):
            logits, _ = model(tokens)
            step = logits[:, -1, :].float()
            step[:, 30:] = float("-inf")
            step = filter_logits(step, top_p=0.9)
            tokens = torch.cat([tokens, torch.multinomial(step.softmax(-1), 1, generator=gen)], dim=1)
    assert torch.equal(a, tokens)


@pytest.mark.parametrize("attention", ["dilution", "softmax"])
def test_generate_past_trained_context_with_raised_cap(attention):
    """Extrapolation sampling raises config.context_length; the decode caches must then hold
    prompt + new tokens even though the layers were built for the trained context."""
    torch.manual_seed(2)
    model = DilutionLM(tiny(attention=attention, context_length=8, positional="none")).eval()
    idx = torch.randint(0, 32, (2, 10))
    model.config.context_length = 16
    out = model.generate(idx, 6)
    with torch.no_grad():
        full, _ = model(out, out)
    assert torch.equal(out[:, 10:], full[:, 9:15].argmax(-1))       # greedy == teacher-forced argmax




def test_dilution_operator_is_parameter_free():
    """No pressure_w/capacity_w: the order asymmetry comes from the cumsum alone."""

    model = DilutionLM(tiny(attention="dilution"))
    attn = model.transformer.h[0].attn
    names = {name for name, _ in attn.named_parameters()}
    assert not any("pressure" in n or "capacity" in n for n in names), names
    # parameter-matched with the softmax control, exactly (no extra scalars)
    n_dilution = DilutionLM(tiny(attention="dilution")).num_parameters()
    n_softmax = DilutionLM(tiny(attention="softmax")).num_parameters()
    assert n_dilution == n_softmax


@pytest.mark.parametrize("share_cumsum", [True, "exclusive", False])
def test_cached_logits_match_full_forward_for_share_modes(share_cumsum):
    # Prefill and decode must use the same denominator as the full forward, including the
    # experimental exclusive-history variant (temporal attention's committed + eps).
    torch.manual_seed(0)
    model = DilutionLM(tiny(attention="dilution", share_cumsum=share_cumsum)).eval()
    idx = torch.randint(0, 32, (2, 10))
    with torch.no_grad():
        full, _ = model(idx, idx)                       # all positions
        logits, caches = model.prefill(idx[:, :6])
        assert torch.allclose(logits[:, -1], full[:, 5], atol=1e-5)
        for pos in range(6, 10):
            step = model.decode_step(idx[:, pos:pos + 1], pos, caches)
            assert torch.allclose(step[:, -1], full[:, pos], atol=1e-5), f"decode differs at {pos}"
