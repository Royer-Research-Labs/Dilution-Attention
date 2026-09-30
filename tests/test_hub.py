"""Saving and loading models in the Hugging Face layout (dilution.hub), and context extension."""

from __future__ import annotations

import json

import pytest
import torch

from dilution.config import ModelConfig
from dilution.extend import ExtendedContext
from dilution.model import DilutionLM

pytest.importorskip("safetensors")
from dilution.hub import CONFIG, WEIGHTS, from_pretrained, save_pretrained  # noqa: E402


def tiny(**kw):
    values = dict(vocab_size=64, n_layers=2, d_model=16, n_heads=2, context_length=32, mlp="swiglu",
                  layer_attention=["dilution", "softmax"], positional="rope_alt")
    values.update(kw)
    return ModelConfig(**values)


def test_round_trip_is_exact_and_keeps_the_tied_embedding(tmp_path):
    torch.manual_seed(0)
    model = DilutionLM(tiny()).eval()
    save_pretrained(model, tmp_path, metadata={"seed": 7})
    loaded = from_pretrained(tmp_path)
    ids = torch.randint(0, 64, (2, 20))
    with torch.no_grad():
        assert torch.equal(model(ids, ids)[0], loaded(ids, ids)[0])
    assert loaded.lm_head.weight is loaded.transformer.wte.weight
    assert loaded.metadata == {"seed": 7}
    from safetensors import safe_open
    with safe_open(str(tmp_path / WEIGHTS), "pt") as fh:
        assert "lm_head.weight" not in fh.keys()          # stored once
    assert json.loads((tmp_path / CONFIG).read_text())["model_config"]["positional"] == "rope_alt"


def test_from_pretrained_rejects_foreign_configs(tmp_path):
    torch.manual_seed(0)
    save_pretrained(DilutionLM(tiny()), tmp_path)
    cfg = json.loads((tmp_path / CONFIG).read_text())
    cfg["format"] = "something-else"
    (tmp_path / CONFIG).write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="not a dilution-attention"):
        from_pretrained(tmp_path)


def test_extended_context_scales_rope_and_restores_it():
    model = DilutionLM(tiny(context_length=16))
    ropes = [m for m in model.modules() if type(m).__name__ == "RopeModule"]
    before = [r.theta for r in ropes]
    with ExtendedContext(model, 64, ntk=True):
        assert model.config.context_length == 64
        dim = ropes[0].dim
        assert ropes[0].theta == pytest.approx(before[0] * 4 ** (dim / (dim - 2)))
    assert model.config.context_length == 16 and [r.theta for r in ropes] == before
    with pytest.raises(ValueError, match="learned position"):
        with ExtendedContext(DilutionLM(tiny(positional="learned")), 64):
            pass
