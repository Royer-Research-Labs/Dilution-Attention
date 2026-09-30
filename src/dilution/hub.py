"""Save and load DilutionLM weights in a Hugging Face Hub layout.

    from dilution.hub import from_pretrained
    model = from_pretrained("Royer-Research-Labs/<model>")        # or a local directory

A saved model is a directory with `model.safetensors` (the weights, fp32 by default; the output
embedding is tied to the LM head and stored once), `config.json` (the model configuration under
"model_config", plus free-form "metadata" such as the training run it came from) and usually a
`README.md` model card. `save_pretrained` writes the first two; `scripts/export_hf.py` builds the
published models from training checkpoints, verifies them, and writes their cards.

Requires `pip install -e ".[hub]"` (safetensors, huggingface_hub).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .config import ModelConfig, build_model

WEIGHTS = "model.safetensors"
CONFIG = "config.json"
TIED = "lm_head.weight"          # tied to transformer.wte.weight; not stored separately
FORMAT = "dilution-attention/1"


def save_pretrained(model, out_dir: str | Path, *, metadata: Mapping[str, Any] | None = None,
                    dtype: torch.dtype = torch.float32) -> Path:
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().to("cpu", dtype).contiguous() for k, v in model.state_dict().items() if k != TIED}
    save_file(state, str(out / WEIGHTS), metadata={"format": FORMAT})
    config = {"format": FORMAT, "model_config": model.config.to_dict(),
              "dtype": str(dtype).replace("torch.", ""), "metadata": dict(metadata or {})}
    (out / CONFIG).write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    return out


def from_pretrained(repo_or_path: str | Path, *, revision: str | None = None, device: str | torch.device = "cpu",
                    dtype: torch.dtype | None = None):
    """A DilutionLM in eval mode from a local directory or a Hugging Face Hub repo id. On CUDA the
    fused dilution kernels are enabled (the eager path materialises T x T attention per head)."""
    from safetensors.torch import load_file

    path = Path(repo_or_path)
    if not (path / CONFIG).is_file():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(str(repo_or_path), revision=revision, allow_patterns=[WEIGHTS, CONFIG]))
    config = json.loads((path / CONFIG).read_text(encoding="utf-8"))
    if config.get("format") != FORMAT:
        raise ValueError(f"{path / CONFIG} is not a {FORMAT} model (format {config.get('format')!r})")
    model = build_model(ModelConfig.from_dict(config["model_config"]))
    state = load_file(str(path / WEIGHTS))
    state[TIED] = state["transformer.wte.weight"]
    model.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
    model.metadata = config.get("metadata", {})
    if dtype is not None:
        model.to(dtype)
    model.to(device).eval()
    if torch.device(device).type == "cuda":
        model.enable_dilution_kernel()
    return model
