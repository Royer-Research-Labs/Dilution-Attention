"""Dilution attention: committed-claim competition as a language-model operator.

The operator spec is `dilution.kernels.dilution_attention_reference`; the
model is `dilution.model.DilutionLM`; training runs via `dilution.train`.
"""

from .config import ConfigurationError, ModelConfig, build_model
from .model import DilutionLM

__all__ = ["ConfigurationError", "ModelConfig", "build_model", "DilutionLM"]
