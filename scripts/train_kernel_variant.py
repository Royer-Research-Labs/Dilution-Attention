"""Screen a kernel module with the existing training harness and record its source.

    PYTHONPATH=src python scripts/train_kernel_variant.py dilution.kernels_frozen \
        --config configs/12l_8192_600m/all_dilution.yaml \
        --output-dir runs_repro/frozen_screen1000 --stop-after-steps 1000

All arguments after the module name belong to dilution.train. A module must
expose dilution_attention(q, k, v, dot_precision=...). The normal model path
is patched only within this process. Run records include the selected
module source and hash so an experimental backward cannot be mistaken for
the production training rule.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys

from dilution import kernels
from dilution.train import build_parser, main as train_main
from dilution.runtime import load_experiment_config


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    module = importlib.import_module(sys.argv[1])
    argv = sys.argv[2:]
    args = build_parser().parse_args(argv)
    cfg = load_experiment_config(args.config)
    output_dir = Path(args.output_dir or cfg["training"]["output_dir"])
    # The trainer enforces refuse-to-clobber. Write provenance only after it
    # has accepted and completed the run, avoiding changes to an existing run.
    source = Path(module.__file__).read_bytes()
    production_hash = hashlib.sha256(Path(kernels.__file__).read_bytes()).hexdigest()
    kernels.dilution_attention = module.dilution_attention
    train_main(argv)
    (output_dir / "kernel_variant.py").write_bytes(source)
    (output_dir / "kernel_variant.json").write_text(json.dumps({
        "module": module.__name__,
        "sha256": hashlib.sha256(source).hexdigest(),
        "production_source_sha256": production_hash,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
