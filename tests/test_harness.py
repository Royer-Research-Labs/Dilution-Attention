"""Tests for the experiment harness: config validation, data streams, training loop."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest
import torch
import yaml

from dilution.config import ConfigurationError, ModelConfig, build_model
from dilution.data import SyntheticTokenStream, UInt16TokenStream
from dilution.runtime import (
    learning_rate_for_step,
    load_experiment_config,
    resolve_experiment_config,
    verify_token_manifest,
)
from dilution.train import run_training


def smoke_config(tmp_path: Path, **overrides) -> dict:
    config = {
        "model": {
            "vocab_size": 64,
            "n_layers": 2,
            "d_model": 16,
            "n_heads": 2,
            "context_length": 8,
            "attention": "dilution",
        },
        "data": {
            "synthetic": True,
            "vocab_size": 64,
            "train_tokens": 2048,
            "val_tokens": 512,
        },
        "training": {
            "output_dir": str(tmp_path / "run"),
            "device": "cpu",
            "dtype": "float32",
            "batch_size": 2,
            "sequence_length": 8,
            "max_steps": 3,
            "warmup_steps": 1,
            "eval_interval": 3,
            "eval_batches": 2,
            "log_interval": 1,
        },
    }
    for section, values in overrides.items():
        config[section].update(values)
    return config


def write_config(tmp_path: Path, config: dict, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


# --- configuration ---


def test_unknown_keys_are_rejected(tmp_path):
    for section, key in (("model", "flux_capacitor"), ("data", "shard"), ("training", "foo")):
        config = smoke_config(tmp_path)
        config[section][key] = 1
        with pytest.raises(ConfigurationError, match=f"{section}.{key}"):
            resolve_experiment_config(config)

    config = smoke_config(tmp_path)
    config["extras"] = {}
    with pytest.raises(ConfigurationError, match="extras"):
        resolve_experiment_config(config)


def test_cross_section_validation(tmp_path):
    config = smoke_config(tmp_path, data={"vocab_size": 32})
    with pytest.raises(ConfigurationError, match="vocab_size"):
        resolve_experiment_config(config)

    config = smoke_config(tmp_path, training={"sequence_length": 16})
    with pytest.raises(ConfigurationError, match="sequence_length"):
        resolve_experiment_config(config)

    config = smoke_config(tmp_path, training={"warmup_steps": 99})
    with pytest.raises(ConfigurationError, match="warmup_steps"):
        resolve_experiment_config(config)


def test_learning_rate_schedule_shape():
    training = {"learning_rate": 1.0, "warmup_steps": 10, "max_steps": 100, "min_lr_ratio": 0.1}
    assert learning_rate_for_step(0, training) == pytest.approx(0.1)
    assert learning_rate_for_step(9, training) == pytest.approx(1.0)
    assert learning_rate_for_step(99, training) == pytest.approx(0.1)
    mid = learning_rate_for_step(54, training)
    assert 0.1 < mid < 1.0


# --- data ---


def test_synthetic_stream_is_deterministic_and_checkpointable():
    stream = SyntheticTokenStream(1024, 64, seed=7, pattern_length=16)
    generator = torch.Generator().manual_seed(3)
    state = generator.get_state()
    a = stream.sample_batch(4, 8, generator)
    generator.set_state(state)
    b = stream.sample_batch(4, 8, generator)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    # Targets are inputs shifted by one.
    window = stream.read_tokens(0, 9)
    x, y = stream._make_batch([0], 8, None)
    assert torch.equal(x[0], window[:-1]) and torch.equal(y[0], window[1:])


def make_prepared_corpus(directory: Path, train_tokens: int = 200, val_tokens: int = 100,
                         vocab_size: int = 64) -> dict:
    """Write a tiny uint16 corpus with the sibling-harness manifest schema."""

    directory.mkdir(parents=True, exist_ok=True)
    splits = {}
    for manifest_split, name, count in (
        ("train", "train.bin", train_tokens),
        ("validation", "val.bin", val_tokens),
    ):
        tokens = [i % vocab_size for i in range(count)]
        payload = struct.pack(f"<{count}H", *tokens)
        path = directory / name
        path.write_bytes(payload)
        splits[manifest_split] = {
            "path": name,
            "num_tokens": count,
            "num_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest = {
        "manifest_version": 1,
        "token_format": {"dtype": "uint16", "byte_order": "little",
                         "header_bytes": 0, "bytes_per_token": 2},
        "splits": splits,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_uint16_stream_roundtrip_and_shift(tmp_path):
    make_prepared_corpus(tmp_path / "corpus", train_tokens=50, vocab_size=16)
    stream = UInt16TokenStream(tmp_path / "corpus" / "train.bin", vocab_size=16)
    assert stream.num_tokens == 50
    assert stream.read_tokens(0, 50).tolist() == [i % 16 for i in range(50)]
    x, y = stream._make_batch([3], 8, None)
    window = stream.read_tokens(3, 9)
    assert torch.equal(x[0], window[:-1]) and torch.equal(y[0], window[1:])


def test_uint16_stream_rejects_out_of_vocab(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(struct.pack("<4H", 1, 2, 63, 64))
    with pytest.raises(ValueError, match="outside"):
        UInt16TokenStream(path, vocab_size=64)


def test_manifest_verification(tmp_path):
    corpus = tmp_path / "corpus"
    make_prepared_corpus(corpus, train_tokens=200, val_tokens=100)
    result = verify_token_manifest(corpus / "train.bin", "train", expected_num_tokens=200)
    assert result["num_tokens"] == 200 and result["kind"] == "prepared_uint16"
    verify_token_manifest(corpus / "val.bin", "val", expected_num_tokens=100)

    with pytest.raises(ConfigurationError, match="tokens"):
        verify_token_manifest(corpus / "train.bin", "train", expected_num_tokens=999)

    # Tampered content must fail the sha256 check.
    payload = struct.pack("<200H", *([1] * 200))
    (corpus / "train.bin").write_bytes(payload)
    with pytest.raises(ConfigurationError, match="mismatch"):
        verify_token_manifest(corpus / "train.bin", "train", expected_num_tokens=200)


def test_training_on_prepared_binary_data(tmp_path):
    corpus = tmp_path / "corpus"
    make_prepared_corpus(corpus, train_tokens=400, val_tokens=120, vocab_size=64)
    config = smoke_config(tmp_path, data={
        "synthetic": False,
        "train_path": str(corpus / "train.bin"),
        "val_path": str(corpus / "val.bin"),
        "train_tokens": 400,
        "val_tokens": 120,
    })
    config_path = write_config(tmp_path, config, name="binary.yaml")
    summary = run_training(config_path)
    assert summary is not None and summary["event"] == "complete"
    provenance = json.loads((tmp_path / "run" / "data_provenance.json").read_text())
    assert provenance["kind"] == "prepared_uint16"
    assert provenance["splits"]["train"]["num_tokens"] == 400
    assert (tmp_path / "run" / "data_manifest.json").is_file()


# --- training loop ---


def test_training_smoke_and_run_records(tmp_path):
    config_path = write_config(tmp_path, smoke_config(tmp_path))
    summary = run_training(config_path)
    assert summary is not None and summary["event"] == "complete"
    run_dir = tmp_path / "run"
    for name in ("metrics.jsonl", "latest.pt", "resolved_config.yaml",
                 "environment.json", "data_provenance.json"):
        assert (run_dir / name).is_file(), name
    contents = (run_dir / "metrics.jsonl").read_text(encoding="utf-8")
    assert '"event": "complete"' in contents
    assert '"event": "evaluation"' in contents


def test_fresh_run_refuses_to_clobber(tmp_path):
    config_path = write_config(tmp_path, smoke_config(tmp_path))
    run_training(config_path)
    with pytest.raises(ConfigurationError, match="already contains run records"):
        run_training(config_path)
    # And overwrite_run discards them and trains again.
    summary = run_training(config_path, overwrite_run=True)
    assert summary is not None and summary["event"] == "complete"


def test_resume_refuses_config_drift(tmp_path):
    config_path = write_config(tmp_path, smoke_config(tmp_path))
    run_training(config_path)
    drifted = smoke_config(tmp_path, training={"learning_rate": 9.9e-4, "max_steps": 6})
    drifted_path = write_config(tmp_path, drifted, name="drifted.yaml")
    with pytest.raises(ConfigurationError, match="learning_rate"):
        run_training(drifted_path, resume_override=tmp_path / "run" / "latest.pt")


def test_resume_refuses_completed_run(tmp_path):
    config_path = write_config(tmp_path, smoke_config(tmp_path))
    run_training(config_path)
    with pytest.raises(ConfigurationError, match="already completed"):
        run_training(config_path, resume_override=tmp_path / "run" / "latest.pt")


def test_screening_mode(tmp_path):
    config = smoke_config(
        tmp_path,
        # checkpoint_interval below stop_after_steps: screens must STILL write
        # no checkpoints (they are throwaway).
        training={"max_steps": 5, "checkpoint_interval": 1,
                  "output_dir": str(tmp_path / "screen")},
    )
    config_path = write_config(tmp_path, config, name="screen.yaml")
    summary = run_training(config_path, stop_after_steps=2)
    assert summary is None
    screen_dir = tmp_path / "screen"
    contents = (screen_dir / "metrics.jsonl").read_text(encoding="utf-8")
    assert '"event": "screen_complete"' in contents
    assert not list(screen_dir.glob("*.pt"))


def test_screening_refuses_full_schedule(tmp_path):
    config_path = write_config(tmp_path, smoke_config(tmp_path))
    with pytest.raises(ConfigurationError, match="stop-after-steps"):
        run_training(config_path, stop_after_steps=3)  # == max_steps


def test_interval_checkpointing_writes_single_rolling_file(tmp_path):
    config = smoke_config(
        tmp_path,
        training={"max_steps": 4, "checkpoint_interval": 2, "eval_interval": 0},
    )
    config_path = write_config(tmp_path, config)
    run_training(config_path, stop_after_steps=0)
    run_dir = tmp_path / "run"
    # One rolling latest.pt; the numbered-per-step files were retired.
    assert (run_dir / "latest.pt").is_file()
    assert sorted(run_dir.glob("*.pt")) == [run_dir / "latest.pt"]


def test_resume_continues_training(tmp_path):
    config = smoke_config(
        tmp_path,
        training={"max_steps": 4, "checkpoint_interval": 2, "eval_interval": 0},
    )
    config_path = write_config(tmp_path, config)
    run_training(config_path, stop_after_steps=0)
    # Simulate a crash at step 2: doctor a copy of latest.pt back to the
    # mid-run step (the rolling-file mechanism keeps no earlier snapshots).
    latest = tmp_path / "run" / "latest.pt"
    state = torch.load(latest, map_location="cpu", weights_only=True)
    state["step"] = 2
    state["tokens_seen"] = state["tokens_seen"] // 2
    crashed = tmp_path / "crashed.pt"
    torch.save(state, crashed)
    summary = run_training(config_path, resume_override=crashed)
    assert summary is not None and summary["event"] == "complete"
    assert summary["step"] == 4




# --- checkpointed stops and milestones (reproducing the 1x point of a 2x schedule) ---

def test_stop_at_step_keeps_checkpoint_and_resumes(tmp_path):
    config = smoke_config(tmp_path, training={"max_steps": 6, "checkpoint_interval": 4, "eval_interval": 0})
    config_path = write_config(tmp_path, config)
    run_dir = tmp_path / "run"
    stopped = run_training(config_path, stop_at_step=3)
    assert stopped is not None and stopped["event"] == "stopped" and stopped["step"] == 3
    kept = run_dir / "checkpoint_step_0000003.pt"
    assert torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)["step"] == 3
    assert torch.load(kept, map_location="cpu", weights_only=False)["step"] == 3
    done = run_training(config_path, resume_override=run_dir / "latest.pt")
    assert done is not None and done["event"] == "complete" and done["step"] == 6
    assert torch.load(kept, map_location="cpu", weights_only=False)["step"] == 3


def test_keep_checkpoint_at_without_stopping(tmp_path):
    config = smoke_config(tmp_path, training={"max_steps": 4, "checkpoint_interval": 0, "eval_interval": 0})
    config_path = write_config(tmp_path, config)
    done = run_training(config_path, keep_checkpoint_at=[2, 4])
    assert done["event"] == "complete"
    run_dir = tmp_path / "run"
    for step in (2, 4):
        state = torch.load(run_dir / f"checkpoint_step_{step:07d}.pt", map_location="cpu", weights_only=False)
        assert state["step"] == step


def test_stop_at_step_validation(tmp_path):
    config_path = write_config(tmp_path, smoke_config(tmp_path))  # max_steps 3
    with pytest.raises(ConfigurationError, match="strictly inside"):
        run_training(config_path, stop_at_step=3)
    with pytest.raises(ConfigurationError, match="mutually exclusive"):
        run_training(config_path, stop_at_step=1, stop_after_steps=2)


def test_overwrite_keeps_records_when_data_is_missing(tmp_path):
    config_path = write_config(tmp_path, smoke_config(tmp_path))
    run_training(config_path)
    metrics = tmp_path / "run" / "metrics.jsonl"
    before = metrics.read_text(encoding="utf-8")
    broken = smoke_config(tmp_path, data={"synthetic": False, "train_path": str(tmp_path / "missing" / "train.bin"),
                                          "val_path": str(tmp_path / "missing" / "val.bin"),
                                          "train_tokens": 400, "val_tokens": 120})
    broken_path = write_config(tmp_path, broken, name="broken.yaml")
    with pytest.raises(Exception):
        run_training(broken_path, overwrite_run=True)
    assert metrics.read_text(encoding="utf-8") == before   # nothing was deleted


def test_rejected_resume_leaves_the_run_untouched(tmp_path):
    # latest.pt at step 4; resuming the kept step-2 checkpoint with --stop-at-step 2 is
    # impossible and must be rejected before latest.pt is re-anchored or events are appended.
    config = smoke_config(tmp_path, training={"max_steps": 6, "checkpoint_interval": 0, "eval_interval": 0})
    config_path = write_config(tmp_path, config)
    run_dir = tmp_path / "run"
    run_training(config_path, stop_at_step=4, keep_checkpoint_at=[2])
    metrics_before = (run_dir / "metrics.jsonl").read_text(encoding="utf-8")
    with pytest.raises(ConfigurationError, match="not after the resumed step"):
        run_training(config_path, resume_override=run_dir / "checkpoint_step_0000002.pt", stop_at_step=2)
    assert torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)["step"] == 4
    assert (run_dir / "metrics.jsonl").read_text(encoding="utf-8") == metrics_before


def test_stop_between_eval_intervals_saves_its_validation(tmp_path):
    # The stop step is not an eval step: the stop-point evaluation must land in the saved
    # checkpoints, not only in the 'stopped' event.
    config = smoke_config(tmp_path, training={"max_steps": 6, "checkpoint_interval": 0, "eval_interval": 5})
    config_path = write_config(tmp_path, config)
    run_dir = tmp_path / "run"
    stopped = run_training(config_path, stop_at_step=3)
    assert stopped["best_val_loss"] is not None
    for name in ("latest.pt", "checkpoint_step_0000003.pt"):
        state = torch.load(run_dir / name, map_location="cpu", weights_only=False)
        assert state["best_val_loss"] == pytest.approx(stopped["best_val_loss"])


def test_suite_status_comes_from_the_latest_lifecycle_event(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_suite", Path(__file__).resolve().parent.parent / "scripts" / "run_suite.py")
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    metrics = tmp_path / "metrics.jsonl"

    def write(*events):
        metrics.write_text("".join(json.dumps({"event": e}) + "\n" for e in events), encoding="utf-8")

    write("start", "train", "checkpoint", "stopped")
    assert suite.has_terminal_event(metrics)            # a deliberate stop is not resumed
    write("start", "stopped", "resume", "train", "checkpoint")
    assert not suite.has_terminal_event(metrics)        # continued, then interrupted: resumable
    write("start", "stopped", "resume", "train", "complete")
    assert suite.has_terminal_event(metrics)
