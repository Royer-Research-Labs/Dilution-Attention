# Reproducing the results

How to rebuild the data, rerun the evaluations and retrain the models behind the README and
[research-results.md](research-results.md). Every published number traces to a run record under
`runs/` and a result file under `results/`; the evidence ledger ([signals.md](signals.md)) cites
both.

## Requirements

Python 3.10+, PyTorch 2.4+ with CUDA and Triton for the fused kernels (the reference path runs on
CPU). The measurements used PyTorch 2.13, Triton 3.7 and CUDA 13 on an RTX PRO 6000 Blackwell
(sm_120).

```bash
pip install -e ".[dev,data,figures]"
pytest                                          # CPU tests + CUDA-gated kernel parity
python -m dilution.train --config configs/smoke.yaml --overwrite-run   # synthetic-data smoke run
```

## Data

Training corpora are headerless little-endian uint16 GPT-2 token files with a `manifest.json`
recording the source revision, split policy and SHA-256 of every file. The configs expect them
under `data/`. The three FineWeb-Edu corpora used here rebuild exactly:

```bash
# 40B tokens, the 209M runs and the 64M reliability studies
dilution-prepare-data --dataset HuggingFaceFW/fineweb-edu --dataset-config sample-100BT \
    --revision fc9850dff5e2d0f8f776efe41b24a1c49556cfc5 \
    --train-tokens 40000000000 --val-tokens 20000000 --output-dir data/fineweb_edu_100bt_40b
# 1B tokens, the early 64M ladders
dilution-prepare-data --dataset HuggingFaceFW/fineweb-edu --dataset-config sample-10BT \
    --revision fc9850dff5e2d0f8f776efe41b24a1c49556cfc5 \
    --train-tokens 1000000000 --val-tokens 5000000 --output-dir data/fineweb_edu_10bt_1b
# 4B tokens, the equal-wall-clock runs (runs/eqtime_tuned)
dilution-prepare-data --dataset HuggingFaceFW/fineweb-edu --dataset-config sample-10BT \
    --revision fc9850dff5e2d0f8f776efe41b24a1c49556cfc5 \
    --train-tokens 4000000000 --val-tokens 5000000 --output-dir data/fineweb_edu_10bt_4b
```

Each run record's `data_provenance.json` stores the SHA-256 of the corpus it trained on, so a
rebuild can be checked against it. The Wikipedia half of the blend study
(`scripts/blend_corpus.py`) came from a pre-tokenised English Wikipedia dump in the same format;
it is not rebuilt by these commands.

## Training a model

Train into a separate directory, so the archived records under `runs/` (published without
checkpoints) stay untouched. The trainer refuses to overwrite a run directory, rejects unknown
config keys and resumes from the exact batch; `scripts/run_suite.py` skips archived records unless
told otherwise. The headline 209M models were trained on a 2x-Chinchilla cosine schedule and
evaluated at its step-64,000 (1x) point. `--stop-at-step` reproduces that without shortening the
schedule, and keeps the step-64,000 checkpoint when training continues:

```bash
dilution-train --config configs/scale_16384/all_dilution_nope.yaml --seed 1337 \
    --output-dir runs_repro/scale_16384/all_dilution_nope_seed1337 --stop-at-step 64000
# later, the same schedule on to 2x Chinchilla; checkpoint_step_0064000.pt is kept
dilution-train --config configs/scale_16384/all_dilution_nope.yaml --seed 1337 \
    --output-dir runs_repro/scale_16384/all_dilution_nope_seed1337 \
    --resume runs_repro/scale_16384/all_dilution_nope_seed1337/latest.pt
```

Loss comparisons use one common validation slice per scale (`scripts/eval_checkpoint.py`), because
in-run validation windows differ with batch size.

## Retrieval evaluation (NIAH)

Clone the harness next to this repository (or set `DILUTION_NIAH_ROOT`), pinned to the revision
used here:

```bash
git clone https://github.com/Royer-Research-Labs/Niah ../Niah && git -C ../Niah checkout 7cbf29f
dilution-niah --checkpoint runs/<run>/latest.pt --context-length 3840 7936 16000 --samples 20 --num-needles 8
```

What the test is: one fixed 54-token paragraph of office text repeated to length, with the
sentence "The secret keyword is *w*." planted at an evenly spaced depth. The model's likelihood is
compared across 8 fixed single-token keywords, each serving as the needle in turn. The keyword the
model already prefers on a needle-free control prompt is disqualified, so 7 are scored and chance
is 1/7. It measures copying one planted token out of repetitive filler, not reasoning or retrieval
from natural text. Full description: [research-results.md](research-results.md#setup).

Every result row records its provenance: the harness commit, spec version and dirty flag, the
checkpoint's path, step, size and SHA-256, and the evaluation settings.

**Protocol versions.** Results from 2026-09-22 on use the `niah-v4-spec` prompt builder, which
builds each prompt, and the control, to exactly the requested length. The earlier `niah-v3-spec`
ran prompts 12 tokens long (4.7% at 256 tokens, under 1.4% from 896 up) and capped the control
prompt at 5,400 tokens. Every v3 result behind the figures and headline tables was re-scored under
v4 from the same checkpoint (`results/niah_v4/`, `scripts/compare_niah_versions.py`): at 209M no
cell moved by more than 0.09, and the figures and summaries use the v4 values. v4 is not
systematically easier (35 cells up, 35 down); the changes are near-tied comparisons flipping, so
scores near 1.0 or chance barely move while mid-range scores moved about 0.05 on average and up to
0.17. The exceptions are two 209M 1x ladders whose checkpoints no longer exist at that step, marked
in the figure. Older ledger results are v3 as recorded, so their mid-range cells carry that extra
uncertainty, and their files predate the provenance fields; the ledger dates each one.

## Sampling evaluation

`scripts/sample_eval.py` has every model continue the same validation prompts with the same random
streams and scores the samples with a judge model (`pip install -e ".[sampling]"`; the judge,
HuggingFaceTB/SmolLM2-1.7B, downloads on first use):

```bash
python scripts/sample_eval.py --model dilution=runs/<run>/latest.pt --model softmax=runs/<run>/latest.pt \
    --baseline softmax --lengths 256 1024 4096 16128 --extrapolate 32768 --prompts 256 \
    --output results/samples/<name>.json
```

What the test is: each prompt is a window of FineWeb-Edu validation text that ends at least 64
tokens into a document, and the real next 256 tokens are kept as the human reference. The prompt
lengths (from 256 tokens up to the trained context, plus lengths beyond it) share one end point, so
only the amount of earlier context changes. Every model writes 256 tokens from every prompt in
three ways (plain sampling, nucleus sampling with p = 0.9, and greedy), with the same random seeds,
so models are compared prompt by prompt. The main score is the judge's bits per byte for the
continuation given the last 4,096 of its tokens of the prompt: lower is more natural, and the human
continuation scores 0.65-0.70. It is read together with repetition (the share of repeated 4-token
sequences), because repetitive text also scores low; greedy decoding loops in every model at this
scale. Each model also scores the others' samples and the human text, a check that does not depend
on the judge. The judge sees only local context, so the test measures fluency and local coherence,
not long-range consistency, and it resolves loss differences of about 0.2 nats per token, not a few
hundredths.

It writes a summary (`.json`: aggregates, paired differences with bootstrap intervals, and
provenance), every sample's metrics (`.metrics.jsonl.gz`: text statistics, judge score and
cross-scores, without token ids or text), readable excerpts (`.md`, two prompts at every length with
every model's samples) and every sample in full (`.samples.jsonl`). The summaries and metrics of the
published runs are under `results/samples/`, and `python scripts/verify_samples.py` recomputes every
published statistic from the metrics. The full samples are not published (150 MB); each summary
records the prompts, seeds, decoding settings, checkpoint hashes and judge revision needed to
regenerate them. The judge is pinned to that revision by default (`--judge-revision`), and a
continuation too long for the judge's context is rejected rather than truncated.

## Run records and figures

Each published run's record is under `runs/<group>/<arm>_seed<N>/`: `metrics.jsonl`, the fully
resolved config it ran with, data provenance and the environment. Checkpoints are not published.
Every config under `configs/` has at least one such record; the kernel validation runs
(`runs/validate_s27/`) and short learning-rate probes (`runs/smoke/`) have records but no separate
config. Evaluation outputs are under `results/`, and `python scripts/make_retrieval_figure.py`
regenerates the figures from them. Kernel timings are reproduced with `scripts/step_profile.py`,
`scripts/profile_kernels.py` and `scripts/bench_crossover.py` ([kernels.md](kernels.md)).
