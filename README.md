# microtrain

Micro-scale training with macro-scale techniques.

Reimplementations of the systems, architecture, regularization, and compression
techniques from three frontier training challenges —
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt),
[parameter-golf](https://github.com/openai/parameter-golf), and
[slowrun](https://github.com/qlabs-eng/slowrun) at a scale that fits a single
Apple M4 or two T4 GPUs in one night.

The goal is not to beat the challenge leaderboards. It is to
re-implement each technique, measure its effect on **BPB** with the same
tokenizer-agnostic metric the challenge uses, and keep only what survives.

## Scope

- **Model:** 9 layers, dim 512, 8 heads / 4 KV heads, tied embeddings — **17.1M params**.
- **Data:** FineWeb-10B, SentencePiece vocab 1024 (`sp1024`); each run trains on
  ~819M tokens — 5,000 steps × 163,840 tokens.
- **Metric:** `final_int8_zlib_roundtrip_exact` BPB = `(val_loss / ln 2) × (tokens / bytes)`,
  measured *after* int8 + zlib round-trip of the ~15–16 MB artifact. The challenge
  caps artifacts at 16 MB, so this is the number that would actually be submitted.
- **Hardware:** ablations on **2×T4 (16 GB each)** via Kaggle, capped at 12 h; **Apple M4** for MLX development and smoke tests.
  Both trainers (`train_cuda.py`, `train.py`) share the same architecture and hyperparameters.

## Results

Baseline is the stock parameter-golf architecture trained for 5K iters × 163K tokens on 2×T4.

| Run | BPB | Δ vs baseline | Status |
|---|---:|---:|---|
| baseline (9L, dim 512) | 1.3198 | — | baseline |
| mixed precision (fp16 `tok_emb`/`lm_head` passthrough) | 1.3164 | −0.0034 | win |
| WD sawtooth (500-epoch, WD=0.01) | 1.3155 | −0.0043 | win |
| **WD sawtooth + mixed precision** | **1.3122** | **−0.0076** | **best** |

Ablations across five phases are under [`experiments/`](experiments/). The winning
notebook is [`best_model/wd_sawtooth_mixed_precision.ipynb`](best_model/wd_sawtooth_mixed_precision.ipynb).

## Layout

```
train_cuda.py           CUDA trainer (PyTorch, Kaggle 2xT4)
train.py                MLX trainer
data/                   dataset + tokenizer download utilities
experiments/            ablation notebooks, grouped by phase
  phase1_optimizer/       optimizer hyperparameters
  phase2_architecture/    SmearGate, value_embed, parallel_resid, MTP
  phase3_quantization/    int8, int4, mixed precision, GPTQ
  phase4_slowrun/         WD sawtooth, SWA, sliding window, stochastic depth, logit avg
  phase5_systems/         Nesterov momentum, cautious WD, cosine LR decay
best_model/             winning + reference notebooks
mac/                    Apple-Silicon compile/fused-CE experiments
reference/              vendored upstream projects (see References)
```

Each experiment notebook is self-contained: it embeds the full trainer with the
one technique under test changed, so it runs top-to-bottom on Kaggle unchanged.

## Setup

```bash
pip install -r requirements.txt
```

## Running

**CUDA** Open any notebook in `experiments/` on Kaggle with the
`microtrain-fineweb-sp1024` dataset attached, or run the trainer directly:

```bash
RUN_ID=my_run ITERATIONS=5000 TRAIN_BATCH_TOKENS=163840 \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  torchrun --standalone --nproc_per_node=2 train_cuda.py
```

**MLX**

```bash
RUN_ID=my_run ITERATIONS=1000 \
  python train.py
```

`train_cuda.py` downloads the dataset from the public HF repo automatically if it
is not already present locally; for the MLX trainer, run `data/download.py` first.

## References

The upstream projects are vendored under `reference/` for reproducibility:

- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — GPU systems & training speed (MIT)
- [parameter-golf](https://github.com/openai/parameter-golf) — model compression & architecture density (MIT)
- [slowrun](https://github.com/qlabs-eng/slowrun) — data efficiency & regularization (MIT)

## License

MIT — see [`LICENSE`](LICENSE). The vendored `reference/` projects retain their own licenses.
