# Domain-Adaptive Pre-Training for Indonesian Emotion Tweet Classification

This repository contains the code and results for the experiment `2026-xlmt-emotweet`, which studies Domain-Adaptive Pre-Training (DAPT) strategies for Indonesian emotion tweet classification. The central research question is whether **embedding-only DAPT** achieves comparable performance to full DAPT at a fraction of the computational cost.

## Research Questions

| RQ | Question |
|----|----------|
| RQ1 | Does DAPT on an Indonesian emotion tweet corpus improve performance over no adaptation? |
| RQ2 | Does **embedding-only** DAPT (encoder frozen) match full DAPT? *(core contribution)* |
| RQ3 | How does a frozen-encoder fine-tuning strategy behave in low-resource settings? |
| RQ4 | Does DAPT benefit depend on the initial domain gap of the model? *(motivation for adding Indo-RoBERTa)* |

## Repository Structure

```
.
├── data/
│   ├── dapt/
│   │   └── dapt_tweets_1676k.txt       # 1.676M Indonesian tweets for DAPT (gitignored, ~120MB)
│   │                                   # Download: https://drive.google.com/file/d/13-f7oQTZZb0fbkO5UpIKocnKDDGJyFNH/view?usp=sharing
│   └── fine-tuning/
│       ├── D1/seed{42,123,456,789,999}/  # EmoTweetID  — train/dev/test.csv (6 classes, 337 test)
│       ├── D2/seed{42,123,456,789,999}/  # EmoCMT      — train/dev/test.csv (5 classes,  88 test)
│       ├── D3/seed{42,123,456,789,999}/  # EmoT        — train/dev/test.csv (5 classes, 661 test)
│       └── D4/seed{42,123,456,789,999}/  # EmoPubID    — train/dev/test.csv (6 classes,1063 test)
├── src/
│   ├── dapt_train.py       # Continued MLM pre-training (DAPT) with early stopping
│   ├── finetune_train.py   # Single fine-tuning run with WandB logging & summary.json output
│   └── run_finetune.py     # Runner: iterates all experiment × dataset × seed combinations
├── outputs/
│   ├── dapt/               # DAPT model checkpoints (gitignored, ~4GB total)
│   │   ├── A-full/         # XLM-T A-full best_model
│   │   ├── A-emb/          # XLM-T A-emb best_model
│   │   ├── indo-roberta/   # Indo-RoBERTa 3ep (A-full + A-emb)
│   │   └── indo-roberta-6ep/  # Indo-RoBERTa 6ep (A-full + A-emb)
│   ├── finetune/           # Fine-tuning results — 220 runs (11 conditions × 4 datasets × 5 seeds)
│   │   └── <exp>/<dataset>/seed<N>/
│   │       ├── summary.json        # test_macro_f1, accuracy, best_dev_f1
│   │       ├── test_metrics.json   # Full per-class metrics on test set
│   │       ├── train_log.jsonl     # Per-epoch train/eval log
│   │       ├── run_config.json     # Hyperparameters used
│   │       └── param_stats.json    # Trainable parameter count
│   └── report/
│       ├── gen_figures.py          # Generates fig1–fig8 from aggregated results
│       ├── report_id.md            # Full experiment report (Indonesian)
│       └── figures/                # fig1–fig8 (PNG)
├── requirements.txt
└── README.md
```

## Models

| Model | HuggingFace | Params | Embedding % | Domain Gap |
|-------|-------------|--------|-------------|------------|
| [mBERT](https://aclanthology.org/N19-1423/) | [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased) | 178M | ~22.6% | Large (Wikipedia formal text → informal tweets) |
| Indo-RoBERTa | [indonesian-roberta-base](https://huggingface.co/flax-community/indonesian-roberta-base) | 125M | ~31.8% | Large (OSCAR formal text → informal tweets) |
| [XLM-T](https://aclanthology.org/2022.lrec-1.27/) | [twitter-xlm-roberta-base](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base) | 278M | 69.4% | Small (pre-trained on 198M multilingual tweets) |

mBERT serves as a standard multilingual baseline. Indo-RoBERTa and XLM-T are the primary models for DAPT experiments.

## DAPT Model Weights

Pre-trained DAPT checkpoints are available on HuggingFace: [ksnugroho/xlmt-emotweet-id](https://huggingface.co/ksnugroho/xlmt-emotweet-id)

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Available subfolders: xlmt-a-full, xlmt-a-emb,
#   indo-roberta-a-full-3ep, indo-roberta-a-emb-3ep,
#   indo-roberta-a-full-6ep, indo-roberta-a-emb-6ep
subfolder = "xlmt-a-full"

model = AutoModelForMaskedLM.from_pretrained(
    "ksnugroho/xlmt-emotweet-id", subfolder=subfolder
)
tokenizer = AutoTokenizer.from_pretrained(
    "ksnugroho/xlmt-emotweet-id", subfolder=subfolder
)
```

## Experiment Matrix

| Condition | Model | DAPT | DAPT strategy | Fine-tune |
|-----------|-------|------|---------------|-----------|
| MB-E1 | mBERT | None | — | Full |
| E1 | XLM-T | None | — | Full |
| E2 | XLM-T | 3 ep | A-full (all params) | Full |
| E3 | XLM-T | 3 ep | A-emb (encoder frozen) | Full |
| E4 | XLM-T | 3 ep | A-emb | Encoder frozen |
| E5 | XLM-T | 3 ep | A-full | Encoder frozen |
| IR-E1 | Indo-RoBERTa | None | — | Full |
| IR-E2 | Indo-RoBERTa | 3 ep | A-full | Full |
| IR-E3 | Indo-RoBERTa | 3 ep | A-emb | Full |
| IR-E2-6ep | Indo-RoBERTa | 6 ep | A-full | Full |
| IR-E3-6ep | Indo-RoBERTa | 6 ep | A-emb | Full |

**11 conditions × 4 datasets × 5 seeds = 220 fine-tuning runs** total (seeds: 42, 123, 456, 789, 999). MB-E1 is a no-DAPT external baseline; DAPT is only performed for XLM-T and Indo-RoBERTa conditions.

## Training Parameters

### DAPT

| Parameter | Value |
|-----------|-------|
| Objective | Masked Language Modeling (MLM) |
| MLM probability | 0.15 |
| Max sequence length | 128 |
| Learning rate | 1e-4 |
| LR scheduler | Linear with warmup |
| Warmup steps | 500 |
| Batch size (per device) | 64 |
| Gradient accumulation | 4 (effective batch = 256) |
| Weight decay | 0.01 |
| Max epochs | 3 (XLM-T) / 6 (Indo-RoBERTa) |
| Early stopping patience | 3 (eval every 1,000 steps) |
| Validation split | 2% of corpus |
| Precision | bf16 |

### Fine-tuning

| Parameter | Value |
|-----------|-------|
| Max epochs | 20 (with early stopping) |
| Early stopping patience | 5 epochs |
| Best model metric | Macro-F1 |
| Learning rate | 2e-5 |
| LR scheduler | Linear with warmup |
| Warmup ratio | 10% of total steps |
| Batch size (per device) | 32 |
| Max sequence length | 128 |
| Weight decay | 0.01 |
| Evaluation strategy | Per epoch |
| Precision | bf16 |

## Datasets

| ID | Dataset | Labels | Train | Dev | Test |
|----|---------|--------|-------|-----|------|
| D1 | [EmoTweetID](https://doi.org/10.1016/j.dib.2026.113119) | anger, disgust, fear, joy, sadness, surprise | 1,570 | 336 | 337 |
| D2 | [EmoCMT](https://doi.org/10.14569/IJACSA.2021.0121177) | anger, fear, joy, love, sadness | 407 | 87 | 88 |
| D3 | [EmoT](https://doi.org/10.1109/IALP.2018.8629262) | anger, fear, happy, love, sadness | 3,080 | 660 | 661 |
| D4 | [EmoPubID](https://doi.org/10.1016/j.dib.2022.108465) | anger, fear, joy, love, neutral, sad | 4,955 | 1,062 | 1,063 |

Splits use a stratified 70/15/15 train/dev/test ratio, reproducible per seed.

**DAPT corpus:** The unlabeled tweet corpus used for domain-adaptive pre-training is derived from `EmoTweetID-Corpus.csv` (the full unlabeled tweet collection in the EmoTweetID dataset), filtered to remove duplicates and any samples overlapping with the fine-tuning test sets, resulting in **1,676,843 tweets** (`data/dapt/dapt_tweets_1676k.txt`).

> **Download:** The processed corpus file is available at [Google Drive](https://drive.google.com/file/d/13-f7oQTZZb0fbkO5UpIKocnKDDGJyFNH/view?usp=sharing) (120 MB). Place it at `data/dapt/dapt_tweets_1676k.txt` before running DAPT.

## Environment

```bash
pip install -r requirements.txt
```

Requires Python 3.10 and a CUDA-capable GPU. Tested on NVIDIA L40S (48 GB VRAM) with CUDA 13.0.

## Reproducing the Experiment

### 1. Run DAPT

Download the DAPT corpus first (see [Datasets](#datasets)), then run all 6 DAPT variants:

```bash
# XLM-T — A-full (all parameters)
python src/dapt_train.py \
  --base_model cardiffnlp/twitter-xlm-roberta-base \
  --model_id xlmt \
  --freeze_mode full \
  --max_epochs 3 \
  --output_dir outputs/dapt/xlmt/A-full

# XLM-T — A-emb (encoder frozen)
python src/dapt_train.py \
  --base_model cardiffnlp/twitter-xlm-roberta-base \
  --model_id xlmt \
  --freeze_mode freeze_encoder \
  --max_epochs 3 \
  --output_dir outputs/dapt/xlmt/A-emb

# Indo-RoBERTa — A-full, 3 epochs
python src/dapt_train.py \
  --base_model flax-community/indonesian-roberta-base \
  --model_id indo-roberta \
  --freeze_mode full \
  --max_epochs 3 \
  --output_dir outputs/dapt/indo-roberta/A-full

# Indo-RoBERTa — A-emb, 3 epochs
python src/dapt_train.py \
  --base_model flax-community/indonesian-roberta-base \
  --model_id indo-roberta \
  --freeze_mode freeze_encoder \
  --max_epochs 3 \
  --output_dir outputs/dapt/indo-roberta/A-emb

# Indo-RoBERTa — A-full, 6 epochs
python src/dapt_train.py \
  --base_model flax-community/indonesian-roberta-base \
  --model_id indo-roberta-6ep \
  --freeze_mode full \
  --max_epochs 6 \
  --output_dir outputs/dapt/indo-roberta-6ep/A-full

# Indo-RoBERTa — A-emb, 6 epochs
python src/dapt_train.py \
  --base_model flax-community/indonesian-roberta-base \
  --model_id indo-roberta-6ep \
  --freeze_mode freeze_encoder \
  --max_epochs 6 \
  --output_dir outputs/dapt/indo-roberta-6ep/A-emb
```

### 2. Run fine-tuning

```bash
# All 11 conditions, all datasets, all 5 seeds (220 runs total) — skips already-done runs
python src/run_finetune.py

# Selective run
python src/run_finetune.py --experiments E1 MB-E1 IR-E1 --datasets D1 D2 --seeds 42

# Dry-run — print all pending commands without executing
python src/run_finetune.py --dry_run
```

The runner skips runs where `outputs/finetune/<exp>/<dataset>/seed<N>/summary.json` already exists.

## Source Files

| File | Description |
|------|-------------|
| `src/dapt_train.py` | DAPT training (continued MLM) with early stopping and checkpoint cleanup |
| `src/finetune_train.py` | Single fine-tuning run with WandB logging and `summary.json` output |
| `src/run_finetune.py` | Runner that iterates all experiment × dataset × seed combinations |
| `outputs/report/gen_figures.py` | Generates all matplotlib figures from aggregated results |

## Results Structure

Each completed run writes to `outputs/finetune/<exp>/<dataset>/seed<N>/`:

| File | Contents |
|------|----------|
| `summary.json` | Final macro-F1, accuracy, per-class F1 on test set |
| `test_metrics.json` | Full test metrics |
| `train_log.jsonl` | Per-step training/eval log |
| `run_config.json` | Hyperparameters and paths used for the run |
| `param_stats.json` | Trainable parameter count |

## Key Findings (5-seed, 220 fine-tuning runs)

**XLM-T macro-F1 (mean ± std over 5 seeds):**

| Condition | D1 | D2 | D3 | D4 | Avg |
|-----------|----|----|----|----|-----|
| E1 No DAPT *(baseline)* | 0.838 ± 0.012 | 0.549 ± 0.079 | 0.757 ± 0.014 | 0.773 ± 0.006 | **0.729** |
| E2 A-full DAPT | 0.838 ± 0.010 | 0.520 ± 0.026 | 0.780 ± 0.017 | 0.783 ± 0.007 | **0.730** |
| E3 A-emb DAPT | 0.833 ± 0.021 | 0.523 ± 0.028 | 0.764 ± 0.014 | 0.774 ± 0.011 | **0.723** |
| E4 A-emb + freeze FT | 0.181 ± 0.038 | 0.156 ± 0.051 | 0.485 ± 0.011 | 0.556 ± 0.015 | **0.344** |
| E5 A-full + freeze FT | 0.296 ± 0.027 | 0.215 ± 0.018 | 0.596 ± 0.021 | 0.629 ± 0.014 | **0.434** |

**Indo-RoBERTa macro-F1 (mean ± std over 5 seeds):**

| Condition | D1 | D2 | D3 | D4 | Avg |
|-----------|----|----|----|----|-----|
| IR-E1 No DAPT *(baseline)* | 0.808 ± 0.020 | 0.320 ± 0.038 | 0.752 ± 0.009 | 0.758 ± 0.015 | **0.660** |
| IR-E2 A-full DAPT 3ep | 0.821 ± 0.016 | 0.353 ± 0.050 | 0.758 ± 0.013 | 0.771 ± 0.009 | **0.676** |
| IR-E3 A-emb DAPT 3ep | 0.817 ± 0.020 | 0.323 ± 0.045 | 0.755 ± 0.016 | 0.767 ± 0.015 | **0.666** |
| IR-E2-6ep A-full DAPT 6ep | **0.829 ± 0.017** | **0.401 ± 0.035** | 0.752 ± 0.007 | 0.770 ± 0.007 | **0.688** |
| IR-E3-6ep A-emb DAPT 6ep | 0.820 ± 0.016 | 0.340 ± 0.041 | 0.758 ± 0.018 | 0.764 ± 0.013 | **0.671** |

**mBERT baseline (MB-E1, mean ± std over 5 seeds):**

| Condition | D1 | D2 | D3 | D4 | Avg |
|-----------|----|----|----|----|-----|
| MB-E1 mBERT No DAPT | 0.811 ± 0.009 | 0.321 ± 0.060 | 0.650 ± 0.015 | 0.721 ± 0.009 | **0.626** |

**Summary:**

- **mBERT vs XLM-T**: XLM-T outperforms mBERT by **+10.3%** average Macro-F1 (0.729 vs 0.626), with especially large gaps on D2 (+22.8%) and D3 (+10.7%), confirming the value of tweet-domain pre-training.
- **mBERT vs Indo-RoBERTa**: Indo-RoBERTa with A-full 6ep DAPT outperforms mBERT by **+6.2%** average (0.688 vs 0.626), showing that domain-specific pre-training + DAPT together bridge the gap.
- **XLM-T**: DAPT yields no meaningful gain (avg delta ≈ +0.001). The model is already domain-adjacent via its multilingual tweet pre-training.
- **Indo-RoBERTa**: DAPT improves consistently — A-full 3ep +1.6%, A-full 6ep **+2.8%** average Macro-F1 over no-DAPT. Large domain gap (formal text → informal tweets) concentrates the adaptation signal.
- **A-emb vs A-full**: For XLM-T (embedding = 69.4% of params), A-emb matches A-full within −0.7% while saving **43% DAPT training time** (121 min vs 214 min). For Indo-RoBERTa (embedding = 31.8%), A-full outperforms A-emb by −1.0% to −1.7%.

**Statistical Significance** (Wilcoxon signed-rank, two-sided, n=5 per-seed averages):

| Comparison | Direction consistent? | p-value |
|---|---|---|
| IR E2-6ep vs IR-E1 (RQ1: DAPT gain) | Yes (+) | **0.0625** |
| XLM-T E3 vs E2 (RQ2: A-emb vs A-full) | No (mixed) | 0.3125 |
| IR E3 vs E2 3ep (RQ2: A-emb vs A-full) | Yes (−) | **0.0625** |
| IR E3-6ep vs E2-6ep 6ep (RQ2: A-emb vs A-full) | Yes (−) | **0.0625** |

> With n=5, the minimum achievable two-tailed p-value is **0.0625** (all 5 differences in the same direction). Three comparisons reach this minimum. Achieving p < 0.05 two-tailed requires n ≥ 6.

## WandB

Runs are logged to project `2026-xlmt-emotweet`, entity `bdsrc-team` (hosted on CoreWeave Forge).

[Experiment Report — Domain-Adaptive Pre-Training for Indonesian Emotion Tweet Classification](https://forge.coreweave.com/wandb/bdsrc-team/2026-xlmt-emotweet/reports/Experiment-Report-Domain-Adaptive-Pre-Training-for-Indonesian-Emotion-Tweet-Classification--VmlldzoxODE2NzQ3MQ?accessToken=r34je8car69xaokj19sohney8umu424w2jk03oi5usgubzwc6qtexog371k5weru)

## Acknowledgment

Computational resources were provided through the **Cloudeka Research Acceleration Grant** in the form of GPU cloud services. Experiments were conducted on an **NVIDIA L40S** GPU (48 GB VRAM).

This research was supported by the **NVIDIA AI Research and Development Center (AIRDC)**, Bina Nusantara University, Jakarta, Indonesia.

## Lead Researcher

**Kuncahyo Setyo Nugroho** — Bina Nusantara University, Jakarta, Indonesia
