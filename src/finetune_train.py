#!/usr/bin/env python3
import os, warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
warnings.filterwarnings("ignore")

"""
Fine-tuning (Tahap C) — Emotion classification on Indonesian tweet datasets.

Experiment conditions:
  E1: No DAPT     + fine-tune full
  E2: A-full DAPT + fine-tune full
  E3: A-emb  DAPT + fine-tune full
  E4: A-emb  DAPT + fine-tune freeze_encoder
  E5: A-full DAPT + fine-tune freeze_encoder

Freeze modes:
  full               : all parameters trainable
  freeze_encoder     : roberta.encoder frozen; embeddings + classifier trainable
  freeze_encoder_emb : roberta.encoder + embeddings frozen; classifier only

Primary metric: macro-F1 (not loss — label schemas differ and are imbalanced)

Logged metrics per epoch:
  train_loss, eval_loss, eval_macro_f1, eval_accuracy, eval_f1_<class>
  → wandb (project: 2026-xlmt-emotweet, entity: bdsrc-team) + local JSONL
  Test set evaluated once at the end with best checkpoint.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

logging.basicConfig(
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WANDB_PROJECT = "2026-xlmt-emotweet"
WANDB_ENTITY  = "bdsrc-team"

DATASET_NAMES = {
    "D1": "EmoTweetID",
    "D2": "EmoCMT",
    "D3": "EmoT",
    "D4": "EmoPubID",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tuning XLM-T for emotion classification")
    p.add_argument("--experiment", type=str, required=True,
                   help="Experiment ID (E1–E5), used for logging and output path")
    p.add_argument("--model_path", type=str, required=True,
                   help="HuggingFace model name or path to DAPT best_model/")
    p.add_argument("--dataset", choices=["D1", "D2", "D3", "D4"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--freeze_mode",
                   choices=["full", "freeze_encoder", "freeze_encoder_emb"],
                   required=True)
    p.add_argument("--data_dir", type=str,
                   default=str(PROJECT_ROOT / "data" / "fine-tuning"))
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=20)
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--early_stopping_patience", type=int, default=5)
    p.add_argument("--save_model", action="store_true",
                   help="Save best model to best_model/ (off by default to save disk)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def apply_freeze(model, freeze_mode: str) -> None:
    if freeze_mode == "full":
        return
    for name, param in model.named_parameters():
        if freeze_mode == "freeze_encoder" and "encoder" in name:
            param.requires_grad = False
        elif freeze_mode == "freeze_encoder_emb" and (
            "encoder" in name or "embeddings" in name
        ):
            param.requires_grad = False


def get_param_stats(model, freeze_mode: str) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return {
        "freeze_mode": freeze_mode,
        "trainable_params": trainable,
        "total_params": total,
        "pct_trainable": round(100.0 * trainable / total, 4),
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TweetDataset(TorchDataset):
    def __init__(self, encodings: dict, labels: list):
        self.encodings = encodings
        self.labels    = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def load_split(
    path: Path,
    label2id: dict,
    tokenizer,
    max_seq_length: int,
) -> TweetDataset:
    df = pd.read_csv(path)
    texts  = df["tweet"].astype(str).tolist()
    labels = [label2id[str(l).strip()] for l in df["label"].tolist()]
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
        return_tensors="pt",
    )
    return TweetDataset(encodings, labels)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def build_compute_metrics(id2label: dict):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, preds)

        # Aggregate metrics
        result = {
            "accuracy":        accuracy,
            "macro_f1":        f1_score(labels,        preds, average="macro",    zero_division=0),
            "macro_precision": precision_score(labels, preds, average="macro",    zero_division=0),
            "macro_recall":    recall_score(labels,    preds, average="macro",    zero_division=0),
            "weighted_f1":        f1_score(labels,        preds, average="weighted", zero_division=0),
            "weighted_precision": precision_score(labels, preds, average="weighted", zero_division=0),
            "weighted_recall":    recall_score(labels,    preds, average="weighted", zero_division=0),
            "micro_f1":        f1_score(labels,        preds, average="micro",    zero_division=0),
        }

        # Per-class precision, recall, F1
        per_f1   = f1_score(labels,        preds, average=None, zero_division=0)
        per_prec = precision_score(labels, preds, average=None, zero_division=0)
        per_rec  = recall_score(labels,    preds, average=None, zero_division=0)
        for i in range(len(per_f1)):
            lbl = id2label[i]
            result[f"f1_{lbl}"]        = round(float(per_f1[i]),   4)
            result[f"precision_{lbl}"] = round(float(per_prec[i]), 4)
            result[f"recall_{lbl}"]    = round(float(per_rec[i]),  4)

        return result
    return compute_metrics


# ---------------------------------------------------------------------------
# Trainer subclass — skip optimizer state in checkpoints
# ---------------------------------------------------------------------------

class FinetuneTrainer(Trainer):
    """Saves only model weights in checkpoints (no optimizer/scheduler state).

    Reduces checkpoint size from ~3.3 GB to ~1.1 GB.
    Safe because we never resume fine-tuning from a checkpoint mid-run.
    load_best_model_at_end still works — it only needs the model weights.
    """

    def _save_optimizer_and_scheduler(self, output_dir):
        pass  # Skip — optimizer state not needed; we never resume fine-tuning


# ---------------------------------------------------------------------------
# Local logging callback
# ---------------------------------------------------------------------------

class LocalLogCallback(TrainerCallback):
    def __init__(self, log_path: Path):
        self._path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        entry = {"step": state.global_step, "epoch": round(state.epoch or 0.0, 4)}
        for k, v in logs.items():
            entry["train_loss" if k == "loss" else k] = v
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    seed_str = f"seed{args.seed}"
    run_name  = f"{args.experiment}-{args.dataset}-{seed_str}"

    if args.output_dir is None:
        args.output_dir = str(
            PROJECT_ROOT / "outputs" / "finetune"
            / args.experiment / args.dataset / seed_str
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2))

    # Labels — derived from training split for this seed
    split_dir = Path(args.data_dir) / args.dataset / seed_str
    train_df     = pd.read_csv(split_dir / "train.csv")
    labels_sorted = sorted(train_df["label"].astype(str).str.strip().unique())
    label2id = {l: i for i, l in enumerate(labels_sorted)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(labels_sorted)
    logger.info(
        f"{args.dataset} ({DATASET_NAMES[args.dataset]}): "
        f"{num_labels} classes — {labels_sorted}"
    )

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # DAPT model has no classifier head
    )

    apply_freeze(model, args.freeze_mode)
    param_stats = get_param_stats(model, args.freeze_mode)
    logger.info(
        f"Parameters — trainable: {param_stats['trainable_params']:,} / "
        f"{param_stats['total_params']:,} ({param_stats['pct_trainable']:.2f}%)"
    )
    (output_dir / "param_stats.json").write_text(json.dumps(param_stats, indent=2))

    # WandB — group by (experiment, dataset) so all seeds are grouped together
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        group=f"{args.experiment}-{args.dataset}",
        config={
            **vars(args),
            "dataset_name": DATASET_NAMES[args.dataset],
            "num_labels": num_labels,
            "labels": labels_sorted,
            **param_stats,
        },
    )

    # Datasets
    train_ds = load_split(split_dir / "train.csv", label2id, tokenizer, args.max_seq_length)
    dev_ds   = load_split(split_dir / "dev.csv",   label2id, tokenizer, args.max_seq_length)
    test_ds  = load_split(split_dir / "test.csv",  label2id, tokenizer, args.max_seq_length)
    logger.info(
        f"Splits loaded — train: {len(train_ds)}, dev: {len(dev_ds)}, test: {len(test_ds)}"
    )

    # Training arguments
    # Note: evaluation_strategy (not eval_strategy) for transformers 4.40
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.max_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        bf16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="epoch",
        report_to=["wandb"],
        run_name=run_name,
        seed=args.seed,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    trainer = FinetuneTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=build_compute_metrics(id2label),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            ),
            LocalLogCallback(output_dir / "train_log.jsonl"),
        ],
    )

    logger.info(f"Starting fine-tuning — {run_name}")
    trainer.train()

    # Delete checkpoints after training — best model already loaded in memory
    # (load_best_model_at_end=True). We only need metrics, not saved weights.
    for ckpt in output_dir.glob("checkpoint-*"):
        import shutil
        shutil.rmtree(ckpt, ignore_errors=True)
    logger.info("Checkpoints deleted — keeping metrics only")

    # Evaluate on test set once with best model.
    # Remove EarlyStoppingCallback first — it would fire a spurious warning because
    # the test prefix ("test_") doesn't match the training metric ("eval_macro_f1").
    trainer.remove_callback(EarlyStoppingCallback)
    logger.info("Evaluating on test set with best model...")
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    logger.info(f"Test results: {json.dumps(test_metrics, indent=2)}")

    # Run summary
    summary = {
        "experiment": args.experiment,
        "dataset": args.dataset,
        "dataset_name": DATASET_NAMES[args.dataset],
        "seed": args.seed,
        "freeze_mode": args.freeze_mode,
        **param_stats,
        "best_dev_macro_f1": trainer.state.best_metric,
        "test_macro_f1": test_metrics.get("test_macro_f1"),
        "test_accuracy": test_metrics.get("test_accuracy"),
        "total_epochs": trainer.state.epoch,
        "total_steps": trainer.state.global_step,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    wandb.run.summary.update({**test_metrics, "best_dev_macro_f1": trainer.state.best_metric})

    if args.save_model:
        best_model_dir = output_dir / "best_model"
        trainer.save_model(str(best_model_dir))
        tokenizer.save_pretrained(str(best_model_dir))
        logger.info(f"Model saved → {best_model_dir}")

    wandb.finish()
    logger.info(
        f"Done — {run_name} | "
        f"test macro-F1: {test_metrics.get('test_macro_f1', 'N/A'):.4f}"
    )


if __name__ == "__main__":
    main()
