#!/usr/bin/env python3
import os, warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
warnings.filterwarnings("ignore")

"""
DAPT — Domain-Adaptive Pre-Training for XLM-T on Indonesian emotion tweets.

Variants:
  A-full : all parameters updated
  A-emb  : encoder frozen; embeddings + lm_head updated

Parameter distribution (cardiffnlp/twitter-xlm-roberta-base):
  embeddings : 192,398,592  (69.1%) — word / position / token-type / LayerNorm
  encoder    :  85,054,464  (30.6%) — 12 transformer layers
  lm_head    :     842,130   (0.3%) — dense / LayerNorm / bias (decoder tied to embeddings)
  total      : 278,295,186

A-emb trains 69.4% of parameters (vs ~21% for BERT-base) — a key finding for RQ2.

Logged metrics:
  train_loss, train_perplexity, learning_rate, grad_norm   (every logging_steps)
  eval_loss,  eval_perplexity, eval_runtime                (every eval_steps)
  → wandb (project: 2026-xlmt-emotweet, entity: bdsrc-team) + local JSONL
"""

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import wandb
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
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

DEFAULT_BASE_MODEL = "cardiffnlp/twitter-xlm-roberta-base"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WANDB_PROJECT = "2026-xlmt-emotweet"
WANDB_ENTITY = "bdsrc-team"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DAPT for XLM-T — Indonesian emotion tweets")
    p.add_argument(
        "--variant", choices=["full", "emb"], required=True,
        help="A-full: all layers trainable | A-emb: encoder frozen",
    )
    p.add_argument(
        "--base_model", type=str, default=DEFAULT_BASE_MODEL,
        help="HuggingFace model id or local path to use as base for DAPT",
    )
    p.add_argument(
        "--model_id", type=str, default=None,
        help="Short name for output dir (e.g. 'xlmt', 'indo-roberta'). "
             "Defaults to 'xlmt' for the default base model, else derived from --base_model.",
    )
    p.add_argument(
        "--data_path", type=str,
        default=str(PROJECT_ROOT / "data" / "dapt" / "dapt_tweets_1676k.txt"),
    )
    p.add_argument("--output_dir", type=str, default=None,
                   help="Defaults to outputs/dapt/{model_id}/A-{variant}")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=64,
                   help="Per-device batch size (L40S 48 GB)")
    p.add_argument("--grad_accum", type=int, default=4,
                   help="Effective batch = batch_size * grad_accum = 256")
    p.add_argument("--max_epochs", type=int, default=3)
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--val_split", type=float, default=0.02,
                   help="Fraction of corpus reserved for validation (~32K tweets)")
    p.add_argument("--eval_steps", type=int, default=1000,
                   help="Evaluate (and potentially save) every N steps")
    p.add_argument("--early_stopping_patience", type=int, default=3,
                   help="Stop if eval_loss does not improve for N evaluations")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def safe_exp(x: float) -> float:
    """exp(x) capped at inf to avoid OverflowError during early unstable steps."""
    try:
        return math.exp(x)
    except OverflowError:
        return float("inf")


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def freeze_encoder(model) -> None:
    """Freeze all encoder (transformer) layers.

    Keeps trainable: roberta.embeddings.* + lm_head.*
    lm_head.decoder.weight is tied to word_embeddings.weight — gradients flow
    through the vocabulary projection without updating encoder weights.
    """
    for name, param in model.named_parameters():
        if "encoder" in name:
            param.requires_grad = False


def get_param_stats(model, variant: str) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "variant": f"A-{variant}",
        "trainable_params": trainable,
        "total_params": total,
        "pct_trainable": round(100.0 * trainable / total, 4),
    }


# ---------------------------------------------------------------------------
# Trainer subclass — adds perplexity to eval metrics
# ---------------------------------------------------------------------------

class DAPTTrainer(Trainer):
    """Extends Trainer to log eval_perplexity alongside eval_loss."""

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        loss_key = f"{metric_key_prefix}_loss"
        if loss_key in metrics:
            metrics[f"{metric_key_prefix}_perplexity"] = safe_exp(metrics[loss_key])
        return metrics


# ---------------------------------------------------------------------------
# Local logging callback
# ---------------------------------------------------------------------------

class LocalLogCallback(TrainerCallback):
    """Appends every Trainer log event to a JSONL file.

    Renames ambiguous 'loss' → 'train_loss' and adds 'train_perplexity'.
    eval_perplexity is already present via DAPTTrainer.evaluate().
    """

    def __init__(self, log_path: Path):
        self._path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        entry = {"step": state.global_step, "epoch": round(state.epoch or 0.0, 4)}
        for k, v in logs.items():
            entry["train_loss" if k == "loss" else k] = v
        if "loss" in logs:
            entry["train_perplexity"] = safe_exp(logs["loss"])
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_corpus(data_path: str) -> list:
    texts = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                texts.append(text)
    logger.info(f"Loaded {len(texts):,} tweets from {data_path}")
    return texts


def build_hf_datasets(texts: list, val_split: float, tokenizer, max_seq_length: int):
    """Shuffle corpus, split into train/val, then tokenize."""
    rng = random.Random(42)
    indices = list(range(len(texts)))
    rng.shuffle(indices)

    n_val = int(len(indices) * val_split)
    val_set = set(indices[:n_val])
    train_texts = [texts[i] for i in range(len(texts)) if i not in val_set]
    val_texts   = [texts[i] for i in val_set]
    logger.info(f"Split — train: {len(train_texts):,} | val: {len(val_texts):,}")

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    train_ds = Dataset.from_dict({"text": train_texts}).map(
        tokenize_fn, batched=True, remove_columns=["text"],
        desc="Tokenising train",
    )
    val_ds = Dataset.from_dict({"text": val_texts}).map(
        tokenize_fn, batched=True, remove_columns=["text"],
        desc="Tokenising val",
    )
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    # Resolve model_id for output dir and WandB naming
    if args.model_id is None:
        if args.base_model == DEFAULT_BASE_MODEL:
            args.model_id = "xlmt"
        else:
            args.model_id = args.base_model.split("/")[-1]

    run_name = f"{args.model_id}-A-{args.variant}"
    if args.output_dir is None:
        args.output_dir = str(
            PROJECT_ROOT / "outputs" / "dapt" / args.model_id / f"A-{args.variant}"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2))

    # Tokenizer (unchanged — no vocab extension)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Model
    model = AutoModelForMaskedLM.from_pretrained(args.base_model)
    if args.variant == "emb":
        freeze_encoder(model)
        logger.info("Encoder frozen — only embeddings + lm_head are trainable (A-emb)")

    param_stats = get_param_stats(model, args.variant)
    logger.info(
        f"Parameters — trainable: {param_stats['trainable_params']:,} / "
        f"{param_stats['total_params']:,} ({param_stats['pct_trainable']:.2f}%)"
    )
    (output_dir / "param_stats.json").write_text(json.dumps(param_stats, indent=2))

    # Initialise WandB explicitly so config (including param stats) is logged from the start.
    # The Trainer detects an existing run and reuses it instead of creating a new one.
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        group="dapt",
        config={
            **vars(args),
            "base_model": args.base_model,
            "effective_batch_size": args.batch_size * args.grad_accum,
            **param_stats,
        },
    )

    # Data
    texts = load_corpus(args.data_path)
    train_ds, val_ds = build_hf_datasets(
        texts, args.val_split, tokenizer, args.max_seq_length
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    # Note: use evaluation_strategy (not eval_strategy) for transformers 4.40
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.max_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="linear",
        bf16=True,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        # save_total_limit=1: Trainer always preserves the best checkpoint even at limit=1.
        # In practice there may be 2 checkpoint dirs at once (best + latest) but both are
        # cleaned up after training; only best_model/ is kept for HuggingFace upload.
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=100,
        report_to=["wandb"],
        run_name=run_name,
        seed=args.seed,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    trainer = DAPTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            ),
            LocalLogCallback(output_dir / "train_log.jsonl"),
        ],
    )

    logger.info(f"Starting DAPT — variant=A-{args.variant}, lr={args.lr}")
    train_result = trainer.train()

    # Save best model + tokenizer (single directory, ready for HuggingFace upload)
    best_model_dir = output_dir / "best_model"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    logger.info(f"Best model saved → {best_model_dir}")

    # Delete intermediate checkpoints — best model already saved above
    import shutil
    for ckpt in output_dir.glob("checkpoint-*"):
        shutil.rmtree(ckpt, ignore_errors=True)
    logger.info("Intermediate checkpoints deleted")

    # Persist training summary
    best_eval_loss = trainer.state.best_metric
    summary = {
        **param_stats,
        "best_eval_loss": best_eval_loss,
        "best_eval_perplexity": safe_exp(best_eval_loss) if best_eval_loss else None,
        "total_steps": trainer.state.global_step,
        "train_runtime_s": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        "train_steps_per_second": train_result.metrics.get("train_steps_per_second"),
        "total_flos": train_result.metrics.get("total_flos"),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))

    # Sync summary to WandB
    wandb.run.summary.update(summary)

    logger.info(f"Done.\n{json.dumps(summary, indent=2)}")
    wandb.finish()


if __name__ == "__main__":
    main()
