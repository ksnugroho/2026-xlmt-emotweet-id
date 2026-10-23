#!/usr/bin/env python3
"""
Runner — executes all 60 fine-tuning runs sequentially.

Experiment matrix (5 × 4 datasets × 3 seeds = 60 runs):
  E1: No DAPT      + full
  E2: A-full DAPT  + full
  E3: A-emb  DAPT  + full
  E4: A-emb  DAPT  + freeze_encoder
  E5: A-full DAPT  + freeze_encoder

Usage:
  # Dry-run — print all commands without executing
  python src/run_finetune.py --dry_run

  # Run all
  python src/run_finetune.py

  # Run specific experiments / datasets / seeds
  python src/run_finetune.py --experiments E2 E3 --datasets D1 D2 --seeds 42

  # Pass extra args to finetune_train.py (e.g. save best models)
  python src/run_finetune.py --extra_args --save_model
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT       = Path(__file__).resolve().parent.parent
XLMT_MODEL         = "cardiffnlp/twitter-xlm-roberta-base"
INDO_ROBERTA_MODEL = "flax-community/indonesian-roberta-base"
MBERT_MODEL        = "bert-base-multilingual-cased"

EXPERIMENTS = {
    # ── XLM-T experiments ──────────────────────────────────────────────────
    "E1": {
        "model_path":  XLMT_MODEL,
        "freeze_mode": "full",
        "description": "XLM-T | No DAPT + full fine-tune (baseline)",
    },
    "E2": {
        "model_path":  str(PROJECT_ROOT / "outputs" / "dapt" / "xlmt" / "A-full" / "best_model"),
        "freeze_mode": "full",
        "description": "XLM-T | A-full DAPT + full fine-tune (upper bound)",
    },
    "E3": {
        "model_path":  str(PROJECT_ROOT / "outputs" / "dapt" / "xlmt" / "A-emb"  / "best_model"),
        "freeze_mode": "full",
        "description": "XLM-T | A-emb DAPT + full fine-tune (RQ2 efficiency claim)",
    },
    "E4": {
        "model_path":  str(PROJECT_ROOT / "outputs" / "dapt" / "xlmt" / "A-emb"  / "best_model"),
        "freeze_mode": "freeze_encoder",
        "description": "XLM-T | A-emb DAPT + freeze_encoder fine-tune (RQ3 double efficiency)",
    },
    "E5": {
        "model_path":  str(PROJECT_ROOT / "outputs" / "dapt" / "xlmt" / "A-full" / "best_model"),
        "freeze_mode": "freeze_encoder",
        "description": "XLM-T | A-full DAPT + freeze_encoder fine-tune (comparison)",
    },
    # ── Indo-RoBERTa experiments (3 epoch DAPT) ────────────────────────────
    "IR-E1": {
        "model_path":  INDO_ROBERTA_MODEL,
        "freeze_mode": "full",
        "description": "Indo-RoBERTa | No DAPT + full fine-tune (monolingual baseline)",
    },
    "IR-E2": {
        "model_path":  str(PROJECT_ROOT / "outputs" / "dapt" / "indo-roberta" / "A-full" / "best_model"),
        "freeze_mode": "full",
        "description": "Indo-RoBERTa | A-full DAPT 3ep + full fine-tune",
    },
    "IR-E3": {
        "model_path":  str(PROJECT_ROOT / "outputs" / "dapt" / "indo-roberta" / "A-emb"  / "best_model"),
        "freeze_mode": "full",
        "description": "Indo-RoBERTa | A-emb DAPT 3ep + full fine-tune",
    },
    # ── mBERT baseline ─────────────────────────────────────────────────────
    "MB-E1": {
        "model_path":  MBERT_MODEL,
        "freeze_mode": "full",
        "description": "mBERT | No DAPT + full fine-tune (multilingual baseline)",
    },
    # ── Indo-RoBERTa experiments (6 epoch DAPT) ────────────────────────────
    "IR-E2-6ep": {
        "model_path":  str(PROJECT_ROOT / "outputs" / "dapt" / "indo-roberta-6ep" / "A-full" / "best_model"),
        "freeze_mode": "full",
        "description": "Indo-RoBERTa | A-full DAPT 6ep + full fine-tune",
    },
    "IR-E3-6ep": {
        "model_path":  str(PROJECT_ROOT / "outputs" / "dapt" / "indo-roberta-6ep" / "A-emb"  / "best_model"),
        "freeze_mode": "full",
        "description": "Indo-RoBERTa | A-emb DAPT 6ep + full fine-tune",
    },
}

DATASETS = ["D1", "D2", "D3", "D4"]
SEEDS    = [42, 123, 456, 789, 999]

ALL_EXPERIMENTS = list(EXPERIMENTS.keys())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Runner for all fine-tuning experiments")
    p.add_argument("--experiments", nargs="+", default=ALL_EXPERIMENTS,
                   choices=ALL_EXPERIMENTS,
                   help="Experiments to run (default: all 11)")
    p.add_argument("--datasets", nargs="+", default=DATASETS,
                   choices=DATASETS,
                   help="Datasets to run (default: all)")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                   help="Seeds to run (default: 42 123 456 789 999)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--continue_on_error", action="store_true",
                   help="Continue to next run if one fails (default: stop on error)")
    p.add_argument("--extra_args", nargs=argparse.REMAINDER, default=[],
                   help="Extra args forwarded to finetune_train.py (e.g. --save_model)")
    return p.parse_args()


def is_done(exp_id: str, dataset: str, seed: int) -> bool:
    summary = (
        PROJECT_ROOT / "outputs" / "finetune"
        / exp_id / dataset / f"seed{seed}" / "summary.json"
    )
    return summary.exists()


def build_runs(experiments, datasets, seeds, extra_args):
    script = str(PROJECT_ROOT / "src" / "finetune_train.py")
    runs = []
    for exp_id in experiments:
        exp = EXPERIMENTS[exp_id]
        for dataset in datasets:
            for seed in seeds:
                cmd = [
                    sys.executable, script,
                    "--experiment",  exp_id,
                    "--model_path",  exp["model_path"],
                    "--dataset",     dataset,
                    "--seed",        str(seed),
                    "--freeze_mode", exp["freeze_mode"],
                ] + extra_args
                runs.append({
                    "label": f"{exp_id}-{dataset}-seed{seed}",
                    "description": exp["description"],
                    "cmd": cmd,
                    "done": is_done(exp_id, dataset, seed),
                })
    return runs


def main():
    args = parse_args()
    runs  = build_runs(args.experiments, args.datasets, args.seeds, args.extra_args)
    total = len(runs)

    done_count = sum(1 for r in runs if r["done"])
    print(f"Total runs: {total}  (already done: {done_count}, pending: {total - done_count})\n")
    for i, run in enumerate(runs, 1):
        status = "[DONE]" if run["done"] else "      "
        print(f"  {status} [{i:>2}/{total}] {run['label']}  —  {run['description']}")

    if args.dry_run:
        print("\n-- dry_run: pending commands --")
        for run in runs:
            if not run["done"]:
                print(" ".join(run["cmd"]))
        return

    print()
    failed = []
    for i, run in enumerate(runs, 1):
        if run["done"]:
            print(f"[{i}/{total}] SKIP (already done): {run['label']}")
            continue
        print(f"\n{'='*60}")
        print(f"[{i}/{total}] {run['label']}")
        print(f"{'='*60}")
        result = subprocess.run(run["cmd"])
        if result.returncode != 0:
            msg = f"FAILED: {run['label']} (exit {result.returncode})"
            print(msg)
            failed.append(run["label"])
            if not args.continue_on_error:
                print("Stopping. Use --continue_on_error to skip failures.")
                sys.exit(result.returncode)

    print(f"\n{'='*60}")
    print(f"Completed {total - len(failed)}/{total} runs.")
    if failed:
        print(f"Failed runs ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All runs completed successfully.")


if __name__ == "__main__":
    main()
