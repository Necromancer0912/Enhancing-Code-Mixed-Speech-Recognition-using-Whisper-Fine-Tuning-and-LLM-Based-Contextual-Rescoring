"""Continue fine-tuning from an existing checkpoint at a higher learning rate.

The first 2-epoch run at lr 2e-5 plateaued around 49 percent validation P-WER;
the paper trained far longer (10 epochs at 1e-4), so this picks up the best
checkpoint and pushes on.

Usage:
    python scripts/continue_training.py [--epochs 3] [--lr 5e-5]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ROOT  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from clear_bn import PROMPT_1, SEED, PromptedPrefix, project_paths  # noqa: E402
from clear_bn.training import finetune  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    paths = project_paths(ROOT)
    ckpt_dir = paths["checkpoints"] / "whisper_small_ft_bn"
    resume_from = ckpt_dir / "best"
    assert resume_from.exists(), "no checkpoint to continue from"

    train_df = pd.read_parquet(paths["manifests"] / "train.parquet")
    val_df = pd.read_parquet(paths["manifests"] / "val.parquet")
    print(f"continuing from {resume_from}")
    print(f"train {len(train_df):,} segments, val {len(val_df):,}")

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(resume_from)
    model = WhisperForConditionalGeneration.from_pretrained(
        resume_from, torch_dtype=torch.float32, low_cpu_mem_usage=True).to(device)
    model.config.use_cache = False

    for p in model.model.encoder.parameters():
        p.requires_grad = False

    prefix = PromptedPrefix(processor.tokenizer, PROMPT_1)

    finetune(model, processor, prefix, train_df, val_df, ckpt_dir,
             epochs=args.epochs, batch_size=args.batch_size,
             grad_accum=args.grad_accum, lr=args.lr,
             warmup_updates=100, seed=SEED)


if __name__ == "__main__":
    main()
