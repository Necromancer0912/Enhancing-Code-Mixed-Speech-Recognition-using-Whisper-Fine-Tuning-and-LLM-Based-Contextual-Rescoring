"""Fine-tune Whisper-small on MUCS Bengali-English with the CLEAR recipe:
frozen encoder, decoder-only updates, descriptive prompt in the decoder
context, loss masked to the transcription tokens.

Usage:
    python scripts/train_whisper.py [--epochs 2] [--batch-size 8]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ROOT  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from clear_bn import PROMPT_1, SEED, WHISPER_SMALL, PromptedPrefix, project_paths  # noqa: E402
from clear_bn.training import finetune  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-train-samples", type=int, default=0, help="0 = use all")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    paths = project_paths(ROOT)
    train_df = pd.read_parquet(paths["manifests"] / "train.parquet")
    val_df = pd.read_parquet(paths["manifests"] / "val.parquet")
    if args.max_train_samples:
        train_df = train_df.sample(n=min(args.max_train_samples, len(train_df)),
                                   random_state=SEED).reset_index(drop=True)
    print(f"train {len(train_df):,} segments, val {len(val_df):,}")

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(WHISPER_SMALL)
    model = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_SMALL, torch_dtype=torch.float32, low_cpu_mem_usage=True).to(device)
    model.config.use_cache = False

    # the paper's central design choice: adapt the decoder, keep the encoder
    for p in model.model.encoder.parameters():
        p.requires_grad = False

    prefix = PromptedPrefix(processor.tokenizer, PROMPT_1)
    out_dir = paths["checkpoints"] / "whisper_small_ft_bn"

    finetune(model, processor, prefix, train_df, val_df, out_dir,
             epochs=args.epochs, batch_size=args.batch_size,
             grad_accum=args.grad_accum, lr=args.lr, seed=SEED)


if __name__ == "__main__":
    main()
