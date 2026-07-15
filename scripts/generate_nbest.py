"""Generate n-best hypothesis lists from the fine-tuned Whisper with beam
search, keeping Whisper's own sequence scores for later interpolation.

Usage:
    python scripts/generate_nbest.py [--n-best 5] [--limit 1000]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ROOT  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from clear_bn import PROMPT_1, SEED, PromptedPrefix, project_paths  # noqa: E402
from clear_bn.transcribe import generate_nbest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-best", type=int, default=5)
    ap.add_argument("--limit", type=int, default=1000, help="0 = full test set")
    ap.add_argument("--val-limit", type=int, default=300,
                    help="validation utterances for lambda tuning")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    paths = project_paths(ROOT)
    ckpt = paths["checkpoints"] / "whisper_small_ft_bn" / "best"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} missing - run scripts/train_whisper.py first")

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(ckpt)
    model = WhisperForConditionalGeneration.from_pretrained(
        ckpt, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda").eval()
    model.config.use_cache = True
    prefix = PromptedPrefix(processor.tokenizer, PROMPT_1)

    test_df = pd.read_parquet(paths["manifests"] / "test.parquet")
    eval_df = test_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    if args.limit:
        eval_df = eval_df.head(args.limit).copy()
    eval_df = eval_df.sort_values(["wav_path", "start_sec"]).reset_index(drop=True)

    val_df = pd.read_parquet(paths["manifests"] / "val.parquet")
    val_tune = val_df.sample(n=min(args.val_limit, len(val_df)), random_state=SEED).reset_index(drop=True)

    out = paths["predictions"]
    # references are short (10-20 words), so 128 new tokens is a generous cap
    # and keeps beam search from crawling through repetition loops
    nbest = generate_nbest(eval_df, model, processor, prefix, n_best=args.n_best,
                           batch_size=2, max_new_tokens=128, desc="test n-best")
    nbest.to_parquet(out / f"nbest_test_beam{args.n_best}.parquet", index=False)
    print(f"saved test n-best: {len(nbest):,} rows")

    nbest_val = generate_nbest(val_tune, model, processor, prefix, n_best=args.n_best,
                               batch_size=2, max_new_tokens=128, desc="val n-best")
    nbest_val.to_parquet(out / f"nbest_val_beam{args.n_best}.parquet", index=False)
    print(f"saved val n-best: {len(nbest_val):,} rows")


if __name__ == "__main__":
    main()
