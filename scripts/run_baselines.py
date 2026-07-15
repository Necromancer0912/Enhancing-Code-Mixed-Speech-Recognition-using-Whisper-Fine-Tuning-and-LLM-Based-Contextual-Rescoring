"""Zero-shot baselines on the MUCS Bengali-English test set.

B1  Whisper-small, plain zero-shot
B2  Whisper-small with the descriptive prompt (PromptingWhisper style)
B3  Whisper-large-v3-turbo, zero-shot

Usage:
    python scripts/run_baselines.py [--limit 1000]
"""
import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ROOT  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from clear_bn import PROMPT_1, SEED, WHISPER_SMALL, WHISPER_TURBO, project_paths, wer_variants  # noqa: E402
from clear_bn.transcribe import transcribe  # noqa: E402


def load_whisper(model_id, device):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    proc = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    return proc, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000,
                    help="evaluation subset size, 0 for the full test set")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    paths = project_paths(ROOT)

    test_df = pd.read_parquet(paths["manifests"] / "test.parquet")
    eval_df = test_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    if args.limit:
        eval_df = eval_df.head(args.limit).copy()
    eval_df = eval_df.sort_values(["wav_path", "start_sec"]).reset_index(drop=True)
    print(f"evaluating {len(eval_df):,} utterances")

    proc, model = load_whisper(WHISPER_SMALL, device)
    eval_df["hyp_b1_small_zs"] = transcribe(eval_df, model, proc, desc="B1 whisper-small ZS")
    eval_df["hyp_b2_small_prompt"] = transcribe(
        eval_df, model, proc, prompt=PROMPT_1, desc="B2 whisper-small +prompt")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    proc, model = load_whisper(WHISPER_TURBO, device)
    eval_df["hyp_b3_turbo_zs"] = transcribe(eval_df, model, proc, desc="B3 turbo ZS")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    runs = {
        "B1 Whisper-small (ZS)": "hyp_b1_small_zs",
        "B2 PromptingWhisper-small (ZS+prompt)": "hyp_b2_small_prompt",
        "B3 Whisper-large-v3-turbo (ZS)": "hyp_b3_turbo_zs",
    }
    refs = eval_df.ref_text.tolist()
    rows = [dict(model=name, **{k: round(v, 2) for k, v in wer_variants(refs, eval_df[col].tolist()).items()})
            for name, col in runs.items()]
    results = pd.DataFrame(rows)
    print(results.to_string(index=False))

    results.to_csv(paths["predictions"] / "baseline_results.csv", index=False)
    eval_df.to_parquet(paths["predictions"] / "baseline_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
