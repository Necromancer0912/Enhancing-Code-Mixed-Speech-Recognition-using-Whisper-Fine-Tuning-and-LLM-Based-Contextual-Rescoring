"""Rescore the n-best lists and produce the final comparison table.

Selection strategies evaluated:
  - Whisper beam top-1 (no rescoring)
  - CLEAR with each off-the-shelf LLM (paper Eq. 1-2)
  - length-normalised LLM scores (ours)
  - ASR/LLM score interpolation with lambda tuned on validation (ours)
  - MBR decoding over the n-best list (ours)
  - Domain-Adaptive Rescorer: LoRA-tuned BLOOM on the training transcripts (ours)

Usage:
    python scripts/rescore_and_evaluate.py [--n-best 5] [--limit 1000]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ROOT  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from clear_bn import DAR_BASE_MODEL, RESCORER_MODELS, SEED, project_paths  # noqa: E402
from clear_bn.dar import train_dar_adapter  # noqa: E402
from clear_bn.rescoring import (  # noqa: E402
    llm_score_texts,
    mbr_select,
    select,
    tune_lambda,
    wer_of_selection,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-best", type=int, default=5)
    ap.add_argument("--limit", type=int, default=1000, help="0 = full test set")
    ap.add_argument("--skip-dar", action="store_true", help="skip the LoRA rescorer")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    paths = project_paths(ROOT)
    pred = paths["predictions"]

    nbest = pd.read_parquet(pred / f"nbest_test_beam{args.n_best}.parquet")
    nbest_val = pd.read_parquet(pred / f"nbest_val_beam{args.n_best}.parquet")

    test_df = pd.read_parquet(paths["manifests"] / "test.parquet")
    eval_df = test_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    if args.limit:
        eval_df = eval_df.head(args.limit).copy()

    val_df = pd.read_parquet(paths["manifests"] / "val.parquet")
    val_tune = val_df[val_df.utt_id.isin(nbest_val.utt_id.unique())].reset_index(drop=True)

    # score with every off-the-shelf LLM, caching columns in the parquet
    for name, model_id in RESCORER_MODELS.items():
        col = f"llm_{name}"
        if col not in nbest.columns:
            tot, ln = llm_score_texts(nbest.hyp.tolist(), model_id, desc=f"{name} (test)")
            nbest[col], nbest[col + "_len"] = tot, ln
            tot, ln = llm_score_texts(nbest_val.hyp.tolist(), model_id, desc=f"{name} (val)")
            nbest_val[col], nbest_val[col + "_len"] = tot, ln
            nbest.to_parquet(pred / f"nbest_test_beam{args.n_best}.parquet", index=False)
            nbest_val.to_parquet(pred / f"nbest_val_beam{args.n_best}.parquet", index=False)

    # the domain-adaptive rescorer
    if not args.skip_dar:
        dar_dir = paths["checkpoints"] / "bloom1b1_dar_lora"
        if not (dar_dir / "adapter_model.safetensors").exists():
            train_texts = pd.read_parquet(paths["manifests"] / "train.parquet").clean_text.tolist()
            train_dar_adapter(train_texts, dar_dir)
        if "llm_dar" not in nbest.columns:
            tot, ln = llm_score_texts(nbest.hyp.tolist(), DAR_BASE_MODEL,
                                      adapter_dir=dar_dir, desc="DAR (test)")
            nbest["llm_dar"], nbest["llm_dar_len"] = tot, ln
            tot, ln = llm_score_texts(nbest_val.hyp.tolist(), DAR_BASE_MODEL,
                                      adapter_dir=dar_dir, desc="DAR (val)")
            nbest_val["llm_dar"], nbest_val["llm_dar_len"] = tot, ln
            nbest.to_parquet(pred / f"nbest_test_beam{args.n_best}.parquet", index=False)
            nbest_val.to_parquet(pred / f"nbest_val_beam{args.n_best}.parquet", index=False)

    # evaluate every strategy
    results = {}
    nbest["_neg_rank"] = -nbest["rank"]
    results["FT-Whisper beam-top1"] = wer_of_selection(select(nbest, "_neg_rank"), eval_df)
    results["MBR decoding (ours)"] = wer_of_selection(mbr_select(nbest), eval_df)

    scorer_cols = [(name, f"llm_{name}") for name in RESCORER_MODELS]
    if "llm_dar" in nbest.columns:
        scorer_cols.append(("dar", "llm_dar"))

    for name, col in scorer_cols:
        label = "CLEAR-DAR (ours)" if name == "dar" else f"CLEAR ({name})"
        results[label] = wer_of_selection(select(nbest, col), eval_df)

        nbest["_ln"] = nbest[col] / nbest[col + "_len"]
        results[f"{label} +lennorm"] = wer_of_selection(select(nbest, "_ln"), eval_df)

        lam, vw = tune_lambda(nbest_val, val_tune, col)
        nbest["_mix"] = lam * nbest.asr_score_total + (1 - lam) * nbest[col]
        results[f"{label} +interp (lam={lam:.2f})"] = wer_of_selection(select(nbest, "_mix"), eval_df)
        print(f"{label}: tuned lambda {lam:.2f} (val P-WER {vw:.2f}%)")

    rows = [dict(model=k, **{m: round(v, 2) for m, v in r.items()}) for k, r in results.items()]
    clear_results = pd.DataFrame(rows)

    baseline_csv = pred / "baseline_results.csv"
    parts = [pd.read_csv(baseline_csv)] if baseline_csv.exists() else []
    final = pd.concat(parts + [clear_results], ignore_index=True)
    final.to_csv(pred / "final_results.csv", index=False)
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
