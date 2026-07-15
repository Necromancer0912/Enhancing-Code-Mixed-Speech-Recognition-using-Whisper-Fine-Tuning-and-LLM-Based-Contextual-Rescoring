"""Extract the MUCS archives, parse the Kaldi manifests, compute code-mixing
statistics, filter for training and write the train/val/test parquet manifests.

Usage:
    python scripts/prepare_data.py
"""
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ROOT  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from clear_bn import SEED, norm_basic, parse_mucs_split, project_paths, utterance_mix_stats  # noqa: E402


def extract_archives(paths):
    downloads = paths["data"] / "downloads"
    targets = {
        "Bengali-English_train.tar.gz": paths["train"],
        "Bengali-English_test.tar.gz": paths["test"],
    }
    for tar_name, dest in targets.items():
        if dest.exists() and any(dest.rglob("*.wav")):
            print(f"[skip] {dest} already extracted")
            continue
        tar_path = downloads / tar_name
        if not tar_path.exists():
            raise FileNotFoundError(f"{tar_path} missing - run scripts/download_data.py first")
        dest.mkdir(parents=True, exist_ok=True)
        print(f"extracting {tar_name} ...")
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(dest)


def add_mix_stats(df, desc):
    stats = pd.DataFrame([utterance_mix_stats(t) for t in tqdm(df.ref_text, desc=desc)])
    df[stats.columns] = stats
    df["clean_text"] = df.ref_text.map(norm_basic)
    return df


def main():
    paths = project_paths(ROOT)
    extract_archives(paths)

    train_df = parse_mucs_split(paths["train"])
    test_df = parse_mucs_split(paths["test"])
    print(f"train: {len(train_df):,} utts, {train_df.duration_sec.sum() / 3600:.2f} h")
    print(f"test : {len(test_df):,} utts, {test_df.duration_sec.sum() / 3600:.2f} h")

    train_df = add_mix_stats(train_df, "train mix stats")
    test_df = add_mix_stats(test_df, "test mix stats")

    # honesty check: how much of test appears verbatim in train
    train_sents = set(train_df.clean_text)
    test_df["in_train_overlap"] = test_df.clean_text.isin(train_sents)
    print(f"test sentences verbatim in train: {100 * test_df.in_train_overlap.mean():.2f}%")

    # keep what Whisper can actually consume
    ft = train_df[(train_df.duration_sec.between(0.5, 28.0)) & (train_df.clean_text.str.len() > 0)].copy()
    ev = test_df[(test_df.duration_sec.between(0.3, 30.0)) & (test_df.clean_text.str.len() > 0)].copy()

    # recording-wise validation split so no speaker leaks across the boundary
    rng = np.random.default_rng(SEED)
    recs = np.array(sorted(ft.recording_id.unique()))
    rng.shuffle(recs)
    val_recs = set(recs[: max(1, int(0.05 * len(recs)))])
    val_df = ft[ft.recording_id.isin(val_recs)].copy()
    tr_df = ft[~ft.recording_id.isin(val_recs)].copy()

    m = paths["manifests"]
    tr_df.to_parquet(m / "train.parquet", index=False)
    val_df.to_parquet(m / "val.parquet", index=False)
    ev.to_parquet(m / "test.parquet", index=False)
    print(f"saved manifests: train {len(tr_df):,} / val {len(val_df):,} / test {len(ev):,}")


if __name__ == "__main__":
    main()
