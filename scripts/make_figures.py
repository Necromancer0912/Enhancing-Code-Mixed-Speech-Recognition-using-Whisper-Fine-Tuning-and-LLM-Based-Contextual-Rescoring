"""Render the README figures from the result CSVs.

Usage:
    python scripts/make_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ROOT  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from clear_bn import project_paths  # noqa: E402

COLORS = {"S_WER": "#0ea5e9", "P_WER": "#14b8a6", "T_WER": "#f59e0b"}


def final_comparison(paths):
    df = pd.read_csv(paths["predictions"] / "final_results.csv")
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(df) + 2))
    y = np.arange(len(df))
    for i, met in enumerate(["S_WER", "P_WER", "T_WER"]):
        ax.barh(y + (i - 1) * 0.26, df[met], 0.26, label=met, color=COLORS[met])
    ax.set_yticks(y)
    ax.set_yticklabels(df.model, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("WER (%)")
    ax.set_title("Bengali-English code-mixed ASR: all systems, 1000-utterance test subset")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths["figures"] / "final_comparison.png", dpi=150, bbox_inches="tight")
    print("saved final_comparison.png")


def finetuned_only(paths):
    """Same chart without the zero-shot rows, so the interesting differences
    between rescoring strategies are readable."""
    df = pd.read_csv(paths["predictions"] / "final_results.csv")
    df = df[df.S_WER < 100].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(df) + 2))
    y = np.arange(len(df))
    for i, met in enumerate(["S_WER", "P_WER", "T_WER"]):
        ax.barh(y + (i - 1) * 0.26, df[met], 0.26, label=met, color=COLORS[met])
    ax.set_yticks(y)
    ax.set_yticklabels(df.model, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("WER (%)")
    ax.set_title("Fine-tuned systems and rescoring strategies")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths["figures"] / "rescoring_comparison.png", dpi=150, bbox_inches="tight")
    print("saved rescoring_comparison.png")


if __name__ == "__main__":
    paths = project_paths(ROOT)
    final_comparison(paths)
    finetuned_only(paths)
