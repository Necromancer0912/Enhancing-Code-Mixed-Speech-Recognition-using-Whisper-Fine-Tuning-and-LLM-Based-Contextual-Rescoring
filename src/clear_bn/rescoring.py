"""Hypothesis selection: LLM rescoring (paper Eq. 1-2), length normalisation,
ASR/LLM score interpolation, and MBR decoding."""
import gc

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .metrics import norm_punct, wer_variants


@torch.no_grad()
def llm_score_texts(texts, model_id, device="cuda", batch_size=16, max_length=256,
                    adapter_dir=None, desc=None):
    """Total log-probability of each text under a causal LM (paper Eq. 1),
    plus the token count so callers can length-normalise.

    Loads the model, scores, and frees the GPU before returning, so several
    scorers can run one after another on an 8 GB card. Pass `adapter_dir` to
    stack a LoRA adapter on top of the base model.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # bf16, not fp16: BLOOM overflows to inf logits in fp16, which turns the
    # log-softmax into NaN for most Bengali inputs
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    if adapter_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    totals, lengths = [], []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc or model_id):
        chunk = [t if isinstance(t, str) and t.strip() else " " for t in texts[i:i + batch_size]]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length).to(device)
        logits = model(**enc).logits.float()
        logprobs = torch.log_softmax(logits[:, :-1], dim=-1)
        targets = enc.input_ids[:, 1:]
        mask = enc.attention_mask[:, 1:].bool()
        token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        token_lp = token_lp.masked_fill(~mask, 0.0)
        totals.extend(token_lp.sum(-1).cpu().tolist())
        lengths.extend(mask.sum(-1).clamp(min=1).cpu().tolist())

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return np.array(totals), np.array(lengths)


def select(nbest, score_col):
    """Pick the highest-scoring hypothesis per utterance. NaN scores (a scorer
    numerically failing on a hypothesis) lose against any finite score."""
    filled = nbest[score_col].fillna(float("-inf"))
    idx = filled.groupby(nbest["utt_id"]).idxmax()
    return nbest.loc[idx, ["utt_id", "hyp"]].set_index("utt_id").hyp


def wer_of_selection(selection, ref_df):
    """S/P/T-WER of a per-utterance selection against a manifest DataFrame."""
    merged = ref_df.set_index("utt_id").join(selection.rename("hyp_sel"))
    merged["hyp_sel"] = merged["hyp_sel"].fillna("")
    return wer_variants(merged.ref_text.tolist(), merged.hyp_sel.tolist())


def tune_lambda(nbest, ref_df, llm_col, grid=None):
    """Grid-search the ASR/LLM interpolation weight on held-out data.
    lambda = 0 recovers the paper's pure-LLM selection, 1 pure beam search."""
    if grid is None:
        grid = np.linspace(0.0, 1.0, 21)
    nbest = nbest.copy()
    best_lam, best_wer = 0.0, float("inf")
    for lam in grid:
        nbest["_mix"] = lam * nbest.asr_score_total + (1 - lam) * nbest[llm_col]
        w = wer_of_selection(select(nbest, "_mix"), ref_df)["P_WER"]
        if w < best_wer:
            best_lam, best_wer = float(lam), w
    return best_lam, best_wer


def mbr_select(nbest):
    """Minimum Bayes Risk decoding over the n-best list: choose the hypothesis
    with the lowest expected WER against the other beams, weighted by the ASR
    posterior. A consensus vote that needs no extra model."""
    from jiwer import wer

    picks = {}
    for utt_id, group in tqdm(nbest.groupby("utt_id"), desc="MBR"):
        hyps = group.hyp.tolist()
        weights = np.exp(np.array(group.asr_score_mean.tolist()))
        weights = weights / max(weights.sum(), 1e-9)
        if len(set(hyps)) == 1:
            picks[utt_id] = hyps[0]
            continue
        normed = [norm_punct(h) or "<e>" for h in hyps]
        risk = np.zeros(len(hyps))
        for i in range(len(hyps)):
            for j in range(len(hyps)):
                if i != j:
                    risk[i] += weights[j] * wer(normed[j], normed[i])
        picks[utt_id] = hyps[int(risk.argmin())]
    return pd.Series(picks, name="hyp")
