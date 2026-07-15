"""Batched Whisper inference: plain transcription and n-best generation."""
import pandas as pd
import torch
from tqdm.auto import tqdm

from .audio import load_segment
from .config import TARGET_SR


@torch.no_grad()
def transcribe(df, model, processor, language="bn", prompt=None, num_beams=1,
               batch_size=8, max_new_tokens=200, desc="transcribe"):
    """Transcribe every row of a manifest DataFrame. Returns hypothesis strings
    in row order. `prompt` goes in via prompt_ids and its echo is stripped."""
    device = next(model.parameters()).device
    model.eval()

    prompt_ids = None
    prompt_text = None
    if prompt is not None:
        prompt_ids = torch.tensor(processor.get_prompt_ids(prompt), device=device)
        prompt_text = processor.tokenizer.decode(
            prompt_ids.tolist(), skip_special_tokens=True).strip()

    hyps = []
    for i in tqdm(range(0, len(df), batch_size), desc=desc):
        batch = df.iloc[i:i + batch_size]
        audios = [load_segment(r.wav_path, r.start_sec, r.end_sec) for r in batch.itertuples()]
        feats = processor(audios, sampling_rate=TARGET_SR, return_tensors="pt",
                          return_attention_mask=True)
        input_features = feats.input_features.to(device=device, dtype=model.dtype)

        kwargs = dict(language=language, task="transcribe", max_new_tokens=max_new_tokens,
                      num_beams=num_beams, do_sample=False, use_cache=True)
        if prompt_ids is not None:
            kwargs["prompt_ids"] = prompt_ids
        if "attention_mask" in feats:
            kwargs["attention_mask"] = feats.attention_mask.to(device)

        pred = model.generate(input_features, **kwargs)
        for t in processor.batch_decode(pred, skip_special_tokens=True):
            t = t.strip()
            if prompt_text and t.startswith(prompt_text):
                t = t[len(prompt_text):].strip()
            hyps.append(t)
    return hyps


@torch.no_grad()
def generate_nbest(df, model, processor, prefix, n_best=5, batch_size=4,
                   max_new_tokens=128, temperature=0.8, top_p=0.95, seed=42,
                   desc="nbest"):
    """N-best lists: one greedy hypothesis (rank 0) plus temperature samples.

    Beam search would be the textbook choice, but transformers 5.x returns
    num_return_sequences identical copies of the top beam for Whisper, which
    leaves the rescorer nothing to choose between. Sampling gives genuinely
    diverse candidates; the ASR log-probability of every hypothesis is
    recovered exactly from the per-step scores.

    Returns a long DataFrame: one row per (utterance, rank) with the
    hypothesis text, mean and total ASR log-probability.
    """
    device = next(model.parameters()).device
    model.eval()
    torch.manual_seed(seed)
    prompt_ids = prefix.generation_prompt_ids(device)

    def run(fx, sample, n_seq):
        kwargs = dict(prompt_ids=prompt_ids, language="bn", task="transcribe",
                      max_new_tokens=max_new_tokens, return_dict_in_generate=True,
                      output_scores=True, use_cache=True, num_beams=1)
        if sample:
            kwargs.update(do_sample=True, temperature=temperature, top_p=top_p,
                          num_return_sequences=n_seq)
        else:
            kwargs.update(do_sample=False)
        out = model.generate(fx, **kwargs)
        # exact token log-probs of whatever was generated
        ts = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        finite = torch.isfinite(ts)
        totals = ts.masked_fill(~finite, 0.0).sum(-1)
        lengths = finite.sum(-1).clamp(min=1)
        return out.sequences.cpu().tolist(), totals.cpu().tolist(), lengths.cpu().tolist()

    rows = []
    for i in tqdm(range(0, len(df), batch_size), desc=desc):
        batch = df.iloc[i:i + batch_size]
        audios = [load_segment(r.wav_path, r.start_sec, r.end_sec) for r in batch.itertuples()]
        feats = processor(audios, sampling_rate=TARGET_SR, return_tensors="pt")
        fx = feats.input_features.to(device=device, dtype=model.dtype)

        g_seqs, g_tot, g_len = run(fx, sample=False, n_seq=1)
        s_seqs, s_tot, s_len = run(fx, sample=True, n_seq=n_best - 1)

        for j, r in enumerate(batch.itertuples()):
            rows.append(dict(utt_id=r.utt_id, rank=0, hyp=prefix.strip_decode(g_seqs[j]),
                             asr_score_mean=g_tot[j] / g_len[j], asr_score_total=g_tot[j]))
            for k in range(n_best - 1):
                idx = j * (n_best - 1) + k
                rows.append(dict(utt_id=r.utt_id, rank=k + 1,
                                 hyp=prefix.strip_decode(s_seqs[idx]),
                                 asr_score_mean=s_tot[idx] / s_len[idx],
                                 asr_score_total=s_tot[idx]))
    return pd.DataFrame(rows)
