"""Parsing the MUCS Kaldi-style manifests and code-mixing statistics."""
import re
from pathlib import Path

import pandas as pd

from .config import BENGALI_BLOCK, DEVANAGARI_BLOCK
from .metrics import norm_basic


def _read_kv_lines(path, max_split=1):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=max_split)
        if len(parts) == max_split + 1:
            rows.append(parts)
    return rows


def find_transcript_dir(split_dir):
    """The archives nest things differently per split, so just look for the
    directory that holds text + segments + wav.scp together."""
    split_dir = Path(split_dir)
    for cand in split_dir.rglob("text"):
        if cand.is_file() and (cand.parent / "segments").exists() and (cand.parent / "wav.scp").exists():
            return cand.parent
    raise FileNotFoundError(f"No Kaldi transcript dir (text+segments+wav.scp) under {split_dir}")


def parse_mucs_split(split_dir):
    """Turn one split into a DataFrame with one row per utterance and the
    audio path resolved against the files actually on disk."""
    split_dir = Path(split_dir)
    tdir = find_transcript_dir(split_dir)

    text = {u: t for u, t in _read_kv_lines(tdir / "text")}
    utt2spk = {u: s for u, s in _read_kv_lines(tdir / "utt2spk")}

    seg_rows = []
    for line in (tdir / "segments").read_text(encoding="utf-8").splitlines():
        p = line.strip().split()
        if len(p) >= 4:
            seg_rows.append((p[0], p[1], float(p[2]), float(p[3])))

    wav_map = {}
    for line in (tdir / "wav.scp").read_text(encoding="utf-8").splitlines():
        p = line.strip().split(maxsplit=1)
        if len(p) == 2:
            # the value can be a plain path or a command pipeline; take the
            # last token that looks like a wav file
            toks = [t for t in p[1].split() if t.lower().endswith(".wav")]
            wav_map[p[0]] = toks[-1] if toks else p[1].strip()

    audio_index = {f.stem: f for f in split_dir.rglob("*.wav")}

    rows = []
    for utt_id, rec_id, start, end in seg_rows:
        if utt_id not in text:
            continue
        wav_ref = wav_map.get(rec_id, "")
        wav_path = None
        if wav_ref and Path(wav_ref).exists():
            wav_path = Path(wav_ref)
        if wav_path is None:
            stem = Path(wav_ref).stem if wav_ref else rec_id
            wav_path = audio_index.get(stem) or audio_index.get(rec_id)
        if wav_path is None:
            continue
        rows.append(dict(
            utt_id=utt_id,
            recording_id=rec_id,
            speaker_id=utt2spk.get(utt_id, "unk"),
            start_sec=start,
            end_sec=end,
            duration_sec=round(end - start, 3),
            ref_text=text[utt_id],
            wav_path=str(wav_path),
        ))
    return pd.DataFrame(rows).sort_values("utt_id").reset_index(drop=True)


def script_of_word(word):
    if re.search(rf"[{BENGALI_BLOCK}]", word):
        return "bengali"
    if re.search(rf"[{DEVANAGARI_BLOCK}]", word):
        return "devanagari"
    if re.search(r"[a-zA-Z]", word):
        return "latin"
    return "other"


def utterance_mix_stats(text):
    """Per-utterance word counts by script plus the Code-Mixing Index of
    Das & Gambäck (2014)."""
    words = norm_basic(text).split()
    n = len(words)
    counts = {"bengali": 0, "latin": 0, "devanagari": 0, "other": 0}
    for w in words:
        counts[script_of_word(w)] += 1
    is_mixed = counts["bengali"] > 0 and counts["latin"] > 0
    lang_total = n - counts["other"]
    cmi = 0.0
    if lang_total > 0:
        cmi = 100.0 * (1.0 - max(counts["bengali"], counts["latin"], counts["devanagari"]) / lang_total)
    return dict(n_words=n, n_bengali=counts["bengali"], n_english=counts["latin"],
                n_other=counts["other"], is_code_mixed=is_mixed, cmi=round(cmi, 2))
