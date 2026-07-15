"""Audio segment loading. The MUCS recordings are long tutorial wavs; each
utterance is a (start, end) slice of one, so we seek instead of reading the
whole file."""
import math

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import TARGET_SR


def load_segment(wav_path, start_sec, end_sec, target_sr=TARGET_SR):
    """Read one utterance slice as mono float32 at 16 kHz."""
    with sf.SoundFile(wav_path) as f:
        sr = int(f.samplerate)
        start = max(0, int(start_sec * sr))
        end = max(start + 1, int(end_sec * sr))
        f.seek(start)
        audio = f.read(frames=end - start, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        g = math.gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
    return audio
