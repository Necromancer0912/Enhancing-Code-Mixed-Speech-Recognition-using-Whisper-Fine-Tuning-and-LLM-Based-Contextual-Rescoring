"""Project-wide constants, paths and the descriptive prompts.

CLEAR-Bn: Bengali-English code-switched ASR with LLM-driven rescoring,
following Kumar & Akhtar (ICNLSP 2025) and extended to a new language pair.
"""
from pathlib import Path

TARGET_SR = 16000
SEED = 42

# Unicode ranges used by the normalisers and the script detector
BENGALI_BLOCK = "ঀ-৿"
DEVANAGARI_BLOCK = "ऀ-ॿ"
INDIC_BLOCKS = BENGALI_BLOCK + DEVANAGARI_BLOCK

# Descriptive prompts, adapted from Table 1 of the paper (Hindi -> Bengali).
# Prompt 1 was the paper's best performer and is what we fine-tune with.
PROMPT_1 = (
    "This transcript is a code-switched text. Mix of Bengali and English words "
    "are present. Text is related to tutorials on academic or technical subjects."
)

PROMPT_2 = (
    "This transcript is a code-switched text. Mix of Bengali and English words are "
    "present. Text is related to tutorials on academic or technical subjects. "
    "Few examples look like this: "
    "এখানে file menu তে click করুন "
    "এই option টি select করে enter চাপুন"
)

WHISPER_SMALL = "openai/whisper-small"
WHISPER_TURBO = "openai/whisper-large-v3-turbo"

# Rescoring language models. GPT-2 is the paper's best scorer but has never
# seen Bengali script; the other two are multilingual and did.
RESCORER_MODELS = {
    "gpt2": "gpt2",
    "bloom-1b1": "bigscience/bloom-1b1",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
}
DAR_BASE_MODEL = "bigscience/bloom-1b1"


def project_paths(root=None):
    """Resolve every directory the pipeline reads or writes, creating the
    output directories on first use. `root` defaults to the repo root."""
    if root is None:
        root = Path(__file__).resolve().parents[2]
    root = Path(root)
    p = dict(
        root=root,
        data=root / "Data",
        train=root / "Data" / "bn-en" / "train",
        test=root / "Data" / "bn-en" / "test",
        outputs=root / "outputs_bn",
        manifests=root / "outputs_bn" / "manifests",
        checkpoints=root / "outputs_bn" / "checkpoints",
        predictions=root / "outputs_bn" / "predictions",
        figures=root / "outputs_bn" / "figures",
    )
    for key in ("outputs", "manifests", "checkpoints", "predictions", "figures"):
        p[key].mkdir(parents=True, exist_ok=True)
    return p
