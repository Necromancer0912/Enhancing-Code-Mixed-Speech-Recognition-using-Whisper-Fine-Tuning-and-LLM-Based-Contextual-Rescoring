"""Text normalisation and the three WER variants from the paper.

S-WER  strict surface-level WER
P-WER  WER after mapping enunciated punctuation to spoken words
T-WER  WER after romanising Indic-script words, so a hypothesis is not
       penalised for writing a word in the "wrong" script
"""
import re
import unicodedata

from .config import BENGALI_BLOCK, DEVANAGARI_BLOCK, INDIC_BLOCKS

# Symbols the spoken-tutorial speakers enunciate. Mapping them to words keeps
# "greater than" in the reference from mismatching ">" in a hypothesis.
PUNCT_WORD_MAP = {
    "<": "less than",
    ">": "greater than",
    "=": "equal to",
    "+": "plus",
    "-": "minus",
    "/": "slash",
    "_": "underscore",
    "%": "percent",
    "&": "and",
    "@": "at the rate",
    "*": "star",
}


def norm_basic(text):
    """S-WER normalisation: NFC, lowercase, collapse whitespace. Nothing else."""
    text = unicodedata.normalize("NFC", str(text)).lower()
    return re.sub(r"\s+", " ", text).strip()


def norm_punct(text):
    """P-WER normalisation: spell out symbols, then drop remaining punctuation."""
    text = norm_basic(text)
    for sym, word in PUNCT_WORD_MAP.items():
        text = text.replace(sym, f" {word} ")
    text = re.sub(rf"[^\w\s{INDIC_BLOCKS}]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_translit_cache = {}


def _translit_word(word):
    if word in _translit_cache:
        return _translit_cache[word]
    out = word
    src = None
    if re.search(rf"[{BENGALI_BLOCK}]", word):
        src = "bengali"
    elif re.search(rf"[{DEVANAGARI_BLOCK}]", word):
        src = "devanagari"
    if src is not None:
        try:
            from indic_transliteration import sanscript
            from indic_transliteration.sanscript import transliterate

            scheme = sanscript.BENGALI if src == "bengali" else sanscript.DEVANAGARI
            out = transliterate(word, scheme, sanscript.ITRANS).lower()
        except Exception:
            out = word
    out = re.sub(r"[^a-z0-9]", "", out) or word
    _translit_cache[word] = out
    return out


def norm_translit(text):
    """T-WER normalisation: P-WER normalisation plus word-level romanisation."""
    text = norm_punct(text)
    return " ".join(_translit_word(w) for w in text.split())


def wer_variants(refs, hyps):
    """Corpus-level S/P/T-WER in percent. Empty references are skipped; empty
    hypotheses count as a full error rather than crashing jiwer."""
    from jiwer import wer

    out = {}
    for name, norm in [("S_WER", norm_basic), ("P_WER", norm_punct), ("T_WER", norm_translit)]:
        r_list, h_list = [], []
        for ref, hyp in zip(refs, hyps):
            r = norm(ref)
            if not r:
                continue
            r_list.append(r)
            h_list.append(norm(hyp) or "<empty>")
        out[name] = 100.0 * wer(r_list, h_list)
    return out
