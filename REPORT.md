# Enhancing Code-Mixed Speech Recognition using Whisper Fine-Tuning and LLM-Based Contextual Rescoring

Course project report. Base paper: Shivam Kumar and Md. Shad Akhtar,
"CLEAR: Code-Mixed ASR with LLM-Driven Rescoring", ICNLSP 2025.

Team: Sayan Das (MT25041) and Senjuti Ghosal.

## 1. Problem Statement

Code-mixing (alternating between two languages inside a single utterance) is the
default speaking style of multilingual communities in India. It breaks standard
ASR systems in specific, well-understood ways:

- Cross-lingual homophone ambiguity: the same sound maps to different words in
  the two languages, and only sentence-level context disambiguates them.
- Switch-point detection: the recogniser must decide, word by word, which
  language it is hearing, with no explicit marker in the audio.
- Script ambiguity: a word such as "file" may legitimately be written in Latin
  script or in Bengali script, and a fair evaluation has to account for that.
- Data scarcity: transcribed code-mixed speech is rare, and Bengali-English
  is significantly lower-resourced than Hindi-English.

The base paper addresses these problems for Hindi-English with a pipeline of
descriptive prompting, decoder-only fine-tuning of Whisper, and rescoring of
n-best hypotheses with large language models. This project reproduces that
pipeline end to end and extends it to Bengali-English, a pair the paper did not
study, under a strict hardware budget of a single 8 GB consumer GPU
(NVIDIA RTX 4060 Ti).

## 2. Dataset

MUCS 2021 subtask-2 Bengali-English (OpenSLR-104), derived from spoken
tutorials on technical subjects (LibreOffice, programming, GIMP, Blender and
similar). Audio is 16 kHz, 16-bit mono. Statistics computed by our pipeline:

| Split | Utterances | Recordings / speakers | Hours | Code-mixed % | Mean CMI |
|-------|-----------:|----------------------:|------:|-------------:|---------:|
| Train | 26,606 | 267 | 45.94 | 84.98 | 25.77 |
| Test | 4,275 | 40 | 7.00 | 86.76 | 27.19 |

CMI is the Code-Mixing Index of Das and Gambaeck (2014). Additional findings
from our exploratory analysis:

- 15.25 percent of test sentences appear verbatim in the training text. This is
  much lower than the 33.9 percent train/test overlap the base paper reports for
  the Hindi-English pair, so evaluation on this test set is comparatively honest.
- References mix scripts freely ("libreoffice impress এর উপর এই কথ্য tutorial এ
  আপনাদের স্বাগত"), which motivates the transliterated WER metric below.
- After filtering to 0.5-28 s segments with non-empty transcripts, 25,358
  training segments (43.1 h) remain; a recording-wise 5 percent split gives
  1,187 validation segments with no speaker leakage.

## 3. Methodology

### 3.1 Pipeline

```text
audio (16 kHz)
   |
   v
Whisper-small encoder  (frozen)
   |
   v
Whisper-small decoder  (fine-tuned)
   ^ descriptive prompt after <|startofprev|>
   |
   |  beam search, n = 5
   v
n-best hypotheses + Whisper sequence scores
   |
   v
LLM rescoring  (GPT-2 / BLOOM-1b1 / Qwen2.5-1.5B / DAR)
   |
   v
selection: argmax LLM score  |  + length norm  |  + score interpolation  |  MBR
   |
   v
final transcription -> S-WER / P-WER / T-WER
```

### 3.2 Descriptive prompting

Whisper accepts a textual prompt through its `<|startofprev|>` context slot.
Following the paper's best prompt (their Prompt-1), adapted to Bengali:

> "This transcript is a code-switched text. Mix of Bengali and English words are
> present. Text is related to tutorials on academic or technical subjects."

The full decoder context during training is

```text
<|startofprev|> {prompt} <|startoftranscript|> <|bn|> <|transcribe|> <|notimestamps|> {text} <|endoftext|>
```

with the loss masked (label -100) on everything before the transcription
tokens, so the model is never trained to reproduce the prompt, only to condition
on it.

### 3.3 Decoder-only fine-tuning

As in the paper, the encoder is frozen and only the decoder is updated: the
encoder already encodes acoustics well after 630k hours of pre-training, and
freezing it halves memory and prevents overfitting on 43 h of data. Training
configuration (8 GB budget):

| Setting | Value |
|---|---|
| Base model | openai/whisper-small (244 M params, 153 M trainable) |
| Schedule | phase 1: 2 epochs at lr 2e-5; phase 2: 3 epochs at lr 5e-5 |
| Batch size | 8, gradient accumulation 2 (effective 16, as in the paper) |
| Optimiser | AdamW, weight decay 0.01, cosine schedule with warmup |
| Precision | fp16 autocast with gradient scaling |
| Augmentation | light additive noise (p=0.3) and gain jitter (p=0.3) |
| Checkpointing | best validation P-WER on a 120-utterance probe every 400 updates |

Training took about 90 minutes total on the RTX 4060 Ti. Validation P-WER fell
from 90.1 percent (first eval) to 49.2 percent after phase 1 and 46.7 percent at
its best in phase 2, after which the model began to overfit (train loss 0.03,
validation flat).

### 3.4 LLM rescoring (paper Eq. 1 and 2)

The fine-tuned model produces n = 5 hypotheses per utterance: one greedy
hypothesis plus four temperature samples (T 0.8, top-p 0.95), with each
hypothesis's exact ASR log-probability recovered from the generation scores
(see the implementation note in section 5 on why beam search was not usable).
Each hypothesis x is scored by a causal LM as the sum of token log-probabilities

    log P(x) = sum_t log P(x_t | x_<t)

and the highest-scoring hypothesis is selected. The paper used GPT-2 as its
best scorer. GPT-2 never saw Bengali script, so we additionally evaluate two
small multilingual LMs that did: BLOOM-1b1 and Qwen2.5-1.5B.

### 3.5 Our extensions

1. New language pair: the entire recipe evaluated on Bengali-English.
2. Multilingual rescorers, as motivated above.
3. Length-normalised LLM scores: raw log-probability sums favour short
   hypotheses; dividing by token count removes this bias.
4. ASR-LLM score interpolation: final score = lambda * logP_ASR +
   (1 - lambda) * logP_LLM, with lambda tuned by grid search on 300 held-out
   validation utterances. lambda = 0 recovers the paper exactly and lambda = 1
   recovers pure beam search, so this strictly generalises both.
5. MBR decoding: pick the hypothesis with the lowest expected WER against the
   other beams, weighted by ASR posterior. A training-free consensus baseline.
6. Domain-Adaptive Rescorer (DAR): BLOOM-1b1 LoRA-tuned (r=16, ~1.6 M trainable
   parameters) on the 25k training transcripts, text only, one epoch. The
   rescorer thus speaks the exact code-mixed technical register it judges.

### 3.6 Evaluation metrics

Three WER variants, as in the paper:

- S-WER: WER on lightly normalised text (NFC, lower-case, whitespace).
- P-WER: enunciated punctuation is mapped to spoken words ("greater than" vs
  ">") before scoring.
- T-WER: on top of P-WER normalisation, Bengali/Devanagari words are romanised
  (indic-transliteration, ITRANS) on both sides, so script choice is not
  penalised.

## 4. Results

Evaluation on a fixed, seeded 1,000-utterance subset of the MUCS Bn-En test set.

| System | S-WER | P-WER | T-WER |
|---|---:|---:|---:|
| B1 Whisper-small (zero-shot) | 195.24 | 249.53 | 246.15 |
| B2 PromptingWhisper-small (zero-shot + prompt) | 135.62 | 174.63 | 174.13 |
| B3 Whisper-large-v3-turbo (zero-shot) | 101.02 | 100.56 | 97.32 |
| Fine-tuned Whisper, greedy top-1 | 41.22 | 40.25 | 39.67 |
| MBR decoding (ours) | 40.10 | 39.09 | 38.48 |
| CLEAR, GPT-2 (paper's scorer, raw) | 43.11 | 42.20 | 41.64 |
| CLEAR, GPT-2 + interpolation (ours, lambda 0.95) | 40.22 | 39.29 | 38.69 |
| CLEAR, BLOOM-1b1 (raw) | 41.63 | 40.74 | 40.19 |
| CLEAR, BLOOM-1b1 + interpolation (ours, lambda 0.90) | 39.59 | 38.65 | 38.10 |
| CLEAR, Qwen2.5-1.5B (raw) | 42.07 | 41.08 | 40.51 |
| CLEAR, Qwen2.5-1.5B + interpolation (ours, lambda 0.85) | 39.76 | 38.79 | 38.22 |
| CLEAR-DAR (ours, raw) | 42.79 | 41.85 | 41.28 |
| CLEAR-DAR + interpolation (ours, lambda 0.95) | 40.02 | 39.09 | 38.52 |

Length-normalised LLM scores consistently hurt by 2 to 4 points and are omitted
here (full table in outputs_bn/predictions/final_results.csv). The best system is
BLOOM-1b1 rescoring interpolated with the ASR score at lambda 0.90: 39.59 S-WER,
a 4.0 percent relative reduction over the fine-tuned greedy output.

Reference points from the literature (different training regimes, shown for
context): on the full Bn-En test set the MUCS 2021 challenge baselines reach
25.89 percent WER (Kaldi TDNN hybrid with LM decoding) and 23.15 percent WER
(end-to-end transformer with CTC-attention). On Hindi-English, the base paper's
CLEAR reaches 26.9 S-WER, and the MUCS organisers note Bengali-English is
roughly 10 WER points harder than Hindi-English across systems.

## 5. Analysis

Fine-tuning is the dominant factor: 195 to 41.2 S-WER with about 90 minutes of
decoder-only training over two phases (2 epochs at lr 2e-5, 3 epochs at lr 5e-5;
best validation P-WER 46.7 percent at the end of the fourth overall epoch, after
which the model began to overfit).

On this language pair, raw LLM argmax selection alone is worse than the ASR top-1
for all four scorers. GPT-2 degrades most (+1.9 S-WER) - it has essentially no
Bengali in its training data, so its scores on majority-Bengali hypotheses are
noise. The multilingual BLOOM-1b1 is the least-bad raw scorer, which
supports the language-coverage explanation. Interpolating LLM and ASR scores (tuned
lambda 0.85 to 0.95) turns every scorer into a net improvement; the high lambda
values indicate the LLM works best as a light fluency prior rather than the primary
decision maker. MBR consensus decoding, which needs no extra model, recovers about
two thirds of the best rescoring gain.

The domain-adaptive rescorer was a mixed result: LoRA-tuning BLOOM on reference
transcripts made raw scoring worse (42.79 vs 41.63), plausibly because a scorer
tuned toward reference-style fluency assigns uniformly high scores to all plausible
hypotheses, flattening the ranking contrast. It still helps when interpolated
(40.02). A discriminative objective over hypothesis pairs is the natural next step.

Rescoring altered 37.3 percent of the 1,000 selections. Qualitative inspection
shows exactly the phenomena the paper targets, for example "up arrow সরূপ"
(greedy) corrected to the reference word "উদাহরণস্বরূপ", and spelled-out digit
sequences repaired toward the reference formatting.

An implementation finding worth recording: transformers 5.x Whisper beam search
returns num_return_sequences identical copies of the top beam, which silently makes
any n-best rescoring a no-op (all candidates equal). The n-best list used here is
therefore one greedy hypothesis plus temperature samples (T 0.8, top-p 0.95), with
exact per-hypothesis log-probabilities recovered from the generation scores. A
second finding: BLOOM-1b1 overflows to NaN on most Bengali inputs in fp16; scoring
runs in bfloat16.

## 6. Reproducibility

All code is released as a Python package (src/clear_bn) with runnable scripts
(scripts/), covering data download, preparation, baselines, fine-tuning, n-best
generation and rescoring. See README.md for exact commands. Seeds are fixed
(42) throughout; the evaluation subset is deterministic.

## 7. References

1. Kumar, S. and Akhtar, M.S. CLEAR: Code-Mixed ASR with LLM-Driven Rescoring. ICNLSP 2025.
2. Diwan, A. et al. MUCS 2021: Multilingual and Code-Switching ASR Challenges for Low Resource Indian Languages. Interspeech 2021.
3. Radford, A. et al. Robust Speech Recognition via Large-Scale Weak Supervision (Whisper). ICML 2023.
4. Das, A. and Gambaeck, B. Identifying Languages at the Word Level in Code-Mixed Indian Social Media Text. ICON 2014.
5. Hu, E. et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
6. Peng, P. et al. Prompting the Hidden Talent of Web-Scale Speech Models for Zero-Shot Task Generalization. Interspeech 2023.
