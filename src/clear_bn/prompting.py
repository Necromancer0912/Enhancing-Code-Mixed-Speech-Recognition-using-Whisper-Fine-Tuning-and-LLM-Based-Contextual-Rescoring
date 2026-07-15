"""Building Whisper's prompted decoder prefix and undoing the prompt echo.

The trained format is
    <|startofprev|> {prompt} <|startoftranscript|><|bn|><|transcribe|><|notimestamps|> {text} <|endoftext|>
with the loss masked up to the transcription tokens.
"""
import torch


class PromptedPrefix:
    """Everything derived from one (tokenizer, prompt) pair, in one place."""

    def __init__(self, tokenizer, prompt, language_token="<|bn|>"):
        self.tok = tokenizer
        self.prompt = prompt.strip()
        self.sop = tokenizer.convert_tokens_to_ids("<|startofprev|>")
        self.eot = tokenizer.eos_token_id
        self.sot_seq = tokenizer.convert_tokens_to_ids(
            ["<|startoftranscript|>", language_token, "<|transcribe|>", "<|notimestamps|>"])
        self.prompt_toks = tokenizer.encode(" " + self.prompt, add_special_tokens=False)
        self.prefix = [self.sop] + self.prompt_toks + self.sot_seq
        # what the prompt looks like once it has been decoded back to text,
        # used to strip the echo from generated sequences
        self.prompt_text = tokenizer.decode(self.prompt_toks).strip()

    def generation_prompt_ids(self, device):
        """prompt_ids for model.generate(): <|startofprev|> plus prompt tokens.
        Whisper appends its own task tokens after these."""
        return torch.tensor([self.sop] + self.prompt_toks, dtype=torch.long, device=device)

    def training_pair(self, text, max_len=384):
        """decoder_input_ids and loss-masked labels for one training example."""
        text_toks = self.tok.encode(" " + text.strip(), add_special_tokens=False)
        text_toks = text_toks[: max_len - len(self.prefix) - 1]
        full = self.prefix + text_toks + [self.eot]
        dec_in = full[:-1]
        labels = [-100] * (len(self.prefix) - 1) + full[len(self.prefix):]
        return dec_in, labels

    def strip_decode(self, seq):
        """Decode one generated sequence, cutting off the echoed prompt.
        transformers keeps the prompt at the front of beam outputs (with
        <|startofprev|>) and greedy outputs (without), so string matching on
        the decoded text is the reliable way to remove it."""
        text = self.tok.decode([t for t in seq if t >= 0], skip_special_tokens=True).strip()
        if text.startswith(self.prompt_text):
            text = text[len(self.prompt_text):].strip()
        return text

    def transcript_token_count(self, seq):
        """Approximate token length of the transcription part, used to turn
        Whisper's mean per-token score back into a total log-probability."""
        specials = set(self.tok.all_special_ids)
        n = sum(1 for t in seq if t >= 0 and t not in specials)
        return max(1, n - len(self.prompt_toks))
