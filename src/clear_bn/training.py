"""Decoder-only Whisper fine-tuning with the descriptive prompt, sized for a
single 8 GB GPU (fp16 autocast + gradient accumulation, encoder frozen)."""
import math
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .audio import load_segment
from .config import TARGET_SR
from .metrics import norm_punct


class PromptedMUCSDataset(Dataset):
    """Loads one utterance slice, extracts log-mel features and builds the
    prompt-conditioned decoder inputs. Light noise/gain augmentation on the
    training side only."""

    def __init__(self, df, processor, prefix, train_mode=False, max_label_len=384):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.prefix = prefix
        self.train_mode = train_mode
        self.max_label_len = max_label_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        audio = load_segment(r.wav_path, r.start_sec, r.end_sec)
        if self.train_mode:
            if np.random.rand() < 0.3:
                audio = np.clip(audio + np.random.normal(0, 0.003, audio.shape).astype(np.float32), -1, 1)
            if np.random.rand() < 0.3:
                audio = np.clip(audio * np.random.uniform(0.85, 1.15), -1, 1)
        feats = self.processor.feature_extractor(audio, sampling_rate=TARGET_SR, return_tensors="np")
        dec_in, labels = self.prefix.training_pair(r.clean_text, self.max_label_len)
        return dict(input_features=feats.input_features[0],
                    decoder_input_ids=dec_in, labels=labels)


def make_collator(eot_id):
    def collate(batch):
        feats = torch.tensor(np.stack([b["input_features"] for b in batch]))
        maxlen = max(len(b["decoder_input_ids"]) for b in batch)
        dec, lab = [], []
        for b in batch:
            pad = maxlen - len(b["decoder_input_ids"])
            dec.append(b["decoder_input_ids"] + [eot_id] * pad)
            lab.append(b["labels"] + [-100] * pad)
        return dict(input_features=feats,
                    decoder_input_ids=torch.tensor(dec, dtype=torch.long),
                    labels=torch.tensor(lab, dtype=torch.long))
    return collate


@torch.no_grad()
def prompted_val_wer(model, processor, prefix, val_df, batch_size=8, fp16=True):
    """Greedy prompted decoding on a validation sample, scored with P-WER."""
    from jiwer import wer

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    prompt_ids = prefix.generation_prompt_ids(device)

    hyps = []
    for i in range(0, len(val_df), batch_size):
        batch = val_df.iloc[i:i + batch_size]
        audios = [load_segment(r.wav_path, r.start_sec, r.end_sec) for r in batch.itertuples()]
        feats = processor(audios, sampling_rate=TARGET_SR, return_tensors="pt")
        fx = feats.input_features.to(device)
        with torch.autocast("cuda", enabled=fp16):
            out = model.generate(fx, prompt_ids=prompt_ids, language="bn", task="transcribe",
                                 max_new_tokens=180, do_sample=False, num_beams=1, use_cache=True)
        hyps.extend(prefix.strip_decode(seq) for seq in out.cpu().tolist())
    if was_training:
        model.train()

    refs = [norm_punct(t) for t in val_df.ref_text]
    hyps = [norm_punct(h) or "<empty>" for h in hyps]
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    return 100.0 * wer([p[0] for p in pairs], [p[1] for p in pairs])


def finetune(model, processor, prefix, train_df, val_df, out_dir, epochs=2,
             batch_size=8, grad_accum=2, lr=2e-5, weight_decay=0.01,
             warmup_updates=200, max_grad_norm=1.0, eval_every=400,
             eval_samples=120, fp16=True, seed=42):
    """Train the decoder (encoder frozen by the caller) and keep the best
    checkpoint by validation P-WER. Returns the history DataFrame."""
    device = next(model.parameters()).device
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PromptedMUCSDataset(train_df, processor, prefix, train_mode=True)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, collate_fn=make_collator(prefix.eot), drop_last=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    updates_per_epoch = math.ceil(len(loader) / grad_accum)
    total_updates = updates_per_epoch * epochs

    def lr_lambda(step):
        if step < warmup_updates:
            return (step + 1) / warmup_updates
        p = (step - warmup_updates) / max(1, total_updates - warmup_updates)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    val_probe = val_df.sample(n=min(eval_samples, len(val_df)), random_state=seed).reset_index(drop=True)

    history = []
    best_wer = float("inf")
    update = 0
    t0 = time.time()

    def checkpoint_if_best(val_wer, epoch, losses):
        nonlocal best_wer
        history.append(dict(update=update, epoch=epoch, train_loss=float(np.mean(losses[-400:])),
                            val_pwer=val_wer, minutes=round((time.time() - t0) / 60, 1)))
        if val_wer < best_wer:
            best_wer = val_wer
            model.save_pretrained(out_dir / "best")
            processor.save_pretrained(out_dir / "best")
            print(f"  new best val P-WER {val_wer:.2f}%, checkpoint saved")

    model.train()
    for epoch in range(1, epochs + 1):
        losses = []
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(loader, desc=f"epoch {epoch}/{epochs}")
        for step, batch in enumerate(pbar):
            fx = batch["input_features"].to(device)
            dec = batch["decoder_input_ids"].to(device)
            lab = batch["labels"].to(device)

            with torch.autocast("cuda", enabled=fp16):
                out = model(input_features=fx, decoder_input_ids=dec, labels=lab)
                loss = out.loss / grad_accum
            scaler.scale(loss).backward()
            losses.append(float(out.loss.detach()))

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                update += 1

                if update % 50 == 0:
                    pbar.set_postfix(loss=f"{np.mean(losses[-200:]):.3f}",
                                     lr=f"{sched.get_last_lr()[0]:.2e}", upd=update)
                if update % eval_every == 0:
                    vw = prompted_val_wer(model, processor, prefix, val_probe, fp16=fp16)
                    print(f"  update {update}: loss {np.mean(losses[-400:]):.3f}, val P-WER {vw:.2f}%")
                    checkpoint_if_best(vw, epoch, losses)

        vw = prompted_val_wer(model, processor, prefix, val_probe, fp16=fp16)
        print(f"end of epoch {epoch}: val P-WER {vw:.2f}%")
        checkpoint_if_best(vw, epoch, losses)

    model.save_pretrained(out_dir / "last")
    processor.save_pretrained(out_dir / "last")
    hist = pd.DataFrame(history)
    hist.to_csv(out_dir / "history.csv", index=False)
    print(f"done in {(time.time() - t0) / 60:.1f} min, best val P-WER {best_wer:.2f}%")
    return hist
