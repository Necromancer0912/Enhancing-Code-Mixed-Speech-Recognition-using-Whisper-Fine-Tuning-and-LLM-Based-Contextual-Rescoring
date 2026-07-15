"""Domain-Adaptive Rescorer: LoRA fine-tuning of a small multilingual LM on
the MUCS training transcripts (text only). The paper rescoresonly with
off-the-shelf LLMs; adapting the scorer to the exact code-switched,
spoken-tutorial register is our extension."""
import gc

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import DAR_BASE_MODEL, SEED


def train_dar_adapter(train_texts, out_dir, base_model=DAR_BASE_MODEL, device="cuda",
                      epochs=1, batch_size=16, lr=2e-4, max_length=64):
    """LoRA-tune the base LM as a causal LM over the training transcripts and
    save the adapter (a few MB) to `out_dir`."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # bf16 to match scoring and avoid BLOOM's fp16 overflow problem
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=["query_key_value"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = np.random.default_rng(SEED)

    model.train()
    losses = []
    for epoch in range(epochs):
        order = rng.permutation(len(train_texts))
        batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        for idx in tqdm(batches, desc=f"DAR epoch {epoch + 1}/{epochs}"):
            chunk = [train_texts[j] for j in idx]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_length).to(device)
            labels = enc.input_ids.masked_fill(~enc.attention_mask.bool(), -100)
            loss = model(**enc, labels=labels).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())

    print(f"final DAR loss: {np.mean(losses[-50:]):.3f}")
    model.save_pretrained(out_dir)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out_dir
