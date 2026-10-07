from pathlib import Path

import numpy as np
import tiktoken
import torch
from datasets import load_dataset

enc = tiktoken.get_encoding("gpt2")
EOS_ID = enc.eot_token

DATA_DIR = Path("data")


def load_tokens(split: str, n_samples: int) -> torch.Tensor:
    """Tokenize TinyStories once, cache to data/<split>_<n>.npy, memory-map afterwards."""
    cache = DATA_DIR / f"{split}_{n_samples}.npy"
    if not cache.exists():
        print(f"Tokenizing TinyStories {split}[:{n_samples}]...")
        texts = load_dataset("roneneldan/TinyStories", split=f"{split}[:{n_samples}]")["text"]
        docs = enc.encode_batch(texts, allowed_special={"<|endoftext|>"})
        tokens = np.fromiter((t for doc in docs for t in (*doc, EOS_ID)), dtype=np.uint16)  # vocab < 65536
        DATA_DIR.mkdir(exist_ok=True)
        np.save(cache, tokens)
    tokens = torch.from_numpy(np.load(cache).astype(np.int64))
    print(f"{split} tokens: {len(tokens):,}")
    return tokens


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str):
    ix = torch.randint(len(data) - block_size, (batch_size, 1))
    offsets = ix + torch.arange(block_size + 1)  # (B, T+1) gather in one shot, no Python loop
    chunk = data[offsets]
    x, y = chunk[:, :-1], chunk[:, 1:]
    if device == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)
