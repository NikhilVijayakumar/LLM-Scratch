from dataclasses import dataclass

import torch


@dataclass
class GPTConfig:
    block_size: int = 256   # max context window length
    vocab_size: int = 50257  # GPT-2 BPE vocab
    n_embd: int = 384
    n_layer: int = 6
    n_head: int = 6         # head_dim = 384 / 6 = 64
    dropout: float = 0.1


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def autocast(device: str):
    """Mixed precision context: bf16 where supported, fp16 on older CUDA (T4), off on CPU/MPS."""
    if device != "cuda":
        return torch.autocast("cpu", enabled=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)
