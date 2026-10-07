from dataclasses import asdict
from pathlib import Path

import torch

from tinygpt.config import GPTConfig
from tinygpt.model import TinyGPT


def save(path: Path, model: TinyGPT, optimizer, step: int, val_loss: float, lr: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": float(val_loss),
        "learning_rate": lr,
        "config": asdict(model.cfg),
    }, path)


def load(path: Path, device: str):
    """Returns (model, checkpoint dict). Works with the original Colab checkpoints too."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = GPTConfig(**ckpt.get("config", {}))  # Colab checkpoints have no config: defaults match them
    # Colab model stored the causal mask as an `attn.bias` buffer; SDPA doesn't need it.
    state = {k: v for k, v in ckpt["model_state_dict"].items() if not k.endswith(".attn.bias")}
    model = TinyGPT(cfg).to(device)
    model.load_state_dict(state)
    return model, ckpt
