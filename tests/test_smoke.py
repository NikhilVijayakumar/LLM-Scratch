"""Run: uv run python tests/test_smoke.py"""
import math
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F

from tinygpt import checkpoint
from tinygpt.config import GPTConfig
from tinygpt.data import get_batch
from tinygpt.model import TinyGPT
from tinygpt.train import get_lr

torch.manual_seed(0)
cfg = GPTConfig(block_size=16, vocab_size=100, n_embd=32, n_layer=2, n_head=4, dropout=0.0)
model = TinyGPT(cfg).eval()

# SDPA must equal the original masked-softmax attention from the Colab notebook
attn, x = model.blocks[0].attn, torch.randn(2, 10, 32)
q, k, v = (t.view(2, 10, 4, 8).transpose(1, 2) for t in attn.c_attn(x).split(32, dim=2))
att = (q @ k.transpose(-2, -1)) / math.sqrt(8)
att = att.masked_fill(torch.tril(torch.ones(10, 10)) == 0, float("-inf"))
ref = attn.c_proj((F.softmax(att, -1) @ v).transpose(1, 2).reshape(2, 10, 32))
assert torch.allclose(attn(x), ref, atol=1e-5)

# Colab-format checkpoint (has attn.bias buffers, no config) must load
state = model.state_dict()
for i in range(cfg.n_layer):
    state[f"blocks.{i}.attn.bias"] = torch.tril(torch.ones(1, 1, 16, 16))
opt = torch.optim.AdamW(model.parameters())
with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "old.pt"
    torch.save({"step": 5, "model_state_dict": state, "optimizer_state_dict": opt.state_dict(),
                "val_loss": torch.tensor(1.5)}, path)
    import tinygpt.checkpoint as ck
    ck.GPTConfig = lambda **kw: cfg  # old ckpt has no config; use the tiny one
    loaded, _ = checkpoint.load(path, "cpu")
    ck.GPTConfig = GPTConfig
    checkpoint.save(Path(d) / "new.pt", loaded, opt, 6, 1.4, 1e-4)
    reloaded, ckpt = checkpoint.load(Path(d) / "new.pt", "cpu")
    assert ckpt["config"]["n_embd"] == 32

# batches: y is x shifted by one
data = torch.arange(1000)
xb, yb = get_batch(data, 4, 16, "cpu")
assert xb.shape == (4, 16) and torch.equal(xb + 1, yb)

# training forward/backward + generate
logits, loss = reloaded.train()(xb % 100, yb % 100)
loss.backward()
out = reloaded.generate(torch.zeros(1, 3, dtype=torch.long), max_new_tokens=20)
assert out.shape == (1, 23)

# LR: warmup up to peak, cosine down to min
assert get_lr(0, 0, 10000, 200, 6e-4, 6e-5) < 6e-6
assert abs(get_lr(200, 0, 10000, 200, 6e-4, 6e-5) - 6e-4) < 1e-9
assert abs(get_lr(9999, 0, 10000, 200, 6e-4, 6e-5) - 6e-5) < 1e-8
assert abs(get_lr(10000, 10000, 20000, 0, 6e-5, 1e-5) - 6e-5) < 1e-9
print("ok")
