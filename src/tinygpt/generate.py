"""Sample stories from a trained checkpoint.

    tinygpt-generate checkpoints/model_step_19999.pt "One day, Tim went to the forest to find"
"""
import argparse
from pathlib import Path

import torch

from tinygpt import checkpoint
from tinygpt.config import get_device
from tinygpt.data import EOS_ID, enc

DEFAULT_PROMPTS = [
    "One day, Tim went to the forest to find",
    "Once upon a time, there was a big friendly dragon who",
    "Mia had a magic box. When she opened it,",
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("prompts", nargs="*", default=DEFAULT_PROMPTS)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.75)
    p.add_argument("--top-k", type=int, default=40)
    args = p.parse_args()

    device = get_device()
    model, ckpt = checkpoint.load(args.checkpoint, device)
    model.eval()
    print(f"Loaded step {ckpt['step']} (val loss {float(ckpt['val_loss']):.4f}) on {device}")

    for prompt in args.prompts:
        idx = torch.tensor([enc.encode(prompt)], device=device)
        out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k, eos_id=EOS_ID)
        print("=" * 60)
        print(enc.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
