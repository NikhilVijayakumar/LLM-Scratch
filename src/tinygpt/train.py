"""Pretrain TinyGPT on TinyStories, or resume from a checkpoint.

Fresh run (original Colab phase 1):
    tinygpt-train --max-steps 10000
Low-LR continuation (original Colab phase 2):
    tinygpt-train --resume checkpoints/model_step_9999.pt --max-steps 20000 --lr 6e-5 --min-lr 1e-5 --warmup 0
"""
import argparse
import math
import time
from pathlib import Path

import torch

from tinygpt import checkpoint
from tinygpt.config import GPTConfig, autocast, get_device
from tinygpt.data import EOS_ID, enc, get_batch, load_tokens
from tinygpt.model import TinyGPT

PREVIEW_PROMPT = "Once upon a time, a little girl named Lily found"


def get_lr(step, start_step, max_steps, warmup, lr, min_lr):
    """Linear warmup then cosine decay from lr to min_lr over [start_step, max_steps)."""
    step -= start_step
    total = max_steps - start_step
    if step < warmup:
        return lr * (step + 1) / (warmup + 1)
    ratio = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (lr - min_lr)


@torch.no_grad()
def estimate_loss(model, splits, args, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(data, args.batch_size, model.cfg.block_size, device)
            with autocast(device):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resume", type=Path, help="checkpoint to continue from")
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--train-samples", type=int, default=100_000)
    p.add_argument("--val-samples", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=10_000)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr", type=float, default=6e-5)
    p.add_argument("--compile", action="store_true", help="torch.compile the model (GPU, faster after warmup)")
    args = p.parse_args()

    device = get_device()
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    torch.manual_seed(1337)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    splits = {
        "train": load_tokens("train", args.train_samples),
        "val": load_tokens("validation", args.val_samples),
    }

    if args.resume:
        model, ckpt = checkpoint.load(args.resume, device)
        start_step = ckpt["step"] + 1
        print(f"Resumed {args.resume} (step {ckpt['step']}, val loss {float(ckpt['val_loss']):.4f})")
    else:
        model, ckpt, start_step = TinyGPT(GPTConfig()).to(device), None, 0
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95), fused=device == "cuda"
    )
    if ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # GradScaler only matters for fp16; it's a no-op when disabled (bf16 / CPU).
    use_fp16 = device == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    fwd = torch.compile(model) if args.compile else model
    prompt_ids = torch.tensor([enc.encode(PREVIEW_PROMPT)], device=device)

    model.train()
    start_time = time.time()
    for step in range(start_step, args.max_steps):
        lr = get_lr(step, start_step, args.max_steps, args.warmup, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            losses = estimate_loss(model, splits, args, device)
            print(f"\n--- Step {step} | Train {losses['train']:.4f} | Val {losses['val']:.4f} "
                  f"| LR {lr:.2e} | {(time.time() - start_time) / 60:.1f}m ---")
            sample = model.generate(prompt_ids, max_new_tokens=80, eos_id=EOS_ID)
            print(enc.decode(sample[0].tolist()))
            path = args.out_dir / f"model_step_{step}.pt"
            checkpoint.save(path, model, optimizer, step, losses["val"], lr)
            print(f"Saved {path}")

        x, y = get_batch(splits["train"], args.batch_size, model.cfg.block_size, device)
        with autocast(device):
            _, loss = fwd(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

    print("Training complete.")


if __name__ == "__main__":
    main()
