"""Convert a TinyGPT .pt checkpoint into a Hugging Face GPT-2 folder.

TinyGPT is architecturally GPT-2 (learned positions, pre-LayerNorm, GELU MLP, tied
embeddings, same tokenizer), so the weights map 1:1 onto GPT2LMHeadModel. That folder
is what llama.cpp's convert_hf_to_gguf.py understands.

    tinygpt-export-hf checkpoints/model_step_19999.pt exports/tinygpt-hf
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from tinygpt import checkpoint


def to_hf_state_dict(sd: dict) -> dict:
    """Rename TinyGPT keys to HF GPT-2 keys.

    HF GPT-2 uses Conv1D (weight shape [in, out]) instead of nn.Linear ([out, in]),
    so every Linear weight inside a block gets transposed.
    """
    out = {
        "transformer.wte.weight": sd["token_embedding.weight"],
        "transformer.wpe.weight": sd["position_embedding.weight"],
        "transformer.ln_f.weight": sd["ln_f.weight"],
        "transformer.ln_f.bias": sd["ln_f.bias"],
        "lm_head.weight": sd["lm_head.weight"],  # tied to wte
    }
    rename = {
        "ln_1": "ln_1", "ln_2": "ln_2",
        "attn.c_attn": "attn.c_attn", "attn.c_proj": "attn.c_proj",
        "mlp.0": "mlp.c_fc", "mlp.2": "mlp.c_proj",
    }
    n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    for i in range(n_layer):
        for ours, theirs in rename.items():
            w, b = sd[f"blocks.{i}.{ours}.weight"], sd[f"blocks.{i}.{ours}.bias"]
            if not ours.startswith("ln"):
                w = w.t()
            out[f"transformer.h.{i}.{theirs}.weight"] = w.contiguous()
            out[f"transformer.h.{i}.{theirs}.bias"] = b
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("out_dir", type=Path)
    args = p.parse_args()

    model, ckpt = checkpoint.load(args.checkpoint, "cpu")
    model.eval()
    cfg = model.cfg

    hf_cfg = GPT2Config(
        vocab_size=cfg.vocab_size,
        n_positions=cfg.block_size,
        n_ctx=cfg.block_size,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        layer_norm_epsilon=1e-5,         # nn.LayerNorm default
        activation_function="gelu",      # nn.GELU() = exact erf GELU (GPT-2 default is tanh approx)
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        bos_token_id=50256, eos_token_id=50256,
        tie_word_embeddings=True,
    )
    hf_model = GPT2LMHeadModel(hf_cfg).eval()
    missing, unexpected = hf_model.load_state_dict(to_hf_state_dict(model.state_dict()), strict=False)
    # HF keeps only the causal-mask buffers out of the checkpoint; anything else missing is a bug
    assert not unexpected and all(k.endswith((".attn.bias", ".attn.masked_bias")) for k in missing), (missing, unexpected)

    # Sanity check: both models must produce the same logits
    idx = torch.randint(0, cfg.vocab_size, (2, 32))
    with torch.no_grad():
        ours, _ = model(idx)
        theirs = hf_model(idx).logits
    diff = (ours - theirs).abs().max().item()
    print(f"Max logit diff TinyGPT vs HF GPT-2: {diff:.2e}")
    assert diff < 1e-3, "weight mapping is wrong"

    hf_model.save_pretrained(args.out_dir)  # config.json + model.safetensors (fp32)
    AutoTokenizer.from_pretrained("openai-community/gpt2").save_pretrained(args.out_dir)  # same BPE as tiktoken "gpt2"
    print(f"Saved HF model (step {ckpt['step']}) to {args.out_dir}")


if __name__ == "__main__":
    main()
