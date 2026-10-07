#!/usr/bin/env bash
# Checkpoint (.pt) -> HF GPT-2 folder -> GGUF f32 -> f16/bf16/Q8_0/.../Q4 -> perplexity table.
#
#   scripts/make_gguf.sh checkpoints/model_step_19999.pt
#
# Needs: uv sync --group export, and llama.cpp built in tools/llama.cpp (see README in that dir):
#   git clone --depth 1 https://github.com/ggml-org/llama.cpp tools/llama.cpp
#   cmake -S tools/llama.cpp -B tools/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
#   cmake --build tools/llama.cpp/build -j --target llama-quantize llama-completion llama-perplexity
set -euo pipefail

CKPT=${1:?usage: $0 <checkpoint.pt> [out_dir]}
OUT=${2:-exports}
LLAMA=tools/llama.cpp
BIN=$LLAMA/build/bin
NAME=tinygpt
TYPES=(F16 BF16 Q8_0 Q6_K Q5_K_M Q5_0 Q4_K_M Q4_0)

mkdir -p "$OUT"

echo "== 1. PyTorch checkpoint -> Hugging Face GPT-2 format"
uv run tinygpt-export-hf "$CKPT" "$OUT/hf"

echo "== 2. HF -> GGUF f32 (lossless master copy)"
uv run python $LLAMA/convert_hf_to_gguf.py "$OUT/hf" --outtype f32 --outfile "$OUT/$NAME-F32.gguf"

echo "== 3. Quantize from the f32 master"
for t in "${TYPES[@]}"; do
  $BIN/llama-quantize "$OUT/$NAME-F32.gguf" "$OUT/$NAME-$t.gguf" "$t" > "$OUT/quantize-$t.log" 2>&1
  # Which formats each tensor actually got. 384-wide rows aren't divisible by 256 (k-quant block),
  # so K types fall back to legacy q5_0/q8_0 for most tensors.
  echo "$t: $(grep -oP 'converting to \K\w+' "$OUT/quantize-$t.log" | sort | uniq -c | xargs)"
done

echo "== 4. Perplexity on held-out TinyStories (lower = better; compare against F32)"
uv run python -c "
from datasets import load_dataset
ds = load_dataset('roneneldan/TinyStories', split='validation[-500:]')
open('$OUT/val.txt', 'w').write('\n\n'.join(ds['text']))
"
printf '%-8s %8s  %s\n' TYPE SIZE PPL
for t in F32 "${TYPES[@]}"; do
  f="$OUT/$NAME-$t.gguf"
  ppl=$($BIN/llama-perplexity -m "$f" -f "$OUT/val.txt" -c 256 -b 256 2>&1 | grep -oP 'Final estimate: PPL = \K[0-9.]+' || echo "?")
  printf '%-8s %8s  %s\n' "$t" "$(du -h "$f" | cut -f1)" "$ppl"
done
