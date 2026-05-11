#!/usr/bin/env bash
# Wrapper to run llama-bench inside the llama-server container
# Usage: ./bench.sh [llama-bench args]

set -euo pipefail

# Get current model from config or env
MODEL_FILE="${LLAMA_BENCH_MODEL:-Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
MODEL_PATH="/models/${MODEL_FILE}"

echo "=== Running llama-bench in container ==="
echo "  Model: ${MODEL_PATH}"

docker compose exec -it llama-server llama-bench \
    -m "${MODEL_PATH}" \
    -ngl 99 \
    "$@"
