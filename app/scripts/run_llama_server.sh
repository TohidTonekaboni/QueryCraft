#!/usr/bin/env bash
# Serve a GGUF model with llama.cpp on the host (Metal/MPS on macOS, CUDA if built with it).
# Needed for *_BACKEND=remote when the app runs in Docker on macOS: containers cannot use MPS.
#   brew install llama.cpp
# Usage (two terminals):
#   app/scripts/run_llama_server.sh sql     # SQLCoder on :8080, all layers on GPU/MPS
#   app/scripts/run_llama_server.sh gemma   # Gemma on :8081, CPU by default
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
dir="${SQL_MODEL_DIR:-./models}"
case "${1:-}" in
  sql)   exec llama-server -m "$dir/$SQL_MODEL_FILE" --host 0.0.0.0 --port 8080 \
           -ngl "${SQL_N_GPU_LAYERS:--1}" -c "${SQL_N_CTX:-4096}" ;;
  gemma) exec llama-server -m "$dir/$GEMMA_MODEL_FILE" --host 0.0.0.0 --port 8081 \
           -ngl "${GEMMA_N_GPU_LAYERS:-0}" -c "${GEMMA_N_CTX:-2048}" ;;
  *) echo "usage: $0 sql|gemma" >&2; exit 1 ;;
esac
