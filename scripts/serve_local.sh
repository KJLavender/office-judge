#!/usr/bin/env bash
# Start local decision models behind llama-server's /v1/systemone (CPU build).
# The binary is copied and renamed to oj-llama-server.exe so another project
# stopping "llama-server.exe" by name does not take these down.
#   ports: 8191 julia-1, 8192 laya (English), 8193 laya-multilingual, 8195 kev-4b
#   KEV_GPU=1 runs kev-4b on the CUDA build (p50 ~60 ms instead of ~1.3 s on CPU)
#   ONLY=kev-4b starts just that model
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/.cache/llama/cpu/oj-llama-server.exe"
IEKA="${IEKA_MODELS:-$HOME/Desktop/IEKA/.cache/llama/models}"
LOCAL="$ROOT/.cache/llama/models"
LOG="$ROOT/data/out/llama"
mkdir -p "$LOG"
start() {  # name port model extra-args...
  local name=$1 port=$2 model=$3; shift 3
  [ -n "${ONLY:-}" ] && [ "$ONLY" != "$name" ] && return 0
  local bin="$BIN"
  if [ "$name" = kev-4b ] && [ "${KEV_GPU:-0}" = 1 ]; then
    bin="$ROOT/.cache/llama/cuda/llama-server.exe"; set -- "$@" -ngl 99
  fi
  "$bin" -m "$model" --port "$port" --alias "$name" "$@" > "$LOG/$name.log" 2>&1 &
  echo "$name -> http://127.0.0.1:$port (pid $!)"
}
start julia-1           8191 "$IEKA/Julia-1-Q8_0.gguf"            -b 8192 -ub 8192
start laya              8192 "$IEKA/Laya-Q8_0.gguf"               -b 8192 -ub 8192
start laya-multilingual 8193 "$LOCAL/Laya-Multilingual-Q8_0.gguf" -c 1024 -b 1024 -ub 1024
start kev-4b            8195 "$IEKA/Kev-4B-Q4_K_M.gguf"           -c 4096 -b 2048 -ub 2048
wait
