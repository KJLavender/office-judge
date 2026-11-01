#!/usr/bin/env bash
# One-command live demo: Kev-4B (local decision model) + Claude fallback + web page.
#   ./scripts/demo.sh            # Kev-4B on GPU when >= 4 GB VRAM is free, else CPU
#   ./scripts/demo.sh --offline  # no models at all: keyword heuristic, for when nothing else works
# Ctrl+C stops everything.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PORT="${PORT:-8000}"
URL="http://127.0.0.1:$PORT/?sample=indirect&auto=1"

cleanup() { [ -n "${KEV_PID:-}" ] && kill "$KEV_PID" 2>/dev/null || true; }
trap cleanup EXIT

if [ "${1:-}" = "--offline" ]; then
  export OFFICE_JUDGE_PRIMARY=demo
else
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
  if [ "${free:-0}" -ge 4096 ]; then export KEV_GPU=1; echo "Kev-4B on GPU (${free} MiB free)"
  else echo "Kev-4B on CPU (GPU busy: ${free:-0} MiB free); first answers take ~1 s per sentence"; fi
  ONLY=kev-4b ./scripts/serve_local.sh & KEV_PID=$!
  until curl -s 127.0.0.1:8195/health | grep -q ok; do sleep 1; done
  export TYPESAFE_BASE_URL=http://127.0.0.1:8195 TYPESAFE_DEFAULT_MODEL=kev-4b
fi

echo "opening $URL"
(sleep 3; cmd.exe /c start "" "$URL" >/dev/null 2>&1 || true) &
uv run uvicorn office_judge.app:app --port "$PORT"
