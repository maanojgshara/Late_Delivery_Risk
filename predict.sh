#!/usr/bin/env bash

# --------------- Inference runner  ------------
# Score new deliveries with the trained model
# -------------------------------------------

set -euo pipefail
cd "$(dirname "$0")"

PY=""

for cand in "${PYTHON:-}" "py -3" "python" "python3"; do
  [ -z "$cand" ] && continue
  if $cand -m pip --version >/dev/null 2>&1; then PY="$cand"; break; fi
done

[ -z "$PY" ] && { echo "[predict.sh] ERROR ❌: no Python with pip found."; exit 1; }

export PYTHONUNBUFFERED=1
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-4}"   

echo "[predict.sh] interpreter: $PY"
echo "[run.sh] Launching Predictor  🔄🔄🔄"

$PY -u -m src.predict "$@"

echo "[predict.sh] Done ✅ ✅ ✅ . "