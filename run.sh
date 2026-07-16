#!/usr/bin/env bash

# ------ Pipeline runner  ------------
# Forwards all CLI flags to the pipeline.
# Auto-detects a Python with pip
# Streams output live
# Forwards all CLI flags to the pipeline
# ---------------------------------------------------------------------------

set -euo pipefail
cd "$(dirname "$0")"

PY=""

for cand in "${PYTHON:-}" "py -3" "python" "python3"; do
  [ -z "$cand" ] && continue
  if $cand -m pip --version >/dev/null 2>&1; then PY="$cand"; break; fi
done

[ -z "$PY" ] && { echo "[run] ERROR ❌: no Python with pip found (set PYTHON=...)."; exit 1; }

export PYTHONUNBUFFERED=1
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-4}"   

echo "[run.sh] Using interpreter: $PY"
echo "[run.sh] Launching Machine Learning pipeline  🔄🔄🔄"

$PY -u -m src.main "$@"

echo "[run.sh] Done ✅ ✅ ✅ . Refer to artifacts/ (figures, tables, logs, runs) and models/."