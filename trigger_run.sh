#!/usr/bin/env bash

set -euo pipefail
export PYTHONUNBUFFERED=1
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-4}"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -u -m src.main 2>&1 | tee output.txt