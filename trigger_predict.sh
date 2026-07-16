#!/usr/bin/env bash

set -euo pipefail
export PYTHONUNBUFFERED=1
python -u -m src.predict 2>&1 | tee output_predict.txt