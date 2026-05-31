#!/usr/bin/env bash
set -euo pipefail

exec bash scripts/run-torch-python.sh scripts/gpu_train_alphazero.py "$@"
