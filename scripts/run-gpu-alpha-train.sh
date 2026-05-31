#!/usr/bin/env bash
set -euo pipefail

if python3 - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  exec python3 scripts/gpu_train_alphazero.py "$@"
fi

for py in \
  /home/kyle/Documents/GitHub/compass-of-es/.venv/bin/python \
  /home/kyle/Documents/GitHub/humpback-annotation/.venv/bin/python \
  /home/kyle/Documents/GitHub/adsb-research/.venv/bin/python \
  /home/kyle/Documents/GitHub/pencil-upscaling/.venv/bin/python \
  /home/kyle/Documents/GitHub/hamer/.venv/bin/python \
  /home/kyle/Documents/GitHub/transformirror-web/.venv/bin/python
do
  if [ -x "$py" ] && "$py" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    exec "$py" scripts/gpu_train_alphazero.py "$@"
  fi
done

echo "No CUDA-enabled PyTorch Python was found." >&2
exit 1
