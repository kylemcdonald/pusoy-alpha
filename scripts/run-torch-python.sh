#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <python-script> [args...]" >&2
  exit 2
fi

for py in \
  "$(command -v python3 || true)" \
  /home/kyle/Documents/GitHub/compass-of-es/.venv/bin/python \
  /home/kyle/Documents/GitHub/humpback-annotation/.venv/bin/python \
  /home/kyle/Documents/GitHub/adsb-research/.venv/bin/python \
  /home/kyle/Documents/GitHub/pencil-upscaling/.venv/bin/python \
  /home/kyle/Documents/GitHub/hamer/.venv/bin/python \
  /home/kyle/Documents/GitHub/transformirror-web/.venv/bin/python
do
  if [ -n "$py" ] && [ -x "$py" ] && "$py" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    exec "$py" "$@"
  fi
done

echo "No CUDA-enabled PyTorch Python was found." >&2
exit 1
