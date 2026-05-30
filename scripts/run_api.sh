#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
pip install -q -r requirements.txt -r requirements-api.txt
exec python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
