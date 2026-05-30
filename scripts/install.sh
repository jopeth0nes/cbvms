#!/usr/bin/env bash
# Install core CBVMS dependencies.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if ! command -v "$PYTHON_BIN" &>/dev/null; then
  echo "Error: $PYTHON_BIN not found. Install: brew install python@3.12"
  exit 1
fi

# CustomTkinter needs tkinter (_tkinter). Homebrew Python requires python-tk.
if [[ "$(uname -s)" == "Darwin" ]]; then
  if ! "$PYTHON_BIN" -c "import tkinter" 2>/dev/null; then
    echo "Installing python-tk@3.12 (required for CustomTkinter GUI)..."
    brew install python-tk@3.12
  fi
  if ! "$PYTHON_BIN" -c "import tkinter" 2>/dev/null; then
    echo "Error: tkinter still missing after python-tk install."
    echo "Try: brew install python@3.12 python-tk@3.12"
    exit 1
  fi
fi

if [[ ! -d .venv ]]; then
  echo "Creating venv with $PYTHON_BIN..."
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip wheel
# torch (via ultralytics) requires setuptools<82
python -m pip install "setuptools>=70,<82"
pip install -r requirements.txt

if ! python -c "import tkinter; import customtkinter" 2>/dev/null; then
  echo "Error: customtkinter/tkinter not available in venv."
  exit 1
fi

echo "Core dependencies installed."
echo "Run the app:  ./scripts/run.sh"
echo "Or:          source .venv/bin/activate && python main.py"
echo "Optional:    ./scripts/install_face_deps.sh"
