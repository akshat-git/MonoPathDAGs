#!/bin/bash
# First-time setup for a fresh clone of Monopath DAGs.
#
# Creates the Python venv + installs deps, and (on Sherlock) builds the local
# GPU Ollama container image. It does NOT fetch the case-report dataset — that
# is deliberately not committed; see step (data) printed at the end.
#
# Heavy steps (pip install, container pull) need internet and real CPU, so run
# this inside a compute allocation, NOT on the login node:
#     salloc -p dev -c 4 --mem 8G -t 60      # or sh_dev
#     bash scripts/setup.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
echo "=== Monopath DAGs setup in $REPO ==="

# --- warn if on a login node (Sherlock) ---
if command -v squeue >/dev/null 2>&1 && [ -z "${SLURM_JOB_ID:-}" ]; then
  case "$(hostname)" in
    sh*-ln*|*login*) echo "WARNING: looks like a login node. Heavy steps should run in sh_dev/salloc." ;;
  esac
fi

# --- 1. Python (>=3.9); prefer the Sherlock module ---
if command -v ml >/dev/null 2>&1; then
  ml python/3.12.1 2>/dev/null || ml python 2>/dev/null || true
fi
PY="$(command -v python3.12 || command -v python3 || command -v python)"
echo "Using python: $PY ($("$PY" --version 2>&1))"

# --- 2. venv + dependencies ---
if [ ! -d .venv ]; then
  echo "=== creating .venv ==="
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
echo "=== pip install -r requirements.txt ==="
pip install --only-binary=:all: pillow || true    # avoid source build (libjpeg headers)
pip install --prefer-binary -r requirements.txt
python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null \
  || python -m spacy download en_core_web_sm || true
echo "deps installed."

# --- 3. Local GPU Ollama container (Sherlock / Apptainer) ---
SIF="$REPO/.ollama_dist/ollama.sif"
if command -v apptainer >/dev/null 2>&1; then
  if [ ! -f "$SIF" ]; then
    echo "=== building Ollama container image (one-time, ~2.8GB) ==="
    export APPTAINER_CACHEDIR="${SCRATCH:-$REPO}/.apptainer"
    mkdir -p "$APPTAINER_CACHEDIR" "$REPO/.ollama_dist"
    apptainer pull "$SIF" docker://ollama/ollama || echo "WARNING: SIF build failed; build later."
  else
    echo "Ollama SIF already present."
  fi
else
  echo "NOTE: apptainer not found — install/point at an Ollama server yourself, or use a hosted model (Gemini)."
fi

# --- 4. .config/.env template ---
mkdir -p .config
if [ ! -f .config/.env ]; then
  echo "=== writing .config/.env template ==="
  cat > .config/.env <<'ENV'
# Model backend for the pipeline. Local (no key):
DSPY_MODEL="ollama_chat/llama3.1:8b"
# Hosted alternative (uncomment + set a key):
# DSPY_MODEL="gemini/gemini-2.0-flash"
# GEMINI_APIKEY=""
# GPTKEY=""
ENV
else
  echo ".config/.env already exists — leaving it."
fi

cat <<EOF

=== setup complete ===
Still needed (not committed):
  DATA: put PMC case-report HTMLs in webapp/static/pmc_htmls/ — either
        - re-scrape:  set \$ncbi_api_key and \$email, then run src/data/web/scrape.py, or
        - copy an existing corpus into that folder.
Then run (from a GPU allocation):
  sbatch --export=ALL,MONO_MODEL=llama3.1:8b scripts/run_smoke.sbatch     # one case
  sbatch --export=ALL,MONO_MODEL=llama3.1:8b scripts/run_all.sbatch       # full corpus
The first run pulls the model into .ollama/ (cached thereafter).
See PIPELINE.md for the full data flow.
EOF
