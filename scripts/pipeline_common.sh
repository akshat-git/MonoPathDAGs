#!/bin/bash
# Shared runner for Monopath DAG generation on Sherlock.
#
# Runs the DSPy pipeline against a local, GPU-accelerated Llama 3.1 served by a
# containerized Ollama (Apptainer + --nv, which sidesteps the host glibc 2.17 vs
# ollama's glibc 2.28 mismatch). Sourced by run_smoke.sbatch and run_all.sbatch,
# which set RUN_KIND, then call mono_setup, populate "$IN_DIR", and call mono_run.
#
# Folder layout (all gitignored, separated by usage):
#   logs/slurm/            scheduler stdout/stderr        (SBATCH -o/-e)
#   logs/ollama/           ollama server logs + gpu check
#   logs/runs/<kind>_<jid>/input   case-report HTMLs for this run
#   logs/runs/<kind>_<jid>/output  graph_*.json + graph_metadata.csv
set -uo pipefail

REPO=/scratch/users/akshatg/MonopathDAGs
cd "$REPO"
: "${RUN_KIND:=run}"
: "${MONO_MODEL:=llama3.1:8b}"   # override, e.g. MONO_MODEL=qwen2.5:14b
JID="${SLURM_JOB_ID:-manual}"

SLURM_DIR="$REPO/logs/slurm"
OLLAMA_DIR="$REPO/logs/ollama"
RUN_DIR="$REPO/logs/runs/${RUN_KIND}_${JID}"
IN_DIR="$RUN_DIR/input"
OUT_DIR="$RUN_DIR/output"
SERVE_LOG="$OLLAMA_DIR/serve_${JID}.log"
GPUCHECK="$OLLAMA_DIR/gpucheck_${JID}.txt"
mkdir -p "$SLURM_DIR" "$OLLAMA_DIR" "$IN_DIR" "$OUT_DIR"

SIF="$REPO/.ollama_dist/ollama.sif"
OLLAMA_PID=""

mono_setup() {
  echo "=== [$RUN_KIND $JID] setup on $(hostname) $(date) ==="
  ml purge 2>/dev/null || true
  ml python/3.12.1
  source "$REPO/.venv/bin/activate"
  export PYTHONUNBUFFERED=1
  export APPTAINER_CACHEDIR="$SCRATCH/.apptainer" SINGULARITY_CACHEDIR="$SCRATCH/.apptainer"
  mkdir -p "$APPTAINER_CACHEDIR"
  nvidia-smi --query-gpu=name,memory.total --format=csv 2>&1 | head -2 || true

  [ -f "$SIF" ] || apptainer pull "$SIF" docker://ollama/ollama || { echo "!!! SIF build failed"; exit 41; }

  export OLLAMA_MODELS="$L_SCRATCH/ollama_models"
  mkdir -p "$OLLAMA_MODELS"
  [ -d "$REPO/.ollama/models/blobs" ] && cp -r "$REPO/.ollama/models/." "$OLLAMA_MODELS/" || true

  echo "=== starting containerized ollama (--nv) ==="
  apptainer exec --nv --bind "$SCRATCH","$L_SCRATCH" \
    --env OLLAMA_HOST=127.0.0.1:11434 --env OLLAMA_MODELS="$OLLAMA_MODELS" --env OLLAMA_FLASH_ATTENTION=1 \
    "$SIF" ollama serve > "$SERVE_LOG" 2>&1 &
  OLLAMA_PID=$!
  for i in $(seq 1 90); do curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break; sleep 2; done
  echo "=== ensuring model present: $MONO_MODEL ==="
  apptainer exec --bind "$L_SCRATCH" --env OLLAMA_HOST=127.0.0.1:11434 --env OLLAMA_MODELS="$OLLAMA_MODELS" \
    "$SIF" ollama pull "$MONO_MODEL" || true
  # persist any newly pulled blobs back to the project store for reuse
  cp -rn "$OLLAMA_MODELS/." "$REPO/.ollama/models/" 2>/dev/null || true

  curl -s http://127.0.0.1:11434/api/generate \
    -d "{\"model\":\"$MONO_MODEL\",\"prompt\":\"ok\",\"stream\":false}" -o /dev/null 2>/dev/null
  sleep 2
  grep -a "inference compute" "$SERVE_LOG" | tail -1 | tee "$GPUCHECK"
  if ! grep -aiq "cuda\|gpu" "$GPUCHECK" 2>/dev/null; then
    echo "!!! GPU backend not detected; aborting."; tail -n 20 "$SERVE_LOG"
    [ -n "$OLLAMA_PID" ] && kill "$OLLAMA_PID" 2>/dev/null; exit 30
  fi
  echo "GPU backend confirmed."
}

mono_run() {
  local n; n=$(ls "$IN_DIR"/*.html 2>/dev/null | wc -l)
  echo "=== [$RUN_KIND $JID] generate-graphs ($MONO_MODEL) on $n case(s) START $(date) ==="
  export DSPY_MODEL="ollama_chat/$MONO_MODEL"
  time python -u "$REPO/main.py" generate-graphs \
    --input_dir "$IN_DIR" --output_dir "$OUT_DIR" --env_file .config/.env
  echo "=== generate-graphs END $(date) ==="
  echo "--- output ---"; ls -la "$OUT_DIR"
  python -u "$REPO/scripts/summarize_graphs.py" "$OUT_DIR" || true
  [ -n "$OLLAMA_PID" ] && kill "$OLLAMA_PID" 2>/dev/null || true
  echo "=== [$RUN_KIND $JID] done $(date) ==="
}
