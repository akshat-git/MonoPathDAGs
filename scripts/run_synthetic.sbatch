#!/bin/bash
# Synthetic case generation: DAGs -> synthetic narratives (control + sample).
# Consumes a graph_metadata.csv produced by a prior generate-graphs run.
#
# Usage:
#   sbatch --export=ALL,GRAPH_CSV=logs/runs/all_<jid>/output/graph_metadata.csv scripts/run_synthetic.sbatch
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:16GB
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -J mono_synth
#SBATCH -o /scratch/users/akshatg/MonopathDAGs/logs/slurm/synth_%j.out
#SBATCH -e /scratch/users/akshatg/MonopathDAGs/logs/slurm/synth_%j.err

export RUN_KIND=synth
source /scratch/users/akshatg/MonopathDAGs/scripts/pipeline_common.sh

mono_setup   # brings up the containerized GPU Ollama server + confirms backend

GRAPH_CSV="${GRAPH_CSV:?set GRAPH_CSV to a graph_metadata.csv from a generate-graphs run}"
SYNTH_OUT="$RUN_DIR/synthetic"
mkdir -p "$SYNTH_OUT"
echo "=== generate-synthetic ($MONO_MODEL) from $GRAPH_CSV START $(date) ==="
python -u "$REPO/main.py" generate-synthetic \
  --csv "$GRAPH_CSV" \
  --model "ollama_chat/$MONO_MODEL" \
  --output_dir "$SYNTH_OUT" \
  --env_file .config/.env
echo "=== generate-synthetic END $(date) ==="
echo "--- outputs ---"; ls -la "$SYNTH_OUT" 2>/dev/null
echo "--- index.jsonl ---"; cat "$SYNTH_OUT/index.jsonl" 2>/dev/null | head

[ -n "${OLLAMA_PID:-}" ] && kill "$OLLAMA_PID" 2>/dev/null || true
echo "=== [synth $JID] done $(date) ==="
