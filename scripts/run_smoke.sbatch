#!/bin/bash
# Smoke test: run the DAG pipeline on ONE case report (end-to-end validation).
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:15:00
#SBATCH -J mono_smoke
#SBATCH -o /scratch/users/akshatg/MonopathDAGs/logs/slurm/smoke_%j.out
#SBATCH -e /scratch/users/akshatg/MonopathDAGs/logs/slurm/smoke_%j.err

export RUN_KIND=smoke
source /scratch/users/akshatg/MonopathDAGs/scripts/pipeline_common.sh

mono_setup

# One case report. Override with:  sbatch --export=ALL,SMOKE_CASE=<file.html> scripts/run_smoke.sbatch
SMOKE_CASE="${SMOKE_CASE:-Metachronous_development_of_L858R_and_T790M_EGFR_mutations_following_ALK_inhibit_PMC12000955.html}"
cp "$REPO/webapp/static/pmc_htmls/$SMOKE_CASE" "$IN_DIR/"
echo "smoke case: $SMOKE_CASE"

mono_run
