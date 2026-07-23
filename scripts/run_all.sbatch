#!/bin/bash
# Full corpus: run the DAG pipeline on EVERY case report.
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1-00:00:00
#SBATCH -J mono_all
#SBATCH -o /scratch/users/akshatg/MonopathDAGs/logs/slurm/all_%j.out
#SBATCH -e /scratch/users/akshatg/MonopathDAGs/logs/slurm/all_%j.err

export RUN_KIND=all
source /scratch/users/akshatg/MonopathDAGs/scripts/pipeline_common.sh

mono_setup

# Every case report in the corpus.
cp "$REPO"/webapp/static/pmc_htmls/*.html "$IN_DIR/"

mono_run
