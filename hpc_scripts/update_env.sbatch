#!/bin/bash
#SBATCH --job-name=nexrad-env-update
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=env_update_%j.out
#SBATCH --error=env_update_%j.err

set -euo pipefail
set -x

module load miniforge/24.11.2-py312

# Find the conda base directory and use absolute paths (works in Slurm)
BASE="$(conda info --base)"
CONDA="${BASE}/bin/conda"
MAMBA="${BASE}/bin/mamba"

# Ensure mamba exists in base (safe to run repeatedly)
if [ ! -x "$MAMBA" ]; then
  "$CONDA" install -y -n base -c conda-forge mamba
fi

cd "${SLURM_SUBMIT_DIR}"

# Update the env from your recipe
"$MAMBA" env update -n nexrad_env -f environment.yml --prune
