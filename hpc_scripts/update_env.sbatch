#!/bin/bash
#SBATCH --job-name=nexrad-env-update
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=env_update_%j.out
#SBATCH --error=env_update_%j.err

set -euo pipefail

module load miniforge/24.11.2-py312

# Make sure conda functions are available in non-interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"

# IMPORTANT: don't update while inside the env
conda deactivate || true

# If mamba isn't installed in base yet, uncomment:
# conda activate base
# conda install -y -n base mamba
# conda deactivate

cd "${SLURM_SUBMIT_DIR}"

mamba env update -n nexrad_env -f environment.yml --prune
