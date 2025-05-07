#!/bin/bash

# load_modules.sh
# Authors: Team Celestial Blue
# Spring 2025
# Overview: Load all modules and active specified environment.
# Important: Must be run as `source load_modules.sh`

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Error: Please run this script as: source ${0}"
    exit 1
fi

echo "Resetting environment"
source "$(dirname ${BASH_SOURCE[0]})/unload_modules.sh"

echo "Loading miniforge/24.11.2-py312..."
module load miniforge/24.11.2-py312

echo "Loading cuda/12.2..."
module load cuda/12.2

echo "Activating cbenv conda environment..." 
source activate $ENV_PATH

echo "Done! Environment has now been set up"