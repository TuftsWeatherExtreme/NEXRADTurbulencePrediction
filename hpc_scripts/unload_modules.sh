#!/bin/bash

# unload_modules.sh
# Team Celestial Blue
# Spring 2025

echo "Deactivating conda environment (if any)"
conda deactivate >& /dev/null

echo "Purging all modules"
module purge

echo "Done! Environment has been torn down"