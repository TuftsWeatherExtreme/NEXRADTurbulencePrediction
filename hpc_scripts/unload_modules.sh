#!/bin/bash

# unload_modules.sh
# Authors: Team Celestial Blue
# Spring 2025
# Overview: Purge all modules and deactive specified environment.
# Run as `source unload_modules.sh`


echo "Deactivating conda environment (if any)"
conda deactivate >& /dev/null

echo "Purging all modules"
module purge

echo "Done! Environment has been torn down"