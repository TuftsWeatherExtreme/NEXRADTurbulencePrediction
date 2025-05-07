#!/bin/bash -l

# generate_dataloader.sh
# Team Celestial Blue
# Spring 2025

#SBATCH -J generate_dataloder
#SBATCH --time=02-00:00:00
#SBATCH -p batch  
#SBATCH -n 1  
#SBATCH --mem=64g
#SBATCH --output=generate_dataloder.%j.%N.out
#SBATCH --error=generate_dataloder.%j.%N.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

DATALOADER_NAME=$1

cd $REPO_PATH
source $REPO_PATH/hpc_scripts/load_modules.sh

echo "Calling python create_datasets.py"
python -u $REPO_PATH/model_training/create_datasets.py $REPO_PATH/model_training/$DATALOADER_NAME

echo "Finished running create_datasets.py and created $DATALOADER_NAME! Deleting decompressed dir"
rm -r "decompressed"

source $REPO_PATH/hpc_scripts/unload_modules.sh

echo "All done!"
