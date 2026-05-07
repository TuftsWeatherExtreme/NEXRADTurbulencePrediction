#!/bin/bash -l

# train_and_test_model.sh
# Authors: Team Celestial Blue Spring 2025
# Edited by Razzle Dazzle Rose Fall 25/Spring 26
# Overview: Run train_and_test_model on HPC using GPU resources
# Usage: sbatch train_and_test_model.sh <model_type> <seed>
#    Ex: sbatch train_and_test_model.sh hybrid 42

#SBATCH -J train_and_test_model
#SBATCH --time=02-00:00:00
#SBATCH -p preempt                 
#SBATCH --gres=gpu:1        
# THIS LINE IS COMMENTED OUT, BUT CAN UNCOMMENT AND ADD VALID AVAILABLE GPUS: #SBATCH --constraint="a100-80G|a100-40G|l40|rtx_a6000|rtx_a6000ada"
#SBATCH -n 8                         
#SBATCH --mem=32g                  
#SBATCH --output=train_and_test_model.%j.%N.out
#SBATCH --error=train_and_test_model.%j.%N.err
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=

cd $REPO_PATH
source $REPO_PATH/hpc_scripts/load_modules.sh 
nvidia-smi

model_type=$1
seed=$2

# Note: Heatmap model was an attempt to train a model more suited for displaying
# heatmap predictions. It doesn't currently work, but the infrastructure is there :)
if [ "$model_type" == "heatmap" ]; then
    echo "About to train the heatmap model with seed $seed"
    python -u $REPO_PATH/model_training/train_heatmap.py $seed
else
    echo "About to train the $model_type model with seed $seed"
    python -u $REPO_PATH/model_training/train_and_test_model.py $model_type $seed
fi
echo "Finished training and testing the model!"

source $REPO_PATH/hpc_scripts/unload_modules.sh

echo "All done!"
