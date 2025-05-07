#!/bin/bash -l

# train_and_test_model.sh
# Authors: Team Celestial Blue
# Spring 2025
# Overview: Run train_and_test_model on using GPU resources

#SBATCH -J train_and_test_model
#SBATCH --time=02-00:00:00
#SBATCH -p preempt                 
#SBATCH --gres=gpu:1        
#SBATCH --constraint="a100-80G|a100-40G|l40|rtx_a6000|rtx_a6000ada"
#SBATCH -n 8                         
#SBATCH --mem=32g                  
#SBATCH --output=train_and_test_model.%j.%N.out
#SBATCH --output=train_and_test_model.%j.%N.out
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=

cd $REPO_PATH
source $REPO_PATH/hpc_scripts/load_modules.sh 
nvidia-smi

model_type=$1
loss_function=$2
seed=$3

echo "About to train the $model_type model with $loss_function loss with seed $seed"
python -u $REPO_PATH/model_training/train_and_test_model.py $model_type $loss_function $seed
echo "Finished training and testing the model!"

source $REPO_PATH/hpc_scripts/unload_modules.sh

echo "All done!"