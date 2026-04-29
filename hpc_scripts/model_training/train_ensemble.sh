#!/bin/bash -l

# train_ensemble.sh
# Authors: Team Celestial Blue
# Spring 2025
# Overview: Train a small ensemble model that combines radar + satellite predictions.
# Usage: sbatch train_ensemble.sh <seed> <radar_model_type> <radar_weights> [sat_weights] [sat_data_dir]

#SBATCH -J ensemble
#SBATCH --time=01:00:00
#SBATCH -p preempt
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=32g
#SBATCH --output=ensemble.%j.%N.out
#SBATCH --error=ensemble.%j.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

cd $REPO_PATH
source $REPO_PATH/hpc_scripts/load_modules.sh

seed=$1
radar_model_type=$2
radar_weights=$3
sat_weights=$4
sat_data_dir=$5

RADAR_DATALOADER=$REPO_PATH/model_training/eleanor_dataloader.pth

echo "Training ensemble model"
echo "  Seed: $seed"
echo "  Radar: $radar_model_type ($radar_weights)"

CMD="python -u $REPO_PATH/model_training/train_ensemble.py $seed \
    --radar-model-type $radar_model_type \
    --radar-weights $radar_weights \
    --radar-dataloader $RADAR_DATALOADER"

if [ -n "$sat_weights" ] && [ -n "$sat_data_dir" ]; then
    echo "  Satellite: $sat_weights (data: $sat_data_dir)"
    CMD="$CMD --sat-weights $sat_weights --sat-data-dir $sat_data_dir --sat-repo $SAT_REPO_PATH/src"
else
    echo "  Satellite: not included (radar-only ensemble)"
fi

eval $CMD
echo "Python exit code: $?"

source $REPO_PATH/hpc_scripts/unload_modules.sh
echo "All done!"
