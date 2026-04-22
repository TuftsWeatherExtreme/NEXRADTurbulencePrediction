#!/bin/bash -l

# predict_radar_demo.sh
# Authors: Team Celestial Blue
# Spring 2025
# Overview: Generate 16 GeoJSON files for 8 hours of radar predictions (demo).
# Usage: sbatch predict_radar_demo.sh <model_type> <weights_path> [start_time]

#SBATCH -J radar_demo
#SBATCH --time=01:00:00
#SBATCH -p preempt
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=16g
#SBATCH --output=radar_demo.%j.%N.out
#SBATCH --error=radar_demo.%j.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

cd $REPO_PATH
source $REPO_PATH/hpc_scripts/load_modules.sh
nvidia-smi

model_type=$1
weights=$2
start_time=$3

OUTPUT_DIR=$REPO_PATH/model_training/demo_radar

if [ -n "$start_time" ]; then
    echo "Generating radar demo GeoJSONs: model=$model_type, start=$start_time"
    python -u $REPO_PATH/model_training/predict_radar_demo.py \
        --model-type $model_type --weights $weights \
        --output-dir $OUTPUT_DIR --start-time "$start_time"
else
    echo "Generating radar demo GeoJSONs: model=$model_type, start=8hrs ago"
    python -u $REPO_PATH/model_training/predict_radar_demo.py \
        --model-type $model_type --weights $weights \
        --output-dir $OUTPUT_DIR
fi

echo "Python exit code: $?"
echo "Output files:"
ls -la $OUTPUT_DIR/

source $REPO_PATH/hpc_scripts/unload_modules.sh
echo "All done!"
