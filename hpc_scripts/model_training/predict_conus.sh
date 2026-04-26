#!/bin/bash -l

# predict_conus.sh
# Authors: Team Celestial Blue
# Spring 2025
# Overview: Two-step CONUS radar prediction pipeline.
#   Step 1: Generate a CONUS grid CSV, pipe through get_radars_for_pirep.py
#           to find nearest radar scans (uses proven aiobotocore S3 listing).
#   Step 2: Download radar data, grid, run model, write GeoJSON.
#
# Usage: sbatch predict_conus.sh <model_type> <weights_path> <prediction_time>
#   model_type: hybrid or resnet
#   weights_path: path to trained .pth file
#   prediction_time: ISO format, e.g. "2024-03-01T12:00:00"
#
# Optional env vars:
#   STEP_DEG: grid spacing in degrees (default: 0.5)
#   ALT: single altitude in ft (default: all levels)

#SBATCH -J conus_pred
#SBATCH --time=02-00:00:00
#SBATCH -p preempt
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=32g
#SBATCH --output=conus_pred.%j.%N.out
#SBATCH --error=conus_pred.%j.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

cd $REPO_PATH
source $REPO_PATH/hpc_scripts/load_modules.sh
nvidia-smi

model_type=$1
weights=$2
pred_time=$3
step_deg=${STEP_DEG:-0.5}
alt_flag=""
if [ -n "$ALT" ]; then
    alt_flag="--alt $ALT"
fi

WORK_DIR=$REPO_PATH/model_training/conus_predictions
mkdir -p $WORK_DIR
GRID_CSV=$WORK_DIR/conus_grid.csv
RADAR_CSV=$WORK_DIR/conus_with_radars.csv
OUTPUT=$WORK_DIR/prediction_00.geojson

echo "=============================================="
echo "CONUS Radar Prediction Pipeline"
echo "  Model: $model_type"
echo "  Weights: $weights"
echo "  Time: $pred_time"
echo "  Grid step: ${step_deg} deg"
echo "=============================================="

# --- Step 1: Generate grid and find radar files ---
echo ""
echo "STEP 1: Generating CONUS grid and finding radar files..."
echo "  Generating grid CSV..."
python -u $REPO_PATH/model_training/generate_conus_grid.py \
    --time "$pred_time" --step-deg $step_deg $alt_flag --output $GRID_CSV

echo "  Grid CSV has $(wc -l < $GRID_CSV) rows"
echo "  Piping through get_radars_for_pirep.py to find radar scans..."
python -u $REPO_PATH/radars/get_radars_for_pirep.py -o FILE < $GRID_CSV

# get_radars_for_pirep.py outputs to pirep_with_radar_data/{year}/{month}.csv
# Move it to our work directory
YEAR=$(echo $pred_time | cut -c1-4)
MONTH=$(echo $pred_time | cut -c6-7)
RADAR_OUTPUT=$REPO_PATH/radars/pirep_with_radar_data/$YEAR/$MONTH.csv

if [ -f "$RADAR_OUTPUT" ]; then
    cp $RADAR_OUTPUT $RADAR_CSV
    echo "  Radar CSV has $(wc -l < $RADAR_CSV) rows"
else
    echo "  ERROR: Radar output not found at $RADAR_OUTPUT"
    echo "  Checking for stdin-based output..."
    # get_radars_for_pirep.py with stdin might output differently
    ls -la $REPO_PATH/radars/pirep_with_radar_data/
fi

# --- Step 2: Download radar data, grid, predict ---
echo ""
echo "STEP 2: Downloading radar data and running predictions..."
python -u $REPO_PATH/model_training/predict_from_csv.py \
    --model-type $model_type \
    --weights $weights \
    --input-csv $RADAR_CSV \
    --output $OUTPUT

echo ""
echo "Python exit code: $?"
echo "Output files:"
ls -la $WORK_DIR/

source $REPO_PATH/hpc_scripts/unload_modules.sh
echo "All done!"
