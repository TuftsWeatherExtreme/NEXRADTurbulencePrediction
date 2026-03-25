#!/bin/bash -l

# generate_model_inputs.sh
# Authors: Team Celestial Blue
# Spring 2025
# Overview: Generate all compressed model inputs by gridding radar data.
# Each array job processes one month/year of PIREPs with radar data.
# Note: YEARS and MONTHS must match those used in generate_radar_data.sh.
# The job array size should be adjusted accordingly.

#SBATCH -J generate_model_inputs
#SBATCH --time=02-00:00:00
#SBATCH -p batch
#SBATCH -n 1
#SBATCH --mem=32g
#SBATCH --output=generate_model_inputs.%j.%a.%N.out
#SBATCH --error=generate_model_inputs.%j.%a.%N.err
#SBATCH --array=0-215
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

cd $REPO_PATH
source $REPO_PATH/hpc_scripts/load_modules.sh

YEARS=("2008" "2009" "2010" "2011" "2012" "2013" "2014" "2015" "2016" "2017" "2018" "2019" "2020" "2021" "2022" "2023" "2024" "2025")
MONTHS=("january" "february" "march" "april" "may" "june" "july" "august" "september" "october" "november" "december")

num_months=${#MONTHS[@]}
idx=$SLURM_ARRAY_TASK_ID

year_idx=$((idx / num_months))
month_idx=$((idx % num_months))

year=${YEARS[$year_idx]}
month_num=$(printf "%02d" $((month_idx + 1)))

INPUT_CSV=$REPO_PATH/radars/pirep_with_radar_data/$year/$month_num.csv
OUTPUT_DIR=$REPO_PATH/model_inputs/${year}_${month_num}

# Skip if input CSV doesn't exist (e.g., future months in 2025)
if [ ! -f "$INPUT_CSV" ]; then
    echo "No input CSV found at $INPUT_CSV, skipping"
    source $REPO_PATH/hpc_scripts/unload_modules.sh
    exit 0
fi

echo "Processing $year/$month_num"
mkdir -p $OUTPUT_DIR

echo "Running radar_data_to_model_input on $INPUT_CSV"
python3 $REPO_PATH/radars/radar_data_to_model_input.py $INPUT_CSV $OUTPUT_DIR
echo "Python script exit code: $?"

echo "Compressing output directory into $REPO_PATH/model_inputs/compressed/${year}_${month_num}.tar.xz"
mkdir -p $REPO_PATH/model_inputs/compressed
tar -cvJf $REPO_PATH/model_inputs/compressed/${year}_${month_num}.tar.xz -C "$REPO_PATH/model_inputs" "${year}_${month_num}"

echo "Removing uncompressed directory $OUTPUT_DIR"
rm -r $OUTPUT_DIR

source $REPO_PATH/hpc_scripts/unload_modules.sh

echo "All done!"
