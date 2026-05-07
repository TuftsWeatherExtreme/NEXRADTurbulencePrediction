#!/bin/bash -l

# generate_radar_data.sh
# Authors: Razzle Dazzle Rose Fall 25/Spring 26
# Overview: For each month/year of cleaned PIREP CSVs, find the best candidate
# NEXRAD radar sites (using beam geometry scoring) and the closest radar scan
# times from S3. Outputs CSVs with nexrad_sites and aws_files columns.
# Note: YEARS and MONTHS must match those used in generate_csv_data.sh.
# The job array size should be adjusted accordingly.


#SBATCH -J radar_gen
#SBATCH --time=24:00:00
#SBATCH -p batch,preempt
#SBATCH -n 1
#SBATCH --mem=32g
#SBATCH --output=radar_gen.%j.%a.%N.out
#SBATCH --error=radar_gen.%j.%a.%N.err
#SBATCH --array=0-215
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

cd $REPO_PATH
source $REPO_PATH/hpc_scripts/load_modules.sh

idx=$SLURM_ARRAY_TASK_ID

YEARS=("2008" "2009" "2010" "2011" "2012" "2013" "2014" "2015" "2016" "2017" "2018" "2019" "2020" "2021" "2022" "2023" "2024" "2025")
MONTHS=("january" "february" "march" "april" "may" "june" "july" "august" "september" "october" "november" "december")

num_months=${#MONTHS[@]}

year_idx=$((idx / num_months))
month_idx=$((idx % num_months))

year=${YEARS[$year_idx]}
month=${MONTHS[$month_idx]}

echo "Processing $month $year"
python $REPO_PATH/radars/get_radars_for_pirep.py -month $month -year $year -o FILE

source $REPO_PATH/hpc_scripts/unload_modules.sh

echo "All done!"
