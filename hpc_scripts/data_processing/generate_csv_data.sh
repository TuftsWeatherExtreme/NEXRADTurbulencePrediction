#!/bin/bash -l

# generate_csv_data.sh
########
# ONCE THIS IS RUN ONCE, THE OUTPUT WILL BE IN pireps/clean_pirep_data
# AND WILL NOT NEED TO BE RUN AGAIN
#######
# Authors: Team Celestial Blue Spring 2025
# Edited by Razzle Dazzle Rose Fall 25/Spring 26
# Overview: Download and clean PIREP data, filtering to SEV+ turbulence only.
# Outputs one CSV per month/year to pireps/clean_pirep_data/{year}/{month}_turb_pireps.csv
# Note: YEARS and MONTHS can be changed to generate for all combinations specified.
# The job array size should be adjusted accordingly.


#SBATCH -J csv_gen
#SBATCH --time=01:00:00
#SBATCH -p batch,preempt
#SBATCH -n 1
#SBATCH --mem=8g
#SBATCH --output=csv_gen.%j.%a.%N.out
#SBATCH --error=csv_gen.%j.%a.%N.err
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
python $REPO_PATH/pireps/clean_pireps.py -month $month -year $year -o FILE

source $REPO_PATH/hpc_scripts/unload_modules.sh

echo "All done!"
