#!/bin/bash -l

# generate_csv_data.sh
# Team Celestial Blue
# Spring 2025

#SBATCH -J csv_gen
#SBATCH --time=03-00:00:00
#SBATCH -p batch,preempt
#SBATCH -n 1
#SBATCH --mem=32g
#SBATCH --output=csv_gen.%j.%a.%N.out
#SBATCH --error=csv_gen.%j.%a.%N.err
#SBATCH --array=0-191
#SBATCH --mail-type=ALL
#SBATCH --mail-user= 

cd $REPO_PATH
source $REPO_PATH/hpc_scripts/load_modules.sh 

idx=$SLURM_ARRAY_TASK_ID

YEARS=("2008" "2009" "2010" "2011" "2012" "2013" "2014" "2015" "2016" "2017" "2018" "2019" "2020" "2021" "2022" "2023" "2024")
MONTHS=("january" "february" "march" "april" "may" "june" "july" "august" "september" "october" "november" "december")

num_months=${#MONTHS[@]}

year_idx=$((idx / num_months))
month_idx=$((idx % num_months))

year=${YEARS[$year_idx]}
month=${MONTHS[$month_idx]}

echo "Processing $month $year"
python $REPO_PATH/pireps/clean_pireps.py -month $month -year $year -o STDOUT | python $REPO_PATH/radars/get_radars_for_pirep.py -o FILE

source $REPO_PATH/hpc_scripts/unload_modules.sh

echo "All done!"
