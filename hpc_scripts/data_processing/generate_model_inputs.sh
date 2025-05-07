#!/bin/bash -l

# generate_model_inputs.sh
# Authors: Team Celestial Blue
# Spring 2025
# Overview: Generate all compressed model inputs by gridding radar data from split csv data sets. 

#SBATCH -J generate_model_inputs       
#SBATCH --time=02-00:00:00   
#SBATCH -p batch   
#SBATCH -n 1   
#SBATCH --mem=32g    
#SBATCH --output=generate_model_inputs.%j.%a.%N.out
#SBATCH --error=generate_model_inputs.%j.%a.%N.err
#SBATCH --array=1-250                
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

cd $REPO_PATH
source $REPO_PATH/hpc_scripts/load_modules.sh

OUTPUT_DIR=$REPO_PATH/model_inputs"

idx=$(printf "%03d" ${SLURM_ARRAY_TASK_ID})
echo "Operating on $idx"

echo "Making directory $OUTPUT_DIR/$idx" 
mkdir -p $OUTPUT_DIR/$idx

echo "Running radar_data_to_model_input on $REPO_PATH/radars/split_radar_data/part_"$idx".csv"
python3 $REPO_PATH/radars/radar_data_to_model_input.py $REPO_PATH/radars/split_radar_data/part_"$idx".csv $OUTPUT_DIR/$idx

echo "Finished running radar_data_to_model_input.py on part! Compressing output directory into $OUTPUT_DIR/compressed/$idx.tar.xz"
tar -cvJf $OUTPUT_DIR/compressed/$idx.tar.xz -C "$OUTPUT_DIR" "$idx"

echo "All done with compression. Removing $REPO_PATH/model_inputs/$idxdirectory"
rm -r $OUTPUT_DIR/$idx 

source $REPO_PATH/hpc_scripts/unload_modules.sh

echo "All done!"
