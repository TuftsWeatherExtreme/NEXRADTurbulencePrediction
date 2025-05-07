#!/bin/bash -l

# template.sh
# Team Celestial Blue
# Spring 2025

#SBATCH -J job_name                 # Job name
#SBATCH --time=02-00:00:00          # Requested time (dd-hh:mm:ss)
#SBATCH -p batch                    # Which partition (can choose multiple and split with ,)
#SBATCH -n 1                        # Number of nodes requested for this task
#SBATCH --mem=64g                   # Minimum memory requirement 
#SBATCH --output=job_name.%j.%N.out # Change to desired output directory/path for all standard output
#SBATCH --error=job_name.%j.%N.err  # Change to desired output directory/path for all standard error

# Optional emailing feature
#SBATCH --mail-type=ALL             # Will include all email options
#SBATCH --mail-user=                # Replace utln with your utln (ex. jzelev01)

# Optional Job Array:
# #SBATCH --array=1-(NUM_NODES)     # Can specify NUM_NODES to be any value (less than 1000) or specify a range or list of indices to use  

cd $REPO_PATH
source $REPO_PATH/hpc_scripts/load_modules.sh 

echo "Calling python [script name]"
python -u script.py                  # Note that using -u allows for unbuffered output from the script to view live results in stdout.txt

echo "Finished running script.py!"

source $REPO_PATH/hpc_scripts/unload_modules.sh

echo "All done!"