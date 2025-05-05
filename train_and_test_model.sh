#!/bin/bash -l
#SBATCH -J train_and_test_model         # job name
#SBATCH --time=02-00:00:00           # requested time (dd-hh:mm:ss) - can request up to 7 days
#SBATCH -p preempt                     # running on "gpu" partition/queue (comma will take first one it gets access to)
#SBATCH --gres=gpu:1          #  if after May 1st, we'll add constraint
#SBATCH --constraint="a100-80G|a100-40G|l40|rtx_a6000|rtx_a6000ada"
##SBATCH --constraint="a100-80G|a100-40G|l40s|l40|h100|rtx_5000|rtx_a6000|rtx_6000ada|rtx_6000"
#SBATCH -n 8                         # using 1 cpu core/task - 8 or 16 at least, want enough to feed 
#SBATCH --mem=32g                   # requesting 32GB of RAM total - don't need to be too stingy, each GPU has 80 gb of memory, at least 32 gb of RAM
#SBATCH --output=/cluster/tufts/capstone25celestialb/shared/model_training_output/train_and_test_model.%j.%N.out  # saving standard output to file (%j: jobID, %N: nodename)
#SBATCH --error=/cluster/tufts/capstone25celestialb/shared/model_training_output/train_and_test_model.%j.%N.err   # saving standard error to file (%j: jobID, %N: nodename)
#SBATCH --mail-type=ALL #email options
#SBATCH --mail-user=

cd /cluster/tufts/capstone25celestialb/
source load_modules.sh
nvidia-smi

model_type=$1
loss_function=$2
seed=$3

cd /cluster/tufts/capstone25celestialb/shared/nexrad_playground
echo "About to train the $model_type model with $loss_function loss with seed $seed"
python -u train_and_test_model.py $model_type $loss_function $seed

cd /cluster/tufts/capstone25celestialb/
source unload_modules.sh
echo "All done!"
