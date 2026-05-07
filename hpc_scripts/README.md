# HPC Scripts

## Overview
To conduct data processing and model training, we utilized the Tufts High Performance Computing Cluster for parallel computing and access to powerful computing resources. This was essential for saving time while handling the large amounts of data processing needed. This also allowed us to utilize GPUs when training/testing our model. We created SLURM sbatch scripts provided in this directory to achieve these goals.

Dave Lillethun, advisor of the Tufts Senior Capstone for 2024-2025, requested access for our team, and we ultimately had a quota of 150 GB allocated for our project. 

Overarchingly, any sbatch script can be generally run as:
**Usage:** `sbatch [script].sh [any command line args]`
This provides consistent specification on what nodes to request, time and memory requirements, where to save output, a script to run once the node has been allocated, and more. 

While our ultimate scripts were formatted as batch scripts, to request a node individually and get command line access:
- Usage (non-GPU): `srun -p batch -n 1 --mem=32g -t 1-0 --pty bash`
- Usage (with GPU): `srun -p preempt -n 8 --mem=32g -t 1-0 --gres=gpu:l40:1 --pty bash`
_Any of those parameters can be changed depending on what is desired. Those options are documented [here](https://rtguides.it.tufts.edu/hpc/slurm/interactive.html)._ 

## Contents

The following scripts are provided in each directory:
- [Data Processing](./data_processing/)
  - [generate_csv_data.sh](./data_processing/generate_csv_data.sh)
  - [generate_radar_data.sh](./data_processing/generate_radar_data.sh)
  - [generate_model_inputs.sh](./data_processing/generate_model_inputs.sh)
  - [generate_dataloader.sh](./data_processing/generate_dataloader.sh)
- [Model Training](./model_training/)
  - [train_and_test_model.sh](./model_training/train_and_test_model.sh)

Notes:
- These scripts assume that the conda environment and environment variables have already been [set up](#setup).
- [Data Processing](./data_processing/README.md) and [Model Training](./model_training/README.md) each contain descriptions for each script and their dependencies. 

### [template.sh](template.sh)
Provides a template with annotated notes about the sbatch options that we have set up. This script shows the overall structure for submitting a script to the HPC. This template includes options for running with a job array if desired.

Additional useful options:
- If you would like to instantiate an array of jobs, the `--array` option can be used.
  - This would be written as: `#SBATCH --array=INDEX_VALUES`, where `INDEX_VALUES` can be any range of indices to be used as an array index (ex. `#SBATCH --array=0-155`) or list of indices to use (ex. `#SBATCH --array=0, 2-6`)
- If using GPUs, the parition used (from #SBATCH -p) should be changed to preempt and/or GPUs. The `-n` option can then be changed to greater than 1 (we tended to use 8). Additionally, TTS recommended that we include a constraint on GPUs that it will utilize by adding the line:
`#SBATCH --constraint="a100-80G|a100-40G|l40|rtx_a6000|rtx_a6000ada"`. An example of a script utilizing GPUs is [train_and_test_model.sh](./model_training/train_and_test_model.sh). In the script itself, it was then recommended to use `nvidia-smi` to output information about the specific GPU allocated in the case of failure.

Additionally, please note:
- The following job reference information can be:
  - %j - job_id (ex. job 13573217), can be seen with `squeue -u [YOUR_UTLN]`
  - %N - which node the job was completed on (ex. d1cmp002)
  - %a - the array index if using a job array script

- This can be included in the outut names for stderr/stdout files. For example, an example output file for us when using [generate_csv_data.sh](./data_processing/generate_csv_data.sh) was:
  - `#SBATCH --output=csv_gen.%j.%a.%N.out`
- We often included full filepaths for output/error to have them redirected to specific folders

Usage: `sbatch template.sh`, or more generally: `sbatch [script].sh`

## Setup

For the full Tech Setup Guide, please see the Google Doc https://docs.google.com/document/d/1TpXVFvKOJJwh6hDFK9GU0aOpVQGVvaJRkTvm8C1XjRU/edit?tab=t.0 

The following environment variables need to be set up for all scripts to work: 

- **ENV_PATH**
  -This is the path to the conda environment used for package management. For example, ours was `/cluster/tufts/capstone25skyblue/condaenv26/nexrad_env`. 
  
  This can be added by running: `export ENV_PATH=" environment path here "` or appended to your bashrc to automatically be loaded in on login: `echo 'export ENV_PATH=" environment path here "' >> ~/.bashrc`

- **REPO_PATH**
  - This is the remote path to this repository on the HPC. For example, ours was `/cluster/tufts/capstone25skyblue/<YOUR_UTLN>/NEXRADTurbulencePrediction`. 
  
  This can be added by running: `export REPO_PATH=" repository path here "` or appended to automatically be loaded in with your bash: `echo 'export REPO_PATH=" repository path here "' >> ~/.bashrc`

Note: we were advised to load `miniforge` instead of `anaconda`. This worked very well. All this requires is `module load miniforge/24.11.2-py312`. 


### [load_modules.sh](load_modules.sh)

Usage: `source load_modules.sh`

Resets environment, loads miniforge, cuda, and activates the environment (`ENV_PATH`). Ideal as the first step of running a script as seen in [template.sh](./template.sh).

*Note: will exit failure and report error if not being run with `source`.*

### [unload_modules.sh](unload_modules.sh)

Usage: `source unload_modules.sh`

Deactivates the conda environment and purges all loaded modules. Ideal as the final step of running a script as seen in [template.sh](./template.sh).


## Additional Resources
We accessed the following resources in order to guide our use of the HPC:
- **Checkpointing**: We were provided [this guide](https://tufts.app.box.com/s/jav14xvd0m25hp7kij1yr908xt2byn9f) from Ryan Veiga, PhD, Data Science Specialist, from Tufts Technology Services. This applies specifically to using the preempt parition and provides insight into using checkpoints to restore work. For an example of checkpointing, see [train_and_test_model.py](/model_training/train_and_test_model.py).

- **General HPC Resources** In March 2025, Tufts Technology Services redid their resources. We previously were using provided PDF guides (such as [this](https://tufts.app.box.com/v/Pax-User-Guide)) but now they have switched to have a more robust resource guide that can be found [here](https://rtguides.it.tufts.edu/hpc/index.html). Most importantly, 

- **Additional Slurm Information** For more information on different possible sbatch options, we referenced the official SLURM documentation [here](https://slurm.schedmd.com/sbatch.html). Additionally, the HPC guides include [this](https://rtguides.it.tufts.edu/hpc/slurm/batchjob.html) information about batch scripts and an example one. We also created our own documented [template](./template.sh) to reference.

- **Technology Support** With questions, we recommend reaching out to [tts-research@tufts.edu](mailto:tts-research@tufts.edu) for all Tufts HPC cluster questions and requests. We specifically received very insightful support from [Delilah Maloney](mailto:Delilah.Maloney@tufts.edu), who is the current Sr. High Performance Computing (HPC) Specialist as of May 2025.

