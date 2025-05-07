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

To create an environment that matches ours, you can setup the conda environment listed in [env_req](#env_req). 

The following environment variables need to be set up for all scripts to work: 

- **ENV_PATH**
  -This is the path to the conda environment used for package management. For example, ours was `/cluster/tufts/capstone25celestialb/cbenv`. 
  
  This can be added by running: `export ENV_PATH=" environment path here "` or appended to your bashrc to automatically be loaded in on login: `echo 'export ENV_PATH=" environment path here "' >> ~/.bashrc`

- **REPO_PATH**
  - This is the remote path to this repository on the HPC. For example, ours was `/cluster/tufts/capstone25celestialb/shared/NEXRADTurbulencePrediction`. 
  
  This can be added by running: `export REPO_PATH=" repository path here "` or appended to automatically be loaded in with your bash: `echo 'export REPO_PATH=" repository path here "' >> ~/.bashrc`

Note: we were advised to load `miniforge` instead of `anaconda`. This worked very well. All this requires is `module load miniforge/24.11.2-py312`. 


### [load_modules.sh](load_modules.sh)

Usage: `source load_modules.sh`

Resets environment, loads miniforge, cuda, and activates the environment (`ENV_PATH`). Ideal as the first step of running a script as seen in [template.sh](./template.sh).

*Note: will exit failure and report error if not being run with `source`.*

### [unload_modules.sh](unload_modules.sh)

Usage: `source unload_modules.sh`

Deactivates the conda environment and purges all loaded modules. Ideal as the final step of running a script as seen in [template.sh](./template.sh).

### [env_req.txt](env_req.txt)

This is the resulting file from running `module load miniforge/24.11.2-py312`, activating our conda environment (`source activate [env_path]`), and running  `conda list -e > env_req.txt`. This provides a list of the environment requirements.

This could potentially be recreated using `conda create -n <environment-name> --file env_req.txt`

If creating a new environment, a couple notable module installation notes:
- PyTorch: Since PyTorch no longer supports conda installation, we used `pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121` to install PyTorch into our environment.
- PyArt: PyArt can be installed with `conda install -c conda-forge arm_pyart`
- Cuda already exists as a module and should be loaded prior to environment activation with `module load cuda/12.2`

## Interfacing with the HPC
- To most effectively edit files directly on the HPC, we utilized the Remote-SSH extension in Visual Studio Code.
- We utilized the following workflow:
  - Connect to login node of HPC (`ssh jzelev01@login.pax.tufts.edu`)
  - Request a compute node (`srun -p preempt -n 1 --mem=32g -t 1-0 --pty bash`)
    - Upon success and resource allocation, you will have the hostname of the compute node that you've been allocated. An example of this would be `d1cmp002` - it will be visible as `jzelev01@d1cmp002` in the terminal dialogue
  - Connect to that node with Remote SSH in VSCode 
    - Press `F1` to bring up the Command Pallete, and start typing Remote-SSH. Select `Remote-SSH: Add New SSH Host...` and type: `ssh jzelev01@d1cmp002.pax.tufts.edu`, replacing the UTLN and node name as appropriate. 
    - Note that it will then ask for your password, after which, a Duo Push notification will need be accepted if not on the Tufts VPN. 
  - Open the appopriate folder for your cluster path (ex. `/cluster/tufts/capstone25celestialb/`) and then you can interact with any files and edit directly in VSCode. To run interactively, you can open a terminal (Terminal --> New Terminal) for command line access to the compute node.
-Additional resources for interfacing with the HPC, such as using a tunnel, can be found [here](https://rtguides.it.tufts.edu/hpc/application/40-vscode.html)


## Additional Resources 
We accessed the following resources in order to guide our use of the HPC:
- **Checkpointing**: We were provided [this guide](https://tufts.app.box.com/s/jav14xvd0m25hp7kij1yr908xt2byn9f) from Ryan Veiga, PhD, Data Science Specialist, from Tufts Technology Services. This applies specifically to using the preempt parition and provides insight into using checkpoints to restore work. For an example of checkpointing, see [train_and_test_model.py](/model_training/train_and_test_model.py).

- **General HPC Resources** In March 2025, Tufts Technology Services redid their resources. We previously were using provided PDF guides (such as [this](https://tufts.app.box.com/v/Pax-User-Guide)) but now they have switched to have a more robust resource guide that can be found [here](https://rtguides.it.tufts.edu/hpc/index.html). Most importantly, 

- **Additional Slurm Information** For more information on different possible sbatch options, we referenced the official SLURM documentation [here](https://slurm.schedmd.com/sbatch.html). Additionally, the HPC guides include [this](https://rtguides.it.tufts.edu/hpc/slurm/batchjob.html) information about batch scripts and an example one. We also created our own documented [template](./template.sh) to reference.

- **Technology Support** With questions, we recommend reaching out to [tts-research@tufts.edu](mailto:tts-research@tufts.edu) for all Tufts HPC cluster questions and requests. We specifically received very insightful support from [Delilah Maloney](mailto:Delilah.Maloney@tufts.edu), who is the current Sr. High Performance Computing (HPC) Specialist as of May 2025.

