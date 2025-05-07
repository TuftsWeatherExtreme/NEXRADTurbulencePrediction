# Model Training

This directory contains the python scripts for generating a dataloader for model inputs as well as for
training and testing the model on that dataloader. Sample model results, sample saved models, and 

## Creating the dataloader - [create_datasets.py](create_datasets.py) using [dataloader_class](dataloader_class.py)

create_datasets creates a dataloder.pth to be used for training and testing of any generated model. 

### [create_datasets.sh](generate_model_inputs.sh)
**Description**: 

**Usage**: `python create_datasets.py <dataloder_name> [existing_dataloader]`
- dataloader_name: Desired name for dataloader to generate
- existing_dataloader: Can optionally include an existing dataloader that it will add to

**Dependencies**:
- [Dataloader Class](dataloader_class.py)
- [/model_inputs/compressed](../model_inputs/) contains tar xz compressed model input files
**Notes**:
- Will add all compressed model inputs located in [`/model_inputs/compressed`](../model_inputs/) into a dataloader.
-

### [dataloader_class.py](dataloader.py)
**Description**: Custom 

**Usage**: `from dataloader_class import RadarDataLoader`
- Provides access to the RadarDataLoader class

**Dependencies**:
- [Dataloader Class](dataloader_class.py)

**Notes**:
- We designed this class to optimize for a slower initialization but faster 'get_item` time. This meant
we did the proecssing into features and label during initialization to be able to get fast indexing.
- Note that in order to maintain the benefits of compression, we ensure that 
- 


## Training and Testing the Model - [train_and_test_model.py](train_and_test_model.py)

[train_and_test_model.py](train_and_test_model.py) is the overarching python script for training and evaluating models on the HPC cluster, which is a High Performance Computing Cluster. In our project, we used the [Tufts HPC cluster](https://it.tufts.edu/high-performance-computing). `train_and_test_model.sh` can be found [here](hpc_scripts/model_training/train_and_test_model.sh).

- `train_and_test_model.sh` (Bash script): Submits jobs to the cluster and allocates resources.
- `train_and_test_model.py` (Python script): Trains, cross-validates, and tests models on the dataloader.

### Usage: 
This should be run using `train_and_test_model.sh` as:
`source train_and_test_model.sh [hybrid|linear] [LOSS_FN] [SEED]`
- Example: `sbatch hpc_scripts/model_training/train_and_test_model.sh hybrid mse 42`

## SLURM Job Script: train_and_test_model.sh (Bash)

### Purpose

This script is designed to submit a job to the HPC cluster using the SLURM workload manager.  
It activates the necessary computing modules and environment using [`load_modules.sh`](./hpc_scripts/load_modules.sh), and runs [`train_and_test_model.py`](train_and_test_model.py) with three 
arguments: 
1. model_type: Type of model to train (linear or hybrid).

2. loss_function: Loss function to use (mse, mae, or nll).

3. seed: Seed for dataset split and reproducibility.

At the end, it unloads the previously loaded modules and deactivates the environment to clean up the job environment after execution.

The script can be found in the [hpc_scripts directory](../hpc_scripts).

## Python Script: train_and_test_model.py (Python)
This script runs the complete machine learning pipeline, from loading and splitting the dataset to selecting and training either a linear or hybrid model. It performs hyperparameter tuning via cross-validation, supports checkpointing for resuming interrupted jobs, retrains the best model on the full training set, and evaluates performance on a hold-out test set. Finally, it saves the best-performing model for deployment.
####  Dataset Preparation

- Loads pre-saved radar dataset using specified dataloader.pth. This can be specified by changing the
`DATALOADER_PATH` variavle in the script. 

- Splits into:
  - **90% training + validation**
  - **10% test set**
- Uses a seed (3rd command-line argument) for reproducibility

#### Model Choices

- `LinearClassifierModel`
- `HybridModel1Out`

Chosen dynamically via **1st** command-line arg (`linear` or `hybrid`).

#### Loss Function Choices

- **MSELoss** (regression)
- **L1Loss/MAE** (regression)
- **NLLLoss** (classification)

Chosen dynamically via command-line argument (`mse`, `mae`, `nll`).

#### Cross-Validation and Hyperparameter Search

- Uses `KFold` with **6 folds**, so eacb fold has 15% of the total dataset
- Searches over `L2` regularization values `[0.10, 0.01, 0.001, 0]`
- For each `L2` value:
  - Trains on 5 folds (for `NUM_EPOCHS` times)
  - Validates on 1 fold
  - Stores average loss
- After CV completes, selects L2 with lowest validation loss as the "best model"

#### Checkpointing

- Saves checkpoints during training to: `<output_dir>/<model>_<loss>_<seed>_model_checkpoint.pth`
- Stores:
  - Model and Optimizer state
  - Epoch, fold, L2 index (in `[0.10, 0.01, 0.001, 0]`)
  - Loss history prior to interruption
- Allows seamless resuming if job is interrupted

#### Retraining on Full Training Data
- Once best L2 is chosen, re-initializes and trains model on full 90% training and validations dataset.
- Trains for 5 epochs to maximize performance using all available data.

#### Final Testing and Evaluation
- Tests retrained model on hold-out test set (10%).

- Collects and prints:

  - Accuracy

  - False Positive Rate

  - False Negative Rate

  - Actual vs Predicted Class Distributions

- Saves final model as `trained_model_outputs/<timestamp>_best_<model_type>_<loss>_model_w_seed_<SEED>.pth`
- Saves final results as `trained_model_outputs/<timestamp>_best_<model_type>_<loss>_model_w_seed_<SEED>_results.txt`

- TODO:
- add that there will be 2  examples in trained_model_outputs that include their best mdoel and reuslts files



