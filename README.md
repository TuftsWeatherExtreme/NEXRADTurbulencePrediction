
# Training and Testing the Model

This directory includes two key files that work together to train and evaluate models on the HPC cluster, which is a High Performance Computing Cluster. In our project, we used the [Tufts HPC cluster](https://it.tufts.edu/high-performance-computing).

- `train_and_test_model.sh` (Bash script): Submits jobs to the cluster and allocates resources.
- `train_and_test_model.py` (Python script): Trains, cross-validates, and tests models on the dataloader.

## Usage: 
This should be run using `train_and_test_model.sh` as:
`python source train_and_test_model.sh [hybrid|linear] [LOSS_FN] [SEED]`
- Example: `python source hpc_scripts/train_and_test_model.sh hybrid mse 42`

## SLURM Job Script: train_and_test_model.sh (Bash)

### Purpose

This script is designed to submit a job to the HPC cluster using the SLURM workload manager.  
It activates the necessary computing modules and environment using `load_modules.sh`, and runs `train_and_test_model.py` with three 
arguments: 
1. model_type: Type of model to train (linear or hybrid).

2. loss_function: Loss function to use (mse, mae, or nll).

3. seed: Seed for dataset split and reproducibility.

At the end, it unloads the previously loaded modules and deactivates the environment to clean up the job environment after execution.

## Python Script: train_and_test_model.py (Python)
This script runs the complete machine learning pipeline, from loading and splitting the dataset to selecting and training either a linear or hybrid model. It performs hyperparameter tuning via cross-validation, supports checkpointing for resuming interrupted jobs, retrains the best model on the full training set, and evaluates performance on a hold-out test set. Finally, it saves the best-performing model for deployment.
####  Dataset Preparation

- Loads pre-saved radar dataset: `dataloader_2008_2025.pth`
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

- Saves final model as `trained_model_outputs/<timestamp>_best_<model_type>_mse_model_w_seed_<SEED>.pth`



