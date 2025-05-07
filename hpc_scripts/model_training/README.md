
# Model Training

This directory includes our sbatch scrupt to run [`train_and_test_model.py`](/model_training/train_and_test_model.py) (Python script), which trains, cross-validates, and tests models on the dataloader.

## train_and_test_model.sh
**Description**: The [README](/model_training/README.md) in the [Model Training](/model_training/) directory describes the arguments to this script in detail as well as the [`train_and_test_model.py`](/model_training/train_and_test_model.py) script.

**Usage:** `sbatch train_and_test_model.sh [hybrid|linear] [LOSS_FN] [SEED]`
- Example: `sbatch train_and_test_model.sh hybrid mse 42`
