
# Model Training

This directory includes our sbatch scrupt to run [`train_and_test_model.py`](../../train_and_test_model.py) (Python script), which trains, cross-validates, and tests models on the dataloader.

## train_and_test_model.sh
**Description**: [train_test_pipeline.md](../../train_test_pipeline.md) describes these arguments in detail as well as the [`train_and_test_model.py`](../../train_and_test_model.py) script.

**Usage:** `sbatch train_and_test_model.sh [hybrid|linear] [LOSS_FN] [SEED]`
- Example: `sbatch train_and_test_model.sh hybrid mse 42`
