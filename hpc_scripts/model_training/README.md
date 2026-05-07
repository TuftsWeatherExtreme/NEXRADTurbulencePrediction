
# Model Training

This directory includes our sbatch script to run [`train_and_test_model.py`](/NEXRADTurbulencePrediction/model_training/train_and_test_model.py) (Python script), which trains, cross-validates, and tests models on the dataloader. It also includes attempts to predict and deploy the models.

## train_and_test_model.sh
**Description**: The [README](/model_training/README.md) in the [Model Training](/model_training/) directory describes the arguments to this script in detail as well as the [`train_and_test_model.py`](/model_training/train_and_test_model.py) script. Trains, cross-validates, and tests turbulence prediction models using GPU resources. Supports multiple model architectures including hybrid CNN-based models and linear baselines.

**Usage:** `sbatch train_and_test_model.sh <model_type> <seed>`
- model_type: Model architecture to train
    - hybrid  - CNN-based hybrid model (recommended)
    - resnet  - ResNet-based architecture
    - linear  - Linear baseline model
    - heatmap - Experimental heatmap model (infrastructure in place, not fully working)
- seed: Random seed for reproducibility (e.g., 42)
- Example: `sbatch train_and_test_model.sh hybrid 42`

# Prediction and Deployment
## predict_conus.sh
**Description**: Two-step pipeline for generating CONUS-wide (Continental United States) turbulence predictions at a specified time using a trained model.

**Step 1**: Grid Generation & Radar Matching
Generates a regular lat/lon grid covering CONUS at specified altitude(s)
For each grid point, finds the nearest NEXRAD radar scans at the prediction time
Uses proven aiobotocore S3 listing to efficiently locate radar files
Outputs CSV with grid points and associated radar file paths

**Step 2**: Radar Processing & Prediction
Downloads radar data for each grid point
Creates 3D reflectivity grids (same format as training data)
Runs trained model to predict turbulence at each location
Writes predictions to GeoJSON format for visualization

**Usage**: `sbatch predict_conus.sh <model_type> <path_to_model> <prediction_time>`
- model_type: Model architecture (hybrid or resnet)
- path_to_model: Path to trained .pth model weights file
- prediction_time: ISO format timestamp, e.g., "2024-03-01T12:00:00"

**Optional Arguments**:
- STEP_DEG: Grid spacing in degrees (default: 0.5°)
  Smaller values = finer resolution, longer runtime
- ALT: Single altitude in feet (default: all standard flight levels)
  Use to generate predictions at one altitude only

**Output**:
GeoJSON file: [prediction_00.geojson](/NEXRADTurbulencePrediction/model_training/conus_predictions)
Contains turbulence predictions at each grid point with lat/lon coordinates
Can be visualized in mapping applications
