#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --output=setup_env.log
#SBATCH --time=02:00:00          # Adjust if needed
#SBATCH --partition=batch        # Use a valid partition on your cluster
#SBATCH --mem=32G                 # Adjust memory if needed
#SBATCH --cpus-per-task=4        # Adjust CPUs if needed

# ===============================
# Environment setup
# ===============================
echo "Loading modules..."
module load miniforge/24.11.2-py312  # Make sure this matches the module on your cluster
module load cuda/12.2                 # Optional: only needed if using GPU/PyTorch CUDA

echo "Activating conda environment..."
conda activate nexrad_env

# ===============================
# Install mamba if missing
# ===============================
if ! command -v mamba &> /dev/null
then
    echo "Installing mamba..."
    conda install -c conda-forge mamba -y
else
    echo "Mamba already installed."
fi

# ===============================
# Install packages from conda_req.txt
# ===============================
if [ -f ./conda_req.txt ]; then
    echo "Installing packages from conda_req.txt..."
    mamba install --file ./conda_req.txt -y
else
    echo "conda_req.txt not found!"
fi

# ===============================
# Install packages from pip_req.txt
# ===============================
if [ -f ./pip_req.txt ]; then
    echo "Installing packages from pip_req.txt..."
    pip install -r ./pip_req.txt
else
    echo "pip_req.txt not found!"
fi

# ===============================
# Install PyTorch (CUDA version if desired)
# ===============================
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122

echo "Environment setup complete!"

