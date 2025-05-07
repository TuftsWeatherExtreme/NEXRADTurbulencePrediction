# PyTDA - Retrieving Eddy Dissipation Rate
### Files:
- `pytda.py`: This is the modified version of the PyTDA library source code that includes a bug fix for handling incomplete data when calculating EDR.
- `calculate_edr.ipynb`: This file contains code demonstrating how to calculate Eddy Dissipation Rate (EDR) using the PyTDA library.
<br><br>
Note: this bug fix was implemented on 02/13/2025.

## Background
### EDR
Eddy Dissipation Rate (EDR) is a measure of turbulence **intensity** in the atmosphere, that is based on the rate at which energy dissapates in the atmosphere. It has units of m^(2/3)s^(–1), and the standard measure for clear-air turbulence (CAT). Planes record EDR values as they fly, but this data is not always publicly available. 

### PyTDA
The [PyTDA](https://github.com/nasa/PyTDA/) library is a Python package for turbulence detection and analysis. It provides tools for calculating Eddy Dissipation Rate (EDR) using the NTDA (Normalized Turbulence Detection Algorithm) method. However, when processing radar sweeps with incomplete data the NTDA algorithm fails. 
### Bug Fix
We thus edited the library to handle this case gracefully by calculating EDR without the NTDA algorithm when a ValueError exception is raised. This allows us to still obtain EDR values even when the input data is not complete. 

## Setup and Usage
### Setting up the Environment:
1. install pyart: ran `conda install -c conda-forge arm_pyart` in the conda environment
2. download the [PyTDA code](https://github.com/nasa/PyTDA) from github by running `git clone https://github.com/nasa/PyTDA.git`
3. Make bug fixes: replace the file at `PyTDA/pytda/pytda.py` with the code in `pytda.py` (the file in this folder) which has bug fixes (can run `cp pytda.py PyTDA/pytda/pytda.py`)
4. Install the modified PyTDA code: navigate into the _PyTDA/_ folder, and run `python -m pip install .`
    1. This may result in the error error "ModuleNotFoundError: No module named 'Cython'", which is resolved by running `pip install Cython` (installation can be verified via `pip show Cython`)
    2. Then `python -m pip install .` can be run again to install PyTDA

### Example Usage
After setting up the environment, and installing a modified version of the PyTDA installation, you can use the library to calculate EDR in a given radar sweeps.
The example code in `calculate_edr.ipynb` demonstrates how to use the PyTDA library to calculate EDR from radar data.