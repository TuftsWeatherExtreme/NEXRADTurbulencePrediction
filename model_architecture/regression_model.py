# regression_model.py
# Team Celestial Blue
# Spring 2025

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

N_ALT = 10
N_LAT = 16
N_LON = 16

NUM_GRID_FEATURES = N_ALT * N_LAT * N_LAT # Number of grid features
NUM_LINEAR_FEATURES = 4 # lat long alt delta_t 
NUM_TOTAL_FEATURES = NUM_GRID_FEATURES + NUM_LINEAR_FEATURES
NUM_FIELDS = 1

class LinearClassifierModel(nn.Module):
    def __init__(self):
        super(LinearClassifierModel, self).__init__()
        # Set a random seed for reproducibility
        torch.manual_seed(42)  # Ensures weight initialization is the same each run
        torch.cuda.manual_seed_all(42)  
        output_len = 10
        self.linear = nn.Linear(NUM_TOTAL_FEATURES, output_len)

    def forward(self, x):
        # print(f"forward called with inputs {x}")
        return self.linear(x)
    




