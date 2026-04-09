# hybrid_model_1_out.py
# Team Celestial Blue
# Spring 2025
# Hybrid model: 3D CNN branch for radar grid + FC branch for metadata
# Binary classification output (2 classes)

import torch
from torch import nn

N_ALT = 10
N_LAT = 16
N_LON = 16

NUM_GRID_FEATURES = N_ALT * N_LAT * N_LON
NUM_LINEAR_FEATURES = 5  # lat, lon, alt, delta_t, in_sigmet
NUM_FIELDS = 1
NUM_CLASSES = 2


class HybridModel1Out(nn.Module):
    def __init__(self):
        super().__init__()

        # FC branch for metadata (lat, lon, alt, delta_t)
        self.fc_branch = nn.Sequential(
            nn.Linear(NUM_LINEAR_FEATURES, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        # 3D CNN branch for radar reflectivity grid
        self.conv_branch = nn.Sequential(
            nn.Conv3d(NUM_FIELDS, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),  # -> (16, 5, 8, 8)

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),  # -> (32, 2, 4, 4)

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # -> (64, 1, 1, 1)

            nn.Flatten()
        )

        # Classifier combining both branches
        self.classifier = nn.Sequential(
            nn.Linear(64 + 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, x):
        x_cnn = x[:, NUM_LINEAR_FEATURES:].reshape(-1, NUM_FIELDS, N_ALT, N_LAT, N_LON)
        x_fc = x[:, :NUM_LINEAR_FEATURES]

        out_fc = self.fc_branch(x_fc)
        out_cnn = self.conv_branch(x_cnn)

        out = torch.cat((out_fc, out_cnn), dim=1)
        out = self.classifier(out)
        return out


if __name__ == "__main__":
    model = HybridModel1Out()
    print(model)
    # Test with dummy input
    x = torch.randn(2, NUM_LINEAR_FEATURES + NUM_GRID_FEATURES)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {model(x).shape}")
