# resnet3d_model.py
# Team Celestial Blue
# Spring 2025
# ResNet-style 3D CNN for radar reflectivity grids
# Uses residual connections to enable deeper networks without vanishing gradients

import torch
from torch import nn

N_ALT = 10
N_LAT = 16
N_LON = 16

NUM_GRID_FEATURES = N_ALT * N_LAT * N_LON
NUM_LINEAR_FEATURES = 4  # lat, lon, alt, delta_t
NUM_FIELDS = 1
NUM_CLASSES = 2


class ResBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class ResNet3DModel(nn.Module):
    def __init__(self):
        super().__init__()

        # FC branch for metadata
        self.fc_branch = nn.Sequential(
            nn.Linear(NUM_LINEAR_FEATURES, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        # ResNet-style 3D CNN branch
        self.conv_stem = nn.Sequential(
            nn.Conv3d(NUM_FIELDS, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )

        self.layer1 = nn.Sequential(
            ResBlock3D(32),
            nn.MaxPool3d(2),  # -> (32, 5, 8, 8)
        )

        self.layer2_conv = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            ResBlock3D(64),
            nn.MaxPool3d(2),  # -> (64, 2, 4, 4)
        )

        self.layer3_conv = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            ResBlock3D(128),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # -> (128, 1, 1, 1)
        )

        self.flatten = nn.Flatten()

        # Classifier combining both branches
        self.classifier = nn.Sequential(
            nn.Linear(128 + 8, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, x):
        x_fc = x[:, :4]
        x_cnn = x[:, 4:].reshape(-1, NUM_FIELDS, N_ALT, N_LAT, N_LON)

        out_fc = self.fc_branch(x_fc)

        out_cnn = self.conv_stem(x_cnn)
        out_cnn = self.layer1(out_cnn)
        out_cnn = self.layer2_conv(out_cnn)
        out_cnn = self.layer2(out_cnn)
        out_cnn = self.layer3_conv(out_cnn)
        out_cnn = self.layer3(out_cnn)
        out_cnn = self.flatten(out_cnn)

        out = torch.cat((out_fc, out_cnn), dim=1)
        out = self.classifier(out)
        return out


if __name__ == "__main__":
    model = ResNet3DModel()
    print(model)
    x = torch.randn(2, NUM_LINEAR_FEATURES + NUM_GRID_FEATURES)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {model(x).shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
