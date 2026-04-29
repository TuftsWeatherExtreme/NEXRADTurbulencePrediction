# model_architecture.py (new file)
import torch
import torch.nn as nn

class SatelliteTurbulenceModel(nn.Module):
    def __init__(self, num_metadata_features=4):
        super().__init__()
        # Process image sequence: (B, 15, 6, 128, 128)
        self.cnn = nn.Sequential(
            nn.Conv3d(6, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),  # -> (B, 32, 15, 64, 64)
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((3, 2, 2)),  # -> (B, 64, 5, 32, 32)
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 4, 4)),  # -> (B, 128, 1, 4, 4)
            nn.Flatten()  # -> (B, 2048)
        )
        # Combine CNN output with metadata
        self.classifier = nn.Sequential(
            nn.Linear(2048 + num_metadata_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # 2-class output for CrossEntropyLoss
        )

    def forward(self, x, metadata):
        # x: (B, 15, 6, 128, 128) -> permute for Conv3d: (B, 6, 15, 128, 128)
        x = x.permute(0, 2, 1, 3, 4)
        cnn_out = self.cnn(x)
        combined = torch.cat([cnn_out, metadata], dim=1)
        return self.classifier(combined)
