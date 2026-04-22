# heatmap_model.py
# Team Celestial Blue
# Spring 2025
# U-Net style encoder-decoder that takes a 10x16x16 radar reflectivity grid
# and outputs a 16x16 turbulence probability heatmap.
# Uses the same dataloader as the binary models but generates spatial predictions.

import torch
import torch.nn as nn
import numpy as np

N_ALT = 10
N_LAT = 16
N_LON = 16
NUM_LINEAR_FEATURES = 5  # lat, lon, alt, delta_t, in_sigmet
NUM_FIELDS = 1


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if sizes don't match after upsampling
        if x.shape != skip.shape:
            diff_d = skip.shape[2] - x.shape[2]
            diff_h = skip.shape[3] - x.shape[3]
            diff_w = skip.shape[4] - x.shape[4]
            x = nn.functional.pad(x, [0, diff_w, 0, diff_h, 0, diff_d])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class HeatmapModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = EncoderBlock(NUM_FIELDS, 32)    # (32, 10, 16, 16)
        self.pool1 = nn.MaxPool3d(2)                 # (32, 5, 8, 8)

        self.enc2 = EncoderBlock(32, 64)              # (64, 5, 8, 8)
        self.pool2 = nn.MaxPool3d(2)                  # (64, 2, 4, 4)

        # Bottleneck
        self.bottleneck = EncoderBlock(64, 128)       # (128, 2, 4, 4)

        # Decoder
        self.dec2 = DecoderBlock(128, 64, 64)         # (64, 5, 8, 8) -- after upsampling (128, 4, 8, 8) then pad
        self.dec1 = DecoderBlock(64, 32, 32)          # (32, 10, 16, 16)

        # Collapse altitude dimension and produce 16x16 heatmap
        self.head = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=(N_ALT, 1, 1)),  # (16, 1, 16, 16)
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size=1),                # (1, 1, 16, 16)
            nn.Sigmoid(),
        )

        # Metadata branch to modulate the heatmap
        self.meta_branch = nn.Sequential(
            nn.Linear(NUM_LINEAR_FEATURES, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_meta = x[:, :NUM_LINEAR_FEATURES]
        x_grid = x[:, NUM_LINEAR_FEATURES:].reshape(-1, NUM_FIELDS, N_ALT, N_LAT, N_LON)

        # Encoder
        e1 = self.enc1(x_grid)          # (B, 32, 10, 16, 16)
        e2 = self.enc2(self.pool1(e1))  # (B, 64, 5, 8, 8)

        # Bottleneck
        b = self.bottleneck(self.pool2(e2))  # (B, 128, 2, 4, 4)

        # Decoder
        d2 = self.dec2(b, e2)   # (B, 64, 5, 8, 8)
        d1 = self.dec1(d2, e1)  # (B, 32, 10, 16, 16)

        # Heatmap output
        heatmap = self.head(d1)  # (B, 1, 1, 16, 16)
        heatmap = heatmap.squeeze(2)  # (B, 1, 16, 16)

        # Modulate by metadata (e.g., scale by SIGMET presence)
        meta_scale = self.meta_branch(x_meta)  # (B, 1)
        meta_scale = meta_scale.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
        heatmap = heatmap * (0.5 + 0.5 * meta_scale)  # Scale between 0.5x and 1x

        return heatmap.squeeze(1)  # (B, 16, 16)


def create_gaussian_target(pirep_lat, pirep_lon, grid_lat_range, grid_lon_range,
                           grid_h=N_LAT, grid_w=N_LON, sigma=2.0):
    """
    Create a 2D Gaussian target heatmap centered at the PIREP location
    within the grid.

    Args:
        pirep_lat, pirep_lon: PIREP coordinates
        grid_lat_range: (min_lat, max_lat) of the grid
        grid_lon_range: (min_lon, max_lon) of the grid
        grid_h, grid_w: grid dimensions
        sigma: Gaussian spread in grid cells

    Returns:
        (grid_h, grid_w) numpy array with values 0-1
    """
    # Normalize PIREP position to grid coordinates
    lat_frac = (pirep_lat - grid_lat_range[0]) / (grid_lat_range[1] - grid_lat_range[0])
    lon_frac = (pirep_lon - grid_lon_range[0]) / (grid_lon_range[1] - grid_lon_range[0])

    cy = lat_frac * (grid_h - 1)
    cx = lon_frac * (grid_w - 1)

    # Create meshgrid
    y = np.arange(grid_h)
    x = np.arange(grid_w)
    xx, yy = np.meshgrid(x, y)

    # Gaussian
    gaussian = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    return gaussian.astype(np.float32)


if __name__ == "__main__":
    model = HeatmapModel()
    print(model)
    x = torch.randn(2, NUM_LINEAR_FEATURES + N_ALT * N_LAT * N_LON)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
