# heatmap_model.py
# Authors: Razzle Dazzle Rose Fall 25/Spring 26
# U-Net style encoder-decoder that takes a 10x16x16 radar reflectivity grid
# and outputs a 16x16 turbulence probability heatmap.

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
        self.enc1 = EncoderBlock(NUM_FIELDS, 32)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = EncoderBlock(32, 64)
        self.pool2 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = EncoderBlock(64, 128)

        # Decoder
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 32, 32)

        # Collapse altitude dimension and produce 16x16 heatmap
        self.head = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=(N_ALT, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size=1),
        )

        # FIX: metadata branch now outputs an additive bias, not a
        # multiplicative gate. This prevents the 0.5 floor collapse.
        # It adds a small learned offset based on SIGMET/altitude context.
        self.meta_branch = nn.Sequential(
            nn.Linear(NUM_LINEAR_FEATURES, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh(),   # output in (-1, 1) — used as additive bias
        )

    def forward(self, x):
        x_meta = x[:, :NUM_LINEAR_FEATURES]
        x_grid = x[:, NUM_LINEAR_FEATURES:].reshape(-1, NUM_FIELDS, N_ALT, N_LAT, N_LON)
    
        e1 = self.enc1(x_grid)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(b, e2)
        d1 = self.dec1(d2, e1)
    
        # Raw logits — NO sigmoid here since we use BCEWithLogitsLoss
        heatmap = self.head(d1)        # (B, 1, 1, 16, 16)
        heatmap = heatmap.squeeze(2)   # (B, 1, 16, 16)
    
        meta_bias = self.meta_branch(x_meta)
        meta_bias = meta_bias.unsqueeze(-1).unsqueeze(-1)
        # Return raw logits — sigmoid applied by loss function during training
        # and explicitly in evaluate() for metrics
        return (heatmap + 0.1 * meta_bias).squeeze(1)  # (B, 16, 16)


def create_gaussian_target(label, grid_h=N_LAT, grid_w=N_LON, sigma=2.0):
    """
    Create a 2D Gaussian target heatmap.

    Since the radar grid is ALWAYS centered on the PIREP by construction
    (see radar_data_to_model_input.py), the PIREP is always at the center
    of the grid. We therefore always place the Gaussian at the grid center
    for severe samples, and return zeros for non-severe samples.

    This is the correct behavior — the model's job is to learn whether
    reflectivity patterns in the grid indicate severe turbulence AT the
    center point, not to predict where in the grid turbulence is.

    Args:
        label: 1 for severe, 0 for non-severe
        grid_h, grid_w: grid dimensions (default 16x16)
        sigma: Gaussian spread in grid cells

    Returns:
        (grid_h, grid_w) numpy array with values 0-1
    """
    if label != 1 and label != 1.0:
        return np.zeros((grid_h, grid_w), dtype=np.float32)

    # Center of the grid
    cy = (grid_h - 1) / 2.0
    cx = (grid_w - 1) / 2.0

    y = np.arange(grid_h)
    x = np.arange(grid_w)
    xx, yy = np.meshgrid(x, y)

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