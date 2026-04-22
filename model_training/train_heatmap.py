# train_heatmap.py
# Team Celestial Blue
# Spring 2025
# Purpose: Train the heatmap U-Net model that predicts a 16x16 turbulence
#          probability grid from radar reflectivity data.
# Usage: python train_heatmap.py <seed>
#
# Uses the same dataloader as the binary classification models.
# For severe PIREPs (label=1), the target is a Gaussian centered at the PIREP
# location within the grid. For non-severe (label=0), the target is all zeros.

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import KFold
import time
import datetime

DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(DIRNAME, ".."))

from model_architecture.heatmap_model import HeatmapModel, create_gaussian_target

NUM_EPOCHS = 30
BATCH_SIZE = 64
NUM_FOLDS = 4
DATALOADER_PATH = os.path.join(DIRNAME, "eleanor_dataloader.pth")
OUTPUT_DIR = os.path.join(DIRNAME, "trained_model_outputs")

N_LAT = 16
N_LON = 16
DEGREES = 0.25  # grid extent in degrees

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HeatmapDataset(Dataset):
    """
    Wraps the existing binary classification dataset and generates
    Gaussian heatmap targets from the PIREP metadata.
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        features = item[0]
        label = item[1]

        # Extract PIREP lat/lon from the first features
        pirep_lat = features[0].item()
        pirep_lon = features[1].item()

        # The grid is centered on the PIREP, so the PIREP is at the center
        # of the 16x16 grid. For severe, we place a Gaussian at center.
        # For non-severe, target is all zeros.
        if label == 1 or label == 1.0:
            target = create_gaussian_target(
                pirep_lat, pirep_lon,
                grid_lat_range=(pirep_lat - DEGREES/2, pirep_lat + DEGREES/2),
                grid_lon_range=(pirep_lon - DEGREES/2, pirep_lon + DEGREES/2),
                sigma=2.0
            )
        else:
            target = np.zeros((N_LAT, N_LON), dtype=np.float32)

        return features, torch.tensor(target, dtype=torch.float32)


def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch_num, (x, target) in enumerate(loader):
        x = x.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        pred = model(x)

        # BCE loss per pixel
        loss = F.binary_cross_entropy(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_num % 50 == 49:
            print(f"    Batch {batch_num+1}: loss={total_loss/50:.6f}", flush=True)
            total_loss = 0

    return total_loss / max(1, len(loader) % 50)


def evaluate(model, loader):
    model.eval()
    total_loss = 0
    total_iou = 0
    total_peak_error = 0
    count = 0

    with torch.no_grad():
        for x, target in loader:
            x = x.to(device)
            target = target.to(device)

            pred = model(x)
            loss = F.binary_cross_entropy(pred, target)
            total_loss += loss.item()

            # IoU at threshold 0.5
            pred_bin = (pred > 0.5).float()
            target_bin = (target > 0.5).float()
            intersection = (pred_bin * target_bin).sum(dim=(1, 2))
            union = ((pred_bin + target_bin) > 0).float().sum(dim=(1, 2))
            iou = (intersection / union.clamp(min=1)).mean().item()
            total_iou += iou

            # Peak location error (in grid cells)
            for i in range(pred.shape[0]):
                if target[i].max() > 0.1:  # Only for severe samples
                    pred_peak = torch.argmax(pred[i])
                    true_peak = torch.argmax(target[i])
                    pred_y, pred_x = pred_peak // N_LON, pred_peak % N_LON
                    true_y, true_x = true_peak // N_LON, true_peak % N_LON
                    dist = ((pred_y - true_y)**2 + (pred_x - true_x)**2).float().sqrt()
                    total_peak_error += dist.item()
                    count += 1

    n_batches = len(loader)
    avg_loss = total_loss / n_batches
    avg_iou = total_iou / n_batches
    avg_peak = total_peak_error / max(1, count)

    return avg_loss, avg_iou, avg_peak


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <seed>")
        sys.exit(1)

    SEED = int(sys.argv[1])
    torch.manual_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Using device: {device}", flush=True)
    print(f"Loading dataloader from {DATALOADER_PATH}", flush=True)

    base_dataset = torch.load(DATALOADER_PATH, weights_only=False)
    dataset = HeatmapDataset(base_dataset)
    print(f"Dataset size: {len(dataset)}", flush=True)

    # Split 85/15
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [0.85, 0.15], generator=torch.Generator().manual_seed(SEED)
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # K-fold CV for L2 selection
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    l2_values = [0.01, 0.001, 0.0001, 0]
    l2_results = []

    print(f"\n{'='*60}", flush=True)
    print(f"Beginning {NUM_FOLDS}-fold CV over L2 values: {l2_values}", flush=True)
    print(f"{'='*60}\n", flush=True)

    for l2_alpha in l2_values:
        fold_losses = []
        print(f"\n--- L2 alpha = {l2_alpha} ---", flush=True)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
            print(f"\n  Fold {fold+1}/{NUM_FOLDS}", flush=True)
            train_loader = DataLoader(Subset(train_dataset, train_idx),
                                      batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(Subset(train_dataset, val_idx),
                                    batch_size=BATCH_SIZE)

            model = HeatmapModel().to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=l2_alpha)

            for epoch in range(NUM_EPOCHS):
                t0 = time.time()
                train_epoch(model, train_loader, optimizer)
                val_loss, val_iou, val_peak = evaluate(model, val_loader)
                elapsed = time.time() - t0
                print(f"    Epoch {epoch+1}/{NUM_EPOCHS}: "
                      f"val_loss={val_loss:.6f}, IoU={val_iou:.4f}, "
                      f"peak_err={val_peak:.2f} cells ({elapsed:.1f}s)", flush=True)

            fold_losses.append(val_loss)
            print(f"  Fold {fold+1} final val_loss: {val_loss:.6f}", flush=True)

        avg = np.mean(fold_losses)
        l2_results.append(avg)
        print(f"\n  L2={l2_alpha} avg CV loss: {avg:.6f}", flush=True)

    best_idx = np.argmin(l2_results)
    best_l2 = l2_values[best_idx]
    print(f"\n{'='*60}", flush=True)
    print(f"Best L2: {best_l2} (loss: {l2_results[best_idx]:.6f})", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Retrain on full training set
    print("Retraining on full training set...", flush=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    best_model = HeatmapModel().to(device)
    optimizer = optim.AdamW(best_model.parameters(), lr=1e-3, weight_decay=best_l2)

    for epoch in range(NUM_EPOCHS):
        train_epoch(best_model, train_loader, optimizer)
        if (epoch + 1) % 5 == 0:
            print(f"  Retrain epoch {epoch+1}/{NUM_EPOCHS}", flush=True)

    # Test evaluation
    print(f"\n{'='*60}", flush=True)
    print("TESTING ON HELD-OUT SET", flush=True)
    print(f"{'='*60}", flush=True)

    test_loss, test_iou, test_peak = evaluate(best_model, test_loader)
    print(f"Test BCE Loss: {test_loss:.6f}", flush=True)
    print(f"Test IoU (threshold=0.5): {test_iou:.4f}", flush=True)
    print(f"Test Peak Location Error: {test_peak:.2f} grid cells", flush=True)
    print(f"  (~{test_peak * DEGREES/N_LAT * 111:.1f} km at mid-latitudes)", flush=True)

    # Per-sample analysis on test set
    print(f"\nSample predictions:", flush=True)
    best_model.eval()
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_loader):
            if batch_idx > 0:
                break
            x = x.to(device)
            pred = best_model(x)
            for i in range(min(5, pred.shape[0])):
                p_max = pred[i].max().item()
                t_max = target[i].max().item()
                label = "SEV" if t_max > 0.5 else "NONE"
                print(f"  Sample {i}: label={label}, pred_max={p_max:.4f}, "
                      f"target_max={t_max:.4f}", flush=True)

    # Save model
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    model_path = os.path.join(OUTPUT_DIR, f"{timestamp}_heatmap_seed_{SEED}.pth")
    torch.save(best_model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}", flush=True)

    # Save results
    results_path = os.path.join(OUTPUT_DIR, f"{timestamp}_heatmap_seed_{SEED}_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Heatmap Model Results\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Best L2: {best_l2}\n")
        f.write(f"Test BCE Loss: {test_loss:.6f}\n")
        f.write(f"Test IoU: {test_iou:.4f}\n")
        f.write(f"Test Peak Error: {test_peak:.2f} cells (~{test_peak * DEGREES/N_LAT * 111:.1f} km)\n")
        f.write(f"Epochs: {NUM_EPOCHS}, Batch Size: {BATCH_SIZE}, Folds: {NUM_FOLDS}\n")
    print(f"Results saved to {results_path}", flush=True)


if __name__ == "__main__":
    main()
