# train_ensemble.py
# Team Celestial Blue
# Spring 2025
# Purpose: Train a small ensemble model that combines radar and satellite
#          predictions into a single turbulence probability.
#
# Usage: python train_ensemble.py <seed> \
#            --radar-weights resnet_model.pth \
#            --sat-weights sat_model.pth \
#            --radar-dataloader radar_dataloader.pth \
#            --sat-data-dir /path/to/sat/model_inputs
#
# The ensemble model takes as input:
#   - radar_severe_prob (from radar model)
#   - sat_severe_prob (from satellite model)
#   - has_radar (1 if radar coverage exists, 0 otherwise)
#   - lat, lon, alt
# And outputs a combined turbulence probability.

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from pathlib import Path
import time
import datetime
import json

DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(DIRNAME, ".."))

from model_architecture.hybrid_model_1_out import HybridModel1Out
from model_architecture.resnet3d_model import ResNet3DModel

RADAR_MODELS = {
    "hybrid": HybridModel1Out,
    "resnet": ResNet3DModel,
}

OUTPUT_DIR = os.path.join(DIRNAME, "trained_model_outputs")


class EnsembleModel(nn.Module):
    """Small MLP that combines radar + satellite predictions."""
    def __init__(self, input_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)


def get_radar_predictions(radar_model, dataloader, device):
    """Run radar model on its dataloader, return (probs, labels, metadata)."""
    radar_model.eval()
    all_probs = []
    all_labels = []
    all_meta = []  # lat, lon, alt

    with torch.no_grad():
        for batch in dataloader:
            features = batch[0].to(device)
            labels = batch[1]

            output = radar_model(features)
            probs = F.softmax(output, dim=-1)[:, 1].cpu().numpy()

            # Extract metadata (first 5 features: lat, lon, alt, delta_t, in_sigmet)
            meta = features[:, :3].cpu().numpy()  # lat, lon, alt

            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            all_meta.extend(meta)

    return np.array(all_probs), np.array(all_labels), np.array(all_meta)


def get_sat_predictions(sat_model_path, sat_data_dir, device, sat_repo_src=None):
    """Run satellite model on its data, return (probs, labels, metadata)."""
    # Try multiple paths for satellite imports
    search_paths = []
    if sat_repo_src:
        search_paths.append(sat_repo_src)
    if os.environ.get("SAT_REPO_PATH"):
        search_paths.append(os.path.join(os.environ["SAT_REPO_PATH"], "src"))
    search_paths.append(os.path.join(os.path.dirname(DIRNAME), "..",
                                      "SatelliteTurbulencePrediction", "src"))

    for path in search_paths:
        if os.path.exists(path):
            sys.path.insert(0, path)
            print(f"  Added satellite src path: {path}", flush=True)
            break
    else:
        print(f"WARNING: No satellite src found. Searched: {search_paths}", flush=True)

    try:
        from dataloader_class import SatelliteDataLoader
        from model_architecture import SatelliteTurbulenceModel
    except ImportError as e:
        print(f"WARNING: Satellite model import failed: {e}. Using dummy predictions.", flush=True)
        return None, None, None

    dataset = SatelliteDataLoader(sat_data_dir)
    loader = DataLoader(dataset, batch_size=32)

    model = SatelliteTurbulenceModel().to(device)
    raw = torch.load(sat_model_path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        raw = raw["model_state_dict"]
    model.load_state_dict(raw, strict=True)
    model.eval()

    all_probs = []
    all_labels = []
    all_meta = []

    with torch.no_grad():
        for batch in loader:
            features, labels, weights = batch
            # SatelliteTurbulenceModel needs (x, metadata)
            metadata = torch.zeros(features.shape[0], 4).to(device)
            output = model(features.to(device), metadata)
            probs = F.softmax(output, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    return np.array(all_probs), np.array(all_labels), None


def main():
    parser = argparse.ArgumentParser(description="Train ensemble model combining radar + satellite")
    parser.add_argument("seed", type=int)
    parser.add_argument("--radar-model-type", default="resnet", choices=sorted(RADAR_MODELS.keys()))
    parser.add_argument("--radar-weights", required=True, type=Path)
    parser.add_argument("--radar-dataloader", required=True, type=Path,
                        help="Path to radar .pth dataloader")
    parser.add_argument("--sat-weights", type=Path, default=None,
                        help="Path to satellite model weights (optional)")
    parser.add_argument("--sat-data-dir", type=Path, default=None,
                        help="Path to satellite model_inputs dir (optional)")
    parser.add_argument("--sat-repo", type=str, default=None,
                        help="Path to SatelliteTurbulencePrediction/src (for imports)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}", flush=True)

    # --- Step 1: Get radar predictions on training data ---
    print("\nLoading radar model and generating predictions...", flush=True)
    radar_dataset = torch.load(str(args.radar_dataloader), weights_only=False)
    radar_loader = DataLoader(radar_dataset, batch_size=64)

    RadarModelClass = RADAR_MODELS[args.radar_model_type]
    radar_model = RadarModelClass().to(device)
    raw = torch.load(str(args.radar_weights), map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        raw = raw["model_state_dict"]
    radar_model.load_state_dict(raw, strict=True)

    radar_probs, labels, meta = get_radar_predictions(radar_model, radar_loader, device)
    print(f"Radar predictions: {len(radar_probs)} samples", flush=True)
    print(f"  severe_prob: min={radar_probs.min():.4f}, max={radar_probs.max():.4f}, "
          f"mean={radar_probs.mean():.4f}", flush=True)

    # --- Step 2: Get satellite predictions (if available) ---
    has_sat = False
    if args.sat_weights and args.sat_data_dir:
        print("\nLoading satellite model and generating predictions...", flush=True)
        sat_probs, sat_labels, _ = get_sat_predictions(
            str(args.sat_weights), str(args.sat_data_dir), device,
            sat_repo_src=args.sat_repo
        )
        if sat_probs is not None:
            has_sat = True
            print(f"Satellite predictions: {len(sat_probs)} samples", flush=True)

    # --- Step 3: Build ensemble features ---
    print("\nBuilding ensemble features...", flush=True)

    n = len(radar_probs)
    # Features: radar_prob, sat_prob, has_radar, lat, lon, alt
    ensemble_features = np.zeros((n, 6), dtype=np.float32)
    ensemble_features[:, 0] = radar_probs
    ensemble_features[:, 1] = 0.0  # default sat prob
    ensemble_features[:, 2] = 1.0  # has_radar = always true for radar dataloader
    ensemble_features[:, 3] = meta[:, 0]  # lat
    ensemble_features[:, 4] = meta[:, 1]  # lon
    ensemble_features[:, 5] = meta[:, 2]  # alt

    if has_sat:
        # Match satellite predictions to radar by index (assumes same PIREP ordering)
        n_match = min(n, len(sat_probs))
        ensemble_features[:n_match, 1] = sat_probs[:n_match]

    ensemble_labels = labels.astype(np.int64)

    X = torch.tensor(ensemble_features, dtype=torch.float32)
    y = torch.tensor(ensemble_labels, dtype=torch.long)

    print(f"Ensemble dataset: {len(X)} samples, {X.shape[1]} features", flush=True)
    print(f"Label distribution: 0={int((y==0).sum())}, 1={int((y==1).sum())}", flush=True)

    # --- Step 4: Train ensemble with k-fold CV ---
    print("\nTraining ensemble model...", flush=True)

    # Split 85/15
    n_test = int(0.15 * len(X))
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(SEED))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    NUM_EPOCHS = 50
    BATCH_SIZE = 256
    NUM_FOLDS = 4
    l2_values = [0.01, 0.001, 0.0001, 0]
    best_l2 = 0
    best_loss = float('inf')

    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    for l2 in l2_values:
        fold_losses = []
        for fold, (tr_idx, val_idx) in enumerate(kfold.split(X_train)):
            model = EnsembleModel().to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=l2)

            tr_loader = DataLoader(
                TensorDataset(X_train[tr_idx], y_train[tr_idx]),
                batch_size=BATCH_SIZE, shuffle=True
            )
            val_loader = DataLoader(
                TensorDataset(X_train[val_idx], y_train[val_idx]),
                batch_size=BATCH_SIZE
            )

            for epoch in range(NUM_EPOCHS):
                model.train()
                for xb, yb in tr_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    loss = F.cross_entropy(model(xb), yb)
                    loss.backward()
                    optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss += F.cross_entropy(model(xb), yb).item()
            fold_losses.append(val_loss / len(val_loader))

        avg = np.mean(fold_losses)
        print(f"  L2={l2}: avg CV loss={avg:.4f}", flush=True)
        if avg < best_loss:
            best_loss = avg
            best_l2 = l2

    print(f"Best L2: {best_l2} (loss: {best_loss:.4f})", flush=True)

    # --- Step 5: Retrain on full training set ---
    print("\nRetraining on full training set...", flush=True)
    best_model = EnsembleModel().to(device)
    optimizer = optim.AdamW(best_model.parameters(), lr=1e-3, weight_decay=best_l2)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )

    for epoch in range(NUM_EPOCHS):
        best_model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(best_model(xb), yb)
            loss.backward()
            optimizer.step()

    # --- Step 6: Evaluate on test set and collect per-sample predictions ---
    print("\nEvaluating on test set...", flush=True)
    best_model.eval()
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

    num_correct = 0
    tp = fp = tn = fn = 0
    all_test_probs = []
    per_sample_rows = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            output = best_model(xb)
            probs = F.softmax(output, dim=-1)[:, 1].cpu().numpy()
            preds = output.argmax(dim=1)
            all_test_probs.extend(probs)

            for i in range(len(yb)):
                true = yb[i].item()
                pred = preds[i].item()
                if true == pred:
                    num_correct += 1
                if true == 1 and pred == 1: tp += 1
                elif true == 0 and pred == 1: fp += 1
                elif true == 0 and pred == 0: tn += 1
                elif true == 1 and pred == 0: fn += 1

                # Collect per-sample data
                per_sample_rows.append({
                    "radar_prob": float(xb[i, 0].item()),
                    "sat_prob": float(xb[i, 1].item()),
                    "has_radar": float(xb[i, 2].item()),
                    "lat": float(xb[i, 3].item()),
                    "lon": float(xb[i, 4].item()),
                    "alt": float(xb[i, 5].item()),
                    "combined_prob": float(probs[i]),
                    "pred_class": int(pred),
                    "true_label": int(true),
                    "correct": int(true == pred),
                })

    total = len(y_test)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    print(f"\nEnsemble Results:", flush=True)
    print(f"  Accuracy: {num_correct}/{total} ({100*num_correct/total:.2f}%)", flush=True)
    print(f"  Precision: {precision:.4f}", flush=True)
    print(f"  Recall: {recall:.4f}", flush=True)
    print(f"  F1: {f1:.4f}", flush=True)
    print(f"  Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}", flush=True)
    print(f"  Prob stats: min={min(all_test_probs):.4f}, max={max(all_test_probs):.4f}, "
          f"mean={np.mean(all_test_probs):.4f}", flush=True)

    # --- Step 7: Save model, results, and per-sample predictions ---
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    model_path = os.path.join(OUTPUT_DIR, f"{timestamp}_ensemble_seed_{SEED}.pth")
    torch.save(best_model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}", flush=True)

    results = {
        "seed": SEED,
        "best_l2": best_l2,
        "accuracy": num_correct / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "has_satellite": has_sat,
        "radar_model_type": args.radar_model_type,
        "n_train": len(X_train),
        "n_test": total,
    }
    results_path = os.path.join(OUTPUT_DIR, f"{timestamp}_ensemble_seed_{SEED}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}", flush=True)

    # Per-sample predictions CSV
    import csv
    csv_path = os.path.join(OUTPUT_DIR, f"{timestamp}_ensemble_seed_{SEED}_predictions.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_sample_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_sample_rows)
    print(f"Per-sample predictions saved to {csv_path} ({len(per_sample_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
