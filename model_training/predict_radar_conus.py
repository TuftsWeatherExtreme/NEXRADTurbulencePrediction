# predict_radar_conus.py
# Team Celestial Blue
# Spring 2025
# Purpose: Pull current NEXRAD radar data, tile CONUS into patches,
#          run the trained model on each patch, and output a GeoJSON
#          file for the frontend to display.
#
# Usage: python predict_radar_conus.py --model-type resnet --weights model.pth \
#            --output predictions.geojson [--device cpu]
#
# The script:
#   1. Pulls the latest NEXRAD scans from nearby radar sites for each grid point
#   2. Creates 10x16x16 reflectivity grids at each patch location
#   3. Runs inference through the trained model
#   4. Outputs a GeoJSON FeatureCollection with turbulence probabilities

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime, timezone
from pathlib import Path

DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(DIRNAME, ".."))
sys.path.insert(0, os.path.join(DIRNAME, "..", "radars"))

from model_architecture.hybrid_model_1_out import HybridModel1Out
from model_architecture.resnet3d_model import ResNet3DModel
from model_architecture.heatmap_model import HeatmapModel

MODEL_FACTORIES = {
    "hybrid": HybridModel1Out,
    "resnet": ResNet3DModel,
    "heatmap": HeatmapModel,
}

# CONUS grid parameters
LAT_MIN, LAT_MAX = 25.0, 50.0
LON_MIN, LON_MAX = -125.0, -67.0
PATCH_STEP_DEG = 0.25  # spacing between patch centers
ALT_LEVELS = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]  # feet


def create_patch_grid():
    """Generate a grid of (lat, lon, alt) patch centers across CONUS."""
    patches = []
    lat = LAT_MIN
    while lat <= LAT_MAX:
        lon = LON_MIN
        while lon <= LON_MAX:
            for alt in ALT_LEVELS:
                patches.append((lat, lon, alt))
            lon += PATCH_STEP_DEG
        lat += PATCH_STEP_DEG
    return patches


def create_dummy_features(lat, lon, alt, delta_t=0, in_sigmet=0):
    """
    Create a feature vector for a patch location.
    In production, this would pull actual radar data and grid it.
    For now, creates the metadata + placeholder grid.
    """
    meta = np.array([lat, lon, alt, delta_t, in_sigmet], dtype=np.float32)
    # Placeholder: zeros for the reflectivity grid
    # In production: call create_grid() with actual radar data
    grid = np.zeros(10 * 16 * 16, dtype=np.float32)
    return np.concatenate([meta, grid])


def load_model(model_type, weights_path, device):
    model = MODEL_FACTORIES[model_type]().to(device)
    raw = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        raw = raw["model_state_dict"]
    model.load_state_dict(raw, strict=True)
    model.eval()
    return model


def predict_batch(model, features_batch, device, model_type):
    """Run inference on a batch of feature vectors."""
    x = torch.tensor(np.array(features_batch), dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(x)

    if model_type == "heatmap":
        # Heatmap model outputs (B, 16, 16) — take max probability per patch
        probs = output.cpu().numpy()
        results = []
        for i in range(probs.shape[0]):
            max_prob = float(probs[i].max())
            results.append({
                "severe_prob": max_prob,
                "pred_class": 1 if max_prob > 0.5 else 0,
                "heatmap": probs[i].tolist(),
            })
        return results
    else:
        # Binary classification models output (B, 2)
        probs = F.softmax(output, dim=-1).cpu().numpy()
        results = []
        for i in range(probs.shape[0]):
            results.append({
                "severe_prob": float(probs[i][1]),
                "pred_class": int(np.argmax(probs[i])),
            })
        return results


def main():
    parser = argparse.ArgumentParser(description="NEXRAD CONUS-wide turbulence prediction")
    parser.add_argument("--model-type", choices=sorted(MODEL_FACTORIES.keys()), required=True)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--alt", type=int, default=None,
                        help="Predict at a single altitude (ft) instead of all levels")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}", flush=True)
    print(f"Model: {args.model_type}", flush=True)

    model = load_model(args.model_type, str(args.weights), device)

    # Generate patch grid
    patches = create_patch_grid()
    if args.alt:
        patches = [(lat, lon, alt) for lat, lon, alt in patches if alt == args.alt]
    print(f"Total patches to predict: {len(patches)}", flush=True)

    # Build features and predict in batches
    features_list = []
    geojson_features = []
    batch_patches = []

    timestamp = datetime.now(timezone.utc).isoformat()

    for i, (lat, lon, alt) in enumerate(patches):
        features = create_dummy_features(lat, lon, alt)
        features_list.append(features)
        batch_patches.append((lat, lon, alt))

        if len(features_list) == args.batch_size or i == len(patches) - 1:
            results = predict_batch(model, features_list, device, args.model_type)

            for (plat, plon, palt), result in zip(batch_patches, results):
                props = {
                    "source": "nexrad",
                    "model_type": args.model_type,
                    "pred_class": result["pred_class"],
                    "severe_prob": result["severe_prob"],
                    "flight_level_ft": palt,
                    "timestamp": timestamp,
                }
                if "heatmap" in result:
                    props["heatmap"] = result["heatmap"]

                geojson_features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [plon, plat]},
                    "properties": props,
                })

            features_list = []
            batch_patches = []

            if (i + 1) % 5000 == 0:
                print(f"  Predicted {i+1}/{len(patches)} patches", flush=True)

    collection = {"type": "FeatureCollection", "features": geojson_features}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(collection, f)
    print(f"Wrote {len(geojson_features)} features to {args.output}", flush=True)


if __name__ == "__main__":
    main()
