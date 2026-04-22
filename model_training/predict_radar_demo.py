# predict_radar_demo.py
# Team Celestial Blue
# Spring 2025
# Purpose: Generate 16 GeoJSON files representing 8 hours of radar
#          turbulence predictions at 30-minute intervals, for frontend demo.
#
# Usage: python -u predict_radar_demo.py --model-type resnet \
#            --weights model.pth --output-dir demo_geojsons/ \
#            [--start-time "2024-03-01T12:00:00"] [--alt 35000]

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time as time_module

DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(DIRNAME, ".."))

from model_architecture.hybrid_model_1_out import HybridModel1Out
from model_architecture.resnet3d_model import ResNet3DModel
from model_architecture.heatmap_model import HeatmapModel

MODEL_FACTORIES = {
    "hybrid": HybridModel1Out,
    "resnet": ResNet3DModel,
    "heatmap": HeatmapModel,
}

# CONUS grid
LAT_MIN, LAT_MAX = 25.0, 50.0
LON_MIN, LON_MAX = -125.0, -67.0
PATCH_STEP_DEG = 0.5  # coarser for demo speed
ALT_LEVELS = [10000, 20000, 30000, 40000]


def create_dummy_features(lat, lon, alt, delta_t=0, in_sigmet=0):
    meta = np.array([lat, lon, alt, delta_t, in_sigmet], dtype=np.float32)
    grid = np.zeros(10 * 16 * 16, dtype=np.float32)
    return np.concatenate([meta, grid])


def predict_batch(model, features_batch, device, model_type):
    x = torch.tensor(np.array(features_batch), dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(x)

    if model_type == "heatmap":
        probs = output.cpu().numpy()
        results = []
        for i in range(probs.shape[0]):
            max_prob = float(probs[i].max())
            results.append({
                "severe_prob": max_prob,
                "pred_class": 1 if max_prob > 0.5 else 0,
            })
        return results
    else:
        probs = F.softmax(output, dim=-1).cpu().numpy()
        results = []
        for i in range(probs.shape[0]):
            results.append({
                "severe_prob": float(probs[i][1]),
                "pred_class": int(np.argmax(probs[i])),
            })
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate 16 GeoJSON files for 8 hours of radar predictions (demo)"
    )
    parser.add_argument("--model-type", choices=sorted(MODEL_FACTORIES.keys()), required=True)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--alt", type=int, default=None,
                        help="Single altitude level (ft). Default: all levels.")
    parser.add_argument("--start-time", type=str, default=None,
                        help="Start of 8-hour window (ISO format). Default: 8 hours ago.")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.start_time:
        start_time = datetime.fromisoformat(args.start_time).replace(tzinfo=timezone.utc)
    else:
        start_time = datetime.now(timezone.utc) - timedelta(hours=8)

    print(f"Device: {device}", flush=True)
    print(f"Model: {args.model_type}", flush=True)
    print(f"8-hour window: {start_time.isoformat()} to "
          f"{(start_time + timedelta(hours=8)).isoformat()}", flush=True)

    # Load model
    model = MODEL_FACTORIES[args.model_type]().to(device)
    raw = torch.load(str(args.weights), map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        raw = raw["model_state_dict"]
    model.load_state_dict(raw, strict=True)
    model.eval()

    # Generate patches
    alt_levels = [args.alt] if args.alt else ALT_LEVELS
    patches = []
    lat = LAT_MIN
    while lat <= LAT_MAX:
        lon = LON_MIN
        while lon <= LON_MAX:
            for alt in alt_levels:
                patches.append((lat, lon, alt))
            lon += PATCH_STEP_DEG
        lat += PATCH_STEP_DEG
    print(f"Patches per step: {len(patches)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time_module.time()

    for step in range(16):
        prediction_time = start_time + timedelta(minutes=step * 30)
        delta_t = step * 30 * 60  # seconds from start

        print(f"\n{'='*60}", flush=True)
        print(f"Step {step+1}/16: {prediction_time.isoformat()}", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time_module.time()
        features_list = []
        batch_patches = []
        all_results = []

        for i, (lat, lon, alt) in enumerate(patches):
            features = create_dummy_features(lat, lon, alt, delta_t=delta_t)
            features_list.append(features)
            batch_patches.append((lat, lon, alt))

            if len(features_list) == args.batch_size or i == len(patches) - 1:
                results = predict_batch(model, features_list, device, args.model_type)
                for (plat, plon, palt), result in zip(batch_patches, results):
                    all_results.append({
                        "lat": plat, "lon": plon, "alt": palt,
                        "severe_prob": result["severe_prob"],
                        "pred_class": result["pred_class"],
                    })
                features_list = []
                batch_patches = []

        n_severe = sum(1 for r in all_results if r["pred_class"] == 1)
        print(f"  Predicted {len(all_results)} patches in {time_module.time()-t0:.1f}s", flush=True)
        print(f"  Severe: {n_severe}/{len(all_results)} ({100*n_severe/len(all_results):.1f}%)", flush=True)

        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
                    "properties": {
                        "source": "nexrad",
                        "model_type": args.model_type,
                        "pred_class": r["pred_class"],
                        "severe_prob": r["severe_prob"],
                        "flight_level_ft": r["alt"],
                        "timestamp": prediction_time.isoformat(),
                        "step": step,
                    },
                }
                for r in all_results
            ],
        }

        filename = args.output_dir / f"prediction_{step:02d}.geojson"
        with open(filename, "w") as f:
            json.dump(geojson, f)
        print(f"  Wrote {filename}", flush=True)

    total_time = time_module.time() - total_start
    print(f"\n{'='*60}", flush=True)
    print(f"Demo complete. 16 GeoJSONs in {total_time/60:.1f} min", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
