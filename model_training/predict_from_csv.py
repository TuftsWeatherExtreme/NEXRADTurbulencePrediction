# predict_from_csv.py
# Team Celestial Blue
# Spring 2025
# Purpose: Read a CSV with radar file paths (output of get_radars_for_pirep.py),
#          download radar data, grid it, run model inference, output GeoJSON.
#          This is Job B of the two-step CONUS prediction pipeline.
#
# Usage: python -u predict_from_csv.py --model-type resnet --weights model.pth \
#            --input-csv grid_with_radars.csv --output predictions.geojson

import argparse
import json
import os
import sys
import signal
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
import time as time_module

DIRNAME = os.path.dirname(os.path.abspath(__file__))
RADAR_DIR = os.path.join(DIRNAME, "..", "radars")
sys.path.insert(0, os.path.join(DIRNAME, ".."))
sys.path.insert(0, RADAR_DIR)

from model_architecture.hybrid_model_1_out import HybridModel1Out
from model_architecture.resnet3d_model import ResNet3DModel

import quiet_pyart as pyart
from create_grid import create_grid
from get_radars_for_pirep import get_file_time

MODEL_FACTORIES = {
    "hybrid": HybridModel1Out,
    "resnet": ResNet3DModel,
}

# Grid parameters (same as training)
NUM_Z_POINTS = 10
NUM_Y_POINTS = 16
NUM_X_POINTS = 16
DEGREES = 0.25
Z_SIZE = 3048
GRID_SHAPE = (NUM_Z_POINTS, NUM_Y_POINTS, NUM_X_POINTS)
ALT_LIMITS = (-Z_SIZE / 2.0, Z_SIZE / 2.0)
LAT_LIMITS = (-DEGREES / 2.0, DEGREES / 2.0)
LON_LIMITS = (-DEGREES / 2.0, DEGREES / 2.0)
NUM_RADARS = 5
MAX_NAN_FRACTION = 0.95

nexrad_sites_path = os.path.join(RADAR_DIR, "nexrad_sites.csv")
nexrad_sites_df = pd.read_csv(nexrad_sites_path)


def ft_to_meters(ft):
    return ft / 3.281


def timeout_handler(signum, frame):
    raise TimeoutError("read_nexrad_archive timed out")


def fix_radar_longitude(radar, radar_file):
    if radar.longitude['data'] == 0:
        site_code = radar_file.split("/")[6]
        site_longitude = nexrad_sites_df.loc[
            nexrad_sites_df['Site Code'] == site_code, 'Longitude'
        ].iloc[0]
        radar.gate_longitude['data'] += site_longitude
        radar.longitude['data'] = site_longitude


# Global cache: S3 path -> loaded pyart radar object (or None if failed)
_radar_file_cache = {}
_cache_stats = {"hits": 0, "misses": 0, "errors": 0}


def load_radar_cached(rf):
    """Download and cache a radar file. Returns radar object or None."""
    if rf in _radar_file_cache:
        _cache_stats["hits"] += 1
        return _radar_file_cache[rf]

    _cache_stats["misses"] += 1
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        radar = pyart.io.read_nexrad_archive(rf)
        signal.alarm(0)
        fix_radar_longitude(radar, rf)
        _radar_file_cache[rf] = radar
        return radar
    except TimeoutError:
        print(f"    Timeout reading {rf}", flush=True)
        _cache_stats["errors"] += 1
        _radar_file_cache[rf] = None
        return None
    except Exception as e:
        print(f"    Error reading {rf}: {e}", flush=True)
        _cache_stats["errors"] += 1
        _radar_file_cache[rf] = None
        return None


def process_row(row):
    """
    Process a single CSV row: download radars (with caching), grid,
    return feature vector or None.
    """
    radar_files_str = row.get('aws_files', '[]')
    radar_files = radar_files_str.strip("[]").replace("'", "").replace(" ", "").split(',')
    radar_files = [f.replace('noaa-nexrad-level2', 'unidata-nexrad-level2') for f in radar_files]
    radar_files = [f for f in radar_files if f]  # remove empty strings
    radar_files = radar_files[:NUM_RADARS]

    if not radar_files:
        return None

    lat = row['LAT']
    lon = row['LON']
    alt_ft = row['FL']
    alt_m = ft_to_meters(alt_ft)

    # Load radars from cache
    radars = []
    for rf in radar_files:
        radar = load_radar_cached(rf)
        if radar is not None:
            radars.append(radar)

    if len(radars) == 0:
        return None

    # Grid
    try:
        grid = create_grid(
            radars=tuple(radars),
            grid_shape=GRID_SHAPE,
            alt_range=ALT_LIMITS,
            lat_range=LAT_LIMITS,
            lon_range=LON_LIMITS,
            grid_origin=(alt_m, lat, lon),
            fields=["reflectivity"],
            map_roi=False,
            verbose=False
        )
    except Exception as e:
        print(f"    Grid error: {e}", flush=True)
        return None

    if not grid:
        return None

    nan_frac = grid.attrs.get("nan_fraction", 1.0)
    if nan_frac > MAX_NAN_FRACTION:
        return None

    # Compute delta_t
    try:
        basename = os.path.basename(radar_files[0])
        dt = datetime(year=int(basename[4:8]), month=int(basename[8:10]), day=int(basename[10:12]))
        radar_t = get_file_time(radar_files[0], dt)
        pirep_t = datetime.fromisoformat(str(row['datetime']))
        delta_t = abs((pirep_t - radar_t).total_seconds()) if radar_t else 0
    except Exception:
        delta_t = 0

    in_sigmet = row.get('in_sigmet', 0)

    meta = np.array([lat, lon, alt_ft, delta_t, in_sigmet], dtype=np.float32)
    flattened = np.concatenate([grid[var].values.flatten() for var in grid.data_vars])
    features = np.concatenate([meta, flattened])
    features = np.nan_to_num(features, nan=-32.0)
    return features


def main():
    parser = argparse.ArgumentParser(
        description="Run radar model inference on CSV with radar file paths"
    )
    parser.add_argument("--model-type", choices=sorted(MODEL_FACTORIES.keys()), required=True)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}", flush=True)
    print(f"Model: {args.model_type}", flush=True)

    # Load model
    model = MODEL_FACTORIES[args.model_type]().to(device)
    raw = torch.load(str(args.weights), map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        raw = raw["model_state_dict"]
    model.load_state_dict(raw, strict=True)
    model.eval()

    # Read CSV
    df = pd.read_csv(args.input_csv)
    print(f"Input CSV: {len(df)} rows", flush=True)

    # --- Preload: download all unique radar files upfront ---
    all_radar_paths = set()
    for _, row in df.iterrows():
        radar_files_str = row.get('aws_files', '[]')
        files = radar_files_str.strip("[]").replace("'", "").replace(" ", "").split(',')
        files = [f.replace('noaa-nexrad-level2', 'unidata-nexrad-level2') for f in files]
        files = [f for f in files if f][:NUM_RADARS]
        all_radar_paths.update(files)

    print(f"Preloading {len(all_radar_paths)} unique radar files...", flush=True)
    preload_start = time_module.time()
    for i, rf in enumerate(sorted(all_radar_paths)):
        load_radar_cached(rf)
        if (i + 1) % 20 == 0:
            print(f"  Preloaded {i+1}/{len(all_radar_paths)} files "
                  f"({_cache_stats['errors']} errors) "
                  f"[{time_module.time()-preload_start:.0f}s]", flush=True)
    print(f"Preload complete: {len(_radar_file_cache)} files cached, "
          f"{_cache_stats['errors']} errors "
          f"({time_module.time()-preload_start:.0f}s)", flush=True)

    all_results = []
    features_batch = []
    batch_rows = []
    num_processed = 0
    num_no_radar = 0
    start_time = time_module.time()

    for idx, row in df.iterrows():
        features = process_row(row)

        if features is None:
            num_no_radar += 1
            all_results.append({
                "lat": row['LAT'], "lon": row['LON'], "alt": row['FL'],
                "severe_prob": 0.0, "pred_class": 0,
                "probs": [1.0, 0.0], "prob_max": 1.0,
                "pirep_time": str(row['datetime']),
                "patch_id": f"grid_{idx:06d}_no_radar",
            })
        else:
            features_batch.append(features)
            batch_rows.append(row)

        if len(features_batch) == args.batch_size or idx == len(df) - 1:
            if features_batch:
                x = torch.tensor(np.array(features_batch), dtype=torch.float32).to(device)
                with torch.no_grad():
                    output = model(x)
                probs = F.softmax(output, dim=-1).cpu().numpy()

                for i, r in enumerate(batch_rows):
                    p0, p1 = float(probs[i][0]), float(probs[i][1])
                    all_results.append({
                        "lat": r['LAT'], "lon": r['LON'], "alt": r['FL'],
                        "severe_prob": p1, "pred_class": int(np.argmax(probs[i])),
                        "probs": [p0, p1], "prob_max": float(max(p0, p1)),
                        "pirep_time": str(r['datetime']),
                        "patch_id": f"grid_{idx:06d}",
                    })

                features_batch = []
                batch_rows = []

        num_processed += 1
        if num_processed % 500 == 0:
            elapsed = time_module.time() - start_time
            print(f"  Processed {num_processed}/{len(df)} "
                  f"({num_no_radar} no radar) [{elapsed:.0f}s] "
                  f"radar_cache: {len(_radar_file_cache)} files, "
                  f"hits={_cache_stats['hits']} misses={_cache_stats['misses']} "
                  f"errors={_cache_stats['errors']}", flush=True)

    # Stats
    model_preds = [r for r in all_results if "no_radar" not in r["patch_id"]]
    n_severe = sum(1 for r in all_results if r["pred_class"] == 1)
    print(f"\nResults: {len(all_results)} total points", flush=True)
    print(f"  With radar data: {len(model_preds)}", flush=True)
    print(f"  No radar coverage: {num_no_radar}", flush=True)
    print(f"  Severe predictions: {n_severe}", flush=True)
    if model_preds:
        model_probs = [r["severe_prob"] for r in model_preds]
        print(f"  Model prob stats: min={min(model_probs):.4f}, max={max(model_probs):.4f}, "
              f"mean={np.mean(model_probs):.4f}", flush=True)
        print(f"  >0.1: {sum(1 for p in model_probs if p > 0.1)}, "
              f">0.3: {sum(1 for p in model_probs if p > 0.3)}, "
              f">0.5: {sum(1 for p in model_probs if p > 0.5)}", flush=True)

    # Write GeoJSON
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
                "properties": {
                    "source": "nexrad",
                    "pred_class": r["pred_class"],
                    "probs": r["probs"],
                    "severe_prob": r["severe_prob"],
                    "prob_max": r["prob_max"],
                    "true_class": None,
                    "flight_level_ft": r["alt"],
                    "delta_t_seconds": 0.0,
                    "pirep_time": r["pirep_time"],
                    "patch_id": r["patch_id"],
                },
            }
            for r in all_results
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(geojson, f)

    elapsed = time_module.time() - start_time
    print(f"\nWrote {len(all_results)} features to {args.output} ({elapsed/60:.1f} min)",
          flush=True)


if __name__ == "__main__":
    main()
