# predict_radar_demo.py
# Team Celestial Blue
# Spring 2025
# Purpose: Generate 16 GeoJSON files representing 8 hours of radar
#          turbulence predictions at 30-minute intervals, for frontend demo.
#          Pulls real NEXRAD radar data from AWS S3.
#
# Usage: python -u predict_radar_demo.py --model-type resnet \
#            --weights model.pth --output-dir demo_geojsons/ \
#            [--start-time "2024-03-01T12:00:00"] [--alt 35000]

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time as time_module
from scipy.spatial import cKDTree

DIRNAME = os.path.dirname(os.path.abspath(__file__))
RADAR_DIR = os.path.join(DIRNAME, "..", "radars")
sys.path.insert(0, os.path.join(DIRNAME, ".."))
sys.path.insert(0, RADAR_DIR)

from model_architecture.hybrid_model_1_out import HybridModel1Out
from model_architecture.resnet3d_model import ResNet3DModel
from model_architecture.heatmap_model import HeatmapModel

import quiet_pyart as pyart
from create_grid import create_grid
from beam_geometry import score_radar_for_pirep, get_num_candidates
from haversine import haversine

MODEL_FACTORIES = {
    "hybrid": HybridModel1Out,
    "resnet": ResNet3DModel,
    "heatmap": HeatmapModel,
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
MAX_NAN_FRACTION = 0.95  # more lenient for demo — show something rather than nothing

# CONUS grid for predictions
LAT_MIN, LAT_MAX = 25.0, 50.0
LON_MIN, LON_MAX = -125.0, -67.0
PATCH_STEP_DEG = 0.25  # coarser for demo speed; changed from 0.5 to 0.25 to quadruple the number of patches
ALT_LEVELS = [10000, 20000, 30000, 40000]

NEXRAD_BUCKET = 'unidata-nexrad-level2'


def ft_to_meters(ft):
    return ft / 3.281


def load_nexrad_sites():
    sites_path = os.path.join(RADAR_DIR, "nexrad_sites.csv")
    sites_df = pd.read_csv(sites_path)
    coords = sites_df[['Latitude', 'Longitude']].to_numpy()
    tree = cKDTree(np.radians(coords))
    codes = sites_df['Site Code'].to_numpy()
    elevations = ft_to_meters(sites_df['Elevation'].to_numpy())
    return tree, codes, coords, elevations, sites_df


def find_nearest_radars(lat, lon, alt_ft, tree, codes, coords, elevations):
    """Find the best 5 radar sites for a given location using beam geometry."""
    alt_m = ft_to_meters(alt_ft)
    num_to_query = max(NUM_RADARS, get_num_candidates(alt_ft))
    point = np.radians([lat, lon])
    _dists, indices = tree.query(point, k=num_to_query)

    scored = []
    for idx in indices:
        site_lat, site_lon = coords[idx]
        dist_m = haversine((lat, lon), (site_lat, site_lon), unit='m')
        score = score_radar_for_pirep(dist_m, alt_m, elevations[idx])
        scored.append((score, codes[idx]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [code for _score, code in scored[:NUM_RADARS]]


def fetch_radar_scan(site_code, scan_time):
    """Fetch the nearest NEXRAD scan to scan_time for a given site from S3."""
    prefix = f"{scan_time.year}/{scan_time.month:02d}/{scan_time.day:02d}/{site_code}"
    s3_path = f"s3://{NEXRAD_BUCKET}/{prefix}"

    try:
        radar = pyart.io.read_nexrad_archive(s3_path)
        return radar
    except Exception:
        return None


# Cache for radar objects: (site_code, rounded_time) -> radar object
radar_cache = {}


def get_radar_cached(site_code, scan_time):
    """Fetch a radar scan with caching."""
    cache_key = (site_code, scan_time.replace(minute=0, second=0, microsecond=0))
    if cache_key in radar_cache:
        return radar_cache[cache_key]

    radar = fetch_radar_scan(site_code, scan_time)
    radar_cache[cache_key] = radar
    return radar


def create_features_for_point(lat, lon, alt_ft, scan_time, tree, codes, coords, elevations, sites_df):
    """
    Create a feature vector for a single CONUS grid point by:
    1. Finding nearest radar sites
    2. Fetching radar data
    3. Gridding reflectivity
    Returns feature vector or None if insufficient radar coverage.
    """
    site_codes = find_nearest_radars(lat, lon, alt_ft, tree, codes, coords, elevations)
    alt_m = ft_to_meters(alt_ft)

    # Load radar data for each site
    radars = []
    for code in site_codes:
        radar = get_radar_cached(code, scan_time)
        if radar is not None:
            # Fix longitude if needed
            if radar.longitude['data'] == 0:
                site_lon = sites_df.loc[sites_df['Site Code'] == code, 'Longitude'].iloc[0]
                radar.gate_longitude['data'] += site_lon
                radar.longitude['data'] = site_lon
            radars.append(radar)

    if len(radars) == 0:
        return None

    # Create reflectivity grid
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
    except Exception:
        return None

    if not grid:
        return None

    nan_frac = grid.attrs.get("nan_fraction", 1.0)
    if nan_frac > MAX_NAN_FRACTION:
        return None

    # Build feature vector: metadata + flattened grid
    delta_t = 0
    in_sigmet = 0
    meta = np.array([lat, lon, alt_ft, delta_t, in_sigmet], dtype=np.float32)
    flattened = np.concatenate([grid[var].values.flatten() for var in grid.data_vars])
    features = np.concatenate([meta, flattened])
    features = np.nan_to_num(features, nan=-32.0)
    return features


def predict_batch(model, features_batch, device, model_type):
    x = torch.tensor(np.array(features_batch), dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(x)

    if model_type == "heatmap":
        probs = output.cpu().numpy()
        results = []
        for i in range(probs.shape[0]):
            max_prob = float(probs[i].max())
            # For heatmap, interpret "severe_prob" as max probability in the heatmap.
            results.append({
                "severe_prob": max_prob,
                "pred_class": 1 if max_prob > 0.5 else 0,
                "probs": [1.0 - max_prob, max_prob],
                "prob_max": max_prob,
            })
        return results
    else:
        probs = F.softmax(output, dim=-1).cpu().numpy()
        results = []
        for i in range(probs.shape[0]):
            p0 = float(probs[i][0]) if probs.shape[1] > 0 else 0.0
            p1 = float(probs[i][1]) if probs.shape[1] > 1 else 0.0
            results.append({
                "severe_prob": p1,
                "pred_class": int(np.argmax(probs[i])),
                "probs": [p0, p1],
                "prob_max": float(max(p0, p1)),
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
    parser.add_argument("--batch-size", type=int, default=64)
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

    # Load NEXRAD site data
    tree, codes, coords, elevations, sites_df = load_nexrad_sites()
    print(f"Loaded {len(codes)} NEXRAD sites", flush=True)

    # Generate patch grid
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

        print(f"\n{'='*60}", flush=True)
        print(f"Step {step+1}/16: {prediction_time.isoformat()}", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time_module.time()
        features_list = []
        batch_patches = []
        all_results = []
        skipped = 0

        for i, (lat, lon, alt) in enumerate(patches):
            features = create_features_for_point(
                lat, lon, alt, prediction_time,
                tree, codes, coords, elevations, sites_df
            )

            if features is None:
                skipped += 1
                continue

            features_list.append(features)
            batch_patches.append((lat, lon, alt))

            if len(features_list) == args.batch_size or i == len(patches) - 1:
                if features_list:
                    results = predict_batch(model, features_list, device, args.model_type)
                    for (plat, plon, palt), result in zip(batch_patches, results):
                        all_results.append({
                            "lat": plat, "lon": plon, "alt": palt,
                            "severe_prob": result["severe_prob"],
                            "pred_class": result["pred_class"],
                            "probs": result.get("probs"),
                            "prob_max": result.get("prob_max"),
                            "delta_t_seconds": 0.0,
                            "true_class": None,
                            "pirep_time": prediction_time.replace(tzinfo=None).isoformat(),
                            "patch_id": f"demo_{step:02d}_{plat:.3f}_{plon:.3f}_{int(palt)}",
                        })
                    features_list = []
                    batch_patches = []

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{len(patches)} patches...", flush=True)

        n_severe = sum(1 for r in all_results if r["pred_class"] == 1)
        print(f"  Predicted {len(all_results)} patches, skipped {skipped} "
              f"({time_module.time()-t0:.1f}s)", flush=True)
        print(f"  Severe: {n_severe}/{len(all_results)} "
              f"({100*n_severe/max(1,len(all_results)):.1f}%)", flush=True)
        print(f"  Radar cache size: {len(radar_cache)}", flush=True)

        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
                    "properties": {
                        "source": "nexrad",
                        "pred_class": r["pred_class"],
                        "probs": r.get("probs"),
                        "severe_prob": r.get("severe_prob"),
                        "prob_max": r.get("prob_max"),
                        "true_class": r.get("true_class"),
                        "flight_level_ft": r["alt"],
                        "delta_t_seconds": r.get("delta_t_seconds"),
                        "pirep_time": r.get("pirep_time"),
                        "patch_id": r.get("patch_id"),
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
