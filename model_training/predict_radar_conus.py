# predict_radar_conus.py
# Team Celestial Blue
# Spring 2025
# Purpose: Pull current NEXRAD radar data, tile CONUS into patches,
#          run the trained model on each patch, and output a GeoJSON
#          file for the frontend to display. Supports continuous mode.
#
# Usage: python -u predict_radar_conus.py --model-type resnet --weights model.pth \
#            --output predictions.geojson [--cycle-minutes 30] [--alt 35000]

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
MAX_NAN_FRACTION = 0.95

# CONUS grid
LAT_MIN, LAT_MAX = 25.0, 50.0
LON_MIN, LON_MAX = -125.0, -67.0
PATCH_STEP_DEG = 0.5
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


class RadarCache:
    """Cache radar objects with time-based eviction."""
    def __init__(self, max_age_minutes=60):
        self.cache = {}
        self.max_age = timedelta(minutes=max_age_minutes)

    def get(self, site_code, scan_time, sites_df):
        cache_key = (site_code, scan_time.replace(minute=0, second=0, microsecond=0))
        if cache_key in self.cache:
            return self.cache[cache_key]

        prefix = f"{scan_time.year}/{scan_time.month:02d}/{scan_time.day:02d}/{site_code}"
        s3_path = f"s3://{NEXRAD_BUCKET}/{prefix}"
        try:
            radar = pyart.io.read_nexrad_archive(s3_path)
            if radar.longitude['data'] == 0:
                site_lon = sites_df.loc[sites_df['Site Code'] == site_code, 'Longitude'].iloc[0]
                radar.gate_longitude['data'] += site_lon
                radar.longitude['data'] = site_lon
            self.cache[cache_key] = radar
            return radar
        except Exception:
            self.cache[cache_key] = None
            return None

    def evict_old(self, current_time):
        cutoff = current_time - self.max_age
        expired = [k for k, v in self.cache.items() if k[1] < cutoff]
        for k in expired:
            del self.cache[k]
        if expired:
            print(f"  Evicted {len(expired)} old radar scans from cache", flush=True)


def create_features_for_point(lat, lon, alt_ft, scan_time, tree, codes, coords,
                               elevations, sites_df, radar_cache):
    site_codes = find_nearest_radars(lat, lon, alt_ft, tree, codes, coords, elevations)
    alt_m = ft_to_meters(alt_ft)

    radars = []
    for code in site_codes:
        radar = radar_cache.get(code, scan_time, sites_df)
        if radar is not None:
            radars.append(radar)

    if len(radars) == 0:
        return None

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

    meta = np.array([lat, lon, alt_ft, 0, 0], dtype=np.float32)
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
        return [{"severe_prob": float(probs[i].max()),
                 "pred_class": 1 if probs[i].max() > 0.5 else 0}
                for i in range(probs.shape[0])]
    else:
        probs = F.softmax(output, dim=-1).cpu().numpy()
        return [{"severe_prob": float(probs[i][1]),
                 "pred_class": int(np.argmax(probs[i]))}
                for i in range(probs.shape[0])]


def run_prediction_cycle(model, patches, scan_time, device, model_type, batch_size,
                         tree, codes, coords, elevations, sites_df, radar_cache):
    features_list = []
    batch_patches = []
    all_results = []
    skipped = 0

    for i, (lat, lon, alt) in enumerate(patches):
        features = create_features_for_point(
            lat, lon, alt, scan_time,
            tree, codes, coords, elevations, sites_df, radar_cache
        )

        if features is None:
            skipped += 1
            continue

        features_list.append(features)
        batch_patches.append((lat, lon, alt))

        if len(features_list) == batch_size or i == len(patches) - 1:
            if features_list:
                results = predict_batch(model, features_list, device, model_type)
                for (plat, plon, palt), result in zip(batch_patches, results):
                    all_results.append({
                        "lat": plat, "lon": plon, "alt": palt,
                        "severe_prob": result["severe_prob"],
                        "pred_class": result["pred_class"],
                    })
                features_list = []
                batch_patches = []

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(patches)} patches...", flush=True)

    return all_results, skipped


def main():
    parser = argparse.ArgumentParser(description="NEXRAD CONUS-wide turbulence prediction")
    parser.add_argument("--model-type", choices=sorted(MODEL_FACTORIES.keys()), required=True)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--alt", type=int, default=None,
                        help="Predict at a single altitude (ft) instead of all levels")
    parser.add_argument("--cycle-minutes", type=int, default=0,
                        help="If >0, run continuously every N minutes")
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

    # Load NEXRAD sites
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
    print(f"Total patches: {len(patches)}", flush=True)

    radar_cache = RadarCache(max_age_minutes=60)

    while True:
        cycle_start = time_module.time()
        now = datetime.now(timezone.utc)
        print(f"\n{'='*60}", flush=True)
        print(f"Prediction cycle at {now.isoformat()}", flush=True)
        print(f"{'='*60}", flush=True)

        results, skipped = run_prediction_cycle(
            model, patches, now, device, args.model_type, args.batch_size,
            tree, codes, coords, elevations, sites_df, radar_cache
        )

        n_severe = sum(1 for r in results if r["pred_class"] == 1)
        print(f"Predicted {len(results)} patches, skipped {skipped}", flush=True)
        print(f"Severe: {n_severe}/{len(results)} "
              f"({100*n_severe/max(1,len(results)):.1f}%)", flush=True)
        print(f"Radar cache: {len(radar_cache.cache)} scans", flush=True)

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
                        "timestamp": now.isoformat(),
                    },
                }
                for r in results
            ],
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(geojson, f)
        print(f"Wrote {len(results)} features to {args.output}", flush=True)

        cycle_time = time_module.time() - cycle_start
        print(f"Cycle completed in {cycle_time/60:.1f} min", flush=True)

        # Evict old radar scans
        radar_cache.evict_old(now)

        if args.cycle_minutes <= 0:
            break

        wait = max(0, args.cycle_minutes * 60 - cycle_time)
        print(f"Next cycle in {wait:.0f}s...", flush=True)
        time_module.sleep(wait)


if __name__ == "__main__":
    main()
