# export_predictions_geojson.py
# Load a trained NEXRAD model, run inference on NetCDF patches, write GeoJSON points.
#
# Usage:
#   python export_predictions_geojson.py --model-type resnet --weights path/to/model.pth \
#       --input-dir path/to/netcdfs --output predictions.geojson

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(DIRNAME, ".."))

from model_architecture.regression_model import LinearClassifierModel
from model_architecture.hybrid_model_1_out import HybridModel1Out
from model_architecture.resnet3d_model import ResNet3DModel

MODEL_FACTORIES = {
    "linear": LinearClassifierModel,
    "hybrid": HybridModel1Out,
    "resnet": ResNet3DModel,
}


def _features_label_from_attrs(ds: xr.Dataset) -> tuple[np.ndarray, float]:
    """
    Same four scalars + label as RadarDataLoader (named attrs only).
    PIREP_TIME must exist on the file but is not part of the tensor.
    """
    a = ds.attrs
    required = ("LAT", "LON", "ALT", "DELTA_T", "TURB", "PIREP_TIME", "IN_SIGMET")
    missing = [k for k in required if k not in a]
    if missing:
        raise ValueError(f"NetCDF missing required attrs: {missing}")
    meta = np.array(
        [float(a["LAT"]), float(a["LON"]), float(a["ALT"]), float(a["DELTA_T"])],
        dtype=np.float64,
    )
    return meta, float(a["TURB"])


def tensor_and_meta_from_nc(ds: xr.Dataset) -> tuple[torch.Tensor, dict]:
    meta_features, label = _features_label_from_attrs(ds)
    flattened = np.concatenate([ds[var].values.flatten() for var in ds.data_vars])
    vec = np.concatenate((meta_features, flattened))
    t = torch.tensor(vec, dtype=torch.float32)
    t = torch.nan_to_num(t, nan=-32.0)

    a = ds.attrs
    meta = {
        "true_class": label,
        "lat": float(meta_features[0]),
        "lon": float(meta_features[1]),
        "flight_level_ft": float(meta_features[2]),
        "delta_t_seconds": float(meta_features[3]),
        "pirep_time": str(a["PIREP_TIME"]),
    }
    return t, meta


def load_state_dict_flexible(model: torch.nn.Module, path: str, device: torch.device) -> None:
    raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        raw = raw["model_state_dict"]
    model.load_state_dict(raw, strict=True)


def collect_nc_paths(input_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(input_dir.rglob("*.nc"))
    return sorted(input_dir.glob("*.nc"))


def prediction_to_properties(meta: dict, logits_1d: torch.Tensor) -> dict:
    if logits_1d.dim() == 0:
        logits_1d = logits_1d.unsqueeze(0)
    probs = F.softmax(logits_1d, dim=-1).cpu().tolist()
    pred_class = int(torch.argmax(logits_1d, dim=-1).item())
    severe_prob = float(probs[1]) if len(probs) > 1 else None
    return {
        "source": "nexrad",
        "pred_class": pred_class,
        "probs": probs,
        "severe_prob": severe_prob,
        "prob_max": float(max(probs)) if probs else None,
        "true_class": meta["true_class"],
        "flight_level_ft": meta["flight_level_ft"],
        "delta_t_seconds": meta["delta_t_seconds"],
        "pirep_time": meta["pirep_time"],
        "patch_id": meta["patch_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NEXRAD inference → GeoJSON FeatureCollection (points for Mapbox, etc.).",
    )
    parser.add_argument(
        "--model-type",
        choices=sorted(MODEL_FACTORIES.keys()),
        required=True,
        help="Must match the architecture used to train the .pth file.",
    )
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory of NetCDF files (must include PIREP_TIME and LAT/LON/ALT/DELTA_T/TURB attrs).",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories for *.nc",
    )
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: not a directory: {args.input_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.weights.is_file():
        print(f"Error: weights not found: {args.weights}", file=sys.stderr)
        sys.exit(1)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = MODEL_FACTORIES[args.model_type]().to(device)
    load_state_dict_flexible(model, str(args.weights), device)
    model.eval()

    features: list[dict] = []
    for nc_path in collect_nc_paths(args.input_dir, args.recursive):
        try:
            with xr.open_dataset(nc_path) as ds:
                tensor, meta = tensor_and_meta_from_nc(ds)
        except Exception as e:
            print(f"Skip {nc_path}: {e}", file=sys.stderr)
            continue

        meta["patch_id"] = nc_path.name
        x = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
        logits = logits.squeeze(0)

        props = prediction_to_properties(meta, logits)
        lon, lat = meta["lon"], meta["lat"]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": props,
            }
        )

    collection = {"type": "FeatureCollection", "features": features}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(collection, f, indent=2)
    print(f"Wrote {len(features)} features to {args.output}")


if __name__ == "__main__":
    main()
