# find_peak_turbulence_window.py
# Purpose: Scan all clean PIREP CSVs and find the 8-hour window with the
#          highest count of severe turbulence reports (turbulence_intensity >= 5).
# Usage: python find_peak_turbulence_window.py [--data-dir PATH] [--top N]
#
# Defaults to $REPO_PATH/pireps/clean_pirep_data if --data-dir is not provided.

import os
import sys
import glob
import argparse
import pandas as pd
from datetime import timedelta


SEVERE_THRESHOLD = 5
WINDOW_HOURS = 8


def load_all_pireps(data_dir):
    """Load all monthly CSV files from all year subdirectories."""
    pattern = os.path.join(data_dir, "*", "*_turb_pireps.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"ERROR: No PIREP CSV files found under {data_dir}", file=sys.stderr)
        print(f"  Expected pattern: {pattern}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} monthly CSV files across all years", flush=True)

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df["_source_file"] = os.path.relpath(f, data_dir)
            dfs.append(df)
        except Exception as e:
            print(f"  WARNING: Could not read {f}: {e}", file=sys.stderr)

    if not dfs:
        print("ERROR: No files could be loaded.", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined):,} total PIREPs", flush=True)
    return combined


def filter_severe(df):
    """Keep only PIREPs with turbulence_intensity >= SEVERE_THRESHOLD."""
    df["turbulence_intensity"] = pd.to_numeric(df["turbulence_intensity"], errors="coerce")
    severe = df[df["turbulence_intensity"] >= SEVERE_THRESHOLD].copy()
    print(f"Severe PIREPs (intensity >= {SEVERE_THRESHOLD}): {len(severe):,}", flush=True)
    return severe


def find_peak_window(severe_df, top_n=5):
    """
    Sliding window search: for each severe PIREP timestamp t, count how many
    severe PIREPs fall within [t, t + 8h]. Returns the top_n windows.

    Using a sorted array + two-pointer approach for efficiency.
    """
    severe_df = severe_df.sort_values("datetime").reset_index(drop=True)
    times = severe_df["datetime"].values  # numpy datetime64 array
    window = pd.Timedelta(hours=WINDOW_HOURS)

    n = len(times)
    best_windows = []

    right = 0
    for left in range(n):
        # Advance right pointer while within the window
        while right < n and (times[right] - times[left]) <= window:
            right += 1
        count = right - left
        best_windows.append((count, times[left], times[right - 1]))

    # Sort by count descending
    best_windows.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate: skip windows whose start is within 1 hour of an already-selected window
    deduped = []
    selected_starts = []
    for count, start, end in best_windows:
        start_ts = pd.Timestamp(start)
        if all(abs((start_ts - s).total_seconds()) > 3600 for s in selected_starts):
            deduped.append((count, start_ts, pd.Timestamp(end)))
            selected_starts.append(start_ts)
        if len(deduped) >= top_n:
            break

    return deduped


def summarise_window(severe_df, window_start, count):
    """Print a breakdown of PIREPs in the best window."""
    # Ensure window boundaries are tz-aware to match severe_df["datetime"]
    window_start = pd.Timestamp(window_start).tz_localize("UTC") if window_start.tzinfo is None else window_start
    window_end = window_start + timedelta(hours=WINDOW_HOURS)
    mask = (severe_df["datetime"] >= window_start) & (severe_df["datetime"] <= window_end)
    window_pireps = severe_df[mask]

    print(f"\n{'='*60}")
    print(f"Window: {window_start}  →  {window_end} UTC")
    print(f"Severe PIREP count: {count}")
    print(f"{'='*60}")
    print(f"Intensity distribution:")
    print(window_pireps["turbulence_intensity"].value_counts().sort_index().to_string())
    print(f"\nAltitude (FL) stats:")
    fl = pd.to_numeric(window_pireps["FL"], errors="coerce")
    print(f"  Min: {fl.min():.0f} ft   Max: {fl.max():.0f} ft   Mean: {fl.mean():.0f} ft")
    print(f"\nGeographic spread:")
    lat = pd.to_numeric(window_pireps["LAT"], errors="coerce")
    lon = pd.to_numeric(window_pireps["LON"], errors="coerce")
    print(f"  Lat: {lat.min():.2f} → {lat.max():.2f}")
    print(f"  Lon: {lon.min():.2f} → {lon.max():.2f}")
    if "in_sigmet" in window_pireps.columns:
        n_sigmet = pd.to_numeric(window_pireps["in_sigmet"], errors="coerce").sum()
        print(f"\nIn-SIGMET PIREPs: {int(n_sigmet)} / {count}")
    print(f"\nSource files: {window_pireps['_source_file'].unique()}")


def main():
    parser = argparse.ArgumentParser(
        description="Find the 8-hour window with the most severe turbulence PIREPs."
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to clean_pirep_data directory. Defaults to $REPO_PATH/pireps/clean_pirep_data",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top windows to display (default: 5)",
    )
    args = parser.parse_args()

    # Resolve data directory
    if args.data_dir:
        data_dir = args.data_dir
    else:
        repo_path = os.environ.get("REPO_PATH")
        if not repo_path:
            print("ERROR: $REPO_PATH not set. Use --data-dir to specify the path.", file=sys.stderr)
            sys.exit(1)
        data_dir = os.path.join(repo_path, "pireps", "clean_pirep_data")

    if not os.path.isdir(data_dir):
        print(f"ERROR: Directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning: {data_dir}", flush=True)

    # Load and filter
    df = load_all_pireps(data_dir)

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    n_bad = df["datetime"].isna().sum()
    if n_bad > 0:
        print(f"  WARNING: Dropped {n_bad} rows with unparseable datetime", file=sys.stderr)
    df = df.dropna(subset=["datetime"])

    severe_df = filter_severe(df)

    if len(severe_df) == 0:
        print("No severe PIREPs found. Exiting.")
        sys.exit(0)

    # Find top windows
    print(f"\nSearching for top {args.top} {WINDOW_HOURS}-hour windows...", flush=True)
    top_windows = find_peak_window(severe_df, top_n=args.top)

    print(f"\n{'='*60}")
    print(f"TOP {args.top} PEAK 8-HOUR WINDOWS FOR SEVERE TURBULENCE")
    print(f"{'='*60}")
    for rank, (count, start, end) in enumerate(top_windows, 1):
        print(f"  #{rank}  {start}  →  {end}   ({count} severe PIREPs)")

    # Full breakdown of the best window
    best_count, best_start, _ = top_windows[0]
    summarise_window(severe_df, best_start, best_count)


if __name__ == "__main__":
    main()
