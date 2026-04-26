# generate_conus_grid.py
# Team Celestial Blue
# Spring 2025
# Purpose: Generate a CSV of CONUS grid points formatted like a PIREP CSV
#          so it can be piped directly into get_radars_for_pirep.py.
#
# Usage: python generate_conus_grid.py --time "2024-03-01T12:00:00" \
#            [--step-deg 0.25] [--alt 35000] > conus_grid.csv
#
# Output columns match clean_pireps.py format:
#   URGENT, AIRCRAFT, REPORT, TURBULENCE, PRODUCT_ID, FL, LAT, LON,
#   datetime, turbulence_intensity, Plane Weight

import argparse
import sys
from datetime import datetime, timezone

# CONUS bounds
LAT_MIN, LAT_MAX = 25.0, 50.0
LON_MIN, LON_MAX = -125.0, -67.0
ALT_LEVELS = [10000, 20000, 30000, 40000]


def main():
    parser = argparse.ArgumentParser(
        description="Generate a CONUS grid CSV formatted like PIREP data"
    )
    parser.add_argument("--time", required=True,
                        help="Prediction time (ISO format, e.g. 2024-03-01T12:00:00)")
    parser.add_argument("--step-deg", type=float, default=0.25,
                        help="Grid spacing in degrees (default: 0.25)")
    parser.add_argument("--alt", type=int, default=None,
                        help="Single altitude level (ft). Default: all levels.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path. Default: stdout")
    args = parser.parse_args()

    pred_time = datetime.fromisoformat(args.time).replace(tzinfo=timezone.utc)
    dt_str = pred_time.strftime("%Y-%m-%d %H:%M:%S")
    alt_levels = [args.alt] if args.alt else ALT_LEVELS

    out = open(args.output, "w") if args.output else sys.stdout

    # Header matching clean_pireps.py output
    print("URGENT,AIRCRAFT,REPORT,TURBULENCE,PRODUCT_ID,FL,LAT,LON,"
          "datetime,turbulence_intensity,Plane Weight", file=out)

    count = 0
    lat = LAT_MIN
    while lat <= LAT_MAX:
        lon = LON_MIN
        while lon <= LON_MAX:
            for alt in alt_levels:
                # Fake PIREP row — model only uses LAT, LON, FL, datetime
                print(f"F,GRID,CONUS_GRID,NONE,grid_{count},{alt},{lat},{lon},"
                      f"{dt_str},0.0,M", file=out)
                count += 1
            lon += args.step_deg
        lat += args.step_deg

    if args.output:
        out.close()

    print(f"Generated {count} grid points", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
