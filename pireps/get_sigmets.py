# get_sigmets.py
# Authors: Team Razzle Dazzle Rose x Claude
# Purpose: This script downloads Convective SIGMETs for a given month and year
#          from the IEM AWC SIGMET archive, parses polygon geometry and altitude
#          information, and outputs a CSV of SIGMETs that can be used to annotate
#          PIREPs with an `in_sigmet` feature in clean_pireps.py.
#
# Run with: python get_sigmets.py -month MONTH -year YEAR [-o {FILE/STDOUT}]
#
# Output columns:
#   label       - SIGMET label (e.g. "57C")
#   issue       - UTC issuance time
#   expire      - UTC expiration time
#   fl_low      - Lower flight level bound (feet), parsed from raw text
#   fl_high     - Upper flight level bound (feet), parsed from raw text
#   polygon     - WKT string of the SIGMET polygon boundary
#
# Dependencies:
#   pip install requests geopandas shapely fiona
#   (geopandas is needed to read the shapefile format returned by IEM)

import sys
import os
import re
import requests
import zipfile
import io
import pandas as pd
import geopandas as gpd
import calendar
from datetime import date
from shapely.geometry import Point
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL)


# ── helpers ──────────────────────────────────────────────────────────────────

def eprint(*args, **kwargs):
    """Print to stderr (mirrors clean_pireps.py convention)."""
    print(*args, file=sys.stderr, **kwargs)


def usage():
    eprint(f"Usage: {sys.argv[0]} -month MONTH -year YEAR [-o {{FILE/STDOUT}}]")
    exit(1)


MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]

BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/sigmets.py"


def read_command_line_args():
    """Parse -month, -year, -o flags (same style as clean_pireps.py)."""
    month_str = None
    year = None
    output = None
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "-month":
            i += 1
            if i < len(sys.argv):
                month_str = sys.argv[i].lower()
                if month_str not in MONTHS:
                    eprint(f"Invalid month: {month_str}")
                    usage()
        elif sys.argv[i] == "-year":
            i += 1
            if i < len(sys.argv):
                year = int(sys.argv[i])
                if year not in range(2003, 2027):
                    eprint(f"Invalid year: {year}")
                    usage()
        elif sys.argv[i] == "-o":
            i += 1
            if i < len(sys.argv):
                output = sys.argv[i]
                if output.lower() == "stdout":
                    output = sys.stdout
        else:
            eprint(f"Unexpected argument: {sys.argv[i]}")
            usage()
        i += 1

    if month_str is None or year is None:
        eprint("Missing month or year.")
        usage()

    month_idx = MONTHS.index(month_str) + 1
    return year, month_idx, output


# ── flight-level parsing ──────────────────────────────────────────────────────

def parse_flight_levels(text):
    """
    Extract lower and upper flight level bounds (in feet) from raw SIGMET text.

    SIGMETs encode altitude in two common patterns:
        "FL180/FL350"  →  18 000 ft / 35 000 ft
        "FL180/350"    →  18 000 ft / 35 000 ft
        "TOPS FL350"   →  0 ft (surface) / 35 000 ft

    Returns (fl_low_ft, fl_high_ft) as ints, or (None, None) if unparseable.
    """
    if not isinstance(text, str):
        return None, None

    # Pattern 1: FL###/FL### or FL###/###
    match = re.search(r'FL(\d{2,3})[/ ]+(?:FL)?(\d{2,3})', text)
    if match:
        return int(match.group(1)) * 100, int(match.group(2)) * 100

    # Pattern 2: TOPS FL### (surface to tops)
    match = re.search(r'TOPS\s+FL(\d{2,3})', text)
    if match:
        return 0, int(match.group(1)) * 100

    # Pattern 3: single FL### (treat as tops, surface base)
    match = re.search(r'FL(\d{2,3})', text)
    if match:
        return 0, int(match.group(1)) * 100

    return None, None


# ── IEM download ─────────────────────────────────────────────────────────────

def fetch_sigmets(year, month_idx):
    # Build start and end dates for the month
    last_day = calendar.monthrange(year, month_idx)[1]
    sts = date(year, month_idx, 1).strftime("%Y-%m-%dT00:00:00Z")
    ets_year = year + 1 if month_idx == 12 else year
    ets_month = 1 if month_idx == 12 else month_idx + 1
    ets = date(ets_year, ets_month, 1).strftime("%Y-%m-%dT00:00:00Z")

    params = {
        "sts": sts,
        "ets": ets,
        "fmt": "shp",
    }

    eprint(f"Fetching SIGMETs from IEM for {MONTHS[month_idx - 1]} {year}...")
    eprint(f"URL params: sts={sts}, ets={ets}")
    r = requests.get(BASE_URL, params=params, timeout=120)
    
    # Check for non-zip / invalid response
    if not r.content.startswith(b'PK'):
        eprint("ERROR: Response is not a valid zip file. First 200 bytes:")
        eprint(r.content[:200])
        return gpd.GeoDataFrame()

    eprint(f"Response status: {r.status_code}")
    r.raise_for_status()
    eprint(f"Response was {len(r.content) / 1024:.1f} KB")

    if len(r.content) < 100:
        eprint(f"WARNING: Response appears empty: {r.text[:200]}")
        return gpd.GeoDataFrame()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            z.extractall(tmpdir)
            shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
            if not shp_files:
                eprint("ERROR: No .shp file found in zip response.")
                return gpd.GeoDataFrame()
            gdf = gpd.read_file(os.path.join(tmpdir, shp_files[0]))

    eprint(f"Loaded {len(gdf)} raw SIGMET records")
    return gdf


# ── processing ────────────────────────────────────────────────────────────────

def process_sigmets(gdf):
    """
    Clean the raw GeoDataFrame and return a tidy DataFrame with:
        label, issue, expire, fl_low, fl_high, polygon (WKT)
    """
    if gdf.empty:
        return pd.DataFrame(columns=["label", "issue", "expire",
                                     "fl_low", "fl_high", "polygon"])

    # Normalise column names to uppercase for safety, but preserve geometry column
    gdf.columns = [c.upper() if c != "geometry" else c for c in gdf.columns]
    
    # Debugging and error checking for columns
    required_cols = ["ISSUE", "EXPIRE", "TEXT", "LABEL"]
    missing = [c for c in required_cols if c not in gdf.columns]
    if missing:
        eprint(f"ERROR: Missing expected columns: {missing}")
        eprint(f"Available columns: {list(gdf.columns)}")
        return pd.DataFrame(columns=["label", "issue", "expire", "fl_low", "fl_high", "polygon"])
    if "geometry" not in gdf.columns:
        eprint("ERROR: No geometry column found")
        return pd.DataFrame(columns=["label", "issue", "expire", "fl_low", "fl_high", "polygon"])

    # Parse issue / expire to datetime
    gdf["issue"] = pd.to_datetime(gdf["ISSUE"], utc=True, errors="coerce")
    gdf["expire"] = pd.to_datetime(gdf["EXPIRE"], utc=True, errors="coerce")

    # Drop rows without valid geometry or times
    len_before = len(gdf)
    gdf = gdf.dropna(subset=["issue", "expire", "geometry"])
    eprint(f"Dropped {len_before - len(gdf)} rows with missing time/geometry")

    # Parse flight levels from raw text
    fl_parsed = gdf["TEXT"].apply(parse_flight_levels)
    gdf["fl_low"] = fl_parsed.apply(lambda x: x[0])
    gdf["fl_high"] = fl_parsed.apply(lambda x: x[1])

    # Convert polygon geometry to WKT string for easy CSV storage
    gdf["polygon"] = gdf["geometry"].apply(lambda g: g.wkt if g is not None else None)

    result = gdf[["LABEL", "issue", "expire", "fl_low", "fl_high", "polygon"]].copy()
    result = result.rename(columns={"LABEL": "label"})
    result = result.reset_index(drop=True)

    eprint(f"Processed {len(result)} valid SIGMET records")
    return result


# ── spatial lookup helper (used by clean_pireps.py) ──────────────────────────

def is_in_sigmet(pirep_lat, pirep_lon, pirep_alt_ft, pirep_time, sigmets_df):
    """
    Return 1 if the PIREP location/time/altitude falls within any active
    Convective SIGMET polygon, 0 otherwise.

    Arguments:
        pirep_lat     - float, latitude of the PIREP
        pirep_lon     - float, longitude of the PIREP
        pirep_alt_ft  - float, altitude of the PIREP in feet (FL column)
        pirep_time    - pandas Timestamp (UTC)
        sigmets_df    - DataFrame produced by process_sigmets()

    Returns:
        int: 1 if inside a SIGMET, 0 otherwise
    """
    from shapely import wkt as shapely_wkt

    point = Point(pirep_lon, pirep_lat)

    # Make pirep_time timezone-aware if needed
    if pirep_time.tzinfo is None:
        pirep_time = pirep_time.tz_localize("UTC")

    for _, sigmet in sigmets_df.iterrows():
        # ── time check ──
        if pirep_time < sigmet["issue"] or pirep_time > sigmet["expire"]:
            continue

        # ── altitude check ── (skip if flight level data is missing)
        if sigmet["fl_low"] is not None and sigmet["fl_high"] is not None:
            if not (sigmet["fl_low"] <= pirep_alt_ft <= sigmet["fl_high"]):
                continue

        # ── spatial check ──
        if sigmet["polygon"] is None:
            continue
        try:
            polygon = shapely_wkt.loads(sigmet["polygon"])
            if polygon.contains(point):
                return 1
        except Exception:
            continue

    return 0


# ── main ──────────────────────────────────────────────────────────────────────

eprint("**** get_sigmets.py: Download and clean Convective SIGMETs ****")

DIRNAME = os.path.dirname(os.path.abspath(__file__))
YEAR, START_MONTH_IDX, OUTPUT = read_command_line_args()

# Download
gdf = fetch_sigmets(YEAR, START_MONTH_IDX)

# Process
sigmets_df = process_sigmets(gdf)

eprint(f"Writing {len(sigmets_df)} SIGMETs to output")

# Determine output path
if OUTPUT is None or OUTPUT != sys.stdout:
    filename = os.path.join(DIRNAME, "sigmet_data", str(YEAR),
                            f"{START_MONTH_IDX:02}_sigmets.csv")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    eprint(f"Writing CSV to: {filename}")
    OUTPUT = open(filename, "w")

csv_str = sigmets_df.to_csv(None, index=False)
print(csv_str, file=OUTPUT)
OUTPUT.close()
eprint("Finished writing SIGMET CSV")