import os
import pandas as pd
import ast

def validate_radar_file(filepath):
    print(f"\nChecking: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}")
        return
    
    # ---- Required columns ----
    required_cols = ['datetime', 'LAT', 'LON', 'FL', 'nexrad_sites', 'aws_files']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        return
    
    total = len(df)
    print(f"Total rows: {total}")
    
    # ---- Parse datetime ----
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)
    nat_count = df['datetime'].isna().sum()
    print(f"NaT datetimes: {nat_count}")
    
    # ---- Parse list-like columns ----
    def safe_parse(x):
        try:
            return ast.literal_eval(x) if isinstance(x, str) else x
        except:
            return []
    
    df['nexrad_sites'] = df['nexrad_sites'].apply(safe_parse)
    df['aws_files'] = df['aws_files'].apply(safe_parse)
    
    # ---- Radar presence ----
    has_radar = df['aws_files'].apply(lambda x: isinstance(x, list) and len(x) > 0)
    num_with_radar = has_radar.sum()
    
    print(f"Rows with radar files: {num_with_radar}/{total} ({num_with_radar/total:.2%})")
    
    # ---- Empty radar rows ----
    empty_radars = df[~has_radar]
    if len(empty_radars) > 0:
        print(f"[WARNING] {len(empty_radars)} rows have NO radar files")
    
    # ---- Check S3 path format ----
    def valid_s3(x):
        if not isinstance(x, list):
            return False
        return all(str(f).startswith("s3://unidata-nexrad-level2/") for f in x)
    
    bad_paths = df[~df['aws_files'].apply(valid_s3)]
    if len(bad_paths) > 0:
        print(f"[WARNING] {len(bad_paths)} rows have invalid S3 paths")
    
    # ---- Site vs radar mismatch ----
    mismatch = df[
        df['nexrad_sites'].apply(len) != df['aws_files'].apply(len)
    ]
    
    print(f"Rows with site/radar count mismatch: {len(mismatch)}")
    
    # ---- Final sanity ----
    if num_with_radar == 0:
        print("[CRITICAL] No PIREPs matched with radar data → pipeline likely broken")
    
    print("Done.\n")


def run_all_tests(base_dir):
    print(f"Scanning directory: {base_dir}")
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".csv"):
                validate_radar_file(os.path.join(root, file))


if __name__ == "__main__":
    BASE_DIR = os.path.join(os.path.dirname(__file__), "pirep_with_radar_data")
    run_all_tests(BASE_DIR)
