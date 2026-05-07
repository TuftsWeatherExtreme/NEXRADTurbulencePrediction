# test_sigmet_coverage.py
# Authors: Razzle Dazzle Rose Spring 26
# Purpose: Gives overall summary of cleaned pireps and how many were associated
#          with a sigmet.
# Run with `python test_signmet_coverage.py`


import os
import pandas as pd

def count_sigmet_flags(base_dir):
    """
    Walk through all CSVs in clean_pirep_data and count how many
    PIREPs are marked as in_sigmet.

    Arguments:
        base_dir (str): path to clean_pirep_data directory

    Returns:
        None (prints results)
    """

    total_rows = 0
    total_in_sigmet = 0

    print(f"Scanning directory: {base_dir}\n")

    for root, _, files in os.walk(base_dir):
        for file in files:
            if not file.endswith(".csv"):
                continue

            filepath = os.path.join(root, file)

            try:
                df = pd.read_csv(filepath)

                if 'in_sigmet' not in df.columns:
                    print(f"[WARNING] Skipping {filepath} (no 'in_sigmet' column)")
                    continue

                file_total = len(df)
                file_in_sigmet = (df['in_sigmet'] == 1).sum()

                total_rows += file_total
                total_in_sigmet += file_in_sigmet

                print(f"{filepath}")
                print(f"  Total rows:     {file_total}")
                print(f"  In SIGMET:      {file_in_sigmet}")
                print(f"  Percentage:     {file_in_sigmet / file_total:.2%}\n")

            except Exception as e:
                print(f"[ERROR] Failed to process {filepath}: {e}")

    print("====== OVERALL SUMMARY ======")
    print(f"Total PIREPs:      {total_rows}")
    print(f"In SIGMET:         {total_in_sigmet}")

    if total_rows > 0:
        print(f"Overall % in SIGMET: {total_in_sigmet / total_rows:.2%}")
    else:
        print("No valid data found.")


if __name__ == "__main__":
    BASE_DIR = os.path.join(os.path.dirname(__file__), "clean_pirep_data")
    count_sigmet_flags(BASE_DIR)
