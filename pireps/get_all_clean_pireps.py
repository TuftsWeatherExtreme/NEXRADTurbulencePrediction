# Small script to run clean_pireps on all valid months and years to store
# the cleaned pireps locally

import numpy as np
import os
import sys

YEARS = np.arange(2003, 2025, 1)
MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]

dirname = os.path.dirname(sys.argv[0])

for year in YEARS:
    for month in MONTHS:
        output_filename = f"{month}_{year}_pireps.csv"
        exe_name = os.path.join(dirname, 'clean_pireps.py')
        print(f"Invoking: python3 {exe_name} -month {month} -year {year} -o {output_filename}")
        os.system(f"python3 {exe_name} -month {month} -year {year} -o {output_filename}")

