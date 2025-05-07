# split_csv.py
# This python splits a given csv into a given number of parts and stores them
# in a given directory
# Author: Sam Hecht + Claude AI
# Last Modified: 5/7/2025

import pandas as pd
import os
import math
import sys

# This function was completely AI generated
def split_csv_file(input_file, output_folder, num_parts=48):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Get total number of rows
    total_rows = len(df)
    
    # Calculate rows per part (rounded up to ensure all rows are included)
    rows_per_part = math.ceil(total_rows / num_parts)
    
    # Split and save parts
    for i in range(num_parts):
        start_idx = i * rows_per_part
        end_idx = min((i + 1) * rows_per_part, total_rows)
        
        # Get the slice of dataframe
        part_df = df.iloc[start_idx:end_idx]
        
        # Create output filename
        output_file = os.path.join(output_folder, f"part_{i+1:03d}.csv")
        
        # Save to CSV
        part_df.to_csv(output_file, index=True)
        
        print(f"Created {output_file} with {len(part_df)} rows")

def usage(error_msg):
    print(f"Error: {error_msg}")
    print(f"Usage: python {sys.argv[0]} <input_file> <output_folder> <num_parts>")
    exit(1)

def is_int(s: str) -> bool:
    """
    Purpose:
        Returns True if the given string s represents an integer and False otherwise
    Arguments:
        s: The string to determine if we can generate an integer from
    Returns:
        A boolean indicating whether the argument string represents an integer
    """
    try: 
        int(s)
    except ValueError:
        return False
    else:
        return True

if __name__ == "__main__":
    if len(sys.argv) != 4:
        usage(f"Incorrect number of command line arguments. Expected 3 but got {len(sys.argv) - 1}")
    DIRNAME = os.path.dirname(sys.argv[0])
    input_file = sys.argv[1]
    output_folder = sys.argv[2]
    os.makedirs(output_folder, exist_ok=True)
    if not is_int(sys.argv[3]):
        usage(f"Expected argument 3 to be an integer but got '{sys.argv[3]}'")
    num_parts = int(sys.argv[3])
    print(f"Splitting CSV: {input_file} into {num_parts} different files named part_###.csv and storing outputs in {output_folder}")
    split_csv_file(input_file, output_folder, num_parts)
