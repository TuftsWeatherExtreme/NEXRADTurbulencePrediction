# example_create_grid.py
# This python program runs the create_grid function with some example arguments
#   to give a sense for how it works
# Author: Team Celestial Blue
# Last Modified: 5/7/2025

import pandas as pd
from create_grid import create_grid
import time
import quiet_pyart as pyart
import sys
import os

DIRNAME = os.path.dirname(sys.argv[0])

def main():
    # Example usage of create_grid
    print("Beginning preprocessing of pireps")
    start_time = time.time()
    pireps_df = pd.read_csv(os.path.join(DIRNAME, "split_radar_data/part_001.csv"))
    row = pireps_df.iloc[17]
    radar_files = row['aws_files'].strip("[]").replace("'", "").replace(" ", "").split(',')
    radar1 = pyart.io.read_nexrad_archive(radar_files[0])
    radar2 = pyart.io.read_nexrad_archive(radar_files[1])
    radars = (radar1, radar2)

    def ft_to_meters(dist_in_ft):
        return dist_in_ft/3.281

    NUM_Z_POINTS = 10
    NUM_Y_POINTS = 16
    NUM_X_POINTS = 16
    grid_shape = (NUM_Z_POINTS, NUM_Y_POINTS, NUM_X_POINTS) # Number of points in the grid (z, y, x).

    # Determine grid limits
    DEGREES = 0.25
    Z_SIZE = 3048 # 3048m (10000 ft)
    alt_limits_meters = (-Z_SIZE/2.0, Z_SIZE/2.0)
    lat_limits_degrees = (-DEGREES/2.0, DEGREES/2.0)
    lon_limits_degrees = (-DEGREES/2.0, DEGREES/2.0)

    # Find origin based on pirep location
    pirep_location = (ft_to_meters(row['FL']), row['LAT'], row['LON'])

    preprocessing_time = time.time()
    print("Finished preprocessing of pireps, now calling create_grid")

    grid = create_grid(radars=radars,
        grid_shape=grid_shape,
        alt_range=alt_limits_meters,
        lat_range=lat_limits_degrees, 
        lon_range=lon_limits_degrees,
        grid_origin=pirep_location, 
        fields=["reflectivity", "spectrum_width"],
        map_roi=True,
        verbose=True)

    create_grid_time = time.time()
    print("Finished calling create_grid")
    print(grid)
    print(f"Preprocessing time: {(preprocessing_time - start_time):.2f}s, create_grid time: {(create_grid_time - preprocessing_time):.2f}s")

if __name__ == "__main__":
    main()