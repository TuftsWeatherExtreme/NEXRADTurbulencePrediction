from pireps.create_grid import create_grid
import netCDF4
import numpy as np
import pandas as pd
# from pandarallel import pandarallel
from contextlib import contextmanager
import sys
import os
import time
from datetime import datetime
import quiet_pyart as pyart
from plane_weights.scale_turbulence import scale_turbulence 


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

nexrad_sites = pd.read_csv("/cluster/tufts/capstone25celestialb/shared/nexrad_playground/pireps/nexrad_sites.csv")

def get_file_time(filename, dt):
    """
        Purpose: Gets the time of a given nexrad file and returns it as a datetime
        Arguments: 
            filename: The filename for a given nexrad file, will have _HHMMSS_
                in it to parse into the datetime
            dt: The datetime containing the day, month, and year of the nexrad file
        Returns: A datetime representing when the nexrad data was collected
    """
    # Example filenames below
    "YEAR/MONTH/DAY/SITE_CODE/{SITE_CODE}{YEAR}{MONTH}{DAY}_{HHMMSS}_VO6"
    "2024/12/05/KAKQ/KAKQ20241205_001256_V06"
    # Split to get just the {HHMMSS} part of the string and put these values in a datetime
    filetime = filename.split("_")[1]
    hour = int(filetime[:2])
    minute = int(filetime[2:4])
    second = int(filetime[4:6])
    return datetime(year=dt.year, month=dt.month, day=dt.day, hour=hour, minute=minute, second=second)



def output_to_netcdf(pirep, part_number, verbose=False):
    radar_files = pirep['aws_files'].strip("[]").replace("'", "").replace(" ", "").split(',')
    radar = pyart.io.read_nexrad_archive(radar_files[0])


    if radar.longitude['data'] == 0:
        site_code = radar_files[0].split("/")[6]
        site_longitude = nexrad_sites.loc[nexrad_sites['Site Code'] == site_code, 'Longitude'].iloc[0]
        radar.gate_longitude['data'] += site_longitude
        radar.longitude['data'] = nexrad_sites[nexrad_sites['Site Code'] == site_code]['Longitude'].iloc[0]
    
    print(radar.gate_longitude['data'])


    # Find origin based on pirep location
    pirep_location = (ft_to_meters(pirep['FL']), pirep['LAT'], pirep['LON'])
    pirep_t = datetime.fromisoformat(pirep['datetime'])

    # TODO: clean up this (band aid for now)
    radar_file = radar_files[0]
    dt = datetime(year=int(radar_file[24:28]), month=int(radar_file[29:31]), day=int(radar_file[32:34]))
    radar_t = get_file_time(radar_file, dt)

    preprocessing_time = time.time()
    grid = create_grid(radars=radar,
        grid_shape=grid_shape,
        alt_range=alt_limits_meters,
        lat_range=lat_limits_degrees, 
        lon_range=lon_limits_degrees,
        grid_origin=pirep_location, 
        fields=["reflectivity"],
        map_roi=False,
        verbose=False)


    # Detect and do not output empty netcdf files!
    if not grid: # TODO: write a counter or some msg
        if verbose:
            print("INFO: no data found for", pirep)
        print("no data found for pirep")
        return
    

    # if there is data for a pirep, output a corresponding netcdf file
    attrs = dict()
    attrs["LAT"] = pirep["LAT"]
    attrs["LON"] = pirep["LON"]
    attrs["ALT"] = pirep["FL"]
    attrs["DELTA_T"] = (pirep_t - radar_t).seconds
    attrs['TURB'] = scale_turbulence(pirep['turbulence_intensity'], pirep['Turbulence_Category'])
    grid.attrs = attrs

    filename = f"{pirep_t.year}_{pirep_t.month}_df_row_{pirep.name}.nc"
    output_filename = f"/cluster/tufts/capstone25celestialb/shared/nexrad_playground/2000_2015_netcdf_model_inputs/{part_number}/{filename}"
    grid.to_netcdf(output_filename)

    print(f"Generating netcdf: {output_filename}")
    
    # TODO: Add some useful logging of completed ones



# pandarallel.initialize(nb_workers=4)
assert(len(sys.argv) == 3)
filename = sys.argv[1]
print(filename)
part_number = sys.argv[2]

print(f"part_number: {part_number}")

# Important to index_col=0 if reading just a part!!!! Otherwise remove
pireps_df = pd.read_csv(filename, index_col=0)

def safe_output_to_netcdf(pirep, part_number):
    try:
        output_to_netcdf(pirep, part_number)
    except Exception as e:
        print(f"Error processing row {pirep.name}: {e}")

pireps_df.apply(safe_output_to_netcdf, args=(part_number,), axis=1)

