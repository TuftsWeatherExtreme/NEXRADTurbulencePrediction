# radar_data_to_model_input.py
# This python program converts rows of a csv with radar and pirep data to 
#   a model input file and outputs it as a netcdf file
# Author: Team Celestial Blue
# Last Modified: 5/7/2025

from create_grid import create_grid
import pandas as pd
import sys
import os
from datetime import datetime
import quiet_pyart as pyart
DIRNAME = os.path.dirname(sys.argv[0])

# Append to sys path to import scale_turbulence from plane_weights
sys.path.append(os.path.join(DIRNAME, ".."))

from plane_weights.scale_turbulence import scale_turbulence
from get_radars_for_pirep import get_file_time


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

nexrad_sites_path = os.path.join(DIRNAME, "nexrad_sites.csv")
nexrad_sites_df = pd.read_csv(nexrad_sites_path)


def output_to_netcdf(pirep, output_dirname, num_inputs, verbose=False):
    radar_files = pirep['aws_files'].strip("[]").replace("'", "").replace(" ", "").split(',')
    radar = pyart.io.read_nexrad_archive(radar_files[0])


    if radar.longitude['data'] == 0:
        site_code = radar_files[0].split("/")[6]
        site_longitude = nexrad_sites_df.loc[nexrad_sites_df['Site Code'] == site_code, 'Longitude'].iloc[0]
        radar.gate_longitude['data'] += site_longitude
        radar.longitude['data'] = nexrad_sites_df[nexrad_sites_df['Site Code'] == site_code]['Longitude'].iloc[0]
    

    # Find origin based on pirep location
    pirep_location = (ft_to_meters(pirep['FL']), pirep['LAT'], pirep['LON'])
    pirep_t = datetime.fromisoformat(pirep['datetime'])

    # Currently only use the closest radar file - could update to use more
    radar_file = radar_files[0]
    dt = datetime(year=int(radar_file[24:28]), month=int(radar_file[29:31]), day=int(radar_file[32:34]))
    radar_t = get_file_time(radar_file, dt)

    grid = create_grid(radars=radar,
        grid_shape=grid_shape,
        alt_range=alt_limits_meters,
        lat_range=lat_limits_degrees, 
        lon_range=lon_limits_degrees,
        grid_origin=pirep_location, 
        fields=["reflectivity"],
        map_roi=False,
        verbose=False)


    global num_completed
    num_completed += 1
    one_twentieth_num_inputs = num_inputs // 20
    if num_completed % (one_twentieth_num_inputs) == 0:
        print(f"Completed {(num_completed // one_twentieth_num_inputs) * 5}% of df")
    # Detect and do not output empty netcdf files!
    if not grid:
        if verbose:
            print("INFO: no data found for", pirep)
        return
    

    # if there is data for a pirep, output a corresponding netcdf file
    attrs = dict()
    attrs["LAT"] = pirep["LAT"]
    attrs["LON"] = pirep["LON"]
    attrs["ALT"] = pirep["FL"]
    attrs["DELTA_T"] = (pirep_t - radar_t).seconds
    attrs['TURB'] = scale_turbulence(pirep['turbulence_intensity'], pirep['Plane Weight'])
    grid.attrs = attrs

    output_filename = f"{pirep.name:07}_{pirep_t.year}_{pirep_t.month}_df_row.nc"
    output_path = os.path.join(output_dirname, output_filename)
    os.makedirs(output_dirname, exist_ok=True)
    
    grid.to_netcdf(output_path)
    if verbose:
        print(f"Generating netcdf: {output_path}")
    

def usage():
    print(f"Error: Incorrect number of command line arguments. Expected 2 but got {len(sys.argv) - 1}")
    print(f"Usage: python {sys.argv[0]} <input_file> <output_dir>")
    exit(1)

if len(sys.argv) != 3:
    usage()
input_filename = sys.argv[1]
output_dirname = sys.argv[2]

print(f"Reading from file: {input_filename} and outputting to directory: {output_dirname}")

# Important to index_col=0 if reading just a part! - Otherwise remove
pireps_df = pd.read_csv(input_filename, index_col=0)
num_completed = 0

def safe_output_to_netcdf(pirep, output_dirname, num_inputs):
    try:
        output_to_netcdf(pirep, output_dirname, num_inputs)
    except Exception as e:
        print(f"Error processing row {pirep.name}: {e}")

pireps_df.apply(safe_output_to_netcdf, args=(output_dirname, len(pireps_df)), axis=1)

