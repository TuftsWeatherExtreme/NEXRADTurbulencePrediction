# reflect_over_cutoff.py
# Authors: Team Celestial Blue
# Date: 11/11/24
# Purpose: This script takes a VO6 file and a cutoff reflectivity value and
#               creates a csv file containing the longitude, latitude, and 
#               altitude of all instances of reflectivity above this cutoff
#               in the given VO6 file

print("**** Welcome! This script will take all reflectivity values for a given VO6 ****")
print("**** file and convert it to a csv for easier viewing                        ****")
print("    Importing necessary libraries")

import sys


# Access all arguments
arguments = sys.argv
if (len(arguments) != 3):
  print("Expected a VO6 file and a cutoff dBZ value")
  exit(1)

FILENAME = arguments[1]
CUTOFF = int(arguments[2])

import os
from contextlib import contextmanager
import pandas as pd
import numpy as np
# To avoid printing the pyart message everytime
@contextmanager
def quiet():
    sys.stdout = sys.stderr = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

with quiet():
   import pyart




print(f"    Preparing to retrieve reflectivity values over {CUTOFF} for file: {FILENAME}")

data = pyart.io.read_nexrad_archive(FILENAME)

# Use np.ma.where with masking to find indices where data is above the cutoff and not masked
# Transpose to get list of (ray, gate) index pairs where above CUTOFF in dbZ
above_cutoff_indices = np.transpose(np.ma.where(data.fields['reflectivity']['data'] > CUTOFF))

print(f"    There are {len(above_cutoff_indices)} such instances of reflectivity over {CUTOFF}")

if (len(above_cutoff_indices) == 0):
  print("        Since there are 0, no csv file will be written. Exiting now...")
  exit(0)

reflect_50_lla_list = list()

# store all location, ray, gate, reflectivity values if reflectivity values
# are above the cutoff
for ray, gate in above_cutoff_indices:
  reflect_50_lla_list.append((                         
                         data.metadata['instrument_name'],
                         data.fields['reflectivity']['data'][ray][gate],
                         data.gate_longitude['data'][ray][gate], 
                         data.gate_latitude['data'][ray][gate], 
                         data.gate_altitude['data'][ray][gate],
                         ray,
                         gate,
                         ))

print("    Long, Lat, and Alt data acquired, about to write to csv")

reflect_50_lla = np.array(reflect_50_lla_list)

df = pd.DataFrame(reflect_50_lla)
df.columns = ["Radar", "Reflectivity", "Longitude", "Latitude", "Altitude", "Ray", "Gate"]

output_dirname = os.path.join(os.path.dirname(sys.argv[0]), "reflectivity_above_cutoff")
output_filename = f"above_{CUTOFF}_{os.path.basename(FILENAME)[:-4]}.csv"
output_filepath = os.path.join(output_dirname, output_filename)
df.to_csv(output_filepath, index=False)
print(f"    Finished writing csv: {output_filename}")