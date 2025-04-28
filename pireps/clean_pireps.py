# clean_pireps.py
# Authors: Team Celestial Blue
# Date: 11/11/24
# Purpose: This script takes an input pireps data file and cleans it to be a csv
#          with only the pireps which contain reports of turbulence, removing
#          extraneous columns
# GET request calls script here:
# https://github.com/akrherz/iem/blob/main/pylib/iemweb/request/gis/pireps.py#L33
# Instructions here:
# https://mesonet.agron.iastate.edu/cgi-bin/request/gis/pireps.py?help

import sys

import requests
from tqdm import tqdm
import pandas as pd
from io import StringIO
import numpy as np
import os
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL) 


def usage():
    eprint(f"Usage: {sys.argv[0]} -month MONTH -year YEAR [-o {{FILE/STDOUT}}]")
    exit(1)

def read_command_line_args():
    global DIRNAME
    DIRNAME = os.path.dirname(sys.argv[0])
    month_str = None
    year = None
    month_idx = None
    output = None
    i = 1
    while (i < len(sys.argv)):
        if sys.argv[i] == "-month":
            i += 1
            if i < len(sys.argv):
                month_str = sys.argv[i]
                if month_str.lower() not in MONTHS:
                    eprint(f"Invalid month received: {month_str}. Expecting one of:\n {MONTHS}")
                    usage()
        elif sys.argv[i] == "-year":
            i += 1
            if i < len(sys.argv):
                year = int(sys.argv[i])
                if year not in range(2003, 2026):
                    eprint(f"Invalid year: {year} not in range [2003, 2025]")
                    usage()
        elif sys.argv[i] == "-o":
            i += 1
            if i < len(sys.argv):
                output = sys.argv[i]
                if output.lower() == "stdout":
                    eprint("Printing to stdout")
                    output = sys.stdout
        else:
            eprint(f"Unexpected command line arg: {sys.argv[i]}")
            usage()
        i += 1

    if month_str is None or year is None:
        eprint("Missing either month or year. Exiting...")
        usage()

    month_idx = MONTHS.index(month_str.lower()) + 1

    return year, month_idx, output

# Prints to stderr
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)



eprint("**** Welcome! This script will take a month and year and return a csv file ****")
eprint("**** with pireps that mention turbulence from this month and year          ****")

# Access all arguments
DIRNAME = None

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/pireps.py"

YEAR, START_MONTH_IDX, OUTPUT = read_command_line_args()

END_MONTH_IDX = 1 if START_MONTH_IDX == len(MONTHS) else START_MONTH_IDX + 1
END_YEAR = YEAR + 1 if START_MONTH_IDX == len(MONTHS) else YEAR
query = {"year1": YEAR, 
         "month1": START_MONTH_IDX, 
         "year2": END_YEAR, 
         "month2":END_MONTH_IDX, 
         "artcc": "_ALL", 
         "fmt": "csv"}
eprint(f"Performing GET request, this may take a moment...")
r = requests.get(f"{BASE_URL}", params=query, stream=True)
eprint(f"Request was {len(r.content)/1024/1024:.2f} Mb")

eprint("Attempting to create dataframe from csv file")

# Skips any malformatted lines while reading the csv
pireps = pd.read_csv(StringIO(r.text), on_bad_lines='skip')

eprint("Successfully created dataframe from CSV file")

eprint("Cleaning dataframe")
pireps['datetime'] = pd.to_datetime(pireps['VALID'], format='%Y%m%d%H%M')
pireps = pireps.drop(["ICING", "ATRCC", "VALID"], axis=1)
# Columns are named incorrectly, rename them as should be
# pireps = pireps.rename(columns={'PRODUCT_ID': 'LON', 'LON': 'LAT'})


def get_turb_intensity(row):
    """
        Purpose: 
           Gets the turbulence information from a pirep and returns it as an integer intensity value
         Arguments:
            row: The row of a pandas dataframe, i.e. a PIREP to get the turbulence intensity of 
         Return: An integer representation of the turbulence intensity or NaN 
    """
    turb_list = str(row['TURBULENCE']).replace('-', ' ').split()

     # Handle cases where PIREP contains two levels of turbulence without a dash
    if 'LGT' in turb_list and 'MOD' in turb_list:
         return 2
    
    if 'MOD' in turb_list and 'SEV' in turb_list:
         return 4
    
    if 'SEV' in turb_list and 'EXTRM' in turb_list:
         return 6

    # Define the turbulence intensity map
    turbulence_map = {
        'NONE': 0,
        'NEG': 0,
        'LGT': 1,
        'MOD': 3,
        'SEV': 5,
        'EXTRM': 7
    }

    # Check for NaN
    if 'nan' in turb_list:
            return np.nan
        
    # Look for turbulence intensity in the map
    for key in turbulence_map:
        if key in turb_list:
            return turbulence_map[key]
        
   
        
     # Return NaN if no known turbulence level is found
    return np.nan

eprint("Classifying levels of turbulence")
pireps['turbulence_intensity'] = pireps.apply(get_turb_intensity, axis = 1)
len_before_drop_na_turb = len(pireps)
only_turb_pireps = pireps.dropna(subset=['turbulence_intensity'])
eprint(f"We dropped {len_before_drop_na_turb - len(only_turb_pireps)}/{len_before_drop_na_turb} pireps with unknown turbulence intensity")


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


# DEPRECATED - Flight level now comes with PIREP
def get_flight_level(row):
    """
    Purpose:
        Gets the flight level from a pirep and returns it as an integer
    Arguments:
        row: The row of a pandas dataframe, i.e. a PIREP to get the flight level
            of
    Return:
        An integer representation of the flight level of the given pirep
    """


    # We want to get the part of the pirep that contains the flight level
    # The pirep comes in the following format:
    #    /FL XXXX /TP YYYY
    # Where FL = flight level, TP = aircraft type
    # So we find the indices of /FL and /TP and then find the substring after
    #   FL but before TP and then attempt to parse that as a flight level
    report = str(row['REPORT'])
    
    # If either FL or TP doesn't exist, our PIREP is faulty, so assume
    # flight level is nan
    if (('/FL' not in report) or ('/TP' not in report)):
        return np.nan

    fl_index = report.index('/FL')
    tp_index = report.index('/TP')
    
    flight_level_str = report[fl_index + 3:tp_index].strip()

    # If the flight level is an int, return 100 times it to convert to feet
    if (is_int(flight_level_str)):
        return int(flight_level_str) * 100
    
    # If the flight level has SFC (surface), DURC (during climb) or DURD (during descent),
    # return 0 as the flight level must be close to the ground
    elif ("DURC" in flight_level_str or "DURD" in flight_level_str or "SFC" in flight_level_str):
        return 0
    # If the flight level has a dash between 2 numbers (e.g., 040-120) average them
    elif ('-' in flight_level_str):
        dash_index = flight_level_str.index('-')
        if (is_int(flight_level_str[:dash_index]) and is_int(flight_level_str[dash_index + 1:])):
            fl = (int(flight_level_str[:dash_index]) + int(flight_level_str[dash_index + 1:])) * (100 / 2)
            return fl
    # If the flight level is unknown, return nan
    elif "UNKN" in flight_level_str or "UKN" in flight_level_str or "UNK" in flight_level_str:
        return np.nan
    # If we cannot parse the flight level, return nan
    else:
        eprint(f"    Unable to parse: {report}")
        return np.nan

# Currently we drop any pireps that contain NA values for flight level
eprint("Dropping pireps w/ NA flight level, latitude, or longitude")
len_before_drop_na_fl = len(only_turb_pireps)
only_turb_pireps_w_altitude = only_turb_pireps.dropna(subset=['FL', 'LAT', 'LON'])
eprint(f"We dropped {len_before_drop_na_fl - len(only_turb_pireps_w_altitude)}/{len_before_drop_na_fl} pireps with unknown flight level, latitude, or longitude")

#add plane weight classification into the dataframe
plane_weight_dict = pd.read_csv(os.path.join(DIRNAME, "../plane_weights/plane_weight_dictionary.csv"))
final_df = pd.merge(only_turb_pireps_w_altitude, plane_weight_dict, on = 'AIRCRAFT', how = 'left').drop(columns='Unnamed: 0')
final_df = final_df.rename(columns={"Turbulence_Category": "Plane Weight"})

eprint(f"Writing {len(final_df)} pireps with altitude and turbulence to csv")
# Output the cleaned pireps to either a csv output file or stdout


if OUTPUT != sys.stdout:
    filename = f"{DIRNAME}/clean_pirep_data/{YEAR}/{START_MONTH_IDX:02}_turb_pireps.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    eprint(f"Writing csv to output file: {filename}")
    OUTPUT = open(filename, "w")

csv_str = final_df.to_csv(None, index=False, columns=final_df.columns)
print(csv_str, file=OUTPUT)
OUTPUT.close()
eprint(f"Finished writing csv output")
