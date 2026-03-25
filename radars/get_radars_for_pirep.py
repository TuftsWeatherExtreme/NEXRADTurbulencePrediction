# get_radars_for_pirep.py
# Authors: Team Celestial Blue
# Last Modified: 5/6/25
# Purpose: This script takes an input pireps data file and cleans it to be a csv
#          with only the pireps which contain reports of turbulence, removing
#          extraneous columns, adding plane weight, and ensuring all location
#          data is valid
# Run with `python clean_pireps.py -month MONTH -year YEAR [-o {FILE/STDOUT}]`

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.spatial import cKDTree
from haversine import haversine
from botocore import UNSIGNED
from aiobotocore.config import AioConfig
import bisect
import asyncio
from aiobotocore.session import get_session
import sys
import os

from beam_geometry import score_radar_for_pirep, get_num_candidates

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december"]
            
# RADAR_DIRNAME = os.path.dirname(sys.argv[0])
# PIREP_DIRNAME = os.path.join(os.path.dirname(RADAR_DIRNAME), "pireps")

RADAR_DIRNAME = os.path.dirname(os.path.abspath(__file__))
PIREP_DIRNAME = os.path.join(os.path.dirname(RADAR_DIRNAME), "pireps")
EARTH_RADIUS_KM = 6371.0

NEXRAD_BUCKET = 'unidata-nexrad-level2'  # Migrated from noaa-nexrad-level2 (deprecated Sep 2025)

def get_file_time(filename, date):
    """
        Purpose: Gets the time of a given nexrad file and returns it as a datetime
        Arguments: 
            filename: The filename for a given nexrad file, will have _HHMMSS_
                in it to parse into the datetime
            date: The date containing the day, month, and year of the nexrad file
        Returns: A datetime representing when this pirep was created
    """
    # Example filenames below
    "YEAR/MONTH/DAY/SITE_CODE/{SITE_CODE}{YEAR}{MONTH}{DAY}_{HHMMSS}_VO6"
    "2024/12/05/KAKQ/KAKQ20241205_001256_V06"
    # Split to get just the {HHMMSS} part of the string and put these values in a datetime
    filetime = filename.split("_")[1]
    if (".tar" in filename or "MDM" in filename or filetime == "NEXRAD"):
        return None
    hour = int(filetime[:2])
    minute = int(filetime[2:4])
    second = int(filetime[4:6])
    return datetime(year=date.year, month=date.month, day=date.day, hour=hour, 
                    minute=minute, second=second)

# Defines helper function to calc the distance of pirep from radar in km
def haversine_km(lat1, lon1, lat2, lon2):
    """
    Great-circle distance between (lat1, lon1) and (lat2, lon2) in kilometers.
    Inputs in degrees.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def ft_to_meters(ft):
    return ft / 3.281

NUM_RADARS = 5

def find_candidate_sites(pirep, nexrad_tree, site_codes, site_coords, site_elevations):
    """
    Purpose: Given a PIREP, find the best 5 radar sites using altitude-aware
        beam geometry scoring. Pulls extra nearby candidates, scores each by
        how well their beams cover the PIREP altitude, and returns the top 5.
    Arguments:
        pirep - The data for a singular pirep
        nexrad_tree - The cKDTree containing the locations of all NEXRAD radar
            stations (in radians)
        site_codes - All of the site codes for the nexrad sites
        site_coords - Lat/lon coordinates of all NEXRAD sites
        site_elevations - Elevation (meters) of all NEXRAD sites
    Return:
        A tuple of the 5 best site codes, ordered by beam geometry score
    """
    altitude_ft = pirep['FL']
    altitude_m = ft_to_meters(altitude_ft)

    # Always pull extra candidates for scoring, then return the best 5
    num_to_query = max(NUM_RADARS, get_num_candidates(altitude_ft))

    # Query KDTree for the N nearest stations
    pirep_coord = np.radians([pirep['LAT'], pirep['LON']])
    _distances, indices = nexrad_tree.query(pirep_coord, k=num_to_query)

    # Score each candidate by beam geometry coverage at the PIREP altitude
    scored = []
    for idx in indices:
        site_lat, site_lon = site_coords[idx]
        distance_m = haversine(
            (pirep['LAT'], pirep['LON']),
            (site_lat, site_lon),
            unit='m'
        )
        score = score_radar_for_pirep(
            distance_m, altitude_m, site_elevations[idx]
        )
        scored.append((score, site_codes[idx]))

    # Sort by score descending, return the best 5
    scored.sort(key=lambda x: x[0], reverse=True)
    return tuple(code for _score, code in scored[:NUM_RADARS])

def get_closest_sites(pireps_df):
    """
    Purpose: This function determines the best candidate radar sites for
        each pilot report, using altitude-aware beam geometry scoring.
    Arguments:
        pireps_df - The dataframe containing all the pirep data
    Return:
        This function returns this dataframe after adding a column which
        contains the site codes of the candidate nexrad radar sites
    """
    nexrad_sites = pd.read_csv(f"{RADAR_DIRNAME}/nexrad_sites.csv")
    nexrad_coords = nexrad_sites[['Latitude', 'Longitude']].to_numpy()
    # Using radians here allows the cKDTree to treat as Euclidean which works pretty well
    nexrad_tree = cKDTree(np.radians(nexrad_coords))

    site_codes = nexrad_sites['Site Code'].to_numpy()
    site_elevations = ft_to_meters(nexrad_sites['Elevation'].to_numpy())

    pireps_df['nexrad_sites'] = pireps_df.apply(
        find_candidate_sites, axis=1,
        args=(nexrad_tree, site_codes, nexrad_coords, site_elevations)
    )
    return pireps_df


def get_nexrad_basename(filename):
    """
    Purpose: Given a file object as returned from a call to s3.list_objects_v2,
        this returns the base name of the nexrad file
    Arguments:
        filename - A filename from an object returned from a call to 
            s3.list_objects_v2
    Returns:
        A string representing the basename of the file object
    """
    filename = filename.rsplit("/", 1)[-1]
    # Remove _MDM from the end of the file if it exists
    if "_MDM" in filename:
        filename = filename[:filename.index("_MDM")]
    return filename

async def s3_list_nexrad_files(date: datetime, site: str, session) -> tuple:
    """
        Purpose: This function accepts a particular date, nexrad radar site,
            and aiobotocore session object and performs a list_objects_v2
            on the unidata-nexrad-level2 bucket looking for all radar files
            on the given date for that specific site.
        Arguments:
            date - A datetime object containing the year, month, and day to find
                radar data for
            site - The site code of the radar object to find data for
            session - An aiobotocore session object with which to query s3
        Return: A tuple where...
            The first element is a tuple of (date, site)
            The second element is a list of all the nexrad filetimes for that
                date and site
    """
    prefix = f"{date.year}/{date.month:02}/{date.day:02}/{site}"
    async with session.create_client('s3', region_name='us-east-1', config=AioConfig(signature_version=UNSIGNED)) as s3:
        try:
            response = await s3.list_objects_v2(Bucket=NEXRAD_BUCKET, Prefix=prefix)
            files = response.get("Contents", [])
            filetimes = []
            if len(files) != 0:
                # Generate a list of (datetimes, nexrad filename) for all listed objects with valid file times
                filetimes = [(dt, get_nexrad_basename(file['Key'])) for file in files if (dt := get_file_time(file['Key'], date)) is not None]
            return ((date, site), filetimes)
        except Exception as e:
            print(f"ERROR: Failed to access S3 bucket for {site} on {date}: {type(e).__name__}: {e}")
            return ((date, site), [])


async def batch_list_nexrad_times(unique_requests: set) -> dict:
    """
        Purpose: Batch fetches available NEXRAD file times for multiple sites and datetimes.
        Arguments:
            unique_requests - A set of all the unique date/site pairings we'll
                need to query s3 for. This limits the required number of
                list_objects_v2 calls required for efficiency
        Return: A map where...
            key - a unique (date, site) pair
            value - a list of all nexrad filetimes for that (date, site) pair
        Note:
            This function uses asyncio for all of the queries since the slowdown
            here is from all of the calls to s3
    """
    results = {}
    session = get_session()
    tasks = [s3_list_nexrad_files(date, site, session) for date, site in unique_requests]
    

    # As tasks complete, collect results
    for count, task in enumerate(asyncio.as_completed(tasks), start=1):
        (date, site), result = await task
        results[(date, site)] = result

    return results


def generate_unique_requests(pireps_df: pd.DataFrame) -> set:
    """
        Purpose: Generates all of the unique (day, site) pairs from the given
            pireps_df to precompute all the s3 requests we'll need to avoid
            repeated requests. For each pirep on a particular day, we add a
            unique request for the prior day, the current day, and the next day
        Arguments: 
            pireps_df - The dataframe containing all the pirep data
        Return:
            A set of all the unique requests we'll need to query the s3 bucket
    """
    unique_requests = {(day, site_code) 
                       for sites, dt in zip(pireps_df['nexrad_sites'], pireps_df['datetime']) 
                       for site_code, dist_km in sites 
                       for day in {dt.date(), (dt - timedelta(days=1)).date(), (dt + timedelta(days=1)).date()}}
    eprint(f"About to perform {len(unique_requests)} list_objects_v2 requests")
    return unique_requests
    


def nearest_time(times: list, pirep_dt: datetime) -> datetime:
    """
        Purpose: Returns the nearest time to pirep_time in times (in the past, within ±30 minutes)
        Arguments:
            times: A list of all times of nexrad files for the previous day
                current day, and next day. Must be sorted by time.
            pirep_dt: The actual time of the pirep
        Returns: The nearest time to pirep_dt that a nexrad data file was
            generated at (in the past, within ±30 minutes window)
        Note: Returns None if no valid time found within window

    """
    if len(times) == 0:
        return None
    
    idx = bisect.bisect_left([time[0] for time in times], pirep_dt)
    if idx == 0:
        # All times are in the future, check if first one is within window
        first_time = times[0]
        time_diff = abs((pirep_dt - first_time[0]).total_seconds())
        if time_diff <= 1800:  # 30 minutes = 1800 seconds
            return first_time
        return None
    
    # Get the closest time in the past
    prev_time = times[idx - 1]
    
    # Verify it's in the past and within ±30 minutes window
    time_diff = abs((pirep_dt - prev_time[0]).total_seconds())
    if prev_time[0] <= pirep_dt and time_diff <= 1800:
        return prev_time
    
    # If prev_time is too far in the past, check if next time (if exists) is within window
    if idx < len(times):
        next_time = times[idx]
        time_diff = abs((pirep_dt - next_time[0]).total_seconds())
        if time_diff <= 1800:
            return next_time
    
    return None


def get_closest_nexrad_files(pireps_df: pd.DataFrame, nexrad_times_dict: dict):
    """
        Purpose: Adds the actual s3 bucket paths for the closest nexrad
            scan (in time) to each pilot report to each row of the df.
            PIREPs with no radar within 150 km are dropped from the dataframe.
        Arguments:
            pireps_df - The dataframe containing all the pirep data
            nexrad_times_dict - A dictionary indexed by date and site code that
                contains all the file times for that nexrad site and that date
        Return:
            Nothing; modifies pireps_df in place: adds 'aws_files' column and
            removes rows that have no valid radar within the distance threshold.
    
    """
    all_radars = []
    missing = 0
    skipped_no_threshold = 0
    DISTANCE_THRESHOLD_KM = 150.0
    
    for index, pirep in pireps_df.iterrows():
        pirep_dt = pirep['datetime']
        radars = list()
        
        # Iterate sites in distance order (already sorted by find_5_closest_sites)
        for site_code, dist_km in pirep['nexrad_sites']:
            # Skip sites beyond distance threshold
            if dist_km > DISTANCE_THRESHOLD_KM:
                continue
            
            # times will store all possible times of nexrad files nearby
            times = list()

            # When we're close to another day, add that day's times too (in order)
            if (pirep_dt - timedelta(minutes=30)).day != pirep_dt.day:
                times += nexrad_times_dict.get(((pirep_dt - timedelta(minutes=30)).date(), site_code), [])
            times += nexrad_times_dict.get((pirep_dt.date(), site_code), [])
            if (pirep_dt + timedelta(minutes=30)).day != pirep_dt.day:
                times += nexrad_times_dict.get(((pirep_dt + timedelta(minutes=30)).date(), site_code), [])
            
            # Sort times by datetime before calling nearest_time
            times.sort(key=lambda x: x[0])
            
            if len(times) == 0:
                missing += 1
                continue
            
            nearest_result = nearest_time(times, pirep_dt)
            if nearest_result is None:
                missing += 1
                continue
            
            nexrad_dt, file_ending = nearest_result
            prefix=f"{nexrad_dt.year}/{nexrad_dt.month:02}/{nexrad_dt.day:02}/{file_ending[:4]}"
            aws_nexrad_level2_file = f"s3://unidata-nexrad-level2/{prefix}/{file_ending}"
            radars.append(aws_nexrad_level2_file)

        # Track PIREPs with no radar within threshold
        if len(radars) == 0:
            skipped_no_threshold += 1
        all_radars.append(radars)
    eprint(f"There were {missing} sites missing data")
    eprint(f"There were {skipped_no_threshold} PIREPs with no radar within {DISTANCE_THRESHOLD_KM} km threshold (removed)")
    pireps_df['aws_files'] = all_radars
    # Drop PIREPs with no radar within threshold:
    pireps_df.drop(pireps_df[pireps_df['aws_files'].apply(len) == 0].index, inplace=True)



def usage():
    eprint(f"Usage: {sys.argv[0]} [-month MONTH] [-year YEAR] [-o {{FILE/STDOUT}}]")
    eprint("If month and year not specified, expects a csv file on stdin")
    exit(1)


def read_command_line_args():
    month_str = None
    year = None
    month_idx = None
    output = None
    read_stdin = False
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
                if year not in range(2003, 2027):
                    eprint(f"Invalid year: {year} not in range [2003, 2026]")
                    usage()
        elif sys.argv[i] == "-o":
            i += 1
            if i < len(sys.argv):
                output = sys.argv[i]
                if output.lower() == "stdout":
                    output = sys.stdout
                elif output.lower() != "file":
                    eprint("Unknown output. Must be one of [STDOUT, FILE]")
                    usage()
                
        else:
            eprint(f"Unexpected command line arg: {sys.argv[i]}")
            usage()
        i += 1
    
    if (month_str is None or year is None):
        read_stdin = True
    else:
        month_idx = MONTHS.index(month_str.lower()) + 1

    return read_stdin, year, month_idx, output

# Prints to stderr
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

async def main():
    read_stdin, year, month_idx, output = read_command_line_args()
    if read_stdin:
        eprint(f"{sys.argv[0]} Waiting to read csv from stdin...")
    else:
        eprint(f"Beginning to get radars for pirep from {MONTHS[month_idx - 1]} {year}")
    pirep_file = sys.stdin if read_stdin else f"{PIREP_DIRNAME}/clean_pirep_data/{year}/{month_idx:02}_turb_pireps.csv"
    pireps_df = pd.read_csv(pirep_file)
    eprint(f"Successfully read in df of pireps:")
    eprint(pireps_df.head(5))
    get_closest_sites(pireps_df)

    pireps_df['datetime'] = pd.to_datetime(pireps_df['datetime'])
    if read_stdin:
        year = pireps_df['datetime'][0].year
        month_idx = pireps_df['datetime'][0].month
    unique_requests = generate_unique_requests(pireps_df)


    # Takes around 5 minutes to process
    eprint("Getting times for all nexrad files. This may take up to 5 minutes")
    eprint(f"About to perform {len(unique_requests)} requests to S3 bucket...")
    nexrad_times_dict = await batch_list_nexrad_times(unique_requests)
    eprint("Processing complete")

    get_closest_nexrad_files(pireps_df, nexrad_times_dict)
    # Output the df to csv for ease of access
    if output == None or (type(output) == str and output.lower()) == "file":
        output = f"{RADAR_DIRNAME}/pirep_with_radar_data/{year}/{month_idx:02}.csv"
        os.makedirs(os.path.dirname(output), exist_ok=True)
        eprint(f"Outputting to csv file: {output}")
    else:
        eprint("Printing csv to stdout")
    pireps_df.to_csv(output, index=False)
    eprint("Done!")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
