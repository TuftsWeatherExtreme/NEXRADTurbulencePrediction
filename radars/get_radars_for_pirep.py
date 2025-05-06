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
import bisect
import asyncio
from aiobotocore.session import get_session
import sys
import os

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december"]
RADAR_DIRNAME = os.path.dirname(sys.argv[0])
PIREP_DIRNAME = os.path.join(os.path.dirname(RADAR_DIRNAME), "pireps")

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
    filetime = filename['Key'].split("_")[1]
    if (len(filename) > 30 or filetime == "NEXRAD"):
        return None
    hour = int(filetime[:2])
    minute = int(filetime[2:4])
    second = int(filetime[4:6])
    return datetime(year=date.year, month=date.month, day=date.day, hour=hour, 
                    minute=minute, second=second)


# Define a helper function to find the closest sites
def find_5_closest_sites(pirep, nexrad_tree, site_codes):
    """
    Purpose: Given a row of a dataframe which contains data about a pirep,
        this function queries the tree of nexrad sites to find the 5 closest
        sites. 
        This function is meant to be applied to a pandas df
    Arguments:
        pirep - The data for a singular pirep 
        nexrad_tree - The cKDTree containing the locations of all NEXRAD radar
            stations
        site_codes - All of the site codes for the nexrad sites in the nexrad tree
    Return: 
        A 5-tuple of the site codes for these closest sites are returned
    """
    pirep_coord = np.radians([pirep['LAT'], pirep['LON']])
    _distances, indices = nexrad_tree.query(pirep_coord, k=5)    
    return tuple(site_codes[indices])

def get_closest_sites(pireps_df):
    """
    Purpose: This function determines which radar sites are the 5 closest to
        each pilot report in the pireps_df argument
    Arguments:
        pireps_df - The dataframe containing all the pirep data
    Return:
        This function returns this dataframe after adding a column which
        contains the site codes of the 5 closest nexrad radar sites
    """
    nexrad_sites = pd.read_csv(f"{RADAR_DIRNAME}/nexrad_sites.csv")
    nexrad_coords = nexrad_sites[['Latitude', 'Longitude']].to_numpy()
    # Using radians here allows the cKDTree to treat as Euclidean which works pretty well
    nexrad_tree = cKDTree(np.radians(nexrad_coords))

    site_codes = nexrad_sites['Site Code'].to_numpy()
    # Apply the helper function to find closest sites
    pireps_df['nexrad_sites'] = pireps_df.apply(find_5_closest_sites, axis=1, args=(nexrad_tree, site_codes))
    return pireps_df


def get_nexrad_basename(file):
    """
    Purpose: Given a file object as returned from a call to s3.list_objects_v2,
        this returns the base name of the nexrad file
    Arguments:
        file - A file object returned from a call to s3.list_objects_v2
    Returns:
        A string representing the basename of the file object
    """
    filename = file["Key"].rsplit("/", 1)[-1]
    # Remove _MDM from the end of the file if it exists
    if "_MDM" in filename:
        filename = filename[:filename.index("_MDM")]
    return filename

async def s3_list_nexrad_files(date: datetime, site: str, session) -> tuple:
    """
        Purpose: This function accepts a particular date, nexrad radar site,
            and aiobotocore session object and performs a list_objects_v2
            on the noaa-nexrad-level2 bucket looking for all radar files
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
    async with session.create_client('s3', region_name='us-east-1') as s3:
        response = await s3.list_objects_v2(Bucket='noaa-nexrad-level2', Prefix=prefix)
        files = response.get("Contents", [])
        filetimes = []
        if len(files) != 0:
            # Generate a list of (datetimes, nexrad filename) for all listed objects with valid file times
            filetimes = [(dt, get_nexrad_basename(file)) for file in files if (dt := get_file_time(file, date)) is not None]
        
        return ((date, site), filetimes)
        # return ((date, site), [(get_file_time(file, date), file["Key"].rsplit("/", 1)[-1]) for file in files if (dt := get_file_time(file, date))]) if files else ((date, site), [])


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
    unique_requests = {(day, site) 
                       for sites, dt in zip(pireps_df['nexrad_sites'], pireps_df['datetime']) 
                       for site in sites 
                       for day in {dt.date(), (dt - timedelta(days=1)).date(), (dt + timedelta(days=1)).date()}}
    eprint(f"About to perform {len(unique_requests)} list_objects_v2 requests")
    return unique_requests
    


def nearest_time(times: list, pirep_dt: datetime) -> datetime:
    """
        Purpose: Returns the nearest time to pirep_time in times
        Arguments:
            times: A list of all times of nexrad files for the previous day
                current day, and next day
            pirep_dt: The actual time of the pirep
        Returns: The nearest time to pirep_dt that a nexrad data file was
            generated at

    """
    idx = bisect.bisect_left([time[0] for time in times], pirep_dt)
    if idx >= len(times) - 1:
        return times[-1]
    # Safe to subtract one because we've added the whole previous day if its close
    prev_time = times[idx - 1]
    _next_time = times[idx][0] # deprecated, we now only get the closest time in the past
    return prev_time # Just return prev_time - OLD: if pirep_dt - prev_time <= next_time - pirep_dt else next_time





def get_closest_nexrad_files(pireps_df: pd.DataFrame, nexrad_times_dict: dict):
    """
        Purpose: Adds the actual s3 bucket paths for the closest nexrad
            scan (in time) to each pilot report to each row of the df
        Arguments:
            pireps_df - The dataframe containing all the pirep data
            nexrad_times_dict - A dictionary indexed by date and site code that
                contains all the file times for that nexrad site and that date
        Return:
            Nothing, but adds a row to pireps_df with the 5 'aws_files' 
    
    """
    all_radars = []
    missing = 0
    for index, pirep in pireps_df.iterrows():
        pirep_dt = pirep['datetime']
        radars = list()
        for site_code in pirep['nexrad_sites']:
            # times will store all possible times of nexrad files nearby
            times = list()

            # When we're close to another day, add that day's times too (in order)
            if (pirep_dt - timedelta(minutes=30)).day != pirep_dt.day:
                times += nexrad_times_dict[((pirep_dt - timedelta(minutes=30)).date(), site_code)]
            times += nexrad_times_dict[(pirep_dt.date(), site_code)]
            if (pirep_dt + timedelta(minutes=30)).day != pirep_dt.day:
                times += nexrad_times_dict[((pirep_dt + timedelta(minutes=30)).date(), site_code)]
            
            if len(times) == 0:
                missing += 1
            else:
                nexrad_dt, file_ending = nearest_time(times, pirep_dt)

                prefix=f"{nexrad_dt.year}/{nexrad_dt.month:02}/{nexrad_dt.day:02}/{file_ending[:4]}"
                aws_nexrad_level2_file = f"s3://noaa-nexrad-level2/{prefix}/{file_ending}"
                radars.append(aws_nexrad_level2_file)
            # break # Uncomment to only add the closest NEXRAD file

        all_radars.append(radars)
    eprint(f"There were {missing} sites missing data")
    pireps_df['aws_files'] = all_radars



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
                if year not in range(2003, 2026):
                    eprint(f"Invalid year: {year} not in range [2003, 2025]")
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
    eprint(f"About to perform {len(nexrad_times_dict)} requests to S3 bucket...")
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

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
loop.run_until_complete(main())
