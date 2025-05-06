import numpy as np
import pandas as pd
from io import StringIO
import requests
import certifi
from bs4 import BeautifulSoup
import ssl



print(certifi.where())

# start_time = time.time()


# def get_days_in_month(month, year):
#     """Returns the number of days in a given month of a given year."""
#     if month in {1, 3, 5, 7, 8, 10, 12}: 
#         return 31
#     elif month in {4, 6, 9, 11}: 
#         return 30
#     elif month == 2:  
#         return 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
#     return 30  # Fallback, should not be needed

def get_site_url(month, day, year, time, code):

    try:
        # Ensure time is midnight or midday
        if time != 0 and time != 12:
            raise ValueError("Invalid time!")

        # Ensure month is valid
        if month < 1 or month > 12:
            raise ValueError("Invalid month!")

        # Ensure day is valid
        if day < 1 or day > 31:
            raise ValueError("Invalid day!")

        # Ensure year is valid 
        if year < 1973 or year > 2025:
            raise ValueError("Invalid year!")

    except ValueError as e:
        print(f"Error: {e}")

    #pad month, day and time to two digits
    month = "{:02}".format(month)
    day = "{:02}".format(day)
    time = "{:02}".format(time)

    
    # code = str(code).upper()
    # get site number from dataframe
    # site_number = codes[codes['code'] == code]['number'].values[0]

    # TODO: figure this out 
    # radar codes csv is stations, number
    # stations.csv has stations but no numbers
    site_number = str(code)
    print("SITE NUMBER: ", site_number)


    site_url = f'https://weather.uwyo.edu/cgi-bin/sounding?region=naconf&TYPE=TEXT%3ALIST&YEAR={year}&MONTH={month}&FROM={day}{time}&TO={day}{time}&STNM={site_number[0:5]}'

    return site_url




cache = {}

def fetch_table_from_website(url):

    # ssl._create_default_https_context = ssl._create_unverified_context  # Bypass SSL verification

    # ssl_context = ssl.create_default_context(cafile=certifi.where())  # Use certifi certificates
    response = requests.get(url, verify=False)  # Explicitly verify using certifi

    if url in cache:
        return cache[url]

    soup = BeautifulSoup(response.text, 'html.parser')
    pre_tag = soup.find('pre')

    if pre_tag:
        raw_text = pre_tag.get_text()
        table = get_table(raw_text)
        cache[url] = table  # Store in cache
        return table
    else:
        print("No <pre> tag found on the page.")
        return None



def get_table(text):
        
    # Remove any separators and whitespace lines
    cleaned_data = "\n".join([line.strip() for line in text.splitlines() if line.strip() and not line.startswith('---')])

    # Use StringIO to simulate a file object
    data = StringIO(cleaned_data)

    # Define the column names and read the data into a DataFrame
    columns = ["PRES", "HGHT", "TEMP", "DWPT", "RELH", "MIXR", "DRCT", "SKNT", "THTA", "THTE", "THTV"]

    # Load the data into a DataFrame
    df = pd.read_csv(data, sep=r'\s+', header=0, names=columns)

    # drop first row as it is extraneous
    df = df.iloc[1:]

    return df


def calculate_horizontal_wind_shear(row, df):
    import numpy as np

    if row.name == 0:
        return None

    # Ensure numeric types
    try:
        orig_height = float(row['HGHT'])
        orig_wind_speed = float(row['SKNT'])
        orig_wind_direction = float(row['DRCT'])
    except ValueError:
        return None  # Skip invalid rows

    #get previous row in dataframe
    prev_row = df.loc[row.name - 1] if (row.name - 1) in df.index else None
    if prev_row is None:
        return None

    #find height, speed and direction
    prev_height = float(prev_row['HGHT'])
    prev_wind_speed = float(prev_row['SKNT'])
    prev_wind_direction = float(prev_row['DRCT'])

    # Convert wind directions to U/V components
    orig_U = orig_wind_speed * np.cos(np.radians(orig_wind_direction))
    orig_V = orig_wind_speed * np.sin(np.radians(orig_wind_direction))

    prev_U = prev_wind_speed * np.cos(np.radians(prev_wind_direction))
    prev_V = prev_wind_speed * np.sin(np.radians(prev_wind_direction))

    # Compute wind shear magnitude
    wind_shear = np.sqrt((orig_U - prev_U) ** 2 + (orig_V - prev_V) ** 2) / (((orig_height - prev_height) / 1000) + 1e-5)

    return wind_shear

def get_measurement_coordinates(station_code):
    try:
        # Enforce station code as a string
        station_code = str(station_code)

        # Get stations df
        stations_df = pd.read_csv('./stations.csv')

        # Find row where station code is a substring of that code
        match_row = stations_df[stations_df['Station_ID'].str.contains(station_code, na=False)].iloc[0]

        # Extract latitude and longitude
        latitude = match_row['Latitude']
        longitude = match_row['Longitude']

        return latitude, longitude
    except Exception as e:
        # Return NaN if the station code is not found or any error occurs
        return np.nan, np.nan
    
#this function takes a latitude, longitude and stations dataframe and find the nearest code and distance
def find_nearest_station(latitude, longitude, stations_df):
    # Create a tuple for the input latitude and longitude
    input_location = (latitude, longitude)

    # List to store differences
    lat_diffs = []
    lon_diffs = []

    # Loop through the stations dataframe and calculate the difference for each station
    for _, row in stations_df.iterrows():
        station_location = (row['Latitude'], row['Longitude'])
        # Calculate the differences in latitude and longitude
        lat_diff = abs(input_location[0] - station_location[0])
        lon_diff = abs(input_location[1] - station_location[1])
        
        lat_diffs.append(lat_diff)
        lon_diffs.append(lon_diff)
    
    # Find the index of the minimum latitude and longitude difference
    nearest_station_index = np.argmin(np.array(lat_diffs) + np.array(lon_diffs))  # Min based on both diffs

    # Get the station code and differences
    nearest_station_code = stations_df.iloc[nearest_station_index]['station_number']
    nearest_lat_diff = lat_diffs[nearest_station_index]
    nearest_lon_diff = lon_diffs[nearest_station_index]
    
    return nearest_station_code, nearest_lat_diff, nearest_lon_diff


def find_nearest_sounding(lat, lon, alt, month, day, year, time):
    # Determine the closest valid time (00 or 12 UTC)
    if time >= 12:
        sounding_time = 12
    else:
        sounding_time = 0
    
    # get time difference between logged time and sounding
    delta_t = time - sounding_time
    
    # Find station code based on nearest latitude and longitude
    station_code, lat_distance, long_distance = find_nearest_station(lat, lon, codes)
    
    # Get associated url based on adjusted time
    url = get_site_url(month, day, year, sounding_time, station_code)
    
    return url

    # # Get table from website by generated url
    # table = fetch_table_from_website(url)

    # #apply function to calculate wind shear
    # table['wind_shear'] = table.apply(lambda row: calculate_horizontal_wind_shear(row, table), axis=1)
    
    # # Ensure 'HGHT' is numeric, and coerce any non-numeric values to NaN
    # table['HGHT'] = pd.to_numeric(table['HGHT'], errors='coerce')
    
    # # Filter out rows with NaN values in critical columns (e.g., TEMP, DWPT, RELH, etc.)
    # table_clean = table.dropna(subset=['TEMP', 'DWPT', 'RELH', 'MIXR', 'DRCT', 'SKNT', 'THTA', 'THTE', 'THTV'])

    # # Ensure HGHT is a sorted NumPy array for fast lookup
    # heights = table_clean['HGHT'].to_numpy()

    # # Find the closest altitude index using binary search
    # idx = np.searchsorted(heights, alt)

    # # Handle edge cases where altitude is out of range
    # if idx == 0:
    #     closest_idx = 0
    # elif idx == len(heights):
    #     closest_idx = len(heights) - 1
    # else:
    #     # Check which of the two nearest values is closer
    #     before = heights[idx - 1]
    #     after = heights[idx]
    #     closest_idx = idx - 1 if abs(before - alt) < abs(after - alt) else idx

    # closest_row = table_clean.iloc[closest_idx]

    # # Calculate height difference
    # height_difference = abs(closest_row['HGHT'] - alt)

    # # If the difference is larger than expected, log a warning
    # if height_difference > 10000:  
    #     print(f"Warning: Large altitude difference {height_difference} feet for requested altitude {alt} feet.")
    
    # return closest_row, delta_t, lat_distance, long_distance, height_difference

    
# find_nearest_sounding(48.97, -93.38, 10000, 2, 12, 2024, 0)

# #end time and print
# end_time = time.time()

# elapsed_time = end_time - start_time

# print("Elapsed time:", elapsed_time)

def get_horizontal_wind_shear(closest_row, previous_row):

    orig_height = float(closest_row['HGHT'])
    orig_wind_speed = float(closest_row['SKNT'])
    orig_wind_direction = float(closest_row['DRCT'])

    prev_height = float(previous_row['HGHT'])
    prev_wind_speed = float(previous_row['SKNT'])
    prev_wind_direction = float(previous_row['DRCT'])

    # Convert wind directions to U/V components
    orig_U = orig_wind_speed * np.cos(np.radians(orig_wind_direction))
    orig_V = orig_wind_speed * np.sin(np.radians(orig_wind_direction))

    prev_U = prev_wind_speed * np.cos(np.radians(prev_wind_direction))
    prev_V = prev_wind_speed * np.sin(np.radians(prev_wind_direction))

    # Compute wind shear magnitude
    wind_shear = np.sqrt((orig_U - prev_U) ** 2 + (orig_V - prev_V) ** 2) / (((orig_height - prev_height) / 1000) + 1e-5)

    return wind_shear


