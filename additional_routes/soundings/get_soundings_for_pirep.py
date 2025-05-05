import pandas as pd
import numpy as np  # Import numpy for NaN values
import soundings_investigation
import cProfile
import time
import asyncio
import aiohttp
from bs4 import BeautifulSoup


# Also updates the pirep dataframe with latitude, longitude, delta_t
def get_urls_from_pirep_df(pirep_df, stations_df): 

    urls = []
    
    for i in range(len(pirep_df)):
        #access necessary variables from row
        longitude = pirep_df.iloc[i]['LON']
        latitude = pirep_df.iloc[i]['LAT']

        dt = pd.to_datetime(pirep_df.iloc[i]['datetime'])
        year = dt.year
        month = dt.month
        day = dt.day
        hour = dt.hour  

        #convert time to sounding time
        if hour >= 12:
            sounding_time = 12
        else:
            sounding_time = 0

        #find time delta from recorded time to dounsing
        delta_t = hour - sounding_time
    
        station_code, lat_distance, long_distance = soundings_investigation.find_nearest_station(latitude, longitude, stations_df)
        
        url = soundings_investigation.get_site_url(month, day, year, sounding_time, station_code)

        #add url to list
        urls.append(url)

        #add other fields directly to dataframe
        pirep_df.iloc[i]['latitude_distance'] = lat_distance
        pirep_df.iloc[i]['longitude_distance'] = long_distance
        pirep_df.iloc[i]['delta_t'] = delta_t

    return urls
            

async def add_soundings_to_pirep_df(pirep_df, urls):
    # repeated calls to get_table_single_pirep, and if any of them must wait / sleep then the next one will get called
    async with aiohttp.ClientSession() as session:
        tasks = [get_table_from_url(i, session, url, pirep_df.at[i, 'FL']) for i, url in enumerate(urls)]
        results = await asyncio.gather(*tasks)
        
        # add empty cols to dataframe to write in later
        pirep_df['wind_shear'] = ''
        pirep_df['temperature'] = ''
        pirep_df['altitude_distance'] = ''
        
        for result in results:
            i = result['index']  # Get the row index
            
            for key in result:
                if key != 'index':  
                    pirep_df.at[i, key] = result[key]  # Assign values to the DataFrame
        
    return pirep_df
    
async def get_table_from_url(i, session, url, alt):
    print(f"Starting task {i}: {url}")
    results = dict()
    results['index'] = i
    async with session.get(url, verify_ssl=False) as response:
        table_as_html = await response.text()
        
        # parse the sounding table data out
        soup = BeautifulSoup(table_as_html, 'html.parser')
        pre_tag = soup.find('pre')
        table = None
        if pre_tag:
            raw_text = pre_tag.get_text()
            table = soundings_investigation.get_table(raw_text)
        else:
            print("No <pre> tag found on the page.")
            results['wind_shear'] = np.nan
            results['temperature'] = np.nan
            results['altitude_distance'] = np.nan
            return results     
        # Filter out rows with NaN values in critical columns (e.g., TEMP, DWPT, RELH, etc.)
        table = table.dropna(subset=['TEMP', 'DWPT', 'RELH', 'MIXR', 'DRCT', 'SKNT', 'THTA', 'THTE', 'THTV'])

        # Ensure HGHT is a sorted NumPy array for fast lookup
        heights = table['HGHT'].to_numpy().astype(int)

        # Find the closest altitude index using binary search
        idx = np.searchsorted(heights, int(alt)) # TODO

        # Handle edge cases where altitude is out of range
        if idx == 0:
            closest_idx = 0
        elif idx == len(heights):
            closest_idx = len(heights) - 1
        else:
            # Check which of the two nearest values is closer
            before = heights[idx - 1]
            after = heights[idx]
            closest_idx = idx - 1 if abs(before - alt) < abs(after - alt) else idx

        #get closest row and previous row for wind shear calculation
        closest_row = table.iloc[closest_idx]
        previous_row = table.iloc[closest_idx - 1] # TODO: account for when closest row is first row
        results['wind_shear'] = soundings_investigation.get_horizontal_wind_shear(closest_row, previous_row)
        results['temperature'] = closest_row['TEMP']
        results['altitude_distance'] = abs(int(closest_row['HGHT']) - alt)
        
        print(f"Finished task: {url}")
        return results

async def load_df_add_soundings(file_path):
    # Load PIREP CSV file
    pirep_df = pd.read_csv(file_path).head(50)
    stations_df = pd.read_csv('./stations.csv')

    print(f'Length of pirep: {len(pirep_df)}')

    # Apply add_soundings and expand dictionary into separate columns for the first 50 rows
    urls = get_urls_from_pirep_df(pirep_df, stations_df)
    pirep_df = await add_soundings_to_pirep_df(pirep_df, urls)

    # soundings_df = pirep_df.apply(add_soundings, axis=1).apply(pd.Series)

    # Concatenate original DataFrame with new sounding data
    # pirep_with_soundings = pd.concat([pirep_df.head(50), soundings_df], axis=1)

    return pirep_df

def add_soundings(row, max_retries=2):

    #if current row is divisible by 100, print row number
    if int(row.name) % 100 == 0:
        print(f'Current row: {row.name}')
    # Extract latitude, longitude, and altitude
    altitude = row['FL']
    latitude = round(row['LAT'], 3)
    longitude = round(row['LON'], 3)

    # Convert datetime
    dt = pd.to_datetime(row['datetime'])
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour  

    for attempt in range(max_retries):
        try:
            closest_row, delta_t, lat_distance, long_distance, altitude_distance = soundings_investigation.find_nearest_sounding(
                latitude, longitude, altitude, month, day, year, hour
            )

            if closest_row is None or closest_row.empty:
                print("Warning: No valid sounding data found. Retrying...")
            else:
                wind_shear = closest_row.get('wind_shear', np.nan)
                temperature = closest_row.get('TEMP', np.nan)
                return {
                    "delta_t": delta_t,
                    "lat_distance": lat_distance,
                    "long_distance": long_distance,
                    "altitude_distance": altitude_distance,
                    "wind_shear": wind_shear,
                    "temperature": temperature
                }
        except Exception as e:
            print(f"Error in finding soundings: {e}. Retrying in {2**attempt} seconds...")
            time.sleep(2 ** attempt)

    print("Error: Failed to fetch valid sounding data after multiple attempts.")
    return {
        "delta_t": np.nan,
        "lat_distance": np.nan,
        "long_distance": np.nan,
        "altitude_distance": np.nan,
        "wind_shear": np.nan,
        "temperature": np.nan
    }

import cProfile
import pstats

async def profile_code():
    pirep_file = '../pireps/clean_pirep_data/2023/01_turb_pireps.csv'
    pirep_with_soundings = await load_df_add_soundings(pirep_file) 
    pirep_with_soundings.to_csv('./test_pirep.csv')
    return pirep_with_soundings

if __name__ == "__main__":

    profiler = cProfile.Profile()
    profiler.enable()

    asyncio.run(profile_code())  # Run your main function here

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats("cumulative").print_stats(20)  # Prints the top 20 slowest function calls

# start_time = time.time()
# # Load PIREP data with added sounding columns (only first 50 rows)
# pirep_with_soundings = load_df_add_soundings('../pireps/clean_pirep_data/2023/01_turb_pireps.csv')

# #dump to csv 
# pirep_with_soundings.to_csv('./test_pirep.csv')

# end_time = time.time()

# elapsed_time = end_time - start_time

# print("Elapsed time for entire pirep:", elapsed_time)

