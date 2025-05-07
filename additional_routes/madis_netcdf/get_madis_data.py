# Team Celestial Blue
# Last Mododified: 04/25/2025
# Description: Obtain a netCDF4 Dataset from the MADIS FTP server, 
# extract the data for a specific time and location, and print the closest 
# points with their details.
# Usage: python get_madis_data.py <minute> <hour> <day> <month> <year> <latitude> <longitude> <altitude>
# Example: python get_madis_data.py 30 14 26 2 2025 37.7 -122.4 1200


from ftplib import FTP
import gzip
import shutil
import numpy as np
import netCDF4
import sys
from haversine import haversine


def get_data(args):
    # FUTURE: use a Datetime object to handle edge cases such as 12/31 at 11:59 PM

    minute = args[1].zfill(2)  # Ensure two-digit format

    # Represent 5AM as 0500
    hour = int(args[2])
    if hour < 10:
        hour = '0' + str(hour) + '00'
    else:
        hour = str(hour) + '00'

    day = args[3].zfill(2)
    month = args[4].zfill(2)
    year = int(args[5])
    print(f"minute: {minute}, hour: {hour}, day: {day}, month: {month}, year: {year}")

    # Gets us started in path /madisPublic1/data/
    ftp_server = "madis-data.ncep.noaa.gov" 

    # Create an FTP object and connect to the server
    ftp = FTP(ftp_server)
    ftp.login('anonymous', '')
    print(f"Connected to FTP server {ftp_server}")

    # Change to correct directory
    path = f'/archive/{year}/{month}/{day}/point/acars/netcdf'
    ftp.cwd(path)
    print(f"Changed to directory {path}")

    # Download the file
    filename = f'{year}{month}{day}_{hour}.gz'
    with open(filename, 'wb') as f:
        ftp.retrbinary(f"RETR {filename}", f.write)
    print(f"Downloaded {filename} successfully.")

    # Remove the .gz extension to get the original filename
    unzipped_file = filename[:-3]

    # Now, gunzip the .gz file 
    with gzip.open(filename, 'rb') as f_in:
        with open(unzipped_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Decompressed {filename} into {unzipped_file}")

    # Close the connection
    ftp.quit()
    print("FTP connection closed.")

    # Return a netCDF4.Dataset object
    return netCDF4.Dataset(unzipped_file)


def main():
    # Expected number of arguments (excluding script name)
    expected_args = 8  

    # Check if the correct number of arguments is provided
    if len(sys.argv) != expected_args + 1:  
        print("Usage: python get_madis_data.py <minute> <hour> <day> <month> <year> <latitude> <longitude> <altitude>")
        print("Example: python get_madis_data.py 30 14 26 2 2025 37.7 -122.4 1200")
        print("This example would download the data for 2:30 PM on February 26, 2025 at (37.7, -122.4) at altitude 1200m.")
        sys.exit(1)


    data = get_data(sys.argv)

    latitude = float(sys.argv[6])
    longitude = float(sys.argv[7])
    altitude = float(sys.argv[8])

    # Flatten in case 2D
    medTurbulence = data.variables['medTurbulence'][:].flatten()
    medEDR = data.variables['medEDR'][:].flatten()

    # Check if the variable is masked
    if np.ma.is_masked(medTurbulence):
        print("Some values are masked, computing only on valid values.")

        # Get indices of non-masked (filled) values
        valid_medTurbulence_idx = np.where(~medTurbulence.mask)

        # Extract only valid values
        valid_medTurbulence_values = medTurbulence.compressed()
        valid_medEDR_values = medEDR.compressed()
    else:
        # If not masked, use full array
        valid_medTurbulence_values = medTurbulence
        valid_medTurbulence_idx = np.arange(valid_medTurbulence_values.size)  # All indices are valid
        valid_medEDR_values = medEDR
        # valid_medEDR_idx = np.arange(valid_medEDR_values.size)  # All indices are valid

    print(f"Number of valid values: {valid_medTurbulence_values.size}")
    print(f"Number of valid values: {valid_medEDR_values.size}")

    # Flatten in case 2D
    valid_fileLat = data.variables['latitude'][valid_medTurbulence_idx].flatten()
    valid_fileLon = data.variables['longitude'][valid_medTurbulence_idx].flatten()
    valid_fileAlt = data.variables['altitude'][valid_medTurbulence_idx].flatten()

    # Compute vectorized Haversine distance (horizontal distance in meters)
    xy_dist = np.array([haversine((latitude, longitude), (lat, lon), unit='m') 
                    for lat, lon in zip(valid_fileLat, valid_fileLon)])
    dist_sq = xy_dist**2 + (altitude - valid_fileAlt)**2
    sorted_dist_idx = np.argsort(dist_sq)

    # Print the Closest point
    print("5 Closest Points:")
    for i in range(5):
        idx = sorted_dist_idx[i]
        distance = np.sqrt(dist_sq[idx])
        # Output the information for the closest point
        print(f"Point {i+1}:")
        print(f"Latitude: {valid_fileLat[idx]}, Longitude: {valid_fileLon[idx]}")
        print(f"Altitude: {valid_fileAlt[idx]}")
        print(f"Median Turbulence: {valid_medTurbulence_values[idx]}")
        print(f"Median EDR: {valid_medEDR_values[idx]}")
        print(f"Distance: {distance:.4f} meters")
        print("\n\n")

    # Close the netCDF file
    data.close()

if __name__ == "__main__":
    main()

