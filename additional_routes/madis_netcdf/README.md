# MADIS Data and netCDF Files

[MADIS data](https://madis-data.ncep.noaa.gov/) is stored in NetCDF files. 
NetCDF is a binary file format for storing scientific data, 
while MADIS is a database and delivery system for meteorological observations.
Luckily, there is a [Python/numpy interface](https://unidata.github.io/netcdf4-python/) 
for the netCDF library.
First, ensure that you go through the ["Developer install"](https://unidata.github.io/netcdf4-python/#developer-install)
directions in the API documentation to ensure all necessary libraries are created.
If you have a conda environment you are using for running Python scripts, we recommend 
installing netCDF4 in that environment. 
<br><br>
From there, we've made a script called `get_madis_data.py` that takes in time 
and location information, and uses that information to extract relevant MADIS data from the archive at
https://madis-data.ncep.noaa.gov/madisPublic1/data/ using File Transfer Protocol (FTP). It then prints
data about the 5 closest points to the location entered by the user in the netCDF file extracted using FTP.
<br><br>
At the moment, the file takes in the time and location data and gets only the closest netCDF file corresponding to the time 
input by the user.
It first downloads the original data file (stored with a .gz extension), and then 
compresses that file and downloads a file with the same name, but without the .gz ending.
`20250226_1400.gz and 20250226_1400` are in this subdirectory as examples from February 26th, 2025 at 2pm.
<br><br>
In future work, a recommended pathway would be to use this script, using information from a PIREP 
as the arguments. Note that this gets the data for the time closest to the time put in by a user 
and does not currently handle edge cases such as 12/31 at 11:59 PM, so future work would need to handle this.
<br><br>
Some of the relevant data that can be extracted from this script that may be useful for a model includes medTurbulence and medEDR. 
There is also a plethora of data about temperature, wind speed, dewpoint, and more.
<br><br>

