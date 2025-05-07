# Radar Scripts

## NEXRAD

For our project, we used Next Generation Weather Radar Data (NEXRAD) as the
input to our machine learning model. Specifically, we used the level 2 product
**reflectivity**. To acquire this data, we had to perform several queries to
the Amazon S3 bucket found 
[here](https://noaa-nexrad-level2.s3.amazonaws.com/index.html). 
*Most* of the data is stored in the following format:
```
{YEAR}/{MONTH}/{DAY}/{SITE_CODE}/{SITE_CODE}{YEAR}{MONTH}{DAY}_{HHMMSS}_VO6"
```
Where `YEAR`, `MONTH`, and `DAY` are numerical, e.g., `YEAR = 2025`, 
`MONTH = 01`, and `DAY = 14` would be one possible setup. Site code is one of
of the 159 NEXRAD site codes. These are all 4 capital letters that begin with
'K'. For example, `KBUF` is the site code of the radar for Buffalo, New York.

## Processing the Data - [get_radars_for_pirep.py](get_radars_for_pirep.py)
We wrote a helpful script called 
[get_radars_for_pirep.py](get_radars_for_pirep.py) which accepts a CSV of pilot
reports (produced from [clean_pireps.py](/pireps/clean_pireps.py)) and 
determines the closest 5 radars to each pirep. This uses the 
[nexrad_sites.csv](nexrad_sites.csv) to find the location of the 159
NEXRAD sites in the Continental United States. One result of
[get_radars_for_pirep.py](get_radars_for_pirep.py) is to add a column called 
`nexrad_sites` to the CSV of pilot reports to store the site codes of the 5 
closest radar sites.

After determining the closest radars (spatially), we must determine the closest
radar scan temporally. This requires listing the contents of the s3 bucket as found
above and storing the filenames locally. We then use the times given in the
filename (the `HHMMSS` portion) to determine the most recent radar scan for
each pilot report, for each of the 5 spatially closest radars. Once we determine
which file represents the most recent radar scan, we store the file path to the
file in the s3 bucket in our CSV.

### Prerequisites
This script needs to be able to perform AWS queries, and thus the user running
this script must have set up AWS access keys. If not, an error will occur
on the call to `list_objects_v2` during execution.

### Usage
Run with
```
python get_radars_for_pirep.py [-month MONTH] [-year YEAR] [-o {FILE/STDOUT}]
```
If month or year are not specified, this script expects a
CSV filled with pireps through stdin. If they are both specified,
this script searches in the [clean_pirep_data](/pireps/clean_pirep_data/) folder
for the data for the given year and month. 

The `-o` should be followed by either `FILE` or `STDOUT`, 
and this indicates the location to output the new CSV to. If `FILE` is
specified, this script will output to 
[radars/pirep_with_radar_data/{YEAR}/{MONTH}.csv](/radars/pirep_with_radar_data/2024/02.csv).
`MONTH` must be one of "january, february, ..., december", and year must be
of the format `YYYY` (e.g., 2025). One example usage of the script would be:
```
python get_radars_for_pirep.py -month february -year 2024 -o FILE
```
The output of this run can be found in [02.csv](/radars/pirep_with_radar_data/2024/02.csv)
Another example can be seen here:
```
cat pireps/clean_pirep_data/2025/03_turb_pireps.csv |
python radars/get_radars_for_pirep.py -o STDOUT > radars/pirep_with_radar_data/2025/03.csv
```
This piping feature is useful and we use it in our [generate_csv_data.sh](/hpc_scripts/data_processing/generate_csv_data.sh)

### Example
Here are some example values for the two columns the [get_radars_for_pirep.py](get_radars_for_pirep.py) script adds to the final CSV:

```
nexrad_sites: "('KJGX', 'KVAX', 'KFFC', 'KCLX', 'KTLH')",
aws_files: "['s3://noaa-nexrad-level2/2024/01/31/KJGX/KJGX20240131_235419_V06',
             's3://noaa-nexrad-level2/2024/01/31/KVAX/KVAX20240131_235731_V06', 
             's3://noaa-nexrad-level2/2024/01/31/KFFC/KFFC20240131_235614_V06', 
             's3://noaa-nexrad-level2/2024/01/31/KCLX/KCLX20240131_235840_V06', 
             's3://noaa-nexrad-level2/2024/01/31/KTLH/KTLH20240131_235416_V06']"
```
As can be seen, for the given pilot report at 29500 feet, 
32.40º latitude, and -83.21º longitude, the 5 closest radar sites are:
```
KJGX: Southeast of Atlanta
KVAX: South Georgia
KFFC: Atlanta
KCLX: Charleston South Carolina
KTLH: Tallahassee
```
The pilot report is in fact located southeast of Atlanta, and this specific
report can be found on the first line of 
[02.csv](/pireps/clean_pirep_data/2024/02.csv). This pilot report took place
at midnight on February 1st 2024, so the closest radar scans in the past are:
```
KJGX: 20240131_235419_V06
KVAX: 20240131_235731_V06
KFFC: 20240131_235614_V06
KCLX: 20240131_235840_V06
KTLH: 20240131_235416_V06
```
That is, 23:54:19 on January 31st for the KJGX station.

## Accessing the Data
We use [PyART](https://arm-doe.github.io/pyart/index.html) to read NEXRAD data. 
To install PyART, you can run `pip install arm_pyart` 
(NOT `pip install pyart`!). PyART has a ton of useful functions and classes
for manipulating, accessing, and visualizing the radar data. Some of these
downloaded radar files can be found in 
[raw_radar_data](raw_radar_data).

### [reflect_over_cutoff.py](reflect_over_cutoff.py)
To get our bearings with the `Radar` object provided by PyART, we wrote
[reflect_over_cutoff.py](reflect_over_cutoff.py). This script takes takes a 
NEXRAD VO6 file and a cutoff reflectivity value and creates a CSV file
containing the longitude, latitude, and altitude of all instances of 
reflectivity above this cutoff in the given VO6 file. 

#### Usage
```
python reflect_over_cutoff.py <input_file> <cutoff>
```
As an example, if we wanted to find all of the reflectivity values above **20**dBZ
in the radar file `raw_radar_data/KJGX20240131_235419_V06`, we would run
`python reflect_over_cutoff.py raw_radar_data/KJGX20240131_235419_V06 20`


## Gridding the data - [radar_data_to_model_input.py](radar_data_to_model_input.py)
The next step in our data pipeline involves gridding our NEXRAD data around
a particular pilot report. After running 
[clean_pireps.py](../pireps/clean_pireps.py) and sending the output to
[get_radars_for_pirep.py](get_radars_for_pirep.py), we have the requisite
data to grid our nexrad data around a pilot report to create an input to our
machine learning model.

### Prerequisites
Before we can run [radar_data_to_model_input.py](radar_data_to_model_input.py),
if we intend to leverage the Tufts HPC, it is important to combine and then 
split all of the CSVs generated in [pirep_with_radar_data](pirep_with_radar_data)
to be equally sized for parallel processing.
The [collapse.sh](collapse.sh) script can be used to combine all of the CSVs
into a single CSV (run with `bash collapse.sh`). Then, the 
[split_csv.py](split_csv.py) script can be run to split the CSV into the
desired number of even parts. This script can be run with the following 
arguments:
```
python split_csv.py <input_file> <output_dir> <num_parts>
```
Once this has been completed, the data can be processed in parallel. An example
output that has split the data for the 2 months in 
[pirep_with_radar_data/](pirep_with_radar_data) into 10 parts can be found in 
[split_radar_data/](split_radar_data).

### Usage
To run the [radar_data_to_model_input.py](radar_data_to_model_input.py)
program to create model inputs, it expects an input filename and output
directory on the command line:
```
python radar_data_to_model_input.py <input_file> <output_dir>
```
This file will use the closest file to query AWS and download the radar
object. Then, it will call the `create_grid` function exported from
[create_grid.py](create_grid.py) to create a grid of reflectivity data around
the pilot report we're computing on.

After calling `create_grid`, we will output the reflectivity data to a NetCDF
file. Some of these files can be found in [model_inputs](model_inputs). If
no reflectivity data is found around a particular pilot report, this script
does not output a NetCDF file.

### [create_grid.py](create_grid.py)
This function is based on the PyART function 
[grid_from_radars](https://arm-doe.github.io/pyart/API/generated/pyart.map.grid_from_radars.html)
The [example_create_grid.py](example_create_grid.py) script demonstrates how
the `create_grid` function can be called. This script is hardcoded to use
the a pirep in the 
[split_radar_data/part_001.csv](split_radar_data/part_001.csv) file for
the example.

Our implementation of the function is very robust, but for the purposes of our
project, we chose to use a grid of size 16x16x10 that represented
a 0.25º (latitude) x 0.25º (longitude) x 10000ft (altitude) volume of air. 

*Note: These decisions were somewhat arbitrary*

Unfortunately, the vast majority of values in the cells of most grids are NAN.
For the one in [example_create_grid.py](example_create_grid.py), only 25 of the
2560 grid cells are populated with valid
reflectivity values. This was the most populated grid we could find out of
the first 20 pireps in the CSV.

One possible improvements could come from using *composite reflectivity* which
finds the highest reflectivity in a column of altitude. 
Additionally, we could choose to only
use grids that are not so sparsely populated (however, the vast majority of grid
were sparsely populated).

Given more time, this is an area we would focus our efforts on more, as having
good input data to the model is crucial for it to be able to learn any patterns.

Either way, since we call this `create_grid.py` function on
every single pirep to create our input data, it was important to optimize this
function. It was profiled and now runs in about 0.1-0.2s on a node on the HPC.

### [quiet_pyart.py](quiet_pyart.py)

This file can be imported as PyART to silence the print statement that comes
with importing PyART. For example:
```
import quiet_pyart as pyart
```
will effectively import PyART without the print statement.

### [plotting_example.ipynb](plotting_example.ipynb)
This script was written by WeatherExtreme employee Ryan Purciel and provided to 
us as an initial visualization tool for reflectivity data.
