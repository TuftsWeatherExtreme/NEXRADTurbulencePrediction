# Data Processing Scripts

## Overarching Notes
- All scripts rely on [load_modules.sh](../load_modules.sh), [unload_modules.sh](../unload_modules.sh), and environment variables (`ENV_PATH`, `REPO_PATH`).
- All standard output and standard error output is stored in files as follows:
  - Standard Output: `[job_name].[job_array_id].[node_name].out`
  - Standard Error: `[job_name].[job_array_id].[node_name].err`
  - Note: job_array_id is ommitted for any non-array scripts.
  - This can be easily changed if the output is desired to be in any specific location.
- All scripts currently do not specify a user to mail with notifications of a script starting/finishing/exiting early. In the final line of each `#SBATCH` specification, `#SBATCH --mail-user=` should have the email(s) appended to that line. Multiple emails can also be specified by splitting with commas. For example, `#SBATCH --mail-user=jzelev01@tufts.edu, shecht02@tufts.edu`.

## Contents:

### [generate_csv_data.sh](generate_csv_data.sh)
**Description**: Generates csv data that include pilot report information for all the combinations of YEARS, MONTHS specified. This generates subdirectories of csvs in [clean_pirep_data](/NEXRADTurbulencePrediction/pireps/clean_pirep_data).

**Usage**: `sbatch generate_csv_data.sh`

**Dependencies**:
- [clean_pireps.py](/NEXRADTurbulencePrediction/pireps/clean_pireps.py)
- [get_sigmets.py](/NEXRADTurbulencePrediction/pireps/get_sigmets.py)

**Notes**:
- This script uses a job array for parallel processing. 
- Currently uses an array of 0-191 and includes years 2008-2025 for all months. All combinations of years and months listed in `YEARS` and `MONTHS` will be run.
- If modfying the `YEARS` or `MONTHS` lists, the job array range should be updated accordingly. This will be `#SBATCH --array=0-[total_jobs_needed - 1]`, where total_jobs_needed = num_months * num_years. 
- Currently outputs to the [clean_pirep_data](/NEXRADTurbulencePrediction/pireps/clean_pirep_data) directory but can be changed in [clean_pireps.py](/NEXRADTurbulencePrediction/pireps/clean_pireps.py) or by changing command line arguments to said script in [generate_csv_data.sh](generate_csv_data.sh).
- For additional details on the script itself, view the README for [pireps](/NEXRADTurbulencePrediction/pireps/README.md)
- [test_sigmet_coverage.py](/NEXRADTurbulencePrediction/pireps/test_sigmet_coverage.py) gives overall summary of cleaned pireps and how many were associated with a SIGMET.

### [generate_radar_data.sh](generate_radar_data.sh)
**Description**: End goal is that all pireps with real radar data are saved (in conjunction with the radar info) in edited csv, still organized by month and year. New paired data is stored in [pirep_with_radar_data](/NEXRADTurbulencePrediction/radars/pirep_with_radar_data) folder.

**Usage**: `sbatch generate_radar_data.sh`

**Dependencies**:
- [get_radars_for_pirep.py](/NEXRADTurbulencePrediction/radars/get_radars_for_pirep.py)

**Notes**:
- Some loss of data is expected as not all pireps will have valid radar data. Shouldn’t have significant loss though.
- Get total number of pireps with radar data by running `find */ -type f | xargs wc -l` in the [pirep_with_radar_data](/radars/pirep_with_radar_data) folder. Should be around 93,000
- [test_radar_output.py](/NEXRADTurbulencePrediction/radars/test_radar_output.py) gives breakdown of job outcome
- For additional details on the script itself, view the README for [radars](/NEXRADTurbulencePrediction/radars/README.md)

### [generate_model_inputs.sh](generate_model_inputs.sh)
**Description**: Processes PIREP data with associated radar observations to create gridded 3D radar data files suitable for machine learning model input. Each output represents a spatial grid of radar reflectivity centered on a turbulence report.

**Usage**: `sbatch generate_model_inputs.sh`

**Dependencies**:
- [radar_data_to_model_input.py](/NEXRADTurbulencePrediction/radars/radar_data_to_model_input.py)
- Relies on the presence of [pirep_with_radar_data](/NEXRADTurbulencePrediction/radars/pirep_with_radar_data)

**Notes**:
- Data flow:
Input:  radars/pirep_with_radar_data/YYYY/MM.csv
        ↓
Process: radar_data_to_model_input.py
        ↓
Temp:   model_inputs/YYYY_MM/*.nc (individual NetCDF files, deleted later)
        ↓
Output: model_inputs/compressed/YYYY_MM.tar.xz (compressed archive)
- The compressed model inputs are unpacked when creating the dataloader (see [dataloader_class.py](/model_training/dataloader_class.py)) and then recompressed.
- More documentation and information about [radar_data_to_model_input.py](/radars/radar_data_to_model_input.py) can be read [here](/radars/README.md). Note, as documented there, this python script does not generate a netcdf file if the gridded reflectivity value is all undetectable (not read in the scan) so the entire grid is empty. This means that the compressed files (and number of model inputs in each part) may understandably vary from part to part and will not be equivalent to the number of data rows in the csv. 
- Test the total number of NetCDF files by running `for f in $REPO_PATH/model_inputs/compressed/*.tar.xz; do tar -tvf "$f"; done | grep "\.nc" | wc -l`. Should be roughly 27,181


### [generate_dataloader.sh](generate_dataloader.sh)
**Description**: Generates pytorch dataloader with [create_datasets.py](/NEXRADTurbulencePrediction/model_training/create_datasets.py) using the [dataloader_class.py](/NEXRADTurbulencePrediction/model_training/dataloader_class.py).

**Usage**: `sbatch generate_dataloader.sh <dataloader_name> [existing_dataloader]`
Note: `dataloader_name`is the desired name of the dataloader to be generated and optionally an existing dataloader can be provided if that data is wished to be included in the
new dataloader being created.

**Dependencies**:
- [create_datasets.py](/model_training/create_datasets.py)
- [dataloader_class.py](/model_training/dataloader_class.py)

**Notes**:
- This dataloader will be generated in the [model_training directory](/NEXRADTurbulencePrediction/model_training/) with the specified name.
- To account for potentially operating in environments with more limited quota, the dataloader will decompress the compressed inputs one file at a time, add it to the dataloader object, and then remove that decompressed version.
- The end of the script removes a "decompressed" directory that is generated in [create_datasets.py](/model_training/clean_pireps.py)

