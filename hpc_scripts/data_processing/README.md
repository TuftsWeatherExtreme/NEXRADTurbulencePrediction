# Data Processing Scripts

## Overarching Notes
- All scripts rely on [load_modules.sh](../load_modules.sh), [unload_modules.sh](../unload_modules.sh), and environment variables (`ENV_PATH`, `REPO_PATH`).
- All standard output and standard error output is stored in files as follows:
  - Standard Output: `[job_name].[job_array_id].[node_name].out`
  - Standard Error: `[job_name].[job_array_id].[node_name].out`
  - Note: job_array_id is ommitted for any non-array scripts.
  - This can be easily changed if the output is desired to be in any specific location.
- All scripts current do not specify a user to mail with notifications of a script starting/finishing/existing early. In the final line of each `#SBATCH` specification, `#SBATCH --mail-user=` should have the email(s) appended to that line. Multiple emails can also be specified by splitting with commas. For example, `#SBATCH --mail-user=jzelev01@tufts.edu, shecht02@tufts.edu`.
- 

## Contents:

### [generate_csv_data.sh](generate_csv_data.sh)
**Description**: Generates csv data that include pilot report information and which corresponding NEXRAD scans are closest for all the combinations of YEARS, MONTHS specified. This generates subdirectories of csvs in [pirep_w_radar_data](../../pireps/pirep_w_radar_data). 

**Usage**: `sbatch generate_csv_data.sh`

**Dependencies**:
- [clean_pireps.py](../../pireps/clean_pireps.py)
- [get_radars_for_pirep.py](../../radars/get_radars_for_pirep.py)

**Notes**:
- This script uses a job array for parallel processing. 
- Currently uses an array of 0-191 and includes years 2008-2024 for all months. All combinations of years and months listed in `YEARS` and `MONTHS` will be run.
- If modfying the `YEARS` or `MONTHS` lists, the job array range should be updated accordingly. This will be `#SBATCH --array=0-[total_jobs_needed - 1]`, where total_jobs_needed = num_months * num_years. 
- Currently outputs to the [pirep_w_radar_data](../../pireps/pirep_w_radar_data) directory but can be changed in [get_radars_for_pirep.py](../../radars/get_radars_for_pirep.py)
- For additional details on the scripts themselves, view the READMEs for [pireps](../../pireps/README.md) and [radars](../../pireps/README.md).

### [generate_model_inputs.sh](generate_model_inputs.sh)
**Description**: Generates netcdf model input files for all split data and compresses them using tar. All output files end up in the format: [part_id].tar.xz, which are a compressed form of all the netcdf's generated from `/radars/split_radar_data/part_[PART_#].csv`, and are located in `/model_inputs/compressed`.

**Usage**: `sbatch generate_model_inputs.sh`

**Dependencies**:
- [radar_data_to_model_input.py](../../radars/radar_data_to_model_input.py)
- Relies on the presence of the following files:
  - /radars/split_radar_data/part_[PART_#].csv, where PART_# ranges from 000-250 (ex. /radars/split_radar_data/part_001.csv).

**Notes**:
- We utilize data parallelism to split this processing into 250 parts to allow each node to process on its data, create all the model input from its CSV, and compress that data for storage purposes. 
  - The SLURM job array id is used to identify which part of data each node will process on. Note that leading 0's are filled in as needed (as shown by converting the `SLURM_ARRAY_TASK_ID` into a `%03d` format id specifier )
- [split_data.py](../../split_data.py) provides the functionality for all the csv data to be split into 250 parts.
- The script is currently using tar.xz, which had the best compression results. We tried using gzip, bzip2, and xz compression, and xz had the best results. Thus, we compress using `tar -cvJf` 
- The specific output directory is specified as `model_inputs` but can be changed on line 21 of the script.
- The overall flow involves making a directory in the `model_inputs` directory for each part and generating all netcdf files, compressing that entire directory into a `tar` file, and then removing the directory that contained all the raw netcdf files. This allowed for effective storage use. 
- The compressed model inputs are unpacked when creating the dataloader (see [dataloader_class.py](../../model_training/dataloader_class.py)) and then recompressed.
- More documentation and information about [radar_data_to_model_input.py](../../radars/radar_data_to_model_input.py) can be read [here](../../radars/README.md). Note, as documented there, this python script does not generate a netcdf file if the gridded reflectivity value is all undetectable (not read in the scan) so the entire grid is empty. This means that the compressed files (and number of model inputs in each part) may understandably vary from part to part and will not be equivalent to the number of data rows in the csv. 

### [generate_dataloader.sh](generate_dataloader.sh)
**Description**: Generates pytorch dataloader with [create_datasets.py](../../model_training/create_datasets.py) using the [dataloader_class.py](../../model_training/dataloader_class.py).

**Usage**: `sbatch generate_dataloader.sh <dataloader_name> [existing_dataloader]`
Note: `dataloader_name`is the desired name of the dataloader to be generated and optionally an existing dataloader can be provided if that data is wished to be included.

**Dependencies**:
- [create_datasets.py](../../model_training/create_datasets.py)
- [dataloader_class.py](../../model_training/dataloader_class.py)

**Notes**:
- This dataloader will be generated in the [model_training directory](../../model_training/) with the specified name.
- To account for potentially operating in environments with more limited quota, the dataloader will decompress the compressed inputs one file at a time, add it to the dataloader object, and then remove that decompressed version.
- The end of the script removes a "decompressed" directory that is generated in [create_datasets.py](../../model_training/clean_pireps.py)

### 
