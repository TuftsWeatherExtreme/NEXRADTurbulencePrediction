# Radar Scripts

## Overview
This directory contains scripts for processing NEXRAD (Next Generation Weather Radar) data to create machine learning model inputs for turbulence prediction. The pipeline matches PIREP (Pilot Report) turbulence observations with corresponding radar reflectivity data from the NEXRAD network.

### Primary Pipeline Entry Point:
All radar data processing is orchestrated through the [generate_radar_data.sh](/NEXRADTurbulencePrediction/hpc_scripts/data_processing/generate_radar_data.sh) SLURM job array script.

## NEXRAD Data
We use NEXRAD Level 2 reflectivity data from the Amazon S3 bucket:
s3://unidata-nexrad-level2/
Important: The bucket URL was updated in Fall 2025. Old scripts referencing noaa-nexrad-level2 have been migrated to unidata-nexrad-level2.

### Data Format
Most data is stored in the following structure:
`{YEAR}/{MONTH}/{DAY}/{SITE_CODE}/{SITE_CODE}{YEAR}{MONTH}{DAY}_{HHMMSS}_V06`
Where:
* YEAR, MONTH, DAY are numerical (e.g., 2024, 01, 31)
* SITE_CODE is a 4-letter code beginning with 'K' (e.g., KJGX for a radar near Brunswick, GA)
* HHMMSS represents the scan time in UTC
Example:
`2024/01/31/KJGX/KJGX20240131_235419_V06`

There are 159 NEXRAD sites in the Continental United States. Site locations are stored in [nexrad_sites.csv](nexrad_sites.csv)

## Current Pipeline Scripts

### 1. generate_radar_data.sh
SLURM job array script [generate_radar_data.sh](/NEXRADTurbulencePrediction/hpc_scripts/data_processing/generate_radar_data.sh) that processes PIREPs month-by-month.

What it does:
* Runs as a job array across all year/month combinations (2008-2025)
* Calls get_radars_for_pirep.py for each month/year to find radar sites and scan times
* Outputs CSVs with nexrad_sites and aws_files columns to [pirep_with_radar_data/{YEAR}/{MONTH}.csv](/NEXRADTurbulencePrediction/radars/pirep_with_radar_data/)

Note: YEARS and MONTHS arrays must match those used in generate_csv_data.sh (the PIREP cleaning pipeline).
Usage: `sbatch generate_radar_data.sh`

### 2. get_radars_for_pirep.py
Core radar matching script [get_radars_for_pirep.py](/NEXRADTurbulencePrediction/radars/get_radars_for_pirep.py) that finds the best candidate NEXRAD sites and closest scan times for each PIREP.

What it does:
* Spatial Matching: Uses beam geometry scoring (not just distance) to find the 5 best radar sites for each PIREP
    * Scores radars based on how well their beam pattern covers the PIREP's altitude, implemented in [beam_geometry.py](/NEXRADTurbulencePrediction/radars/beam_geometry.py)
    * Pulls extra candidates for high-altitude PIREPs (beam spreads with distance)
* Temporal Matching: Queries S3 to find the closest radar scan within ±30 minutes of each PIREP
    * Uses asynchronous batch queries for efficiency (~5 minutes per month)
    * Matches radar scans in the past (or within 30-min window if no past scan exists)

Key Features:
* Beam Geometry Scoring: Altitude-aware ranking ensures radars actually "see" the turbulence
* Distance Threshold: PIREPs with no radar within range are automatically dropped
* Async S3 Queries: Batch processes thousands of S3 requests efficiently using aiobotocore

Prerequisites:
* AWS access to unidata-nexrad-level2 bucket (public, no credentials needed)
* Input CSV from PIREP cleaning pipeline with columns: datetime, LAT, LON, FL, turbulence_intensity

Usage: `python get_radars_for_pirep.py -month <MONTH> -year <YEAR> -o FILE`

### 3. radar_data_to_model_input.py
Gridding script [radar_data_to_model_input.py](/NEXRADTurbulencePrediction/radars/radar_data_to_model_input.py) that converts PIREP + radar data into NetCDF model inputs.

What it does:
* Downloads radar data from S3 using PyART
* Grids reflectivity data around each PIREP location (16×16×10 grid)
* Outputs NetCDF files with reflectivity and PIREP metadata to [model_inputs/compressed](/NEXRADTurbulencePrediction/model_inputs/compressed/)

Grid Configuration:
* Spatial: 0.25° (lat) × 0.25° (lon) × 10,000 ft (alt)
* Resolution: 16×16×10 = 2,560 grid cells
* Field: Reflectivity (dBZ)
* Quality Control: Skips grids with >90% NaN values

Prerequisites:
* Input CSV from [get_radars_for_pirep.py](/NEXRADTurbulencePrediction/radars/get_radars_for_pirep.py) with nexrad_sites and aws_files columns

Usage: `python radar_data_to_model_input.py <input_file> <output_dir>`
Preferred usage is on HPC through [generate_model_inputs.sh](/NEXRADTurbulencePrediction/hpc_scripts/data_processing/generate_model_inputs.sh) with `sbatch generate_model_inputs.sh`

Note on Data Sparsity:
* Most grids are sparse (>90% NaN). This is expected due to:
    1. Radar beam geometry (gaps between elevation angles)
    2. Atmospheric conditions (clear air = no reflectivity)
    3. Distance from radar (beam spreads at range)
* Grids with >90% NaN are automatically excluded from output.

### 4. find_peak_turbulence_window.py
Analysis tool [find_peak_turbulence_window.py](/NEXRADTurbulencePrediction/radars/find_peak_turbulence_window.py) for identifying periods of intense turbulence activity.

What it does:
* Scans PIREPs with confirmed radar data
* Finds the 8-hour windows with the most severe turbulence reports (intensity ≥ 5)
* Outputs top N windows with geographic/altitude statistics

Usage:
* Scan all years: `python find_peak_turbulence_window.py`
* Scan specific year: `python find_peak_turbulence_window.py --year 2024`
* Show top 10 windows: `python find_peak_turbulence_window.py --top 10`
* Custom data directory: `python find_peak_turbulence_window.py --data-dir /path/to/pirep_with_radar_data`

Default Data Directory: [$REPO_PATH/radars/pirep_with_radar_data](/NEXRADTurbulencePrediction/radars/pirep_with_radar_data/)

### 5. test_radar_output.py
Validation script [test_radar_output.py](/NEXRADTurbulencePrediction/radars/test_radar_output.py) for checking the quality of radar matching output.

What it does:
* Scans all CSVs in [pirep_with_radar_data/](/NEXRADTurbulencePrediction/radars/pirep_with_radar_data/)
* Validates column presence, data types, and S3 path formatting
* Reports statistics on radar matching success rate
* Flags common errors (missing radars, malformed paths, site/radar count mismatches)

Usage: `python test_radar_output.py`

## Supporting Files

### nexrad_sites.csv
Reference file [nexrad_sites.csv](/NEXRADTurbulencePrediction/radars/nexrad_sites.csv) containing all 159 NEXRAD site locations.

Columns:
* Site Code: 4-letter identifier (e.g., KJGX)
* Latitude, Longitude: Site coordinates (degrees)
* Elevation: Site elevation (feet)

Used by [get_radars_for_pirep.py](/NEXRADTurbulencePrediction/radars/get_radars_for_pirep.py) for spatial queries and by [radar_data_to_model_input.py](/NEXRADTurbulencePrediction/radars/radar_data_to_model_input.py) for longitude correction.

### beam_geometry.py
Script [beam_geometry.py](/NEXRADTurbulencePrediction/radars/beam_geometry.py) contains scoring functions for altitude-aware radar site ranking. Higher altitude PIREPs require more candidates because beam quality degrades with distance.

### create_grid.py
Gridding function [create_grid.py](/NEXRADTurbulencePrediction/radars/create_grid.py) used by [radar_data_to_model_input.py](/NEXRADTurbulencePrediction/radars/radar_data_to_model_input.py).

Based on: PyART's grid_from_radars function, optimized for our use case.

Grid Parameters:
* Shape: (10, 16, 16) --> Z, Y, X
* Altitude range: ±5,000 ft around PIREP
* Lat/Lon range: ±0.125° around PIREP

Returns: xarray.Dataset with reflectivity field and NaN fraction metadata.

### quiet_pyart.py
Wrapper to import PyART without its startup message, [quiet_pyart.py](/NEXRADTurbulencePrediction/radars/quiet_pyart.py)

Usage: `python import quiet_pyart as pyart`

## Deprecated / Old Scripts
The following scripts are no longer part of the main pipeline and are kept for reference only

### reflect_over_cutoff.py
Status: Deprecated exploration tool
Original Purpose: Extract all reflectivity values above a threshold from a single radar file.
Usage (historical): `python reflect_over_cutoff.py raw_radar_data/KJGX20240131_235419_V06 20`
Why deprecated: Useful for initial data exploration but not needed for the production pipeline.

### collapse.sh
Status: Utility script (not part of main pipeline)
Purpose: Combines all CSVs from pirep_with_radar_data/ into a single file for downstream processing.
Usage: `bash collapse.sh`
When to use: Only if you need to prepare data for parallel gridding jobs. The main pipeline (generate_radar_data.sh) does not call this.

### split_csv.py
Status: Utility script (not part of main pipeline)
Purpose: Splits a large CSV into equal-sized chunks for HPC job arrays.
Usage: `python split_csv.py <input_file> <output_dir> <num_parts>`
When to use: After running collapse.sh, before running radar_data_to_model_input.py in parallel on the HPC.

### plotting_example.ipynb
Status: Reference notebook
Purpose: Jupyter notebook demonstrating NEXRAD reflectivity visualization.
Author: Ryan Purciel (WeatherExtreme)
When to use: Learning how to visualize radar data with PyART.

### example_create_grid.py
Status: Example/tutorial script
Purpose: Demonstrates how to call the create_grid function.
Hardcoded data: Uses split_radar_data/part_001.csv row 0.
When to use: Understanding grid creation workflow before modifying radar_data_to_model_input.py.
