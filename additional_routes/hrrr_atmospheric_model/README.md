# HRRR (High-Resolution Rapid Refresh) Atmospheric Model
### Files:
- `hrrr_model.ipynb`: Jupyter Notebook demonstrating the use of the Herbie library to retrieve and obtain data from an HRRR model.

## Background
The HRRR model is an hourly-updating weather model that uses [radar-reflectivity data, cloud analysis, and a variety of other weather observations to forecast various atmospheric conditions](https://journals.ametsoc.org/view/journals/wefo/37/8/WAF-D-21-0151.1.xml). The HRRR model specializes in predicting precipitation, including thunderstorms and severe weather events. 
<br>
Every 15 minutes, the model assimilates radar data that adds detail to the forecasts. When it came out in 2014, it had 4x higher resolution (3km grids, 1 hr refresh) than previously used models. The model makes predictions for >95 million grid points over the Continental United States (CONUS) and requires a lot of computing power. The grid points are not cartesian, but rather a lambert-conformal projection system (https://github.com/blaylockbk/Herbie/discussions/45#discussioncomment-2161984) 
<br>
As of writing, The HRRR model is on its fourth edition. This version was released in December 2020. 
<br>

## Accessing HRRR Model Runs:
After trying several other libraries, we had the most success in using the Herbie library to access HRRR data, but did not end up using the data in our final project.
<br>
This Notebook demonstrates the use of the [Herbie](https://herbie.readthedocs.io/en/stable/#) library in order to retrieve and obtain data from an HRRR model, which is a high-resolution weather model that provides forecasts for the continental United States. 
<br>

## Usage:
- install with `conda install -c conda-forge herbie-data`, then run this notebook.

## What Data the Model Provides:
### EDR (Eddy Dissipation Rate):
We hoped to use the HRRR model to look at its **forecasted EDR** and compare it to 
our turbulence predictions, or to use it's predicted reflectivity as input to the 
model when the model was making predictions in the future, however, upon exploration, 
*it did not seem like we could get the EDR data directly from the model*.
<br>
The model forecasts many different types of data, including reflectivity. 
There is a possibility of calculating EDR data from other model products however.

### Other Data (could be relevant to turbulence):
The following fields from the data are relevant to turbulence:
- UGRD: U-Component of Wind
- VGRD: V-Component of Wind
- VUCSH: Vertical U-Component Shear
- VVCSH: Vertical V-Component Shear
- MAXUVV: Hourly Max Upward Vertical Velocity
- MAXDVV: Hourly Max Downward Vertical Velocity
- DZDT: Vertical Velocity (Geometric)
- CAPE: Convective Available Potential Energy
- CIN: Convective Inhibition
- HLCY: Storm Relative Helicity
- HPBL: Planetary Boundary Layer Height
- GUST: Wind Speed (Gust)

### Notes on Data Format
You can use the herbie library to access and download HRRR models 
(from AWS, or other places where they are hosted). 
<br>
The data is stored in a format called GRIB2, whose documentation is online. 
There are several python libraries that can be used to parse the data. 
<br>
If we are using herbie, it uses the `cfgrib` library to read GRIB2 data into an 
xarray Dataset. 
<br>
- Exporting all data would be costly in time and space so it seems better to regex matching in order to retrieve only data that is considered relevant.

## Notes on Other Models:
According to the paper "Feasibility Study on Using High Resolution Numerical Models to Forecast Severe Aircraft Turbulence Associated with Thunderstorms" by Donald W. McCann, high-resolution numerical models (examples given were ARW which is 4km and NMM which is 5km) do a better job of predicting pilot reports of turbulence then lower resolution models (RUC2 which is 13km). 
