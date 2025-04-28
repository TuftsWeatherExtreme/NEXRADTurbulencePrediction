# PIREP Scripts

## PIREPs

PIREPS, short for Pilot Reports, are reports filed by pilots while flying that may indicate various weather conditions. Pilot reports come in the following form:

```
KCMH UA /OV APE 230010/TM 1516/FL085/TP BE20/SK BKN065/WX FV03SM HZ FU/TA 20/TB LGT
```

Where each `/XY` indicates the next field. A short description of how Pilot Reports work can be found [here](https://skybrary.aero/articles/pilot-report-pirep). A more thorough description of each field of the Pilot Report and their possible values can be found [here](https://www.faa.gov/air_traffic/publications/atpubs/fs_html/chap8_section_1.html). The common fields that may be present in a PIREP can be found in a table on this [site](https://skybrary.aero/articles/pilot-report-pirep).

### Turbulence Field
The field we care about most is the turbulence field `/TB`, of which some common abbreviations used in this section are listed below:

* `INTMT` - Intermittent
* `OCNL` - Occassional
* `CONS` - Constant
* `NEG` - Negative, or no turbulence
* `LGT` - Light turbulence
* `MOD` - Moderate turbulence
* `SEV` - Severe turbulence
* `EXTRM` - Extreme turbulence
* `LGT-MOD` - Light-to-moderate turbulence
* `MOD-SEV` - Moderate-to-severe turbulence
* `SEV-EXTRM` - Severe-to-extreme turbulence
* `NONE` - Means the pilot left this field blank
* `CAT` - Clear Air Turbulence
* `CHOP` - Turbulence from "rapid bumpiness" ([source](https://www.faa.gov/air_traffic/publications/atpubs/pcg_html/glossary-c.html#$CHOP))

Some examples of reported turbulence can be seen below:
* `/TB CONS LGT` - Continuous light turbulence
* `/TB LGT-MOD CHOP` - Light-to-moderate chop
* `/TB NEG` - Negative or no turbulence

We use this turbulence value along with the location

### Other Fields
There are several other fields that one might find useful from a PIREP, but for us the most important other fields are the fields which indicate the location the PIREP was filed. This is likely a large source of error for our analysis, as the locations listed are from when the pilot report was filed, not when the turbulence occurred. This descrepancy cannot be avoided as there is no way to know how long ago the turbulence occurred or the heading of the plane that reported the turbulence. Thus, these locations should all be taken with a grain of salt, but it is all we have to work with.


## Getting PIREP data
We used Iowa State University's archive of Pilot Reports found [here](https://mesonet.agron.iastate.edu/request/gis/pireps.php) to acquire all of the PIREPs we used in our project. The backend documentation for this resource can be found [here](https://mesonet.agron.iastate.edu/cgi-bin/request/gis/pireps.py?help). We have a useful script called [clean_pireps.py](clean_pireps.py) that can be run with the following contract:

```
Usage: python clean_pireps.py -month MONTH -year YEAR [-o {FILE/STDOUT}]
```
This script will download PIREPs from a particular month and year and can output these pireps to a file called [clean_pirep_data/{YEAR}/{MONTH_NUM}_turb_pireps.csv](pireps/clean_pirep_data/2025/03_turb_pireps.csv) with `-o FILE` or to standard output with `-o STDOUT`. All pireps downloaded from this script will contain turbulence data and latitude, longitude, and altitude data that we can successfully parse. Additionally, we determine the weight of each plane that reported the PIREP as either unknown (`U`), light (`L`), medium (`M`), or heavy (`H`) and add this as a column called `Plane Weight`.

The script [get_all_clean_pireps.py](get_all_clean_pireps.py) runs the [clean_pireps.py](clean_pireps.py) script to download the data for all months and years from 2003-2024. This script has been parallelized to be run on the Tufts HPC in the script [generate_pirep_data_csvs.sh](../hpc_scripts/generate_pirep_data_csvs.sh) which runs both the [clean_pireps.py](clean_pireps.py) script and then pipes the outputs directly to the [get_radars_for_pirep.py](get_radars_for_pirep.py)