## Soundings
### What are soundings?
Soundings are a form of atmopsheric data collected at various weather stations around the USA. The data is collected by releasing a weather balloon into the air every twelve hours. The data collected from soundings includes, but is not limited to, pressure, temperature, wind speed and wind direction.  
The University of Wyoming archives Soundings data and provides some additional documentation. The archives can be found on this [website](https://weather.uwyo.edu/upperair/sounding_legacy.html).
### Why are they valuable to the project?
Soundings could be interesting to this project primarily because of their ability to measure wind shear. Wind shear is a phenomenon when local differences in wind speed and/or direction create atmospheric distrubances, which may cause clear air turbulence. Soundings provide wind speed and direction at various difficult altitudes, and an estimate of wind shear betwene two altitudes can be found by using the following equation:  
$`wind_shear = sqrt((U₁ - U₀)² + (V₁ - V₀)²) / ((H₁ - H₀) / 1000 + 1e-5)`$  
Where:
- U₁ and V₁ are the eastward and northward wind components at the higher altitude.

- U₀ and V₀ are the eastward and northward wind components at the lower altitude.


- H₁ and H₀ are the altitudes (in meters) at the higher and lower observation points, respectively.

- The denominator converts the height difference from meters to kilometers and adds a small number (1e-5) to prevent division by zero. 

Using this equation, we can undertsand the rough wind shear at a certain point, and using wind shear as a feature to our model could provide valuable information for our turbulence predictions. Other potentially informative fields could be temperature, raw wind speed and air pressure.


### What didn't work?

Thye original idea to incooportate soundings into out work was to query the soundings website mentioned above based on the location of a given PIREP. The script aims to find the nearest sounding in terms of both time and location. However, the University of Wyoming soundings website is unable to handle the magnitude of requests in a timely enough manner where we can incoorporate it into our model.  
In order to incoorporate soundings, pre-processing the data, either by compiling an archive or gaining access to the University of Wyoming database, may be necessary.

### What does each file do?

`get_matching_sounding.py`: Wrapper file to be called by other files in order to find Soundings data.
Takes arguments for latitude, longitude, altitude, and a date/time specification (month, day, year, and hour in 24-hour format) to retrieve the nearest matching atmospheric sounding.
`add_soundings_to_pireps.py`: Meant to process sounding data once pirep is found.
Uses geolocation and timestamp information to match PIREPs with the nearest sounding station and observation time. Profiles performance using cProfile for optimization purposes.
`soundings_investigation.py`: Defines several helpful functions for finding sounding data.
`match_radar_code.py`: No longer needed, mathces numerical station identifier to string identifier.  
`stations.csv`: Database of stations with numerical, string and location identifiers.
