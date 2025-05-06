## plane_weights

#### Usage
```
python3 get_turb_categories.py
```

#### Motivation
Planes of different weights experience turbulence differently. A heavy plane may barely experience any turbulence flying through an area where a light plane experiences a bunch of turbulence. We needed to account for this discrepancy in our script, and make sure that our data was properly contextualized. 
#### Overview
This folder creates a list of planes and their weights using the following [ICAO website](https://www.icao.int/publications/doc8643/pages/search.aspx). The get_turb_categories script screen-scrapes the website and dumps the results into plane_weight_dictionary.csv.  

The actual data from the ICAO is the "wake turbulence category" (WTC) of each plane. This is just the measure of a weight of a plane, though it sounds confusing. Essentially, a plane's weight is determined by how it interacts with turbuelence, hence the WTC identifier. WTC is either Light (L), Medium (M) or Heavy (H). If the plane weight is not found, it will be classified as Unknown (U).

In the get_turbulence_categories.py script, planes are searched by model. For instance, if a plane model is B737 (Boeing 737), that search term will be typed into the ICAO website. If no match is found, the script will try the first three characters, in this case, B73. This decision was made to consider cases where more uncommon plane models that are similar to existing planes show up. For instance, if a plane model is B737H, that may not show up in the ICAO directory, but B73 will. 

The resulting csv file will have only two columns: the model of plane and its turbulence category (L, M, H, U).

#### Scaling by plane weight
Our implementation chooses to scale the reported turbulence value in a pirep to the turbulence value of a light plane. If a medium plane reports class n turbulence, we will scale it to n+1. If a heavy plane reports class n turbulence, we will scale it to n+2. This leads to our model predicting 10 output classes, since is it possible, though unlikely, that a heavy plane can experience extreme turbulence.

One potential pitfall with this approach is that pilots may be considering plane weight in their PIREPs. For instance, if a pilot were flying a heavy plane and experienced moderate turbulence, they may report a higher value of turbulence understanding that they are flying a heavy plane. The opposite could also be true: a pilot flying a light plane may underreport their turbulence value.
