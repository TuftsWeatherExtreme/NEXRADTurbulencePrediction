import numpy as np
import pandas as pd
import argparse
import soundings_investigation

'''get_matching_sounding.py

The purpose of this file is to take in a pirep, and return the
most relevant sounding data to add to our input variables.
'''



def main():
    parser = argparse.ArgumentParser(description="Find the nearest sounding data for given parameters.")
    parser.add_argument("lat", type=float, help="Latitude")
    parser.add_argument("lon", type=float, help="Longitude")
    parser.add_argument("alt", type=float, help="Altitude")
    parser.add_argument("month", type=int, help="Month")
    parser.add_argument("day", type=int, help="Day")
    parser.add_argument("year", type=int, help="Year")
    parser.add_argument("time", type=int, help="Time in hours (0-23)")
    args = parser.parse_args()
    
    closest_row, delta_t = soundings_investigation.find_nearest_sounding(args.lat, args.lon, args.alt, args.month, args.day, args.year, args.time)
    
    #TODO: Determine what to extract
    print("Closest Sounding Data:")
    print(closest_row)
    print(f"Delta T: {delta_t} hours")

if __name__ == "__main__":
    main()
