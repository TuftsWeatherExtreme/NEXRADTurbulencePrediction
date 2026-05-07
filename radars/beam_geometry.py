# beam_geometry.py
# Authors: Razzle Dazzle Rose Fall 25/Spring 26
# Purpose: Utility functions for computing NEXRAD radar beam coverage
#          at various altitudes. Used to select optimal radars for
#          high-altitude PIREPs where the 5 closest stations may not
#          have beam coverage.

import numpy as np

# Standard NEXRAD VCP elevation angles (degrees) — VCP 212 (most common clear-air)
NEXRAD_ELEVATION_ANGLES = [
    0.5, 0.9, 1.3, 1.8, 2.4, 3.1, 4.0, 5.1,
    6.4, 8.0, 10.0, 12.0, 14.0, 16.7, 19.5
]

# 4/3 effective earth radius in meters (standard refraction model)
EFFECTIVE_EARTH_RADIUS_M = 6371000 * (4 / 3)

# Max practical NEXRAD range in meters
MAX_RADAR_RANGE_M = 400_000


def beam_height(distance_m, elevation_deg, radar_elevation_m=0.0):
    """
    Compute the center height of a radar beam at a given distance
    using the 4/3 earth radius model for standard atmospheric refraction.

    Parameters:
        distance_m: Slant/ground distance from radar in meters
        elevation_deg: Beam elevation angle in degrees
        radar_elevation_m: Radar site elevation above sea level in meters

    Returns:
        Beam center height above sea level in meters
    """
    theta = np.radians(elevation_deg)
    # Standard refraction formula
    h = (distance_m * np.sin(theta)
         + (distance_m ** 2) / (2 * EFFECTIVE_EARTH_RADIUS_M))
    return h + radar_elevation_m


def count_covering_angles(distance_m, target_alt_m, radar_elevation_m=0.0,
                          alt_half_range_m=1524.0):
    """
    Count how many standard NEXRAD elevation angles have their beam
    passing through the altitude band around the target.

    Parameters:
        distance_m: Horizontal distance from radar to PIREP in meters
        target_alt_m: PIREP altitude in meters above sea level
        radar_elevation_m: Radar site elevation ASL in meters
        alt_half_range_m: Half the grid altitude range (default ±1524m = ±5000ft)

    Returns:
        Number of elevation angles with beam centers within the altitude band
    """
    alt_min = target_alt_m - alt_half_range_m
    alt_max = target_alt_m + alt_half_range_m

    count = 0
    for angle in NEXRAD_ELEVATION_ANGLES:
        h = beam_height(distance_m, angle, radar_elevation_m)
        if alt_min <= h <= alt_max:
            count += 1
    return count


def score_radar_for_pirep(distance_m, target_alt_m, radar_elevation_m=0.0,
                          alt_half_range_m=1524.0):
    """
    Score a radar's suitability for observing a PIREP at a given altitude.
    Higher score = better candidate.

    Combines:
        - Number of elevation angles covering the altitude band (more = better)
        - Distance penalty (closer = better data quality due to beam width)

    Parameters:
        distance_m: Horizontal distance from radar to PIREP in meters
        target_alt_m: PIREP altitude in meters above sea level
        radar_elevation_m: Radar site elevation ASL in meters
        alt_half_range_m: Half the grid altitude range

    Returns:
        Score between 0.0 and 1.0 (0 = unusable, 1 = ideal)
    """
    if distance_m > MAX_RADAR_RANGE_M or distance_m < 1:
        return 0.0

    # Coverage score: fraction of elevation angles that hit the altitude band
    n_covering = count_covering_angles(
        distance_m, target_alt_m, radar_elevation_m, alt_half_range_m
    )
    coverage_score = n_covering / len(NEXRAD_ELEVATION_ANGLES)

    # Distance penalty: linear decay, reaches 0 at MAX_RADAR_RANGE_M
    distance_score = max(0.0, 1.0 - distance_m / MAX_RADAR_RANGE_M)

    # Weighted combination — coverage matters more than closeness
    return 0.7 * coverage_score + 0.3 * distance_score


def get_num_candidates(altitude_ft):
    """
    Determine how many candidate radars to consider based on PIREP altitude.
    Higher altitudes need more candidates since fewer nearby radars will
    have beam coverage.

    Parameters:
        altitude_ft: PIREP flight level in feet

    Returns:
        Number of candidate radars to query from KDTree
    """
    if altitude_ft < 15000:
        return 5   # Low altitude — 5 closest is fine
    elif altitude_ft < 30000:
        return 10  # Mid altitude — need some options
    else:
        return 20  # High altitude — cast a wide net
