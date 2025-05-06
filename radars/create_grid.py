# create_grid.py
# This python file exports the function create_grid for use in other scripts
# which can create a gridded lat/lon/alt XArray given radar(s) and ranges
# in which to create the grid around a point of origin
# Author: Sam Hecht
# Date: 2/26/25

# Import useful libraries
import numpy as np
import sys
import os
from contextlib import contextmanager
from haversine import haversine, haversine_vector
import math
import inspect
from typing import Union
import xarray as xr
@contextmanager
def quiet():
    sys.stdout = sys.stderr = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

with quiet():
   import pyart


def get_barnes2_weights(dist2: np.typing.ArrayLike, r2: np.float32) -> np.typing.ArrayLike:
    """
    Calculates the barnes2 weights based on the distances from all points to
    the center of the grid cell (squared) and the maximum distance of any
    point from the center of the grid cell (squared)
    Parameters:
        dist2 - An array of distances squared where each distance represents a
                point's distance to the center of the grid cell
        r2    - A float representing the distance from the center of the cell
                to the furthest point within the cell
    Returns:
        A numpy array of weights calculated according to the barnes2 algorithm
    """
    return np.exp(-dist2 / (r2 / 4)) + 1e-5


def initialize_mask(min : np.int64, max : np.int64, values : np.typing.ArrayLike)\
    -> np.typing.ArrayLike:
    """
    Initializes a mask on the NDArray values that is True in all places
    where the value in values is >= min and < max
    Parameters:
        min -- The minimum value to include in the mask
        max -- The maximum value to include in the mask
        values -- An array of values to create a mask for

    Returns:
        A mask (boolean array of shape equal to values.shape) that is true
        where the value in values is min <= value < max
    """
    return (min <= values) & (values < max)

def initialize_masks(
    n: np.int64, 
    start: np.float64, 
    step: np.float64, 
    absolute_grid_center: np.float64, 
    values: np.typing.NDArray
) -> np.typing.NDArray:
    """
    Initializes the masks for the grid cells
    Parameters:
        n: The number of masks to initialize, i.e., the number of pts in this
            dimension of the grid
        start: The start value for the mask (center of the 1st cell)
        step: The distance between the centers of 2 grid cells
        absolute_grid_center: The absolute center of the grid (e.g., if
            this is being called for longitude, then it should be the longitude
            value of the center of the grid)
        values: The values to compare to see whether they are within the cell
    Returns:
        A numpy NDArray of size (n, *(values.shape)) which contains
            n masks for the given inputs
    """
    masks = np.empty((n,*(values.shape)), dtype=bool)
    for i in range(n):
        center_of_cell = start + step * i
        max_of_cell = center_of_cell + step / 2
        min_of_cell = center_of_cell - step / 2

        # Finds the absolute values (e.g., 37.82 vs. -0.125)
        absolute_max_of_cell = max_of_cell + absolute_grid_center
        absolute_min_of_cell = min_of_cell + absolute_grid_center

        # Computes the mask
        masks[i] = initialize_mask(absolute_min_of_cell, absolute_max_of_cell, values)

    return masks

def create_grid(
    radars : Union[pyart.core.Radar, tuple[pyart.core.Radar]], 
    grid_shape : tuple[int, int, int],
    alt_range : tuple[float, float],
    lat_range : tuple[float, float], 
    lon_range : tuple[float, float],
    grid_origin : tuple[float, float, float], 
    fields : list[str] = ["reflectivity"],
    map_roi : bool = False,
    verbose : bool = False
) -> dict:
    """
    Given certain pyart radar object(s), creates an altitude/latitude/longitude grid
    about grid_origin of shape grid_shape where each cell represents the
    weighted average of the values for each field given of the radar object

    Parameters:
        radars -- The pyart radar(s) to create the grid with data from
        grid_shape -- (z, y, x) A 3-tuple representing the desired output shape
            of the grid returned from this function
        alt_range -- The minimum and maximum altitude offset (in meters)
            to grid around the grid_origin
        lat_range -- The minimum and maximum latitude offset (in degrees)
            to grid around the grid_origin
        lon_range -- The minimum and maximum longitude offset (in degrees)
            to grid around the grid_origin
        grid_origin -- (alt, lat, lon) A 3-tuple representing the origin of
            the grid. alt should be in meters and lat/lon in degrees
        fields -- A list of all fields of the radar object to grid
        map_roi -- A bool indicating whether or not to return the radius of
            influence used for each grid cell as part of the output grid
            (note that these roi's are not actually used for gridding, but
            rather only for the weighted average)
        verbose -- A bool indicating we should print verbose logging messages

    Returns:
        An XArray where the coordinates are "alt", "lat", and "lon" and 
        the data represents the calculated values for the specified fields
        in each coordinate location
    """

    
    if isinstance(radars, pyart.core.Radar):
        radars = (radars,)

    if len(radars) == 0:
        raise ValueError("Length of radars tuple cannot be zero")

    # Define a print function for verbose printing in this function
    def my_print(*objs, **kwargs):
        frame = inspect.currentframe().f_back
        line_number = frame.f_lineno
        file_name = os.path.basename(inspect.getfile(frame))
        print(f"    {file_name}:{line_number}", *objs, **kwargs)
        
    verboseprint = my_print if verbose else lambda *a, **k: None

    # Unpack the parameters
    grid_origin_alt, grid_origin_lat, grid_origin_lon = grid_origin # Attempting to unpack into 3 variables
    alt_start, alt_stop = alt_range
    lat_start, lat_stop = lat_range
    lon_start, lon_stop = lon_range
    n_alt, n_lat, n_lon = grid_shape

    absolute_lon_start = lon_start + grid_origin_lon
    absolute_lat_start = lat_start + grid_origin_lat
    absolute_alt_start = alt_start + grid_origin_alt

    absolute_lon_stop = lon_stop + grid_origin_lon
    absolute_lat_stop = lat_stop + grid_origin_lat
    absolute_alt_stop = alt_stop + grid_origin_alt
    
    nfields = len(fields)
    nradars = len(radars)
    ngates_per_radar = [r.fields[fields[0]]["data"].size for r in radars]
    total_gates = np.sum(ngates_per_radar)

    # Create output array
    verboseprint(f"Creating output grids of shape: ({n_alt}, {n_lat}, {n_lon})")
    grid_data = np.empty((n_alt, n_lat, n_lon, nfields), dtype=np.float64)

    # If the user requests the rois, create an array for them
    if map_roi:
        roi = np.empty((n_alt, n_lat, n_lon), dtype=np.float64)
    
    # Create a mask for just the gates in the range specified on input that
    # are non-null
    within_range_mask = np.empty(total_gates, dtype=bool)
    filtered_arr_length = 0
    filtered_offsets = np.zeros(nradars + 1, dtype=np.int32)
    unfiltered_arr_length = 0
    unfiltered_offsets = np.zeros(nradars + 1, dtype=np.int32)
    for i, radar in enumerate(radars):
        # Create gatefilter to remove all faulty values
        # https://arm-doe.github.io/pyart/API/generated/pyart.filters.moment_based_gate_filter.html
        gatefilter = pyart.filters.moment_based_gate_filter(radar)
        size = gatefilter.gate_included.size
        # Create mask to represent data that is inside the ranges we expect
        within_range_mask[unfiltered_arr_length:unfiltered_arr_length + size] = \
            (gatefilter.gate_included.ravel()) &\
            initialize_mask(absolute_lon_start, absolute_lon_stop, radar.gate_longitude['data'].ravel()) &\
            initialize_mask(absolute_lat_start, absolute_lat_stop, radar.gate_latitude ['data'].ravel()) &\
            initialize_mask(absolute_alt_start, absolute_alt_stop, radar.gate_altitude ['data'].ravel())

        # Update the length of the filtered and unfiltered arrays and the offsets
        # we'll need for further initialization
        filtered_arr_length += np.count_nonzero(within_range_mask[unfiltered_arr_length:unfiltered_arr_length + size])
        unfiltered_arr_length += size
        unfiltered_offsets[i + 1] = unfiltered_arr_length
        filtered_offsets[i + 1] = filtered_arr_length
    

    verboseprint(f"Filtering {total_gates} values to {filtered_arr_length} values in the grid")

    if filtered_arr_length == 0:
        verboseprint("No values found in range. Returning empty dataset")
        return xr.Dataset()
    
    # Store the data in arrays of filtered_arr_length ahead of time
    gate_lon = np.empty(filtered_arr_length, dtype=np.float32)
    gate_lat = np.empty(filtered_arr_length, dtype=np.float32)
    gate_alt = np.empty(filtered_arr_length, dtype=np.float32)
    field_data = np.empty((nfields, filtered_arr_length), dtype=np.float32)
    # For each radar, add the data for that radar to the 1D array gate_xxx
    for i, radar in enumerate(radars):
        # Get the mask for the current radar
        curr_radar_mask = within_range_mask[unfiltered_offsets[i]:unfiltered_offsets[i + 1]]

        # Get the start and stop idx of the filtered data in the gate_xxx arrs
        start_idx, end_idx = (filtered_offsets[i], filtered_offsets[i+1])
        gate_lon[start_idx:end_idx] = radar.gate_longitude['data'].ravel()[curr_radar_mask]
        gate_lat[start_idx:end_idx] = radar.gate_latitude['data'].ravel()[curr_radar_mask]
        gate_alt[start_idx:end_idx] = radar.gate_altitude['data'].ravel()[curr_radar_mask]
        for j, f in enumerate(fields):
            field_data[j][start_idx:end_idx] = radar.fields[f]['data'].ravel()[curr_radar_mask]


    # Find center of initial grid cell
    alt_step = (alt_stop - alt_start) / n_alt
    center_alt_start = alt_start + alt_step / 2  # on edge, so move start to center

    lat_step = (lat_stop - lat_start) / n_lat
    center_lat_start = lat_start + lat_step / 2  # on edge, so move start to center

    lon_step = (lon_stop - lon_start) / n_lon
    center_lon_start = lon_start + lon_step / 2  # on edge, so move start to center

    verboseprint("Initializing masks to find which data resides in which cells")
    lon_masks = initialize_masks(n_lon, center_lon_start, lon_step, grid_origin_lon, gate_lon)
    lat_masks = initialize_masks(n_lat, center_lat_start, lat_step, grid_origin_lat, gate_lat)
    alt_masks = initialize_masks(n_alt, center_alt_start, alt_step, grid_origin_alt, gate_alt)
    most_of_mask = np.empty(filtered_arr_length, dtype=bool)

    num_nan = 0
    verboseprint("Beginning to calculate grid cells")
    for iz, iy, ix in np.ndindex(n_alt, n_lat, n_lon):
        # Calculate the grid point
        lon = center_lon_start + lon_step * ix
        lat = center_lat_start + lat_step * iy
        alt = center_alt_start + alt_step * iz

        lon_max = lon + lon_step / 2
        lat_max = lat + lat_step / 2
        alt_max = alt + alt_step / 2

        # Finds the absolute lon, lat and altitude (e.g., 37.82 vs. -0.125)
        absolute_lon = lon + grid_origin_lon
        absolute_lat = lat + grid_origin_lat
        absolute_alt = alt + grid_origin_alt

        absolute_lon_max = lon_max + grid_origin_lon
        absolute_lat_max = lat_max + grid_origin_lat
        absolute_alt_max = alt_max + grid_origin_alt

        # Determine distance to furthest point in grid (corner)
        xy_dist = (haversine((absolute_lat, absolute_lon), (absolute_lat_max, absolute_lon_max), unit='m'))

        # Store the total distance squared for Barnes2 weighting algorithm later
        r2 = xy_dist ** 2 + (absolute_alt_max - absolute_alt) ** 2
        if map_roi:
            roi[iz, iy, ix] = math.sqrt(r2)
        
        # If the x index has reset to 0, recompute most of the mask as at least
        # one of iy or iz has changed (optimization)
        if ix == 0:
            most_of_mask = lat_masks[iy] & alt_masks[iz]

        # Compute the mask for which indices are in the cell
        in_cell_mask = lon_masks[ix] & most_of_mask

        total_pts = 0
        # If there are any points inside the current cell
        if np.any(in_cell_mask):
            # Store the relevant lats, lons, and alts for the points in cell
            rel_gate_lats = gate_lat[in_cell_mask]
            rel_gate_lons = gate_lon[in_cell_mask]
            rel_gate_alts = gate_alt[in_cell_mask]

            # Determine how many points are in the cell
            total_pts = rel_gate_lats.shape[0]

            # Create an array of the lat/lon points in the cell, e.g., if the
            # cell contains the lat pts [1, 2] and lon pts [3, 4], this results
            # in the array: [[1, 3], [2, 4]]
            lat_lon_pts_in_cell = np.column_stack((rel_gate_lats, rel_gate_lons))
            
            # Store the center of the cell (lat, lon) in an array
            center_of_cell = np.full((total_pts, 2), (absolute_lat, absolute_lon), dtype=np.float64)
            
            # Calc the xy distance from the center to all pts in cell (in m)
            pt_xy_dist = haversine_vector(center_of_cell, lat_lon_pts_in_cell, unit='m')
            
            # Calc the total distance (incl. vertical) for pts in cell w/ pythag
            pt_total_dist = np.sqrt(pt_xy_dist ** 2 + (rel_gate_alts - absolute_alt) ** 2)
            
            # Store the dists^2 for use in the Barnes2 weighting algo
            dist2 = pt_total_dist ** 2

            # Calculate the weights using Barnes2
            weights = get_barnes2_weights(dist2, r2)
            for i, f in enumerate(fields):
                grid_data[iz, iy, ix, i] = np.average(field_data[i][in_cell_mask], weights=weights, axis=0)
        else:
            grid_data[iz, iy, ix] = np.nan
            num_nan += 1

        # (Possibly) print useful logging message about current cell
        # val_str = ', '.join([f"{field}: {val:>6.2f}" for field, val in zip(fields, grid_data[iz, iy, ix].tolist())])
        # verboseprint(f"Cell ({ix:>2}, {iy:>2}, {iz:>2}): pts in cell: {total_pts:>3} | value(s): [{val_str}] | (Lon: {absolute_lon:.2f}°, Lat: {absolute_lat:.2f}°, Alt: {absolute_alt:.2f}m)")
    verboseprint(f"Finished calculating values for grid cells, {num_nan}/{n_alt * n_lon * n_lat} are nan")
    verboseprint("Converting grid to dictionary")

    # Converts the grid to a dictionary with keys for fields
    grids = {f: (["alt", "lat", "lon"], grid_data[..., i]) for i, f in enumerate(fields)}
    if map_roi:
        grids["ROI"] = (["alt", "lat", "lon"], roi)

    verboseprint(f"Finished creating dictionary with keys: {list(grids.keys())}, now converting to XArray")
    # Create coordinate arrays
    altitudes  = np.linspace(center_alt_start, center_alt_start + alt_step * (n_alt - 1), n_alt) + grid_origin_alt
    longitudes = np.linspace(center_lon_start, center_lon_start + lon_step * (n_lon - 1), n_lon) + grid_origin_lon
    latitudes  = np.linspace(center_lat_start, center_lat_start + lat_step * (n_lat - 1), n_lat) + grid_origin_lat

    # Create an xarray Dataset where the data_vars represent the different
    # grids and the coordinates represent altitude, latitude, and longitude
    xarray_grid = xr.Dataset(
        data_vars=grids,
        coords={
            "alt": altitudes,
            "lat": latitudes,
            "lon": longitudes,
        },
    )

    verboseprint(xarray_grid)
    verboseprint("Finished creating xarray grid, returning from function now...")

    return xarray_grid
