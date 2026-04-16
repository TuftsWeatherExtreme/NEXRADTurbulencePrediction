# dataloader_class.py
# Authors: Team Celestial Blue
# Spring 2025
# Overview: Defines a dataloader class to store all model inputs with fast access time
# Note: assumes that filepath is to a directory of tar xz compressed model inputs

import torch 
import glob
import tarfile
from torch.utils.data import Dataset
import os
import xarray as xr
import numpy as np
import shutil


def decompress_tar_xz(filepath, extract_path):
    """
    Decompresses a .tar.xz file.

    Args:
        filepath (str): The path to the .tar.xz file.
        extract_path (str, optional): The directory to extract the contents to
    """
    try:
        with tarfile.open(filepath, 'r:xz') as tar:
            tar.extractall(extract_path)
        print(f"Successfully extracted '{filepath}' to '{extract_path}'")
    except FileNotFoundError:
        print(f"Error: File not found: '{filepath}'")
    except tarfile.ReadError as e:
         print(f"Error: Could not open '{filepath}' as a tar archive: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


class RadarDataLoader(Dataset):
  
  def __init__(self, dir_path, old_data=None): 
    """
    __init__() loads all the netcdf files from compressed model inputs
    Note that net cdf files must be generated using radar_data_to_model_input
    and be compressed using tar xz compression. 

    There is optionally the choice to add to an existing dataloader (using old_data)
    or can start fresh if None is specified.
    """
    print("Calling init")
    self.data = [] 

    # If we are given old_data, start by loading this
    if (old_data is not None):
        self.data = [radar_data for radar_data in old_data]
        print(type(self.data))
        print(f"Loaded {len(self.data)} old data points")
    count = 0
    tarfiles = glob.glob(f"{dir_path}/*.tar.xz")
    # WORKING ON THIS
    print("DEBUG tarfiles:", tarfiles)
    os.makedirs("decompressed", exist_ok=True)
    for tarfile in tarfiles:
        specific_dirname = os.path.join("decompressed", os.path.basename(tarfile).replace(".tar.xz", ""))
        print(f"Decompressing dir: {specific_dirname}.tar.xz")
        decompress_tar_xz(tarfile, "./decompressed")
        print("Finished decompressing! Time to add to dataloader")
        
        for filename in os.listdir(specific_dirname):
            filepath = os.path.join(specific_dirname, filename)
            print(f"DEBUG: found file: {filename}, is_file: {os.path.isfile(filepath)}")
            if os.path.isfile(filepath):
                try:
                    print(f"DEBUG: opening {filename}")
                    nc_file = xr.open_dataset(filepath)
                    print(f"DEBUG: opened, attrs: {list(nc_file.attrs.keys())}")
                    # Here we are instead of just turning every attribute value 
                    # into numbers for the model, we are just checking if the file
                    # has LAT, LON, ALT, DELTA_T, and TURB. If it does, use only 
                    # those for the four numbers and the label for the model. 
                    # Don’t use the other attributes as model inputs.
                    # The reason why we are doing this is because we added 
                    # PIREP_TIME as text (a timestamp string) so the map / 
                    # GeoJSON can show when the report was.
                    a = nc_file.attrs
                    if all(k in a for k in ("LAT", "LON", "ALT", "DELTA_T", "TURB", "IN_SIGMET")):
                        meta = np.array(
                            [float(a["LAT"]), float(a["LON"]), float(a["ALT"]), float(a["DELTA_T"]), float(a["IN_SIGMET"])],
                            dtype=np.float64,
                        )
                        label = float(a["TURB"])
                    else:
                        attrs_arr = np.array(list(a.values()))
                        meta = attrs_arr[:-1].astype(float)
                        label = float(attrs_arr[-1])
                    print(f"DEBUG: meta shape: {meta.shape}")
                    flattened_data = np.concatenate([nc_file[var].values.flatten() for var in nc_file.data_vars])
                    print(f"DEBUG: flattened_data shape: {flattened_data.shape}")
                    features = np.concatenate((meta, flattened_data))
                    print(f"DEBUG: label: {label}")
                    features = torch.tensor(features, dtype=torch.float32) 
                    features = torch.nan_to_num(features, nan=-32.0)
                    self.data.append((features, label))
                    print(f"DEBUG: appended successfully")
                    count += 1
                except Exception as e:
                    print(f"ERROR loading {filename}: {e}")

  def __len__(self): 
    """
    __len__() returns the number of rows in the dataset.
    """

    # This represents the number of model inputs in all of the compressed files
    return len(self.data) 

  def __getitem__(self, idx): 
    """
    __getitem__ accepts an index (idx) and retrieves the corresponding features
    and label for that input
    """

    return self.data[idx]


