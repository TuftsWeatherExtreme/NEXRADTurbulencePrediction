# dataloader_class.py
# Authors: Team Celestial Blue
# Spring 2025
# Overview: Defines a dataloader class to store all model inputs with fast access time
# Usage: python create_datasets.py <dataloder_name> [existing_dataloader]
#       dataloader_name: Desired name for dataloader to generate
#       existing_dataloader: Optionally can include dataloader to add to

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
    os.makedirs("decompressed", exist_ok=True)
    for tarfile in tarfiles:
        specific_dirname = os.path.join("decompressed", tarfile[3 + tarfile[2:].find("/"):2 + tarfile[2:].find(".")])
        print(f"Decompressing dir: {specific_dirname}.tar.xz")
        decompress_tar_xz(tarfile, "./decompressed")
        print("Finished decompressing! Time to add to dataloader")
        for filename in os.listdir(specific_dirname):
            filepath = os.path.join(specific_dirname, filename)
            if os.path.isfile(filepath):  # Ensure it's a file
                nc_file = xr.open_dataset(filepath)
                attrs_arr = np.array(list(nc_file.attrs.values()))

                # get attributes (lat,long,) from netcdf file and cast them to floats
                # does not keep last element because that is TURB, which is being used as label
                features = attrs_arr[:-1].astype(float)

                #flatten grid-like data from NEXRAD, reflectivity for instance
                flattened_data = np.concatenate([nc_file[var].values.flatten() for var in nc_file.data_vars])

                #concatenate attributes array with flatted grid-like data
                features = np.concatenate((features, flattened_data))
                
                # Single Category Encoding for pilot reported turbulence level
                label = int(attrs_arr[-1])

                features = torch.tensor(features, dtype=torch.float32) 
                
                # To account for all the cells with undetectable reflectivity, we set a reflectivity value out of range (-32 dBz)
                features = torch.nan_to_num(features, nan=-32.0)
                
                self.data.append((features, label))

                count += 1
                if count % 1000 == 0:
                    print(f'Finished {count} total files')
        
        print("Removing subcontents of directory: " + specific_dirname)

        shutil.rmtree(specific_dirname, ignore_errors=True)

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


