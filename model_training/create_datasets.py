# create_datasets.py
# Team Celestial Blue
# Spring 2025

import torch 
from torch.utils.data import DataLoader 
from dataloader_class import RadarDataLoader
import sys


def usage():
    print("Usage: python create_datasets.py [dataloder_name] ([existing_dataloader])")
    print("dataloader_name: Desired name for dataloader to generate")
    print("existing_dataloader: Optionally can include dataloader to add to")
    exit(1)


def main(): 
    # If an existing dataloader is specified, initialize old_data to include data
    if len(sys.argv) == 3:
        print(f"Specified existing dataloder: {sys.argv[2]}, so loading before appending new data...")
        old_data = torch.load('big_dataloader.pth', weights_only=False)
        print("Loaded old data")
    elif len(sys.arv == 2):
        old_data = None
    else:
        usage()

    # Create an instance of the custom dataset 
    print("  create_datasets.py: Initializing RadarDataLoader")
    dataset = RadarDataLoader(dir_path = "./model_inputs/compressed", old_data=old_data)

    # Save the dataset and dataloader to disk
    torch.save(dataset, sys.argv[1])
