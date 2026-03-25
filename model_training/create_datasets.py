# create_datasets.py
# Authors: Team Celestial Blue
# Spring 2025
# Overview: 
# Usage: python create_datasets.py <dataloder_name> [existing_dataloader]
#       dataloader_name: Desired name for dataloader to generate
#       existing_dataloader: Optionally can include dataloader to add to

import torch 
from dataloader_class import RadarDataLoader
import sys
import os

def usage():
    print("Usage: python create_datasets.py [dataloder_name] ([existing_dataloader])")
    print("dataloader_name: Desired name for dataloader to generate")
    print("existing_dataloader: Optionally can include dataloader to add to")
    exit(1)

def main(): 
    # If an existing dataloader is specified, initialize old_data to include data
    if len(sys.argv) == 3:
        print(f"Specified existing dataloder: {sys.argv[2]}, so loading before appending new data...")
        old_data = torch.load(sys.argv[2], weights_only=False)
        print("Loaded old data")
    elif len(sys.argv) == 2:
        old_data = None
    else:
        usage()

    # Trying new script location - WORKING ON THIS
    # script_location = os.path.dirname(sys.argv[0])
    script_location = os.path.dirname(os.path.abspath(__file__))

    # Create an instance of the custom dataset 
    print("create_datasets.py: Initializing RadarDataLoader")
    dataset = RadarDataLoader(dir_path = f"{script_location}/../model_inputs/compressed", old_data=old_data)

    # Save the dataset and dataloader to specififed location
    torch.save(dataset, sys.argv[1])

if __name__ == "__main__":
    main()
