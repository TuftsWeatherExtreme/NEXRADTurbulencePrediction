# hybrid_model_1_out.py
# Team Celestial Blue
# Spring 2025

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
# from https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html

NUM_CHANNELS = 1 # This corresponds to the number of fields of the gridded input
NUM_CLASSES_TO_LEARN = 10 # Final Output should be projected to this
SMALLER_CLASSES_TO_LEARN = 6

N_ALT = 10
N_LAT = 16
N_LON = 16

NUM_GRID_FEATURES = N_ALT * N_LAT * N_LAT # Number of grid features
NUM_LINEAR_FEATURES = 4 # lat long alt delta_t 
NUM_FIELDS = 1

class HybridModel1Out(nn.Module):
    def __init__(self):
        super(HybridModel1Out, self).__init__()

        # ReLU: f(x) = max(0,x) 
        
        # Fully connected branch for the first 4 features
        first_linear_layer_output = second_linear_layer_input = 8 # CHOSEN BY CHAT
        second_linear_layer_output = 2 # CHOSEN BY CHAT
        self.fc_branch = nn.Sequential(
            nn.Linear(NUM_LINEAR_FEATURES, first_linear_layer_output), # Cast up to 8 (arbitrarily) and then back down
            nn.ReLU(),
            nn.Linear(second_linear_layer_input, second_linear_layer_output),
            nn.ReLU()
        )
        
        # 3D CNN branch for the last 2560 features reshaped to (16,16,10)
        self.conv_branch = nn.Sequential(
             # Kernel size of 3 means predicting on 3 x 3 x 3 for 27 weights per spot
             # Padding = same mmeans size of the output feature map is the same as the input feature map
            nn.Conv3d(in_channels=NUM_FIELDS, out_channels=8, kernel_size=3, padding='same'), 

            nn.ReLU(),
            nn.MaxPool3d(2),  # Reduce size to (8,8,5) using max from each 2x2x2 subsection
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv3d(16, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),


            nn.MaxPool3d(2),  # Reduce size to (4,4,2)
            nn.Flatten() # Convert final 3D feature maps into 1D vector
)

        conv_output_size = self._get_conv_output_shape()

        # Final classifier to combine together
        self.fc_final = nn.Sequential(
            # 16 * 4 * 4 * 2 comes from CNN layer
            # 16 output channels from size (4, 4, 2)
            # where does the + 64 come from?
            # nn.Linear(16 * 4 * 4 * 2 + second_linear_layer_output, 64),  # Concatenated size
            # nn.Linear(16 * 4 * 4 * 2 + 64, 64),  # Concatenated size


            nn.Linear(conv_output_size + second_linear_layer_output, 64),
            nn.ReLU(),
            
            nn.Linear(64, NUM_CLASSES_TO_LEARN),
            nn.Linear(NUM_CLASSES_TO_LEARN, 1) # Create 1 output from all 10 classes
        )

    
    def _get_conv_output_shape(self):
        dummy_input = torch.zeros(1, NUM_FIELDS, N_ALT, N_LAT, N_LON)
        out = self.conv_branch(dummy_input)
        return out.view(1, -1).size(1)


    def forward(self, x):
        # Split input:
        x_fc = x[:, :4]  # First 4 features
        x_cnn = x[:, 4:].reshape(-1, NUM_FIELDS, N_ALT, N_LAT, N_LON)  # Reshape last 2560 elements to (B, C, 10, 16, 16), the -1 means to infer based on batch size

        # Forward through both branches
        out_fc = self.fc_branch(x_fc)
        out_cnn = self.conv_branch(x_cnn)

        # Concatenate outputs and pass through final layers
        out = torch.cat((out_fc, out_cnn), dim=1)

        out = self.fc_final(out)
        return out.view(-1) # Flatten the output to 1D

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__== "__main__":
    testModel = HybridModel1Out()
    print(testModel)

