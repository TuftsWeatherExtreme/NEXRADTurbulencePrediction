# understand_hybrid.py
# Team Celestial Blue
# Spring 2025
# Last Modified: 05/06/2025
# This file prints the layers of the hybrid model to help someone new to our
#   project (or someone old to our project :)) familiarize themselves with
#   the model architecture


import torch 
import torch.nn as nn
from hybrid_model import HybridModel

model = HybridModel()

print("Note: There is actually one higher dimension on all this data for the " \
"batch size, but it has been omitted from this script for simplicity.")

conv_input = torch.randn(1, 1, 16, 16, 10)

def get_shape_as_str(input):
    return str(list(input[0].shape))

print("Conv input shape: ", get_shape_as_str(conv_input))

print("Conv Branch:")
for (step, layer) in enumerate(model.conv_branch):
    conv_input = layer(conv_input)
    print(f"After step: {step}| Shape: {get_shape_as_str(conv_input):<18}| Function: {layer}")

fc_input = torch.randn(1, 4)
print("FC input shape: ", get_shape_as_str(fc_input))

print("FC Branch:")
for (step, layer) in enumerate(model.fc_branch):
    fc_input = layer(fc_input)
    print(f"After step: {step}| Shape: {get_shape_as_str(fc_input):<18}| Function: {layer}")

conv_output = conv_input
fc_output = fc_input

final_branch_input = torch.cat((fc_output, conv_output), dim=1)
print("Final branch input shape: ", get_shape_as_str(final_branch_input))

print("FC Final Branch:")
for (step, layer) in enumerate(model.fc_final):
    final_branch_input = layer(final_branch_input)
    print(f"After step: {step}| Shape: {get_shape_as_str(final_branch_input):<18}| Function: {layer}")

final_output_shape = final_branch_input
print("Final output shape: ", get_shape_as_str(final_output_shape))

