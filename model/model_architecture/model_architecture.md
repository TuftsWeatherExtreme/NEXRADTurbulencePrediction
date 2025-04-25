# Model Architecture

## Linear Classification Model


#### Usage
This python file, regression_model.py should not be directly run, rather it should be run using the train_linear_model.py script on a high-performance computing node.

#### Overview

This model is rather simplistic and features a linear regression of all 2564 features, including the 4 linear features (altitude, latitude, longitude, time since last NEXRAD scan) and 2560 reflectivity values shaped in a 16x16x10 grid. This model will not perform very well, and exists more as a proof of concept. It is unlikely that a linear classifier will outperform a neural network given the complexity of this task.

## Hybrid Model

#### Usage
This python file, hybrid_model.py should not be directly run, rather it should be run using the train_hybrid_model.py script on a high-performance computing node.


#### Motivation
Forecasting turbulence requires integrating both atmospheric grid data and location-specific data (such as coordinates and time). This model was designed to merge these two information sources via a hybrid architecture, combining a fully-connected neural network with a 3D convolutional neural network (3D CNN). The goal is to take full advantage of spatial structure in the atmospheric data while still leveraging linear features.

#### Overview
This script defines a PyTorch model called `HybridModel`, structured to handle two different types of input:

1. **Linear Features**: These include 4 scalar values—latitude, longitude, altitude, and time delta since last NEXRAD scan. This can be modified by changing the `NUM_LINEAR_FEATURES` variable.
2. **Gridded Features**: These represent 3D structured reflectivity values, shaped as `(altitude, latitude, longitude)` = `(10, 16, 16)` for a total of 2,560 values.

Each training input is expected to be a tensor of shape `(batch_size, 2564)`, where:
- The first 4 values correspond to linear features,
- The remaining 2,560 are reshaped into a 3D grid and passed through the CNN branch.

### 3x3x3 kernel

In our 3D convolutional layers, we use a `3×3×3` kernel. You can think of a kernel like a small window that slides through the 3D grid of atmospheric data—looking at one small cube of data at a time. In this case, the cube is 3 units wide, 3 units tall, and 3 units deep (across latitude, longitude, and altitude).

Using a 3×3×3 kernel helps the model detect local patterns in the atmosphere—like clusters of high reflectivity or sudden changes. Increasing the kernel size could make the model notice more overaching patters, but risks losing specificity with respect to certain regions. Decreasing the kernel size may have the opposite effect, where small local trends are emphasized over wider turbulence trends.

#### Output
The final output is a 10-class prediction corresponding to turbulence severity levels. This assumes a 0–9 scale for negative turbulence, the 7 turbulence classes and an extra allowance of two categories for plane weight scaling. This can be modified by changing the variable `NUM_CLASSES_TO_LEARN`.
