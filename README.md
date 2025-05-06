# NEXRADTurbulencePrediction

## Overview
This repository hosts code and documentation originally created in the 2024-25 academic year by a team of Tufts Students, as part of a capstone project sponsored by WeatherExtreme Ltd. The aim of the project is to build a model that predicts clear-air turbulence (CAT) and displays that information on a website with a clean user interface.

## Directories

### Model
The model directory contains our models, including a hybrid model that is, as of May 2025, the best performing option. The model directory also contains our dataloader, which efficiently feeds a large amount of data into our model.

### Data Preprocessing
The data preprocessing directory contains all the files we used to put together our datasets. This includes all the work we did to scrape PIREPs (our repsonse variable), NEXRAD data (our input data) and plane weights (used for scaling our response variable)

### Additional Routes
This directory contains all the work we did exploring methods and datasets that didn't make it into our model for various reasons. since incoorporating the data in this folder into the model may significantly improve performance, all the reasons we were unable to use the data are documented, with the idea that if they were solved, new helpful data would be unlocked. 

### HPC Scripts
This directory contains executable files which run the models and scripts we wrote. The executables here are meant to be run on high-performance compute nodes.

## Contacts
For any questions, you can reach out to the following people:  
Leo Kaluzhny - leo.kaluzhny@gmail.com  
Sam Hecht - shechtor18@gmail.com  
Julia Zelevinsky - julia.zelevinsky@gmail.com  
Richard Diamond - richarddiamond.3@gmail.com   
Charlotte Versavel - charlotteversavel21@gmail.com  
