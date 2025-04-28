#!/bin/bash
# Simple script to collapse all of the csv files into 1 large csv file

DIRNAME=$(dirname "$0")
CLEAN_PIREP_PATH="$DIRNAME/clean_pirep_data"

echo "Collapsing cleaned pirep data into file: $CLEAN_PIREP_PATH/cleaned_pireps.csv"

all_files=("$CLEAN_PIREP_PATH/"*/*)
first_file="${all_files[0]}"

# Put the header in once
head -n 1 "$first_file" > "$CLEAN_PIREP_PATH/cleaned_pireps.csv"
tail -q -n +2 "$CLEAN_PIREP_PATH"/*/* >> "$CLEAN_PIREP_PATH/cleaned_pireps.csv"

echo "Finished collapsing pirep data into file: $CLEAN_PIREP_PATH/cleaned_pireps.csv"

echo "Created file with the following amount of lines:"
wc -l "$CLEAN_PIREP_PATH/cleaned_pireps.csv"
