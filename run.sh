#!/usr/bin/bash

# activate the environment
source ./env/bin/activate

# BERT
python src/DataCleaning .py

# close the environment
deactivate