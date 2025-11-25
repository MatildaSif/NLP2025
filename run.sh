#!/usr/bin/bash

# activate the environment
source ./env/bin/activate

# BERT
python src/CosineDistances.py

# close the environment
deactivate