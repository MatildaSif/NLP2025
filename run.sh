#!/usr/bin/bash

# activate the environment
source ./env/bin/activate

# BERT
python src/CosineSimilarities.py

# close the environment
deactivate