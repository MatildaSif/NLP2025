#!/usr/bin/bash

# activate the environment
source ./env/bin/activate

# BERT
python src/SentimentClassification.py

# close the environment
deactivate