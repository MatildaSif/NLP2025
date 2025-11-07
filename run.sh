#!/usr/bin/bash

# activate the environment
source ./env/bin/activate

# BERT
python src/TransformerModel.py

# close the environment
deactivate