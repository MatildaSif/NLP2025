#!/usr/bin/bash

# activate the environment
source ./env/bin/activate

# BERT
python src/Embeddings.py

# close the environment
deactivate