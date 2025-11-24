#!/usr/bin/bash

# activate the environment
source ./env/bin/activate

# BERT
python src/Finetuned_LLM_response.py

# close the environment
deactivate