#!/usr/bin/bash

export OPENAI_API_KEY=$(cat ./key.txt)

# activate the environment
source ./env/bin/activate

# run the script
python /src/Lexical_analysis.py
  
# close the environment
deactivate