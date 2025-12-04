#!/usr/bin/bash

export OPENAI_API_KEY=$(cat ./key.txt)

# activate the environment
source ./env/bin/activate

# BERT
python elephant/sycophancy_scorers.py \
  --input_file data/Human_responses.csv \
  --prompt_column Context \
  --response_column Human_response \
  --output_column_tag Human_sychophancy \
  --output_file data/Human_validation.csv \
  --validation --indirectness --framing \
  --save_interval 10

python elephant/sycophancy_scorers.py \
  --input_file data/FT_responses.csv \
  --prompt_column Context \
  --response_column FT_response \
  --output_column_tag FT_sychophancy \
  --output_file data/FT_validation.csv \
  --validation --indirectness --framing \
  --save_interval 10

  python elephant/sycophancy_scorers.py \
  --input_file data/GPT_responses.csv \
  --prompt_column Context \
  --response_column GPT_response \
  --output_column_tag GPT_sychophancy \
  --output_file data/GPT_validation.csv \
  --validation --indirectness --framing \
  --save_interval 10
  
# close the environment
deactivate