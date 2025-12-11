# NLP2025
NLP Masters Course Exam Project

In order to run the Generic_LLM_response.py, ensure you have a txt-file with a personal API key in order to generate responses. But for the purpose of running the analysis, just use the generated data (i.e. ....) and go from there.
## Generating LLM responses

### Add OpenAI Token
In order to run the Generic_LLM_response.py, ensure you have a OpenAI API-key to generate prompt responses. 
Create a txt-file called .env (with no file-type-name shown) and place it in the folder with the script. This file should not be pushed to GitHub. 

ORRRR Create a file named api_key.txt in the project folder and paste in your API key.

## ELEPHANT
- you need a key.txt file at the NLP2025 level with your API key that begins with sk-proj-
- To run the sycophancy_scorer.py script for all our context-pair responses use the following layout in the run.sh script for the python run section:
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

### Have access to GPUs
In order to run Finetuned_LLM_response.py you need GPUs. It take a long time so be patient. 

## File overview
### data
20200325_counsel_chat.csv = original downloaded name of data from huggingface - scraped from counsel chat

GPT_responses.csv = generated from generic LLM

HF_responses.csv = generated from huggingface model

Human_responses.csv = the filtered data from Original_Human_responses.csv - there's one human response per context, selected randomly from the responses with the most upvotes

Original_Human_responses.csv = same file as 20200325_counsel_chat.csv, but renamed for simplicity

context_ID.csv = the human responses post cleaning (with only context and ID pairs) - made from Human_responses.csv, but just with the responses removed - used to avoid data overlap if the human responses were presented along with the contexts. 

final_df.csv = the final cleaned df with all types of responses per context. Merged from GPT_responses.csv, context_Id.csv, HF_responses.csv, Human_responses.csv

#### emb

### src
DataCleaning.py - find top amount of upvotes, samples one random of them, creates files, renames and reorders columns
- produces context_ID.csv and Human_responses.csv

Finetuned_LLM_response.py
Generic_LLM_response.py
- write this in terminal to make it run:
          ~/anaconda3/bin/python Generic_LLM_response.py

Embeddings.py - merges files to generate final_df.csv based on context and id
loads encoder + produces embeddings and saves them to emb-folder in data folder. The output is numpy arrays, i.e. matrices

Lexical_features.ipynb

CosineSimilarities.py
computes row-wise cosine similarity and average cosine similarity for both on-topic column pairs and human-likeness column pairs + LOTS of plots <33

# Data Preparation
DataCleaning.py
Input: Original_Human_responses.csv
Outputs:
- Human_responses.csv
- context_ID.csv

Generic_LLM_responses.py
Input: context_ID.csv
Requires: API_key.txt (user must create this file with their own API key)
Output: GPT_responses.csv


# Lexical Analysis
Lexical_analysis.py
Input: final_df.csv
Intermediate Output: lexical_analysis.csv
Final Output: lexical_summary.csv

Lexical_plot.py
Input: lexical_summary.csv
Outputs:
- CTTR_density.png
- MTLD_density.png


# Emotional Alignment Analysis
EmotionClassification.py
Input: final_df.csv
Output: emotion_analysis.csv

EmotionStats.py
Input: emotion_analysis.csv
Outputs:
- Prints Wilcoxon test results in terminal
- emotion_distribution.png


# Counsel Chat Copyright notice :))
MIT License

Copyright (c) 2020 nbertagnolli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



