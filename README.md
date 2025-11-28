# NLP2025
NLP Masters Course Exam Project

In order to run the Generic_LLM_response.py, ensure you have a txt-file with a personal API key in order to generate responses. But for the purpose of running the analysis, just use the generated data (i.e. ....) and go from there.
## Generating LLM responses

### Add OpenAI Token
In order to run the Generic_LLM_response.py, ensure you have a OpenAI API-key to generate prompt responses. 
Create a txt-file called .env (with no file-type-name shown) and place it in the folder with the script. This file should not be pushed to GitHub. 

ORRRR Create a file named api_key.txt in the project folder and paste in your API key.

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



