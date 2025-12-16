# NLP2025

## Table of Contents
1. [Project Description](#description)
2. [Project Structure](#structure)
3. [Installation](#installation)
4. [Dependencies](#dependencies)
5. [Data Source](#data)
6. [Usage](#usage)
7. [File and Pipeline Overview](#fileoverview)
8. [Counsel Chat Copyright notice](#copyrightnotice)

## Project Description
This project explores language usage from ChatGPT, a fine-tuned LLM and human therapists in the context of real-life mental health queries. Lexical features, cosine similarity to contexts, emotional alignemnt and sycophancy are the parameters explored.

Results and visualizations of each analysis are saved and stored.

## Repository Structure
```
NLP2025/
│
├── data/
│   ├── emb/
│         ├── emb-context.npy
│         ├── emb-ft_response.npy
│         ├── emb-gpt_response.npy
│         └── emb-human_response.npy
│   ├── 20200325_counsel_chat.csv
│   ├── FT_responses.csv
│   ├── ...
│   └── lexical_analysis.csv
│
├── output/
│   ├── CTTR_density.png
│   ├── Context_Similarity_overall_statistics.csv
│   ├── ...
│   └── lexical_summary.csv
│
├── src/
│   ├── 1_Lexical_analysis.py
│   ├── 2_Lexical_plot.py
│   ├── 3_Embeddings.py
│   ├── 4_Context_CosineSimilarities.py
│   ├── 5_EmotionClassification.py
│   ├── 6_EmotionStats.py
│   ├── 7_Elephant_sycophancy_scorers.py
│   ├── 8_Elephant_compare_to_human.py
│   ├── DataCleaning.npy
│   ├── Finetuned_LLM_response.npy
│   ├── Generic_LLM_response.npy
│   └── utils.py
│
├── .gitignore
├── README.md
├── key.txt
├── requirements.txt
├── run.sh
└── setup.sh
```

## Installation
To get started with this project, follow these steps:

1. change directories into the projects repository: /NLP2025
2. The data has already been created for you and can be found in the data/ folder but if you would like to re-generate the generic and fine-tuned LLM responses then follow the instructions under the Data Source section to generate the data yourself.
3. To run the following files: 7_Elephant_sycophancy_scorers.py and Generic_LLM_response.py, API keys are required. Follow the instructions in #dependencies to continue.
4. To run the Finetuned_LLM_responses.py, GPUs are required. Follow the instructions in #dependencies to continue.
5. In the terminal, run `./setup.sh` to set up the Python virtual environment and install all dependencies from requirements.txt . 
   - If you encounter a permission error, run `chmod +x setup.sh` and try again.
6. Run `./run.sh` to execute the pipeline.
   - Adjust the script if you are using custom paths.
   - If you encounter a permission error, run `chmod +x run.sh` and try again
   - Scripts can be run individually if required, as all outputs are already saved.
  

  
## Dependencies
A full list of package requirements and dependencies can be found in the requirements.txt file.

#### Generic_LLM_response.py
   - Create a txt-file called .env (with no file-type-name shown) and place it in the folder with the script. This file should not be pushed to GitHub.
   - OR Create a file named api_key.txt in the project folder and paste in your API key. --> RECOMMENDED.
   - Now return to the #installation instructions

#### Finetuned_LLM_response.py
   - A machine with GPUs is required for this file to run. It take a long time so be patient, even with the GPUs. We ran the script using 3 GPUs.
   - Now return to the #installation instructions.
  
#### 7_Elephant_sycophancy_scorers.py
   - You need a key.txt file at the NLP2025 level with your API key that begins with sk-proj-
   - To run the 7_Elephant_sycophancy_scorers.py script for all our context-pair responses use the following layout in the run.sh script for the python run section:
   - `python elephant/sycophancy_scorers.py
      --input_file data/Human_responses.csv
      --prompt_column Context
      --response_column Human_response
      --output_column_tag Human_sychophancy
      --output_file data/Human_validation.csv
      --validation --indirectness --framing
      --save_interval 10
      
      python elephant/sycophancy_scorers.py
      --input_file data/FT_responses.csv
      --prompt_column Context
      --response_column FT_response
      --output_column_tag FT_sychophancy
      --output_file data/FT_validation.csv
      --validation --indirectness --framing
      --save_interval 10
      
      python elephant/sycophancy_scorers.py
      --input_file data/GPT_responses.csv
      --prompt_column Context
      --response_column GPT_response
      --output_column_tag GPT_sychophancy
      --output_file data/GPT_validation.csv
      --validation --indirectness --framing
      --save_interval 10
      `
- Now return to the #installation instructions


## Data Source
The data can be found at the link below and the #copyrightnotice can be found at the bottom of this README.md:
(https://github.com/nbertagnolli/counsel-chat/tree/master/data)

_Steps to download the data:_
1. Download the data file called "20200325_counsel_chat.csv" from the above link
2. Add the above data file into the data/ folder in this repository
3. Rename the file to "Original_Human_responses.csv"
4. The final structure will appear like this:
   ```
    NLP2024/
    │
    ├── data/
    │   └── 20200325_counsel_chat.csv
   ```

In the case that other pre-generated data files have been deleted and nothing is found in the data/ folder, then execute the following instructions:

1. Follow the steps in the ##Installation pipeline above, until step 6.
2. In order to run the Generic_LLM_response.py, ensure you have a OpenAI API-key to generate prompt responses and enough GPUs to run the Finetuned_LLM_response.py script. Follow the instruction in ## Dependencies to implement this.

3. Add the following lines to the run.sh script under "# run the script":
    python src/DataCleaning.py
    python src/Finetuned_LLM_response.py
    python src/Generic_LLM_response.py

  The final run.sh script should look like this if you want to run the full pipeline with analyses as well:
  ` #!/usr/bin/bash
    
    # activate the environment
    source ./env/bin/activate

    # run the script
    python src/DataCleaning.py
    python src/Finetuned_LLM_response.py
    python src/Generic_LLM_response.py
    python src/1_Lexical_analysis.py
    python src/2_Lexical_plot.py
    python src/3_Embeddings.py
    python src/4_Context_CosineSimilarities.py
    python src/5_EmotionClassification.py
    python src/6_EmotionStats.py
    
    python elephant/7_Elephant_sycophancy_scorers.py
    --input_file data/Human_responses.csv
    --prompt_column Context
    --response_column Human_response
    --output_column_tag Human_sychophancy
    --output_file data/Human_validation.csv
    --validation --indirectness --framing
    --save_interval 10
    
    python elephant/7_Elephant_sycophancy_scorers.py
    --input_file data/FT_responses.csv
    --prompt_column Context
    --response_column FT_response
    --output_column_tag FT_sychophancy
    --output_file data/FT_validation.csv
    --validation --indirectness --framing
    --save_interval 10
    
    python elephant/7_Elephant_sycophancy_scorers.py
    --input_file data/GPT_responses.csv
    --prompt_column Context
    --response_column GPT_response
    --output_column_tag GPT_sychophancy
    --output_file data/GPT_validation.csv
    --validation --indirectness --framing
    --save_interval 10
    
    python src/8_Elephant_compare_to_human.py
  
    # close the environment
    deactivate

### Overview of data files

   - **20200325_counsel_chat.csv** = original downloaded name of data from huggingface - scraped from counsel chat
   - **FT_responses.csv** = generated from huggingface model using Finetuned_LLM_response.py script
   - **FT_validation.csv** = the FT_responses with their sycophancy metric scores
   - **GPT_responses.csv** = generated from generic LLM using Generic_LLM_response.py script
   - **GPT_validation.csv** = the GPT_responses with their sycophancy metric scores
   - **Human_responses.csv** = the filtered data from Original_Human_responses.csv - there's one human response per context, selected randomly from the responses with the most upvotes
   - **Human_validation.csv** = the Human_responses with their sycophancy metric scores
   - **Original_Human_responses.csv** = same file as 20200325_counsel_chat.csv, but renamed for simplicity
   - **context_ID.csv** = the human responses post cleaning (with only context and ID pairs) - made from Human_responses.csv, but just with the responses removed - used to avoid data overlap if the human responses were presented along with the contexts.
   - **final_df.csv** = the final cleaned df with all types of responses per context. Merged from GPT_responses.csv, context_Id.csv, HF_responses.csv, Human_responses.csv
   - **final_df_sycophancy.csv** = FT_validation.csv, GPT_validation.csv, Human_validation.csv merged
   - **lexical_analysis.csv** = results from lexical analysis saved to csv.
    `

## Usage
Double check the folder_path and output_path (defined under the "Application" section of the main script)are set correctly in the python scripts based on your directory structure.
The output will be saved in the `output/` folder inside the `NLP2025` directory, unless a different output_path is defined.


## File and Pipeline Overview

### Data Preparation and cleaning

**DataCleaning.py**

_Input:_
Original_Human_responses.csv

_Outputs:_
Human_responses.csv
context_ID.csv

**Generic_LLM_responses.py**

_Input:_
context_ID.csv

_Requires:_
API_key.txt (user must create this file with their own API key)

_Output:_
GPT_responses.csv

**Finetuned_LLM_responses.py**

_Input:_
context_ID.csv

_Requires:_
GPUs on the machine the script is run on. Preferably 2-3

_Output:_
FT_responses.csv

**1_Embeddings.py**

_Input:_
context_ID.csv
Human_responses.csv
GPT_responses.csv
FT_responses.csv

_Outputs:_
final_df.csv
emb-context.py
emb-ft_response.py
emb-gpt_response.py
emb-human_response.py

### Lexical Analysis

**2_Lexical_analysis.py**

_Input:_
final_df.csv

_Intermediate Output:_
lexical_analysis.csv

_Final Output:_
lexical_summary.csv

**3_Lexical_plot.py**

_Input:_
lexical_summary.csv

_Outputs:_
CTTR_density.png
MTLD_density.png

### Context Similarity Analysis

**4_Context_CosineSimilarities.py**

_Input:_
Human_responses.csv
embeddings from "data/emb/"

_Outputs:_
Context_similarity_cosine_similarity_kde_by_topic.png
Context_similarity_cosine_similarity_kde.png
Context_Similarity_overall_statistics.csv
Context_Similarity_topic_statistics.csv
Context_similarity_wilcoxon_results.csv

### Emotional Alignment Analysis

**5_EmotionClassification.py**

_Input:_
final_df.csv

_Output:_
emotion_analysis.csv

**6_EmotionStats.py**

_Input:_
emotion_analysis.csv

_Outputs:_
Prints Wilcoxon test results in terminal
emotion_distribution.png
emotion_similarity_overall_statistics.csv


### Sycophancy Analysis

**7_Elephant_sycophancy_scorers.py**
_Input Parameters:_
  For Human responses sycophancy scores:
    Human_responses.csv

  For Fine-tuned responses sycophancy scores:
    FT_responses.csv

  For GPT responses sycophancy scores:
    GPT_responses.csv

_Requires:_
key.txt (user must create this file with their own API key)

_Outputs:_
Human_validation.csv
FT_validation.csv
GPT_validation.csv

**8_Elephant_compare_to_human.py**
_Inputs:_
Human_validation.csv
GPT_validation.csv
FT_validation.csv

_Outputs:_
final_df_sycophancy.csv
elephant_metrics_results.csv
elephant_metrics_plot.png
elephant_metrics_differences.csv


## Counsel Chat Copyright notice
MIT License

Copyright (c) 2020 nbertagnolli

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


